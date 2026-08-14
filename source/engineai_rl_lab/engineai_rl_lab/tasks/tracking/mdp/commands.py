from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def resolve_motion_files(manifest: str | os.PathLike[str]) -> list[str]:
    """Resolve a motion npz or a recursive ``files``/``exclude_files`` YAML manifest.

    Paths and glob patterns are relative to the YAML file containing them.  Included
    manifests contribute both their motions and their exclusions; exclusions are
    applied after the complete include tree has been expanded.
    """

    root = Path(manifest).expanduser().resolve()
    if root.suffix.lower() != ".yaml" and root.suffix.lower() != ".yml":
        if not root.is_file():
            raise FileNotFoundError(f"Motion file does not exist: {root}")
        return [str(root)]

    visiting: set[Path] = set()

    def _walk(path: Path) -> tuple[list[Path], list[str]]:
        path = path.resolve()
        if path in visiting:
            raise ValueError(f"Recursive motion manifest include detected at {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Motion manifest does not exist: {path}")
        visiting.add(path)
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        if not isinstance(document, dict):
            raise ValueError(f"Motion manifest {path} must contain a mapping")

        motions: list[Path] = []
        exclusions: list[str] = []
        manifest_keys = {"files", "exclude_files"}
        direct_mapping = not manifest_keys.intersection(document)
        if direct_mapping:
            if not all(isinstance(name, str) and isinstance(value, str) for name, value in document.items()):
                raise ValueError(
                    f"Motion mapping {path} must contain string motion-name to file-path entries"
                )
            file_entries = list(document.values())
            exclude_entries = []
        else:
            unknown_keys = set(document) - manifest_keys
            if unknown_keys:
                raise ValueError(f"Unknown keys in motion manifest {path}: {sorted(unknown_keys)}")
            file_entries = document.get("files", [])
            exclude_entries = document.get("exclude_files", [])
            if not isinstance(file_entries, list) or not isinstance(exclude_entries, list):
                raise ValueError(f"files and exclude_files in {path} must be lists")

        for entry in file_entries:
            candidate = Path(os.path.abspath(path.parent / os.fspath(entry)))
            # ScaleBFM's legacy direct mapping interpreted relative values from cwd.
            if direct_mapping and not candidate.exists() and not Path(entry).is_absolute():
                cwd_candidate = Path(os.path.abspath(os.fspath(entry)))
                if cwd_candidate.exists():
                    candidate = cwd_candidate
            if candidate.suffix.lower() in {".yaml", ".yml"}:
                child_motions, child_exclusions = _walk(candidate)
                motions.extend(child_motions)
                exclusions.extend(child_exclusions)
            else:
                motions.append(candidate)
        for entry in exclude_entries:
            candidate = Path(os.path.abspath(path.parent / os.fspath(entry)))
            if candidate.suffix.lower() in {".yaml", ".yml"} and candidate.is_file():
                _, child_exclusions = _walk(candidate)
                exclusions.extend(child_exclusions)
            else:
                exclusions.append(candidate.as_posix())
        visiting.remove(path)
        return motions, exclusions

    motions, exclusions = _walk(root)
    exact_exclusions = {pattern for pattern in exclusions if not any(char in pattern for char in "*?[")}
    glob_exclusions = [pattern for pattern in exclusions if pattern not in exact_exclusions]
    unique: list[str] = []
    seen: set[Path] = set()
    for motion in motions:
        motion_path = motion.as_posix()
        if motion in seen or motion_path in exact_exclusions or any(
            fnmatch(motion_path, pattern) for pattern in glob_exclusions
        ):
            continue
        if motion.suffix.lower() != ".npz":
            raise ValueError(f"Unsupported motion file in {root}: {motion}")
        seen.add(motion)
        unique.append(str(motion))
    if not unique:
        raise ValueError(f"Motion manifest {root} did not resolve to any .npz files")
    return unique


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        device: str = "cpu",
        *,
        joint_names: Sequence[str] | None = None,
        body_names: Sequence[str] | None = None,
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        with np.load(motion_file) as data:
            self.fps = data["fps"]
            file_joint_names = data["joint_names"].astype(str).tolist() if "joint_names" in data.files else None
            file_body_names = data["body_names"].astype(str).tolist() if "body_names" in data.files else None
            self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

            # Isaac Lab 3.x uses XYZW quaternions throughout. Motion files produced by the
            # pre-3.0 conversion scripts did not carry format metadata and stored WXYZ.
            quaternion_order = "wxyz"
            if "quaternion_order" in data.files:
                quaternion_order = np.asarray(data["quaternion_order"]).item()
                if isinstance(quaternion_order, bytes):
                    quaternion_order = quaternion_order.decode("ascii")
                quaternion_order = str(quaternion_order).lower()

        if self.joint_pos.shape != self.joint_vel.shape:
            raise ValueError(
                f"Motion file {motion_file!r} has mismatched joint_pos {tuple(self.joint_pos.shape)} and "
                f"joint_vel {tuple(self.joint_vel.shape)} shapes."
            )
        body_shape = self._body_pos_w.shape[:2]
        if any(
            array.shape[:2] != body_shape
            for array in (
                self._body_quat_w,
                self._body_lin_vel_w,
                self._body_ang_vel_w,
            )
        ):
            raise ValueError(f"Motion file {motion_file!r} has inconsistent body data axes.")
        if self._body_quat_w.shape[-1] != 4:
            raise ValueError(f"Motion file {motion_file!r} body_quat_w must have four quaternion components.")

        if file_joint_names is not None and len(file_joint_names) != self.joint_pos.shape[1]:
            raise ValueError(
                f"Motion file {motion_file!r} has {len(file_joint_names)} joint_names entries but "
                f"{self.joint_pos.shape[1]} joint samples."
            )

        joint_names = list(joint_names) if joint_names is not None else None
        if joint_names is not None and file_joint_names is not None:
            joint_indexes = self._resolve_name_indexes(file_joint_names, joint_names, "joint", require_complete=True)
            self.joint_pos = self.joint_pos[:, joint_indexes]
            self.joint_vel = self.joint_vel[:, joint_indexes]
        elif joint_names is not None and self.joint_pos.shape[-1] != len(joint_names):
            raise ValueError(
                f"Motion file {motion_file!r} has {self.joint_pos.shape[-1]} joints but the robot has "
                f"{len(joint_names)}. Add joint_names metadata so the axes can be mapped safely."
            )
        self.joint_names = joint_names if joint_names is not None else file_joint_names

        if file_body_names is not None and len(file_body_names) != self._body_pos_w.shape[1]:
            raise ValueError(
                f"Motion file {motion_file!r} has {len(file_body_names)} body_names entries but "
                f"{self._body_pos_w.shape[1]} body samples."
            )

        if body_names is not None and file_body_names is not None:
            body_names = list(body_names)
            resolved_body_indexes = self._resolve_name_indexes(
                file_body_names, body_names, "body", require_complete=False
            )
            self._body_indexes = torch.tensor(resolved_body_indexes, dtype=torch.long, device=device)
        else:
            # Legacy files omitted body names and used the PhysX articulation-view
            # order. Robot configs pin their public body order to that convention.
            self._body_indexes = torch.as_tensor(body_indexes, dtype=torch.long, device=device)
            if len(self._body_indexes) > 0 and int(self._body_indexes.max()) >= self._body_pos_w.shape[1]:
                raise ValueError(
                    f"Motion file {motion_file!r} has only {self._body_pos_w.shape[1]} bodies, but the "
                    f"requested legacy body index reaches {int(self._body_indexes.max())}."
                )

        if quaternion_order == "wxyz":
            self._body_quat_w = torch.roll(self._body_quat_w, shifts=-1, dims=-1)
        elif quaternion_order != "xyzw":
            raise ValueError(
                f"Unsupported quaternion_order {quaternion_order!r} in motion file {motion_file!r}. "
                "Expected 'wxyz' or 'xyzw'."
            )
        self.time_step_total = self.joint_pos.shape[0]

    @staticmethod
    def _resolve_name_indexes(
        file_names: Sequence[str], requested_names: Sequence[str], kind: str, *, require_complete: bool
    ) -> list[int]:
        if len(file_names) != len(set(file_names)):
            raise ValueError(f"Motion file contains duplicate {kind} names: {file_names}.")
        index_by_name = {name: index for index, name in enumerate(file_names)}
        requested_name_set = set(requested_names)
        missing = [name for name in requested_names if name not in index_by_name]
        extra = [name for name in file_names if name not in requested_name_set] if require_complete else []
        if missing or extra:
            raise ValueError(f"Motion {kind} names do not match the robot. Missing={missing}, extra={extra}.")
        return [index_by_name[name] for name in requested_names]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCollection:
    """A concatenated, variable-length motion set with safe per-motion indexing."""

    _FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")

    def __init__(
        self,
        motion_files: Sequence[str],
        body_indexes: Sequence[int],
        device: str,
        storage_device: str,
        joint_names: Sequence[str],
        body_names: Sequence[str],
    ):
        loaders = [
            MotionLoader(path, body_indexes, device=storage_device, joint_names=joint_names, body_names=body_names)
            for path in motion_files
        ]
        fps = [float(np.asarray(loader.fps).item()) for loader in loaders]
        if any(not math.isclose(value, fps[0]) for value in fps[1:]):
            raise ValueError(f"All motions must have the same fps; found {sorted(set(fps))}")
        self.fps = fps[0]
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device)
        self.names = list(motion_files)
        self.time_totals = torch.tensor([loader.time_step_total for loader in loaders], device=device)
        if torch.any(self.time_totals < 2):
            bad = [path for path, loader in zip(motion_files, loaders) if loader.time_step_total < 2]
            raise ValueError(f"Motion clips must contain at least two frames: {bad}")
        self.time_offsets = torch.zeros(len(loaders), dtype=torch.long, device=device)
        if len(loaders) > 1:
            self.time_offsets[1:] = torch.cumsum(self.time_totals[:-1], dim=0)
        self.time_step_total = int(self.time_totals.sum().item())
        for field in self._FIELDS:
            setattr(self, field, torch.cat([getattr(loader, field) for loader in loaders], dim=0))

    @property
    def num_motions(self) -> int:
        return len(self.names)

    def global_indices(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        lengths = self.time_totals[motion_ids]
        clamped = torch.minimum(torch.clamp_min(time_steps, 0), lengths - 1)
        return self.time_offsets[motion_ids] + clamped

    def sample(self, field: str, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        indexes = self.global_indices(motion_ids, time_steps)
        values = getattr(self, field)[indexes.to(self.storage_device)]
        return values.to(self.device, non_blocking=True)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        motion_files = resolve_motion_files(self.cfg.motion_file)
        storage_device = self.cfg.motion_data_device
        if storage_device == "auto":
            storage_device = "cpu" if len(motion_files) > 1 else self.device
        self.motion = MotionCollection(
            motion_files,
            self.body_indexes,
            device=self.device,
            storage_device=storage_device,
            joint_names=self.robot.joint_names,
            body_names=self.cfg.body_names,
        )
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._has_sampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._motion_cache = {
            field: torch.empty((self.num_envs, *getattr(self.motion, field).shape[1:]), device=self.device)
            for field in self.motion._FIELDS
        }
        self._window_time_steps: torch.Tensor | None = None
        self._window_motion_ids: torch.Tensor | None = None
        self._window_body_pos: torch.Tensor | None = None
        self._window_body_quat: torch.Tensor | None = None
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 3] = 1.0

        if self.cfg.adaptive_bin_size <= 0:
            raise ValueError("adaptive_bin_size must be positive")
        bins: list[tuple[int, int, int]] = []
        motion_bin_offsets = []
        motion_bin_counts = []
        for motion_id, length in enumerate(self.motion.time_totals.tolist()):
            motion_bin_offsets.append(len(bins))
            motion_bins = [
                (motion_id, start, min(start + self.cfg.adaptive_bin_size, length))
                for start in range(0, length, self.cfg.adaptive_bin_size)
            ]
            motion_bin_counts.extend([len(motion_bins)] * len(motion_bins))
            bins.extend(motion_bins)
        bin_table = torch.tensor(bins, dtype=torch.long, device=self.device)
        self.bin_motion_ids, self.bin_starts, self.bin_ends = bin_table.unbind(dim=1)
        self.bin_prior_weights = (self.bin_ends - self.bin_starts).float()
        if self.cfg.adaptive_sequence_length_agnostic:
            self.bin_prior_weights /= torch.tensor(motion_bin_counts, device=self.device)
        self.bin_prior_weights /= self.bin_prior_weights.sum()
        self.motion_bin_offsets = torch.tensor(motion_bin_offsets, dtype=torch.long, device=self.device)
        self.bin_count = len(bins)
        initial_count = float(self.cfg.adaptive_prior_count)
        self.bin_episode_count = torch.full((self.bin_count,), initial_count, device=self.device)
        self.bin_failed_count = torch.full((self.bin_count,), initial_count, device=self.device)
        self._current_bin_episodes = torch.zeros(self.bin_count, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, device=self.device)
        self._adaptive_sync_counter = 0
        if not 0.0 <= self.cfg.adaptive_uniform_ratio <= 1.0:
            raise ValueError("adaptive_uniform_ratio must be in [0, 1]")
        if self.cfg.adaptive_sync_interval <= 0:
            raise ValueError("adaptive_sync_interval must be positive")

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def _refresh_motion_cache(self, env_ids: torch.Tensor | None = None) -> None:
        """Gather the current reference frame once for all downstream MDP terms."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        motion_ids = self.motion_ids[env_ids]
        time_steps = self.time_steps[env_ids]
        for field, cache in self._motion_cache.items():
            cache[env_ids] = self.motion.sample(field, motion_ids, time_steps)

    def sample_body_window(self, time_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a motion window, reusing overlapping frames from the previous step."""
        motion_ids = self.motion_ids[:, None].expand_as(time_steps)
        if self._window_time_steps is None or self._window_time_steps.shape != time_steps.shape:
            self._window_body_pos = self.motion.sample("body_pos_w", motion_ids, time_steps)
            self._window_body_quat = self.motion.sample("body_quat_w", motion_ids, time_steps)
        else:
            same_window = torch.all(time_steps == self._window_time_steps, dim=1) & (
                self.motion_ids == self._window_motion_ids
            )
            can_shift = (
                torch.all(time_steps[:, :-1] == self._window_time_steps[:, 1:], dim=1)
                & (self.motion_ids == self._window_motion_ids)
                & ~same_window
            )
            refresh = ~(same_window | can_shift)
            if torch.any(can_shift):
                self._window_body_pos[can_shift, :-1] = self._window_body_pos[can_shift, 1:]
                self._window_body_quat[can_shift, :-1] = self._window_body_quat[can_shift, 1:]
                next_motion_ids = self.motion_ids[can_shift]
                next_time_steps = time_steps[can_shift, -1]
                self._window_body_pos[can_shift, -1] = self.motion.sample(
                    "body_pos_w", next_motion_ids, next_time_steps
                )
                self._window_body_quat[can_shift, -1] = self.motion.sample(
                    "body_quat_w", next_motion_ids, next_time_steps
                )
            if torch.any(refresh):
                refresh_motion_ids = motion_ids[refresh]
                refresh_time_steps = time_steps[refresh]
                self._window_body_pos[refresh] = self.motion.sample(
                    "body_pos_w", refresh_motion_ids, refresh_time_steps
                )
                self._window_body_quat[refresh] = self.motion.sample(
                    "body_quat_w", refresh_motion_ids, refresh_time_steps
                )
        self._window_time_steps = time_steps.clone()
        self._window_motion_ids = self.motion_ids.clone()
        return self._window_body_pos, self._window_body_quat

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._motion_cache["joint_pos"]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._motion_cache["joint_vel"]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._motion_cache["body_pos_w"] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._motion_cache["body_quat_w"]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._motion_cache["body_lin_vel_w"]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._motion_cache["body_ang_vel_w"]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._motion_cache["body_pos_w"][:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._motion_cache["body_quat_w"][:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._motion_cache["body_lin_vel_w"][:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._motion_cache["body_ang_vel_w"][:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel.torch

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w.torch[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w.torch[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        previously_sampled = self._has_sampled[env_ids]
        if torch.any(previously_sampled):
            previous_env_ids = env_ids[previously_sampled]
            previous_time_steps = torch.minimum(
                self.time_steps[previous_env_ids], self.motion.time_totals[self.motion_ids[previous_env_ids]] - 1
            )
            previous_bins = self.motion_bin_offsets[self.motion_ids[previous_env_ids]] + torch.div(
                previous_time_steps, self.cfg.adaptive_bin_size, rounding_mode="floor"
            )
            self._current_bin_episodes += torch.bincount(previous_bins, minlength=self.bin_count)
            episode_failed = self._env.termination_manager.terminated[previous_env_ids]
            if torch.any(episode_failed):
                self._current_bin_failed += torch.bincount(
                    previous_bins[episode_failed], minlength=self.bin_count
                )

        failure_rate = self.bin_failed_count / self.bin_episode_count.clamp_min(1.0)
        weighted_failure_rate = failure_rate * self.bin_prior_weights
        failure_probabilities = weighted_failure_rate / weighted_failure_rate.sum().clamp_min(1.0e-12)
        uniform_probabilities = self.bin_prior_weights
        sampling_probabilities = (
            (1.0 - self.cfg.adaptive_uniform_ratio) * failure_probabilities
            + self.cfg.adaptive_uniform_ratio * uniform_probabilities
        )

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        self.motion_ids[env_ids] = self.bin_motion_ids[sampled_bins]
        bin_lengths = self.bin_ends[sampled_bins] - self.bin_starts[sampled_bins]
        self.time_steps[env_ids] = self.bin_starts[sampled_bins] + (
            torch.rand(len(env_ids), device=self.device) * bin_lengths
        ).long()
        self._has_sampled[env_ids] = True

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else torch.ones_like(H)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _sync_adaptive_stats(self):
        """Merge newly collected bin statistics, including across DDP ranks."""
        episodes = self._current_bin_episodes
        failures = self._current_bin_failed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            packed = torch.cat((episodes, failures))
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            episodes, failures = packed.chunk(2)
        self.bin_episode_count += episodes
        self.bin_failed_count += failures * self.cfg.adaptive_failure_multiplier
        self._current_bin_episodes.zero_()
        self._current_bin_failed.zero_()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        self._refresh_motion_cache(torch.as_tensor(env_ids, dtype=torch.long, device=self.device))

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim_index(
            position=joint_pos[env_ids], velocity=joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat([root_pos[env_ids], root_ori[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=torch.cat([root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_totals[self.motion_ids])[0]
        self._resample_command(env_ids)
        active_env_ids = torch.where(self.time_steps < self.motion.time_totals[self.motion_ids])[0]
        if len(env_ids) > 0:
            active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            active_mask[env_ids] = False
            active_env_ids = torch.where(active_mask)[0]
        self._refresh_motion_cache(active_env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self._adaptive_sync_counter += 1
        if self._adaptive_sync_counter % self.cfg.adaptive_sync_interval == 0:
            self._sync_adaptive_stats()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    motion_data_device: str = "auto"
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_bin_size: int = 50
    adaptive_prior_count: float = 1.0
    adaptive_failure_multiplier: float = 1.0
    adaptive_sync_interval: int = 200
    adaptive_sequence_length_agnostic: bool = True

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
