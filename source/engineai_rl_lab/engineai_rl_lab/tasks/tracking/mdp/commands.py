from __future__ import annotations

import gc
import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

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

from .motion_data import MotionCollection, read_motion_lengths, resolve_motion_files
from .motion_data import MotionLoader as MotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

        self._validate_adaptive_sampling_cfg()
        self._motion_files = resolve_motion_files(self.cfg.motion_file)
        self._motion_storage_device = self.cfg.motion_data_device
        if self._motion_storage_device == "auto":
            self._motion_storage_device = "cpu" if len(self._motion_files) > 1 else self.device
        global_time_totals = read_motion_lengths(self._motion_files, max_workers=self.cfg.motion_load_workers)
        self._global_time_totals = torch.tensor(global_time_totals, dtype=torch.long, device=self.device)
        self.global_num_motions = len(self._motion_files)
        self._initialize_adaptive_sampling()
        self._adaptive_layout_checked = False
        self._rebuild_global_sampling_distribution()

        max_num_load_motions = self.cfg.max_num_load_motions
        if max_num_load_motions is None:
            max_num_load_motions = min(self.num_envs, 1024)
        if max_num_load_motions <= 0:
            raise ValueError("max_num_load_motions must be positive or None")
        self.max_num_load_motions = min(int(max_num_load_motions), self.global_num_motions)
        self.all_motions_loaded = self.max_num_load_motions >= self.global_num_motions
        if self.all_motions_loaded:
            selected_global_ids = torch.arange(self.global_num_motions, dtype=torch.long, device=self.device)
        else:
            selected_global_ids = torch.multinomial(
                self._motion_sampling_probabilities,
                num_samples=self.max_num_load_motions,
                replacement=self.cfg.working_set_replacement,
            )
        self.motion = self._load_motion_collection(selected_global_ids)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fixed_motion_ids: torch.Tensor | None = None
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._has_sampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._motion_cache = {
            field: torch.empty((self.num_envs, *self.motion.field_shapes[field]), device=self.device)
            for field in self.motion._FIELDS
        }
        self._window_time_steps: torch.Tensor | None = None
        self._window_motion_ids: torch.Tensor | None = None
        self._window_body_pos: torch.Tensor | None = None
        self._window_body_quat: torch.Tensor | None = None
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 3] = 1.0

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self._rebuild_active_motion_mapping()
        self._rebuild_sampling_distribution()

    def _validate_adaptive_sampling_cfg(self) -> None:
        if self.cfg.bin_size <= 0:
            raise ValueError("bin_size must be positive")
        if not 0.0 <= self.cfg.uniform_sampling_rate <= 1.0:
            raise ValueError("uniform_sampling_rate must be in [0, 1]")
        if (
            self.cfg.adp_samp_failure_rate_max_over_mean is not None
            and self.cfg.adp_samp_failure_rate_max_over_mean <= 0.0
        ):
            raise ValueError("adp_samp_failure_rate_max_over_mean must be positive or None")
        if self.cfg.pre_failure_sample_window < 0:
            raise ValueError("pre_failure_sample_window must be non-negative")
        if self.cfg.init_num_failures < 0:
            raise ValueError("init_num_failures must be non-negative")
        if self.cfg.adaptive_sampling_alpha is not None and not 0.0 < self.cfg.adaptive_sampling_alpha <= 1.0:
            raise ValueError("adaptive_sampling_alpha must be in (0, 1] or None")
        if self.cfg.adaptive_kernel_size < 1:
            raise ValueError("adaptive_kernel_size must be at least one")

    def _initialize_adaptive_sampling(self) -> None:
        motion_lengths = self._global_time_totals
        self.motion_bin_counts = torch.div(
            motion_lengths + self.cfg.bin_size - 1,
            self.cfg.bin_size,
            rounding_mode="floor",
        )
        self.bin_count = int(self.motion_bin_counts.sum().item())
        self.motion_bin_offsets = torch.zeros(self.global_num_motions, dtype=torch.long, device=self.device)
        if self.global_num_motions > 1:
            self.motion_bin_offsets[1:] = torch.cumsum(self.motion_bin_counts[:-1], dim=0)
        self.bin_motion_ids = torch.repeat_interleave(
            torch.arange(self.global_num_motions, device=self.device), self.motion_bin_counts
        )
        bin_indexes = torch.arange(self.bin_count, device=self.device)
        local_bin_indexes = bin_indexes - self.motion_bin_offsets[self.bin_motion_ids]
        self.bin_starts = local_bin_indexes * self.cfg.bin_size
        self.bin_ends = torch.minimum(
            self.bin_starts + self.cfg.bin_size,
            motion_lengths[self.bin_motion_ids],
        )
        self.bin_weights = (self.bin_ends - self.bin_starts).float()
        self.bin_weights /= self.bin_weights.mean()
        if self.cfg.sequence_length_agnostic:
            self.bin_weights /= self.motion_bin_counts[self.bin_motion_ids]
        initial_count = 0.0 if self.cfg.adaptive_sampling_alpha is not None else float(self.cfg.init_num_failures)
        self.adp_samp_num_episodes = torch.full((self.bin_count,), initial_count, device=self.device)
        self.adp_samp_num_failures = torch.full((self.bin_count,), initial_count, device=self.device)
        self._current_adp_samp_num_episodes = torch.zeros(self.bin_count, device=self.device)
        self._current_adp_samp_num_failures = torch.zeros(self.bin_count, device=self.device)

    def _load_motion_collection(self, selected_global_ids: torch.Tensor) -> MotionCollection:
        motion = MotionCollection(
            self._motion_files,
            self.body_indexes,
            device=self.device,
            storage_device=self._motion_storage_device,
            joint_names=self.robot.joint_names,
            body_names=self.cfg.body_names,
            shard_by_rank=False,
            selected_global_ids=selected_global_ids.tolist(),
            global_time_totals=self._global_time_totals,
            max_chunk_frames=self.cfg.motion_chunk_frames,
            max_workers=self.cfg.motion_load_workers,
        )
        print(
            "[INFO] SONIC motion working set: "
            f"rank={motion.rank}/{motion.world_size}, "
            f"files={motion.rank_num_files}/{motion.global_num_motions}, "
            f"unique={motion.global_ids.unique().numel()}, "
            f"frames={motion.rank_num_frames:,}, chunks={motion.num_chunks}, "
            f"resident={motion.resident_bytes / 2**30:.2f} GiB, "
            f"storage={motion.storage_device}."
        )
        return motion

    def _rebuild_active_motion_mapping(self) -> None:
        active_bin_ids = []
        for global_motion_id in self.motion.global_ids.tolist():
            begin = int(self.motion_bin_offsets[global_motion_id])
            count = int(self.motion_bin_counts[global_motion_id])
            active_bin_ids.append(torch.arange(begin, begin + count, dtype=torch.long, device=self.device))
        self._active_bin_ids = torch.cat(active_bin_ids)
        self._active_local_motion_ids = torch.repeat_interleave(
            torch.arange(self.motion.num_motions, device=self.device),
            self.motion_bin_counts[self.motion.global_ids],
        )

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
        samples = self.motion.sample_many(self.motion._FIELDS, motion_ids, time_steps)
        for field, cache in self._motion_cache.items():
            cache[env_ids] = samples[field]

    def sample_body_window(self, time_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a motion window, reusing overlapping frames from the previous step."""
        motion_ids = self.motion_ids[:, None].expand_as(time_steps)
        if self._window_time_steps is None or self._window_time_steps.shape != time_steps.shape:
            samples = self.motion.sample_many(("body_pos_w", "body_quat_w"), motion_ids, time_steps)
            self._window_body_pos = samples["body_pos_w"]
            self._window_body_quat = samples["body_quat_w"]
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
                samples = self.motion.sample_many(("body_pos_w", "body_quat_w"), next_motion_ids, next_time_steps)
                self._window_body_pos[can_shift, -1] = samples["body_pos_w"]
                self._window_body_quat[can_shift, -1] = samples["body_quat_w"]
            if torch.any(refresh):
                refresh_motion_ids = motion_ids[refresh]
                refresh_time_steps = time_steps[refresh]
                samples = self.motion.sample_many(("body_pos_w", "body_quat_w"), refresh_motion_ids, refresh_time_steps)
                self._window_body_pos[refresh] = samples["body_pos_w"]
                self._window_body_quat[refresh] = samples["body_quat_w"]
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
        self.metrics["error_anchor_pos"].copy_(torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1))
        self.metrics["error_anchor_rot"].copy_(quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w))
        self.metrics["error_anchor_lin_vel"].copy_(
            torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        )
        self.metrics["error_anchor_ang_vel"].copy_(
            torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        )

        self.metrics["error_body_pos"].copy_(
            torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        )
        self.metrics["error_body_rot"].copy_(
            quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        )
        self.metrics["error_body_lin_vel"].copy_(
            torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        )
        self.metrics["error_body_ang_vel"].copy_(
            torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)
        )

        self.metrics["error_joint_pos"].copy_(torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1))
        self.metrics["error_joint_vel"].copy_(torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1))

    def _bucket_ids(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        global_motion_ids = self.motion.global_ids[motion_ids]
        return self.motion_bin_offsets[global_motion_ids] + torch.div(
            time_steps, self.cfg.bin_size, rounding_mode="floor"
        )

    def _compute_failure_rate(self) -> torch.Tensor:
        failure_rate = torch.where(
            self.adp_samp_num_episodes > 0.0,
            self.adp_samp_num_failures / self.adp_samp_num_episodes.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(self.adp_samp_num_failures),
        )
        if self.cfg.adaptive_sampling_alpha is not None:
            failure_rate = self._smooth_failure_rate(failure_rate)
        if not self.cfg.use_failure_rate_decay:
            return failure_rate
        failure_rate_with_decay = torch.zeros_like(failure_rate)
        for step in reversed(range(self.bin_count)):
            if step == self.bin_count - 1:
                next_failure_rate = 0.0
            elif self.bin_motion_ids[step + 1] == self.bin_motion_ids[step]:
                next_failure_rate = self.cfg.decay_gamma * failure_rate_with_decay[step + 1]
            else:
                next_failure_rate = 0.0
            failure_rate_with_decay[step] = failure_rate[step] + next_failure_rate
        return failure_rate_with_decay

    def _smooth_failure_rate(self, failure_rate: torch.Tensor) -> torch.Tensor:
        """Apply MarmotLab's forward-looking kernel within each motion."""
        if self.cfg.adaptive_kernel_size == 1:
            return failure_rate
        kernel = failure_rate.new_tensor(
            [float(self.cfg.adaptive_kernel_lambda) ** index for index in range(self.cfg.adaptive_kernel_size)]
        )
        kernel /= kernel.sum()
        motion_end_bins = self.motion_bin_offsets[self.bin_motion_ids] + self.motion_bin_counts[self.bin_motion_ids] - 1
        bin_ids = torch.arange(self.bin_count, device=self.device)
        smoothed = torch.zeros_like(failure_rate)
        for offset, coefficient in enumerate(kernel):
            smoothed.add_(coefficient * failure_rate[torch.minimum(bin_ids + offset, motion_end_bins)])
        return smoothed

    @staticmethod
    def _cap_probabilities_at_uniform_ratio(probabilities: torch.Tensor, ratio: float) -> torch.Tensor:
        """Cap final probabilities and redistribute mass exactly as MarmotLab does."""
        probability_cap = ratio / float(probabilities.numel())
        if probability_cap >= 1.0:
            return probabilities
        capped = probabilities.clamp(max=probability_cap)
        deficit = 1.0 - capped.sum()
        if deficit <= 0.0:
            return capped
        available_capacity = (probability_cap - capped).clamp_min(0.0)
        capacity_sum = available_capacity.sum()
        if capacity_sum <= 0.0:
            return capped / capped.sum()
        return capped + deficit * available_capacity / capacity_sum

    def _marmot_probabilities(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.double()
        if scores.sum() <= 0.0:
            probabilities = torch.full_like(scores, 1.0 / len(scores))
        else:
            probabilities = scores / scores.sum()
        probabilities.mul_(1.0 - self.cfg.uniform_sampling_rate)
        probabilities.add_(self.cfg.uniform_sampling_rate / len(probabilities))
        if self.cfg.adp_samp_failure_rate_max_over_mean is not None:
            probabilities = self._cap_probabilities_at_uniform_ratio(
                probabilities, float(self.cfg.adp_samp_failure_rate_max_over_mean)
            )
        return probabilities

    def _clip_failure_rate(self, failure_rate: torch.Tensor) -> torch.Tensor:
        multiplier = self.cfg.adp_samp_failure_rate_max_over_mean
        if multiplier is None:
            return failure_rate
        return failure_rate.clamp(max=failure_rate.mean() * multiplier)

    def _configured_probability_cap(self, value: str | float | None, count: int) -> float | None:
        if value is None:
            return None
        if value == "auto":
            return float(self.cfg.adp_samp_failure_rate_max_over_mean) / count
        return float(value)

    def _apply_probability_caps(self, probabilities: torch.Tensor, active_bin_ids: torch.Tensor) -> torch.Tensor:
        max_prob_per_bin = self._configured_probability_cap(self.cfg.max_prob_per_bin, len(active_bin_ids))
        if max_prob_per_bin is not None and max_prob_per_bin > 0.0 and len(active_bin_ids) > 1.0 / max_prob_per_bin:
            probabilities = probabilities.clamp(max=max_prob_per_bin)
            probabilities /= probabilities.sum()

        active_motion_ids = self.bin_motion_ids[active_bin_ids]
        unique_motion_ids = active_motion_ids.unique()
        max_prob_per_motion = self._configured_probability_cap(self.cfg.max_prob_per_motion, len(unique_motion_ids))
        if (
            max_prob_per_motion is not None
            and max_prob_per_motion > 0.0
            and len(unique_motion_ids) > 1.0 / max_prob_per_motion
        ):
            for motion_id in unique_motion_ids:
                motion_mask = active_motion_ids == motion_id
                motion_probability = probabilities[motion_mask].sum()
                if motion_probability > max_prob_per_motion:
                    probabilities[motion_mask] *= max_prob_per_motion / motion_probability
            probabilities /= probabilities.sum()
        return probabilities

    def _rebuild_global_sampling_distribution(self) -> None:
        """Match SONIC's full-dataset distribution used to draw a working set."""

        failure_rate = self._compute_failure_rate()
        if self.cfg.adaptive_sampling_alpha is not None:
            probabilities = self._marmot_probabilities(failure_rate)
        else:
            failure_rate = self._clip_failure_rate(failure_rate).double()
            failure_probabilities = failure_rate / failure_rate.sum()
            uniform_probabilities = torch.ones_like(failure_probabilities) / self.bin_count
            probabilities = (
                failure_probabilities * (1.0 - self.cfg.uniform_sampling_rate)
                + uniform_probabilities * self.cfg.uniform_sampling_rate
            )
            probabilities *= self.bin_weights
            probabilities /= probabilities.sum()
            all_bin_ids = torch.arange(self.bin_count, device=self.device)
            probabilities = self._apply_probability_caps(probabilities, all_bin_ids)
        self.adp_sampling_prob = probabilities.float()
        motion_sampling_probabilities = torch.zeros(self.global_num_motions, dtype=torch.float32, device=self.device)
        motion_sampling_probabilities.index_add_(0, self.bin_motion_ids, self.adp_sampling_prob)
        self._motion_sampling_probabilities = motion_sampling_probabilities / motion_sampling_probabilities.sum()

    def _rebuild_sampling_distribution(self) -> None:
        """Match SONIC's adaptive distribution over the active working set."""

        active_failure_rate = self._compute_failure_rate()[self._active_bin_ids]
        if self.cfg.adaptive_sampling_alpha is not None:
            probabilities = self._marmot_probabilities(active_failure_rate)
        else:
            active_failure_rate = self._clip_failure_rate(active_failure_rate.double())
            failure_probabilities = active_failure_rate / active_failure_rate.sum()
            uniform_probabilities = torch.ones_like(failure_probabilities) / len(failure_probabilities)
            probabilities = (
                failure_probabilities * (1.0 - self.cfg.uniform_sampling_rate)
                + uniform_probabilities * self.cfg.uniform_sampling_rate
            )
            probabilities *= self.bin_weights[self._active_bin_ids]
            probabilities /= probabilities.sum()
            probabilities = self._apply_probability_caps(probabilities, self._active_bin_ids)
        if not torch.all(torch.isfinite(probabilities)) or torch.any(probabilities < 0.0):
            raise RuntimeError("Adaptive motion sampling produced invalid probabilities")
        self.adp_sampling_active_prob = probabilities.float()

        entropy = -(self.adp_sampling_active_prob * self.adp_sampling_active_prob.clamp_min(1.0e-12).log()).sum()
        entropy_normalized = (
            entropy / math.log(len(self.adp_sampling_active_prob))
            if len(self.adp_sampling_active_prob) > 1
            else torch.ones_like(entropy)
        )
        probability_max = self.adp_sampling_active_prob.max()
        self._sampling_entropy_normalized = entropy_normalized
        self._sampling_probability_max = probability_max
        if "sampling_entropy" in self.metrics:
            self.metrics["sampling_entropy"][:] = entropy_normalized
            self.metrics["sampling_top1_prob"][:] = probability_max

    def _update_adaptive_exposure(self) -> None:
        """Accumulate length-normalized exposure for every active motion bin."""

        active_env_ids = torch.where(self._has_sampled)[0]
        if len(active_env_ids) == 0:
            return
        time_steps = torch.minimum(self.time_steps[active_env_ids], self.motion_lengths[active_env_ids] - 1)
        bin_ids = self._bucket_ids(self.motion_ids[active_env_ids], time_steps)
        if self.cfg.adaptive_sampling_alpha is not None:
            self._current_adp_samp_num_episodes.index_add_(0, bin_ids, torch.ones_like(bin_ids, dtype=torch.float))
        else:
            bin_lengths = (self.bin_ends[bin_ids] - self.bin_starts[bin_ids]).float()
            self.adp_samp_num_episodes.index_add_(0, bin_ids, bin_lengths.reciprocal())

    def set_fixed_motion_ids(self, motion_ids: Sequence[int] | torch.Tensor) -> None:
        """Pin every environment to one process-local motion for deterministic evaluation.

        The assignment is retained across environment resets.  Training and
        normal playback remain adaptive until this method is called.
        """

        fixed_motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        if fixed_motion_ids.shape != (self.num_envs,):
            raise ValueError(
                f"fixed motion assignment must have shape ({self.num_envs},), got {tuple(fixed_motion_ids.shape)}"
            )
        if torch.any(fixed_motion_ids < 0) or torch.any(fixed_motion_ids >= self.motion.num_motions):
            raise ValueError(f"fixed motion ids must be in [0, {self.motion.num_motions})")
        self._fixed_motion_ids = fixed_motion_ids.clone()

    def clear_fixed_motion_ids(self) -> None:
        """Restore adaptive motion selection after deterministic evaluation."""

        self._fixed_motion_ids = None

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        previously_sampled = self._has_sampled[env_ids]
        if torch.any(previously_sampled):
            previous_env_ids = env_ids[previously_sampled]
            previous_time_steps = torch.minimum(
                self.time_steps[previous_env_ids], self.motion_lengths[previous_env_ids] - 1
            )
            previous_bins = self._bucket_ids(self.motion_ids[previous_env_ids], previous_time_steps)
            episode_failed = self._env.termination_manager.terminated[previous_env_ids]
            if torch.any(episode_failed):
                failed_bins = previous_bins[episode_failed]
                failure_buffer = (
                    self._current_adp_samp_num_failures
                    if self.cfg.adaptive_sampling_alpha is not None
                    else self.adp_samp_num_failures
                )
                failure_buffer.index_add_(
                    0,
                    failed_bins,
                    torch.full_like(
                        failed_bins,
                        self.cfg.failure_counts_multiplier,
                        dtype=self.adp_samp_num_failures.dtype,
                    ),
                )

        if self._fixed_motion_ids is None:
            sampled_active_bin_ids = torch.multinomial(
                self.adp_sampling_active_prob,
                num_samples=len(env_ids),
                replacement=True,
            )
            sampled_bins = self._active_bin_ids[sampled_active_bin_ids]
            sampled_motion_ids = self._active_local_motion_ids[sampled_active_bin_ids]
        else:
            sampled_motion_ids = self._fixed_motion_ids[env_ids]
            sampled_global_motion_ids = self.motion.global_ids[sampled_motion_ids]
            sampled_motion_bin_counts = self.motion_bin_counts[sampled_global_motion_ids]
            sampled_local_bin_ids = (torch.rand(len(env_ids), device=self.device) * sampled_motion_bin_counts).long()
            sampled_bins = self.motion_bin_offsets[sampled_global_motion_ids] + sampled_local_bin_ids
        self.motion_ids[env_ids] = sampled_motion_ids
        self.motion_lengths[env_ids] = self.motion.lengths(sampled_motion_ids)
        bin_lengths = self.bin_ends[sampled_bins] - self.bin_starts[sampled_bins]
        if self.cfg.start_at_motion_beginning:
            sampled_time_steps = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        else:
            sampled_time_steps = (
                self.bin_starts[sampled_bins] + (torch.rand(len(env_ids), device=self.device) * bin_lengths).long()
            )
            if self.cfg.pre_failure_sample_window > 0:
                pre_failure_offsets = torch.randint(
                    self.cfg.pre_failure_sample_window,
                    (len(env_ids),),
                    device=self.device,
                )
                sampled_time_steps = (sampled_time_steps - pre_failure_offsets).clamp_min(0)
        self.time_steps[env_ids] = sampled_time_steps
        self._has_sampled[env_ids] = True
        # CommandTerm.reset clears metrics for reset environments before
        # resampling. Restore these global distribution diagnostics so high
        # reset rates do not make them appear to collapse toward zero.
        self.metrics["sampling_entropy"][env_ids] = self._sampling_entropy_normalized
        self.metrics["sampling_top1_prob"][env_ids] = self._sampling_probability_max

    def _check_adaptive_distributed_layout(self) -> None:
        """Fail coherently before any variable-size adaptive collective."""

        if self._adaptive_layout_checked:
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            # The environment is constructed before the runner initializes
            # the process group. Preserve the future cross-rank check.
            self._adaptive_layout_checked = self.motion.world_size == 1
            return
        layout = torch.tensor(
            [
                self.global_num_motions,
                self.bin_count,
                *self.motion.manifest_fingerprint_words,
            ],
            dtype=torch.long,
            device=self.device,
        )
        layout_min = layout.clone()
        layout_max = layout.clone()
        torch.distributed.all_reduce(layout_min, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(layout_max, op=torch.distributed.ReduceOp.MAX)
        if not torch.equal(layout_min, layout_max):
            raise RuntimeError(
                "Distributed workers disagree on global motion count/bin layout or manifest fingerprint: "
                f"local={layout.tolist()}, min={layout_min.tolist()}, max={layout_max.tolist()}"
            )
        self._adaptive_layout_checked = True

    def sync_and_compute_adaptive_sampling(self, *, sync_across_ranks: bool) -> None:
        """Advance EMA every PPO iteration and periodically average it across ranks."""

        if sync_across_ranks:
            self._check_adaptive_distributed_layout()
        if self.cfg.adaptive_sampling_alpha is not None:
            alpha = float(self.cfg.adaptive_sampling_alpha)
            self.adp_samp_num_episodes.mul_(1.0 - alpha).add_(self._current_adp_samp_num_episodes, alpha=alpha)
            self.adp_samp_num_failures.mul_(1.0 - alpha).add_(self._current_adp_samp_num_failures, alpha=alpha)
            self._current_adp_samp_num_episodes.zero_()
            self._current_adp_samp_num_failures.zero_()
        if sync_across_ranks and torch.distributed.is_available() and torch.distributed.is_initialized():
            statistics = (self.adp_samp_num_episodes, self.adp_samp_num_failures)
            packed = torch.cat(statistics)
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            packed /= torch.distributed.get_world_size()
            episodes, failures = packed.chunk(2)
            statistics[0].copy_(episodes)
            statistics[1].copy_(failures)
        self._rebuild_global_sampling_distribution()
        self._rebuild_sampling_distribution()

    def get_adaptive_sampling_state(self) -> dict[str, torch.Tensor]:
        return {
            "adp_samp_num_episodes": self.adp_samp_num_episodes.detach().cpu(),
            "adp_samp_num_failures": self.adp_samp_num_failures.detach().cpu(),
        }

    def load_adaptive_sampling_state(self, state_dict: dict[str, torch.Tensor]) -> bool:
        episodes = state_dict.get("adp_samp_num_episodes")
        failures = state_dict.get("adp_samp_num_failures")
        if episodes is None or failures is None:
            return False
        if episodes.shape != self.adp_samp_num_episodes.shape or failures.shape != self.adp_samp_num_failures.shape:
            print("[WARN] Adaptive sampling state does not match the current dataset; skipping restore.")
            return False
        self.adp_samp_num_episodes.copy_(episodes.to(self.device))
        self.adp_samp_num_failures.copy_(failures.to(self.device))
        self._current_adp_samp_num_episodes.zero_()
        self._current_adp_samp_num_failures.zero_()
        self._rebuild_global_sampling_distribution()
        self._rebuild_sampling_distribution()
        return True

    def resample_motion_working_set(self) -> bool:
        if self.all_motions_loaded:
            return False
        self._rebuild_global_sampling_distribution()
        selected_global_ids = torch.multinomial(
            self._motion_sampling_probabilities,
            num_samples=self.max_num_load_motions,
            replacement=self.cfg.working_set_replacement,
        )
        self._has_sampled.zero_()
        self._fixed_motion_ids = None
        self.motion_ids.zero_()
        self.motion_lengths.zero_()
        del self.motion
        gc.collect()
        self.motion = self._load_motion_collection(selected_global_ids)
        self._rebuild_active_motion_mapping()
        self._rebuild_sampling_distribution()
        self._window_time_steps = None
        self._window_motion_ids = None
        self._window_body_pos = None
        self._window_body_quat = None
        return True

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._adaptive_sampling(env_ids)
        self._refresh_motion_cache(env_ids)

        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat([root_pos, root_ori], dim=-1),
            env_ids=env_ids,
        )
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=torch.cat([root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self._update_adaptive_exposure()
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion_lengths)[0]
        if self.cfg.resample_at_motion_end:
            self._resample_command(env_ids)
        if len(env_ids) > 0:
            active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            active_mask[env_ids] = False
            active_env_ids = torch.where(active_mask)[0]
        else:
            active_env_ids = torch.arange(self.num_envs, device=self.device)
        self._refresh_motion_cache(active_env_ids)

        anchor_pos_w = self.anchor_pos_w[:, None, :]
        anchor_quat_w = self.anchor_quat_w[:, None, :]
        robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :]
        robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :]

        delta_pos_w = robot_anchor_pos_w.expand(-1, len(self.cfg.body_names), -1).clone()
        delta_pos_w[..., 2] = anchor_pos_w[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w))).expand(
            -1, len(self.cfg.body_names), -1
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w)

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
    # SONIC loads all motions when they fit; otherwise each rank independently
    # draws a replacement-sampled working set of min(num_envs, 1024) motions.
    max_num_load_motions: int | None = None
    # Sampling with replacement matches SONIC. Disable it to keep every
    # process-local working set unique while allowing overlap across ranks.
    working_set_replacement: bool = True
    # NPZ files are decoded concurrently but consumed in manifest order.
    motion_load_workers: int = 4
    # Upper bound for one process-local construction/runtime chunk.
    motion_chunk_frames: int = 262_144
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # When disabled, the environment is expected to terminate at the final
    # motion frame instead of silently switching to another motion.
    resample_at_motion_end: bool = True
    # Playback/debug option. Training keeps random motion-time initialization
    # unless this is explicitly enabled by a caller such as play.py.
    start_at_motion_beginning: bool = False

    bin_size: int = 50
    sequence_length_agnostic: bool = True
    init_num_failures: float = 1.0
    uniform_sampling_rate: float = 0.1
    pre_failure_sample_window: int = 200
    use_failure_rate_decay: bool = False
    decay_gamma: float = 0.8
    # None retains the legacy cumulative SONIC sampler. A value enables
    # MarmotLab-style EMA statistics and forward-kernel smoothing.
    adaptive_sampling_alpha: float | None = None
    adaptive_kernel_size: int = 1
    adaptive_kernel_lambda: float = 0.8
    adp_samp_failure_rate_max_over_mean: float | None = 200.0
    failure_counts_multiplier: float = 1.0
    max_prob_per_bin: str | float | None = None
    max_prob_per_motion: str | float | None = None

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
