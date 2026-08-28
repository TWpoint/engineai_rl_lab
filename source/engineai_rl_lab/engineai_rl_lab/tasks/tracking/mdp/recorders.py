from __future__ import annotations

import os
import re
import socket
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from isaaclab.managers import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils.configclass import configclass

_ROBOT_STATE_FIELDS = (
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_pos",
    "joint_vel",
)

_OPTIONAL_ROBOT_FIELDS = (
    # Read only the COM position.  ``body_com_pose_b`` also materializes a
    # synthetic identity quaternion for every body, which is unnecessary for
    # diagnosing COM randomization and adds another backend kernel.
    "body_com_pos_b",
    "body_mass",
    "body_inertia",
    "joint_acc",
    "joint_pos_target",
    "joint_vel_target",
    "joint_effort_target",
    "computed_torque",
    "applied_torque",
    "joint_pos_limits",
    "soft_joint_pos_limits",
    "joint_vel_limits",
    "soft_joint_vel_limits",
    "joint_stiffness",
    "joint_damping",
    "joint_effort_limits",
)

_DEFAULT_THRESHOLDS = {
    "joint_pos_limit_margin": 0.5,
    "joint_vel_limit_scale": 2.0,
    "max_body_linear_velocity": 100.0,
    "max_body_angular_velocity": 200.0,
    "max_body_extent": 5.0,
    "quaternion_norm_tolerance": 0.25,
}


def _as_torch(value: Any) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _clone_data_fields(data: Any, fields: Sequence[str]) -> dict[str, torch.Tensor]:
    values = {}
    for field in fields:
        try:
            value = _as_torch(getattr(data, field))
        except (AttributeError, RuntimeError):
            continue
        if isinstance(value, torch.Tensor):
            values[field] = value.detach().clone()
    return values


def _clone_data_row(data: Any, fields: Sequence[str], env_id: int) -> dict[str, torch.Tensor]:
    """Copy one environment row without first cloning the full vectorized state."""
    values = {}
    for field in fields:
        try:
            value = _as_torch(getattr(data, field))
        except (AttributeError, RuntimeError):
            continue
        if isinstance(value, torch.Tensor):
            values[field] = value[env_id].detach().cpu().clone()
    return values


def _row_to_cpu(values: dict[str, torch.Tensor], env_id: int) -> dict[str, torch.Tensor]:
    return {name: value[env_id].detach().cpu().clone() for name, value in values.items()}


def _finite_abs_max(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return None
    return float(finite.abs().max().item())


def _largest_entry(value: torch.Tensor | None, names: Sequence[str]) -> dict[str, Any] | None:
    if value is None or value.numel() == 0:
        return None
    finite_abs = torch.where(torch.isfinite(value), value.abs(), torch.full_like(value, -1.0))
    flat_index = int(torch.argmax(finite_abs).item())
    if float(finite_abs.flatten()[flat_index].item()) < 0.0:
        return None
    index = list(torch.unravel_index(torch.tensor(flat_index), value.shape))
    leading_index = int(index[0].item()) if index else 0
    return {
        "index": [int(item.item()) for item in index],
        "name": names[leading_index] if leading_index < len(names) else str(leading_index),
        "value": float(value.flatten()[flat_index].item()),
    }


def _nonfinite_detail(value: torch.Tensor, names: Sequence[str], limit: int = 64) -> dict[str, Any] | None:
    indexes = torch.nonzero(~torch.isfinite(value), as_tuple=False)
    if indexes.numel() == 0:
        return None
    examples = []
    for index in indexes[:limit]:
        coordinates = [int(item) for item in index.tolist()]
        leading_index = coordinates[0] if coordinates else 0
        examples.append(
            {
                "index": coordinates,
                "name": names[leading_index] if leading_index < len(names) else str(leading_index),
            }
        )
    return {"count": int(indexes.shape[0]), "examples": examples}


def diagnose_invalid_state_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Classify a diagnostic snapshot without claiming substep-level causality."""
    # Schema v3 names the state captured at the recorder event explicitly.
    # Keep accepting schema-v2 snapshots so existing diagnostics remain
    # readable after this recorder becomes event-only by default.
    event_step = payload.get("event_step", payload.get("post_step"))
    if event_step is None:
        raise KeyError("snapshot must contain 'event_step' or legacy 'post_step'")
    post = event_step["robot_state"]
    joint_names = payload["joint_names"]
    body_names = payload["body_names"]
    root_names = ["root"]
    reasons: list[dict[str, Any]] = []

    for field in _ROBOT_STATE_FIELDS:
        value = post.get(field)
        if value is None:
            continue
        names = joint_names if field.startswith("joint_") else body_names if field.startswith("body_") else root_names
        detail = _nonfinite_detail(value, names)
        if detail is not None:
            reasons.append({"reason": f"nonfinite_{field}", **detail})

    thresholds = {**_DEFAULT_THRESHOLDS, **payload.get("thresholds", {})}
    if payload.get("snapshot_kind") == "finite-runaway":
        thresholds.update(payload.get("finite_runaway_thresholds", {}))
    joint_pos = post.get("joint_pos")
    joint_pos_limits = post.get("joint_pos_limits")
    if joint_pos is not None and joint_pos_limits is not None:
        margin = float(thresholds["joint_pos_limit_margin"])
        lower = joint_pos_limits[..., 0]
        upper = joint_pos_limits[..., 1]
        below = torch.isfinite(joint_pos) & torch.isfinite(lower) & (joint_pos < lower - margin)
        above = torch.isfinite(joint_pos) & torch.isfinite(upper) & (joint_pos > upper + margin)
        for name, mask in (("joint_pos_below_limit", below), ("joint_pos_above_limit", above)):
            indexes = torch.nonzero(mask, as_tuple=False).flatten()
            if indexes.numel():
                reasons.append(
                    {
                        "reason": name,
                        "count": int(indexes.numel()),
                        "joints": [joint_names[int(index)] for index in indexes[:64]],
                    }
                )

    joint_vel = post.get("joint_vel")
    joint_vel_limits = post.get("joint_vel_limits")
    if joint_vel is not None and joint_vel_limits is not None:
        scale = float(thresholds["joint_vel_limit_scale"])
        mask = (
            torch.isfinite(joint_vel)
            & torch.isfinite(joint_vel_limits)
            & (joint_vel_limits > 0.0)
            & (joint_vel.abs() > joint_vel_limits * scale)
        )
        indexes = torch.nonzero(mask, as_tuple=False).flatten()
        if indexes.numel():
            reasons.append(
                {
                    "reason": "joint_velocity_limit",
                    "count": int(indexes.numel()),
                    "joints": [joint_names[int(index)] for index in indexes[:64]],
                }
            )

    body_lin_vel = post.get("body_lin_vel_w")
    body_ang_vel = post.get("body_ang_vel_w")
    root_pos = post.get("root_pos_w")
    body_pos = post.get("body_pos_w")
    body_quat = post.get("body_quat_w")
    root_quat = post.get("root_quat_w")
    threshold_checks = (
        (
            "body_linear_velocity",
            None if body_lin_vel is None else body_lin_vel.abs() > float(thresholds["max_body_linear_velocity"]),
        ),
        (
            "body_angular_velocity",
            None if body_ang_vel is None else body_ang_vel.abs() > float(thresholds["max_body_angular_velocity"]),
        ),
        (
            "body_extent",
            None
            if root_pos is None or body_pos is None
            else (body_pos - root_pos.unsqueeze(0)).abs() > float(thresholds["max_body_extent"]),
        ),
    )
    for name, mask in threshold_checks:
        if mask is not None and torch.any(mask):
            indexes = torch.nonzero(mask, as_tuple=False)
            reasons.append({"reason": name, "count": int(indexes.shape[0])})

    tolerance = float(thresholds["quaternion_norm_tolerance"])
    for name, value in (("root_quaternion_norm", root_quat), ("body_quaternion_norm", body_quat)):
        if value is None:
            continue
        norms = torch.linalg.vector_norm(value, dim=-1)
        mask = torch.isfinite(norms) & ((norms < 1.0 - tolerance) | (norms > 1.0 + tolerance))
        if torch.any(mask):
            reasons.append({"reason": name, "count": int(mask.sum().item())})

    pre = payload.get("pre_step")
    pre_robot_state = pre.get("robot_state", {}) if pre is not None else {}
    action_context = pre if pre is not None else event_step
    metrics = {
        "pre_max_abs_joint_pos": _finite_abs_max(pre_robot_state.get("joint_pos")),
        "pre_max_abs_joint_vel": _finite_abs_max(pre_robot_state.get("joint_vel")),
        "post_max_abs_joint_pos": _finite_abs_max(joint_pos),
        "post_max_abs_joint_vel": _finite_abs_max(joint_vel),
        "post_max_abs_body_lin_vel": _finite_abs_max(body_lin_vel),
        "post_max_abs_body_ang_vel": _finite_abs_max(body_ang_vel),
        "post_max_abs_computed_torque": _finite_abs_max(post.get("computed_torque")),
        "post_max_abs_applied_torque": _finite_abs_max(post.get("applied_torque")),
        "raw_action_max": _largest_entry(action_context.get("raw_action"), payload["action_joint_names"]),
        "processed_action_max": _largest_entry(action_context.get("processed_action"), payload["action_joint_names"]),
    }
    contact = event_step.get("contact", {})
    metrics["post_max_abs_contact_force"] = _finite_abs_max(contact.get("net_forces_w"))
    return {
        "primary_reason": reasons[0]["reason"] if reasons else "termination_without_matching_snapshot_reason",
        "all_reasons": reasons,
        "metrics": metrics,
    }


class InvalidRobotStateSnapshotRecorder(RecorderTerm):
    """Opt-in recorder for first-step, finite-runaway, and invalid pre-reset snapshots."""

    cfg: InvalidRobotStateSnapshotRecorderCfg

    def __init__(self, cfg: InvalidRobotStateSnapshotRecorderCfg, env):
        super().__init__(cfg, env)
        if not cfg.snapshot_dir:
            raise ValueError("Invalid-state snapshot recording requires a non-empty snapshot_dir")
        if cfg.max_snapshots < 0:
            raise ValueError("max_snapshots must be non-negative")
        if cfg.max_finite_runaway_snapshots < 0:
            raise ValueError("max_finite_runaway_snapshots must be non-negative")
        if cfg.finite_runaway_joint_pos_limit_margin < 0.0:
            raise ValueError("finite_runaway_joint_pos_limit_margin must be non-negative")
        if cfg.finite_runaway_joint_vel_limit_scale <= 0.0:
            raise ValueError("finite_runaway_joint_vel_limit_scale must be positive")
        self.snapshot_paths: list[str] = []
        self._pre_step: dict[str, Any] = {}
        self._captured_events: set[tuple[str, str, int, int]] = set()
        self._captured_first_control_step_contexts: set[str] = set()
        self._invalid_snapshot_count = 0
        self._finite_runaway_snapshot_count = 0

    def _snapshot_context(self) -> tuple[str, str]:
        context = str(
            getattr(self._env, "invalid_state_snapshot_context", None) or self.cfg.snapshot_context or "default"
        )
        safe_context = re.sub(r"[^A-Za-z0-9_.-]+", "_", context).strip("._-") or "default"
        return context, safe_context[:80]

    @staticmethod
    def _global_rank() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return int(os.environ.get("RANK", "0"))

    def _command_snapshot(self, env_id: int | None = None) -> dict[str, torch.Tensor]:
        command = self._env.command_manager.get_term(self.cfg.command_name)
        fields = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        values = {}
        for field in fields:
            value = getattr(command, field, None)
            if isinstance(value, torch.Tensor):
                if env_id is None:
                    values[field] = value.detach().clone()
                else:
                    values[field] = value[env_id].detach().cpu().clone()
        return values

    def record_pre_step(self) -> tuple[None, None]:
        _, safe_context = self._snapshot_context()
        needs_invalid_snapshot = self._invalid_snapshot_count < self.cfg.max_snapshots
        needs_finite_runaway_snapshot = (
            self.cfg.capture_finite_runaway
            and self._finite_runaway_snapshot_count < self.cfg.max_finite_runaway_snapshots
        )
        needs_first_control_step = (
            self.cfg.capture_first_control_step and safe_context not in self._captured_first_control_step_contexts
        )
        needs_explicit_pre_step_context = self.cfg.capture_pre_step_context and (
            needs_invalid_snapshot or needs_finite_runaway_snapshot
        )
        if not needs_explicit_pre_step_context and not needs_first_control_step:
            self._pre_step = {}
            return None, None
        asset = self._env.scene[self.cfg.asset_name]
        command = self._env.command_manager.get_term(self.cfg.command_name)
        action_term = self._env.action_manager._terms.get("joint_pos")
        self._pre_step = {
            "robot_state": _clone_data_fields(asset.data, _ROBOT_STATE_FIELDS),
            "reference_state": self._command_snapshot(),
            "raw_action": self._env.action_manager.action.detach().clone(),
            "previous_action": self._env.action_manager.prev_action.detach().clone(),
            "processed_action": (action_term.processed_actions.detach().clone() if action_term is not None else None),
            "episode_length": self._env.episode_length_buf.detach().clone(),
            "motion_steps": command.time_steps.detach().clone(),
        }
        return None, None

    def _actuator_delays(self, env_id: int) -> dict[str, int]:
        asset = self._env.scene[self.cfg.asset_name]
        values = {}
        for name, actuator in asset.actuators.items():
            delay_buffer = getattr(actuator, "positions_delay_buffer", None)
            if delay_buffer is not None:
                values[name] = int(delay_buffer.time_lags[env_id].item())
        return values

    def _contact_snapshot(self, env_id: int) -> dict[str, torch.Tensor]:
        try:
            sensor = self._env.scene[self.cfg.contact_sensor_name]
        except KeyError:
            return {}
        values = {}
        for field in ("net_forces_w", "net_forces_w_history", "force_matrix_w"):
            try:
                value = _as_torch(getattr(sensor.data, field))
            except (AttributeError, RuntimeError):
                continue
            if isinstance(value, torch.Tensor):
                values[field] = value[env_id].detach().cpu().clone()
        return values

    def _actuator_command_snapshot(self, env_id: int) -> dict[str, torch.Tensor]:
        """Read current articulation-wide actuator commands through the public collection API."""
        asset = self._env.scene[self.cfg.asset_name]
        actuators = asset.actuators
        values = {}
        for name, command in (
            ("target_position", actuators.target_command.position),
            ("output_position", actuators.output_command.position),
        ):
            value = _as_torch(command)
            if isinstance(value, torch.Tensor):
                values[name] = value[env_id].detach().cpu().clone()
        return values

    def _thresholds(self) -> dict[str, float]:
        term_cfg = self._env.termination_manager.get_term_cfg(self.cfg.termination_name)
        return {name: float(term_cfg.params.get(name, default)) for name, default in _DEFAULT_THRESHOLDS.items()}

    def _finite_runaway_thresholds(self) -> dict[str, float]:
        return {
            "joint_pos_limit_margin": float(self.cfg.finite_runaway_joint_pos_limit_margin),
            "joint_vel_limit_scale": float(self.cfg.finite_runaway_joint_vel_limit_scale),
        }

    def _finite_runaway_env_ids(self) -> torch.Tensor:
        """Return finite post-step states whose joints are far outside hard limits."""
        asset = self._env.scene[self.cfg.asset_name]
        all_state_finite = torch.ones(self._env.num_envs, dtype=torch.bool, device=self._env.device)
        for field in _ROBOT_STATE_FIELDS:
            try:
                value = _as_torch(getattr(asset.data, field))
            except (AttributeError, RuntimeError):
                return torch.empty(0, dtype=torch.long, device=self._env.device)
            if not isinstance(value, torch.Tensor):
                return torch.empty(0, dtype=torch.long, device=self._env.device)
            all_state_finite &= torch.isfinite(value).flatten(start_dim=1).all(dim=1)

        try:
            joint_pos = _as_torch(asset.data.joint_pos)
            joint_pos_limits = _as_torch(asset.data.joint_pos_limits)
            joint_vel = _as_torch(asset.data.joint_vel)
            joint_vel_limits = _as_torch(asset.data.joint_vel_limits)
        except (AttributeError, RuntimeError):
            return torch.empty(0, dtype=torch.long, device=self._env.device)
        if not all(
            isinstance(value, torch.Tensor) for value in (joint_pos, joint_pos_limits, joint_vel, joint_vel_limits)
        ):
            return torch.empty(0, dtype=torch.long, device=self._env.device)

        margin = self.cfg.finite_runaway_joint_pos_limit_margin
        lower = joint_pos_limits[..., 0]
        upper = joint_pos_limits[..., 1]
        position_runaway = (
            (torch.isfinite(lower) & (joint_pos < lower - margin))
            | (torch.isfinite(upper) & (joint_pos > upper + margin))
        ).any(dim=1)

        velocity_scale = self.cfg.finite_runaway_joint_vel_limit_scale
        valid_velocity_limit = torch.isfinite(joint_vel_limits) & (joint_vel_limits > 0.0)
        velocity_runaway = (valid_velocity_limit & (joint_vel.abs() > joint_vel_limits * velocity_scale)).any(dim=1)
        return torch.where(all_state_finite & (position_runaway | velocity_runaway))[0]

    def _build_snapshot(self, env_id: int, snapshot_kind: str) -> dict[str, Any]:
        asset = self._env.scene[self.cfg.asset_name]
        command = self._env.command_manager.get_term(self.cfg.command_name)
        action_term = self._env.action_manager._terms.get("joint_pos")
        context, safe_context = self._snapshot_context()
        motion_id = int(command.motion_ids[env_id].item())
        global_motion_id = int(command.motion.global_ids[motion_id].item())
        motion_file = command._motion_files[global_motion_id]
        post_state = _clone_data_row(
            asset.data,
            (*_ROBOT_STATE_FIELDS, *_OPTIONAL_ROBOT_FIELDS),
            env_id,
        )
        event_step = {
            "robot_state": post_state,
            "reference_state": self._command_snapshot(env_id),
            "raw_action": self._env.action_manager.action[env_id].detach().cpu().clone(),
            "previous_action": self._env.action_manager.prev_action[env_id].detach().cpu().clone(),
            "processed_action": (
                None if action_term is None else action_term.processed_actions[env_id].detach().cpu().clone()
            ),
            "episode_length": int(self._env.episode_length_buf[env_id].item()),
            "motion_step": int(command.time_steps[env_id].item()),
            "contact": self._contact_snapshot(env_id),
            "actuator": self._actuator_command_snapshot(env_id),
        }
        termination_flags = {
            name: bool(self._env.termination_manager.get_term(name)[env_id].item())
            for name in self._env.termination_manager.active_terms
        }
        pre_step = None
        if self._pre_step:
            pre_step = {
                "robot_state": _row_to_cpu(self._pre_step["robot_state"], env_id),
                "reference_state": _row_to_cpu(self._pre_step["reference_state"], env_id),
                "raw_action": self._pre_step["raw_action"][env_id].detach().cpu().clone(),
                "previous_action": self._pre_step["previous_action"][env_id].detach().cpu().clone(),
                "processed_action": (
                    None
                    if self._pre_step["processed_action"] is None
                    else self._pre_step["processed_action"][env_id].detach().cpu().clone()
                ),
                "episode_length": int(self._pre_step["episode_length"][env_id].item()),
                "motion_step": int(self._pre_step["motion_steps"][env_id].item()),
            }
        payload = {
            "schema_version": 3,
            "snapshot_kind": snapshot_kind,
            "context": context,
            "safe_context": safe_context,
            "host": socket.gethostname(),
            "rank": self._global_rank(),
            "pid": os.getpid(),
            "env_id": env_id,
            "common_step": int(self._env.common_step_counter),
            "simulation_step": int(self._env._sim_step_counter),
            "episode_length": int(self._env.episode_length_buf[env_id].item()),
            "motion_id": motion_id,
            "global_motion_id": global_motion_id,
            "motion_file": motion_file,
            "motion_step": int(command.time_steps[env_id].item()),
            "motion_length": int(command.motion_lengths[env_id].item()),
            "joint_names": list(asset.joint_names),
            "body_names": list(asset.body_names),
            "reference_body_names": list(command.cfg.body_names),
            "action_joint_names": list(getattr(action_term, "_joint_names", [])),
            "termination_flags": termination_flags,
            "thresholds": self._thresholds(),
            "finite_runaway_thresholds": self._finite_runaway_thresholds(),
            "actuator_delays": self._actuator_delays(env_id),
            "pre_step": pre_step,
            "event_step": event_step,
        }
        payload["diagnosis"] = diagnose_invalid_state_snapshot(payload)
        return payload

    def _write_snapshot(self, payload: dict[str, Any]) -> str:
        output_dir = Path(self.cfg.snapshot_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{payload['snapshot_kind']}-ctx-{payload['safe_context']}-"
            f"rank{payload['rank']:03d}-pid{payload['pid']}-"
            f"step{payload['common_step']:08d}-env{payload['env_id']:05d}-"
            f"motion{payload['global_motion_id']:05d}-frame{payload['motion_step']:06d}.pt"
        )
        target = output_dir / filename
        temporary = output_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self.snapshot_paths.append(str(target))
        return str(target)

    def _record_first_control_step(self) -> None:
        if not self.cfg.capture_first_control_step:
            return
        _, safe_context = self._snapshot_context()
        if safe_context in self._captured_first_control_step_contexts:
            return
        first_step_env_ids = torch.where(self._env.episode_length_buf == 1)[0]
        if first_step_env_ids.numel() == 0:
            return
        # One deterministic baseline per context and rank keeps the opt-in useful
        # for large vectorized evaluations without producing thousands of files.
        env_id = int(first_step_env_ids[0].item())
        event_key = (safe_context, "first_control_step", int(self._env.common_step_counter), env_id)
        if event_key in self._captured_events:
            return
        payload = self._build_snapshot(env_id, snapshot_kind="first-control-step")
        self._write_snapshot(payload)
        self._captured_events.add(event_key)
        self._captured_first_control_step_contexts.add(safe_context)

    def _record_finite_runaway(self) -> None:
        if (
            not self.cfg.capture_finite_runaway
            or self._finite_runaway_snapshot_count >= self.cfg.max_finite_runaway_snapshots
        ):
            return
        _, safe_context = self._snapshot_context()
        for env_id in self._finite_runaway_env_ids().detach().cpu().tolist():
            if self._finite_runaway_snapshot_count >= self.cfg.max_finite_runaway_snapshots:
                break
            event_key = (safe_context, "finite_runaway", int(self._env.common_step_counter), int(env_id))
            if event_key in self._captured_events:
                continue
            payload = self._build_snapshot(int(env_id), snapshot_kind="finite-runaway")
            self._write_snapshot(payload)
            self._captured_events.add(event_key)
            self._finite_runaway_snapshot_count += 1

    def record_post_step(self) -> tuple[None, None]:
        if self._pre_step:
            self._record_first_control_step()
        if self.cfg.capture_finite_runaway:
            self._record_finite_runaway()
        return None, None

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[None, None]:
        if self._invalid_snapshot_count >= self.cfg.max_snapshots:
            return None, None
        if env_ids is None:
            env_ids = range(self._env.num_envs)
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self._env.device)
        invalid = self._env.termination_manager.get_term(self.cfg.termination_name)[env_ids_tensor]
        # TerminationManager intentionally keeps its most recently-computed term
        # buffers across reset. Explicit evaluator resets therefore still see the
        # old flag, while the same-step autoreset has already zeroed episode length.
        automatic_invalid = invalid & (self._env.episode_length_buf[env_ids_tensor] > 0)
        invalid_env_ids = env_ids_tensor[automatic_invalid].detach().cpu().tolist()
        _, safe_context = self._snapshot_context()
        for env_id in invalid_env_ids:
            if self._invalid_snapshot_count >= self.cfg.max_snapshots:
                break
            event_key = (safe_context, "invalid", int(self._env.common_step_counter), int(env_id))
            if event_key in self._captured_events:
                continue
            payload = self._build_snapshot(int(env_id), snapshot_kind="invalid")
            self._write_snapshot(payload)
            self._captured_events.add(event_key)
            self._invalid_snapshot_count += 1
        return None, None


@configclass
class InvalidRobotStateSnapshotRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = InvalidRobotStateSnapshotRecorder
    snapshot_dir: str = ""
    snapshot_context: str = ""
    asset_name: str = "robot"
    command_name: str = "motion"
    termination_name: str = "invalid_robot_state"
    contact_sensor_name: str = "contact_forces"
    max_snapshots: int = 512
    capture_first_control_step: bool = False
    capture_pre_step_context: bool = False
    capture_finite_runaway: bool = False
    max_finite_runaway_snapshots: int = 512
    finite_runaway_joint_pos_limit_margin: float = 1.0
    finite_runaway_joint_vel_limit_scale: float = 5.0


@configclass
class InvalidRobotStateRecorderManagerCfg(RecorderManagerBaseCfg):
    invalid_robot_state = InvalidRobotStateSnapshotRecorderCfg()
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    export_in_close = False
    update_observations_before_recording = False
