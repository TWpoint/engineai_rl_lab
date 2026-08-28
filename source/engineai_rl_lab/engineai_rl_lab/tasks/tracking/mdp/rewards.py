from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude

from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_joint_indexes(command: MotionCommand, joint_names: list[str] | None) -> list[int]:
    if joint_names is None:
        return list(range(command.robot.num_joints))
    return [command.robot.joint_names.index(name) for name in joint_names]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_height_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track only the world-frame anchor height, matching ScaleTrack."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[..., 2] - command.robot_anchor_pos_w[..., 2])
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_local_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str],
    body_offsets: list[list[float]] | None = None,
) -> torch.Tensor:
    """Track selected bodies in their respective reference/robot anchor frames.

    This matches SONIC's local key-point reward: global root translation and
    orientation are removed independently from the reference and robot poses.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Preserve the requested point order so each optional offset remains paired
    # with the body at the same position in ``body_names``.
    body_indexes = [command.cfg.body_names.index(name) for name in body_names]
    num_bodies = len(body_indexes)
    ref_body_pos_w = command.body_pos_w[:, body_indexes]
    robot_body_pos_w = command.robot_body_pos_w[:, body_indexes]
    if body_offsets is not None:
        offsets = torch.tensor(body_offsets, dtype=ref_body_pos_w.dtype, device=ref_body_pos_w.device)
        if offsets.shape != (num_bodies, 3):
            raise ValueError(f"Expected body_offsets with shape ({num_bodies}, 3), got {tuple(offsets.shape)}.")
        offsets = offsets.unsqueeze(0).expand(ref_body_pos_w.shape[0], -1, -1)
        ref_body_pos_w = ref_body_pos_w + quat_apply(command.body_quat_w[:, body_indexes], offsets)
        robot_body_pos_w = robot_body_pos_w + quat_apply(command.robot_body_quat_w[:, body_indexes], offsets)
    ref_anchor_quat = command.anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
    robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
    ref_pos_b = quat_apply_inverse(
        ref_anchor_quat,
        ref_body_pos_w - command.anchor_pos_w[:, None, :],
    )
    robot_pos_b = quat_apply_inverse(
        robot_anchor_quat,
        robot_body_pos_w - command.robot_anchor_pos_w[:, None, :],
    )
    error = torch.square(ref_pos_b - robot_pos_b).sum(dim=-1)
    return torch.exp(-error.mean(dim=-1) / std**2)


def motion_global_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Track absolute world-frame body positions, matching ScaleTrack."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Track absolute world-frame body orientations, matching ScaleTrack."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = quat_error_magnitude(command.body_quat_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Reward reference joint-position tracking over the selected joints."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_indexes = _get_joint_indexes(command, joint_names)
    error = torch.mean(
        torch.square(command.joint_pos[:, joint_indexes] - command.robot_joint_pos[:, joint_indexes]),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def motion_joint_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Reward reference joint-velocity tracking over the selected joints."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_indexes = _get_joint_indexes(command, joint_names)
    error = torch.mean(
        torch.square(command.joint_vel[:, joint_indexes] - command.robot_joint_vel[:, joint_indexes]),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def joint_pos_limits_capped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_error_per_joint: float = 1.0,
    max_total_error: float = 1.0,
) -> torch.Tensor:
    """Penalize soft-limit violations without allowing one bad state to dominate a rollout."""
    if max_error_per_joint <= 0.0:
        raise ValueError("max_error_per_joint must be positive")
    if max_total_error <= 0.0:
        raise ValueError("max_total_error must be positive")
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids]
    below_lower = (limits[..., 0] - joint_pos).clamp_(min=0.0, max=max_error_per_joint)
    above_upper = (joint_pos - limits[..., 1]).clamp_(min=0.0, max=max_error_per_joint)
    total_error = torch.sum(below_lower + above_upper, dim=1)
    total_error = torch.nan_to_num(total_error, nan=max_total_error, posinf=max_total_error, neginf=0.0)
    return total_error.clamp_(max=max_total_error)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
