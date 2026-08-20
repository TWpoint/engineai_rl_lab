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
    body_indexes = _get_body_indexes(command, body_names)
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


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
