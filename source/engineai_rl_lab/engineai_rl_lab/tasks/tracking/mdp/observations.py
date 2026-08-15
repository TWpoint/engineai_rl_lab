from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_lin_vel_w.view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_ang_vel_w.view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference body positions expressed in the robot's current anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.body_pos_w,
        command.body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def motion_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference body orientations expressed in the robot's current anchor frame as 6D rotations."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.body_pos_w,
        command.body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def _motion_time_steps(command: MotionCommand, frame_offsets: list[int] | tuple[int, ...]) -> torch.Tensor:
    """Build clamped, frame-major motion indexes for arbitrary relative offsets."""
    if not frame_offsets:
        raise ValueError("frame_offsets must contain at least one frame")
    if any(not isinstance(offset, int) for offset in frame_offsets):
        raise TypeError("frame_offsets must contain integers only")
    offsets = torch.tensor(frame_offsets, dtype=torch.long, device=command.device)
    time_steps = command.time_steps[:, None] + offsets[None, :]
    motion_lengths = command.motion_lengths[:, None]
    return torch.minimum(torch.clamp_min(time_steps, 0), motion_lengths - 1)


def motion_body_pos_b_window(
    env: ManagerBasedEnv, command_name: str, frame_offsets: list[int] | tuple[int, ...]
) -> torch.Tensor:
    """Reference body positions at the requested offsets from the current frame.

    Every pose is expressed in the robot's *current* anchor frame. The result is
    flattened in frame-major order so existing MLP command encoders can consume it.
    Motion indexes outside the clip are clamped to the first or last frame.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    time_steps = _motion_time_steps(command, frame_offsets)
    body_pos_w, body_quat_w = command.sample_body_window(time_steps)
    body_pos_w = body_pos_w + env.scene.env_origins[:, None, None, :]
    num_frames, num_bodies = time_steps.shape[1], len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, None, :].expand(-1, num_frames, num_bodies, -1),
        command.robot_anchor_quat_w[:, None, None, :].expand(-1, num_frames, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def motion_body_ori_b_window(
    env: ManagerBasedEnv, command_name: str, frame_offsets: list[int] | tuple[int, ...]
) -> torch.Tensor:
    """Reference body 6D orientations over a configurable past/future window."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    time_steps = _motion_time_steps(command, frame_offsets)
    body_pos_w, body_quat_w = command.sample_body_window(time_steps)
    body_pos_w = body_pos_w + env.scene.env_origins[:, None, None, :]
    num_frames, num_bodies = time_steps.shape[1], len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, None, :].expand(-1, num_frames, num_bodies, -1),
        command.robot_anchor_quat_w[:, None, None, :].expand(-1, num_frames, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_body_pose_b_window_by_entity(
    env: ManagerBasedEnv, command_name: str, frame_offsets: list[int] | tuple[int, ...]
) -> torch.Tensor:
    """Reference body poses grouped by entity over a temporal window.

    Each pose contains the body position followed by its 6D rotation. The
    returned shape is ``(num_envs, num_bodies, num_frames * 9)`` so that each
    body forms one entity token containing its complete temporal trajectory.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    time_steps = _motion_time_steps(command, frame_offsets)
    body_pos_w, body_quat_w = command.sample_body_window(time_steps)
    body_pos_w = body_pos_w + env.scene.env_origins[:, None, None, :]
    num_frames, num_bodies = time_steps.shape[1], len(command.cfg.body_names)
    pos_b, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, None, :].expand(-1, num_frames, num_bodies, -1),
        command.robot_anchor_quat_w[:, None, None, :].expand(-1, num_frames, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    ori_6d_b = matrix_from_quat(ori_b)[..., :2].reshape(env.num_envs, num_frames, num_bodies, 6)
    pose_b = torch.cat((pos_b, ori_6d_b), dim=-1)
    return pose_b.permute(0, 2, 1, 3).reshape(env.num_envs, num_bodies, num_frames * 9)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
