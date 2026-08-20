from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

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


def robot_body_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Current tracked-body linear velocities expressed in the robot anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
    velocity_b = quat_apply_inverse(anchor_quat_w, command.robot_body_lin_vel_w)
    return velocity_b.reshape(env.num_envs, -1)


def robot_body_ang_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Current tracked-body angular velocities expressed in the robot anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
    velocity_b = quat_apply_inverse(anchor_quat_w, command.robot_body_ang_vel_w)
    return velocity_b.reshape(env.num_envs, -1)


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


def _motion_time_steps_and_validity(
    command: MotionCommand, frame_offsets: list[int] | tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build clamped motion indexes and mark offsets that lie inside the clip."""
    if not frame_offsets:
        raise ValueError("frame_offsets must contain at least one frame")
    if any(not isinstance(offset, int) for offset in frame_offsets):
        raise TypeError("frame_offsets must contain integers only")
    offsets = torch.tensor(frame_offsets, dtype=torch.long, device=command.device)
    raw_time_steps = command.time_steps[:, None] + offsets[None, :]
    motion_lengths = command.motion_lengths[:, None]
    validity = torch.logical_and(raw_time_steps >= 0, raw_time_steps < motion_lengths)
    time_steps = torch.minimum(torch.clamp_min(raw_time_steps, 0), motion_lengths - 1)
    return time_steps, validity


def _motion_time_steps(command: MotionCommand, frame_offsets: list[int] | tuple[int, ...]) -> torch.Tensor:
    """Build clamped, frame-major motion indexes for arbitrary relative offsets."""
    return _motion_time_steps_and_validity(command, frame_offsets)[0]


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


def _motion_body_pose_b_window_components(
    env: ManagerBasedEnv, command_name: str, frame_offsets: list[int] | tuple[int, ...]
) -> tuple[MotionCommand, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a body-pose window and express every target in the current anchor frame."""
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
    return command, num_frames, num_bodies, body_quat_w, pos_b, ori_b


def _pose_by_entity(pos_b: torch.Tensor, ori_b: torch.Tensor) -> torch.Tensor:
    """Pack frame-major position and 6D orientation features into body tokens."""
    num_envs, num_frames, num_bodies = pos_b.shape[:3]
    ori_6d_b = matrix_from_quat(ori_b)[..., :2].reshape(num_envs, num_frames, num_bodies, 6)
    pose_b = torch.cat((pos_b, ori_6d_b), dim=-1)
    return pose_b.permute(0, 2, 1, 3).reshape(num_envs, num_bodies, num_frames * 9)


def _pose_by_entity_xz(pos_b: torch.Tensor, ori_b: torch.Tensor) -> torch.Tensor:
    """Pack poses with ScaleBFM's contiguous rotated-X/rotated-Z encoding."""
    num_envs, num_frames, num_bodies = pos_b.shape[:3]
    rotation = matrix_from_quat(ori_b)
    ori_6d_b = torch.cat((rotation[..., :, 0], rotation[..., :, 2]), dim=-1)
    pose_b = torch.cat((pos_b, ori_6d_b), dim=-1)
    return pose_b.permute(0, 2, 1, 3).reshape(num_envs, num_bodies, num_frames * 9)


def motion_body_pose_b_window_by_entity(
    env: ManagerBasedEnv, command_name: str, frame_offsets: list[int] | tuple[int, ...]
) -> torch.Tensor:
    """Reference body poses grouped by entity over a temporal window.

    Each pose contains the body position followed by its 6D rotation. The
    returned shape is ``(num_envs, num_bodies, num_frames * 9)`` so that each
    body forms one entity token containing its complete temporal trajectory.
    """
    _, _, _, _, pos_b, ori_b = _motion_body_pose_b_window_components(env, command_name, frame_offsets)
    return _pose_by_entity(pos_b, ori_b)


def motion_body_pose_b_window_by_entity_xz(
    env: ManagerBasedEnv, command_name: str, frame_offsets: list[int] | tuple[int, ...]
) -> torch.Tensor:
    """Reference body poses using the X/Z 6D rotation encoding from V9.

    This is exactly the target-trajectory half of
    :func:`motion_body_pose_and_error_b_window_by_entity`, without the
    target-to-current error features.
    """
    _, _, _, _, pos_b, ori_b = _motion_body_pose_b_window_components(env, command_name, frame_offsets)
    return _pose_by_entity_xz(pos_b, ori_b)


def motion_body_pose_b_window_xz_flat(
    env: ManagerBasedEnv, command_name: str, frame_offsets: list[int] | tuple[int, ...]
) -> torch.Tensor:
    """Flatten the target-only V9 trajectory command for an MLP actor."""
    command = motion_body_pose_b_window_by_entity_xz(env, command_name, frame_offsets)
    return command.flatten(start_dim=1)


def motion_body_pose_and_error_b_window_by_entity(
    env: ManagerBasedEnv,
    command_name: str,
    frame_offsets: list[int] | tuple[int, ...],
    zero_invalid_offsets: bool = False,
) -> torch.Tensor:
    """Reference poses and their errors from the current robot pose, grouped by entity.

    The first half of each token is the existing target trajectory in the
    robot's current anchor frame. The second half is the matching trajectory
    error: target-minus-current position and ScaleBFM relative-rotation X/Z
    features, also expressed in the current anchor frame. Targets and errors at
    offsets outside the motion clip can optionally be zeroed to match
    BeyondMinic validity; legacy callers retain endpoint clamping.

    The returned shape is ``(num_envs, num_bodies, num_frames * 18)``.
    """
    command, num_frames, num_bodies, target_quat_w, target_pos_b, target_ori_b = _motion_body_pose_b_window_components(
        env, command_name, frame_offsets
    )

    robot_pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    target_rotation = matrix_from_quat(target_ori_b)
    target_rotation_xz = torch.cat((target_rotation[..., :, 0], target_rotation[..., :, 2]), dim=-1)
    target_features = torch.cat((target_pos_b, target_rotation_xz), dim=-1)
    pos_error_b = target_pos_b - robot_pos_b[:, None, :, :]

    anchor_quat_inv_w = quat_inv(command.robot_anchor_quat_w)[:, None, None, :].expand(-1, num_frames, num_bodies, -1)
    anchor_quat_w = command.robot_anchor_quat_w[:, None, None, :].expand(-1, num_frames, num_bodies, -1)
    current_quat_inv_w = quat_conjugate(command.robot_body_quat_w)[:, None, :, :].expand(-1, num_frames, -1, -1)
    ori_error_w = quat_mul(target_quat_w, current_quat_inv_w)
    ori_error_b = quat_mul(quat_mul(anchor_quat_inv_w, ori_error_w), anchor_quat_w)
    error_rotation = matrix_from_quat(ori_error_b)
    error_rotation_xz = torch.cat((error_rotation[..., :, 0], error_rotation[..., :, 2]), dim=-1)
    error_features = torch.cat((pos_error_b, error_rotation_xz), dim=-1)

    if zero_invalid_offsets:
        _, validity = _motion_time_steps_and_validity(command, frame_offsets)
        validity = validity[:, :, None, None].to(dtype=target_features.dtype)
        target_features = target_features * validity
        error_features = error_features * validity

    target_pose = target_features.permute(0, 2, 1, 3).reshape(env.num_envs, num_bodies, num_frames * 9)
    error_pose = error_features.permute(0, 2, 1, 3).reshape(env.num_envs, num_bodies, num_frames * 9)

    return torch.cat((target_pose, error_pose), dim=-1)


def motion_body_pose_and_error_b_window_flat(
    env: ManagerBasedEnv,
    command_name: str,
    frame_offsets: list[int] | tuple[int, ...],
    zero_invalid_offsets: bool = False,
) -> torch.Tensor:
    """Flatten the V9 entity/trajectory command into one feature vector per environment."""
    command = motion_body_pose_and_error_b_window_by_entity(
        env, command_name, frame_offsets, zero_invalid_offsets=zero_invalid_offsets
    )
    return command.flatten(start_dim=1)


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
