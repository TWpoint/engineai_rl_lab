from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand
from engineai_rl_lab.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    gravity_w = asset.data.GRAVITY_VEC_W.torch
    gravity_dir_w = torch.nn.functional.normalize(gravity_w, dim=-1)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, gravity_dir_w)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, gravity_dir_w)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_global_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate on absolute world-frame body-position error, matching ScaleTrack."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def motion_time_out(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate after reaching the final frame of the sampled motion."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.time_steps >= command.motion.time_totals[command.motion_ids] - 1


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def nonfinite_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate environments whose simulated robot state contains NaN or Inf.

    A diverged physics state must be reset before observations are computed.  In
    particular, comparisons in the geometric termination terms do not catch NaN
    because every ordered comparison with NaN evaluates to false.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    state_tensors = (
        asset.data.root_pos_w.torch,
        asset.data.root_quat_w.torch,
        asset.data.root_lin_vel_w.torch,
        asset.data.root_ang_vel_w.torch,
        asset.data.body_pos_w.torch,
        asset.data.body_quat_w.torch,
        asset.data.body_lin_vel_w.torch,
        asset.data.body_ang_vel_w.torch,
        asset.data.joint_pos.torch,
        asset.data.joint_vel.torch,
    )
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for state in state_tensors:
        invalid |= ~torch.isfinite(state).flatten(start_dim=1).all(dim=1)
    return invalid
