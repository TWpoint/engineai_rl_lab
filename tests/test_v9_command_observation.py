from __future__ import annotations

from types import SimpleNamespace

import torch
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v9 import (
    T800FlatV9ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v8 import T800FlatWoStateEstimationEnvCfgV8Scale
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v9 import (
    V9_FRAME_OFFSETS,
    T800FlatWoStateEstimationEnvCfgV9Scale,
)
from engineai_rl_lab.tasks.tracking.mdp.observations import (
    motion_body_pose_and_error_b_window_by_entity,
    motion_body_pose_b_window_by_entity,
)
from rsl_rl.models import ModelGraph
from tensordict import TensorDict


def _identity_quaternions(*shape: int) -> torch.Tensor:
    quaternions = torch.zeros(*shape, 4)
    quaternions[..., -1] = 1.0
    return quaternions


def _fake_env() -> tuple[SimpleNamespace, SimpleNamespace]:
    env_origins = torch.tensor([[10.0, -3.0, 0.0]])
    body_positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    body_quaternions = _identity_quaternions(3, 2)
    half_sqrt = 2.0**-0.5
    current_body_quaternions = torch.tensor([[[half_sqrt, 0.0, 0.0, half_sqrt], [0.0, half_sqrt, 0.0, half_sqrt]]])
    body_quaternions[1] = current_body_quaternions[0]

    command = SimpleNamespace(
        device=torch.device("cpu"),
        time_steps=torch.tensor([1]),
        motion_lengths=torch.tensor([3]),
        cfg=SimpleNamespace(body_names=["body_a", "body_b"]),
        robot_anchor_pos_w=env_origins.clone(),
        robot_anchor_quat_w=torch.tensor([[0.0, 0.0, half_sqrt, half_sqrt]]),
        robot_body_pos_w=body_positions[1:2] + env_origins[:, None, :],
        robot_body_quat_w=current_body_quaternions,
        sample_calls=0,
    )

    def sample_body_window(time_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        command.sample_calls += 1
        return body_positions[time_steps], body_quaternions[time_steps]

    command.sample_body_window = sample_body_window
    env = SimpleNamespace(
        num_envs=1,
        scene=SimpleNamespace(env_origins=env_origins),
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )
    return env, command


def test_v9_command_doubles_last_dimension_and_preserves_v8_target() -> None:
    env, command = _fake_env()
    frame_offsets = [-1, 0, 1]

    target = motion_body_pose_b_window_by_entity(env, "motion", frame_offsets)
    target_and_error = motion_body_pose_and_error_b_window_by_entity(env, "motion", frame_offsets)

    assert target.shape == (1, 2, 27)
    assert target_and_error.shape == (1, 2, 54)
    torch.testing.assert_close(target_and_error[..., :27], target)
    assert command.sample_calls == 2


def test_v9_current_frame_error_is_zero_position_and_identity_rotation() -> None:
    env, _ = _fake_env()

    target_and_error = motion_body_pose_and_error_b_window_by_entity(env, "motion", [-1, 0, 1])
    error = target_and_error[..., 27:].reshape(1, 2, 3, 9)

    torch.testing.assert_close(error[:, :, 1, :3], torch.zeros(1, 2, 3), atol=1.0e-6, rtol=0.0)
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expand(1, 2, -1)
    torch.testing.assert_close(error[:, :, 1, 3:], identity_6d, atol=1.0e-6, rtol=0.0)


def test_v9_scale_config_uses_asymmetric_window_and_stronger_action_rate_penalty() -> None:
    v8_env_cfg = T800FlatWoStateEstimationEnvCfgV8Scale()
    env_cfg = T800FlatWoStateEstimationEnvCfgV9Scale()
    runner_cfg = T800FlatV9ScalePPORunnerCfg()

    command_term = env_cfg.observations.command.link_pose_b
    assert command_term.func is motion_body_pose_and_error_b_window_by_entity
    assert command_term.params["frame_offsets"] == [-3, -2, -1, 0, 1, 2, 3, 4, 5]
    assert command_term.params["frame_offsets"] == V9_FRAME_OFFSETS
    assert v8_env_cfg.rewards.action_rate_l2.weight == -0.03
    assert env_cfg.rewards.action_rate_l2.weight == -0.075
    assert env_cfg.terminations.body_pos.params["threshold"] == 0.6
    assert env_cfg.commands.motion.motion_shard_across_ranks
    assert runner_cfg.run_name == "v9_scale_lafan"
    assert runner_cfg.num_steps_per_env == 64
    assert runner_cfg.save_interval == 500


def test_v9_actor_projects_162_dimensional_command_tokens() -> None:
    runner_cfg = T800FlatV9ScalePPORunnerCfg()
    actor_cfg = runner_cfg.actor.to_dict()
    actor_cfg.pop("class_name")
    observations = TensorDict(
        {
            "policy": torch.randn(2, 5, 53),
            "action": torch.randn(2, 4, 25),
            "command": torch.randn(2, 14, 162),
        },
        batch_size=[2],
    )

    actor = ModelGraph(observations, runner_cfg.obs_groups, "actor", 25, **actor_cfg)

    assert actor(observations).shape == (2, 25)
    assert actor.nodes["command_projection"].mlp[0].in_features == 162
