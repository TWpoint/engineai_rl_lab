from __future__ import annotations

import math
from types import SimpleNamespace

import engineai_rl_lab.tasks  # noqa: F401
import engineai_rl_lab.tasks.tracking.mdp as mdp
import gymnasium as gym
import torch
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v23 import (
    T800FlatV23ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v24 import (
    T800FlatV24ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v23 import (
    T800FlatWoStateEstimationEnvCfgV23Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v24 import (
    T800FlatWoStateEstimationEnvCfgV24Scale,
)


def _identity_quaternions(*shape: int) -> torch.Tensor:
    quaternions = torch.zeros(*shape, 4)
    quaternions[..., -1] = 1.0
    return quaternions


def test_v24_local_command_uses_current_reference_anchor_position() -> None:
    env_origin = torch.tensor([[10.0, -3.0, 0.0]])
    body_positions = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.5]],
            [[0.2, 0.0, 1.0], [1.2, 0.0, 1.5]],
            [[0.5, 0.0, 1.0], [1.5, 0.0, 1.5]],
        ]
    )
    command = SimpleNamespace(
        device=torch.device("cpu"),
        time_steps=torch.tensor([1]),
        motion_lengths=torch.tensor([3]),
        cfg=SimpleNamespace(body_names=["base", "hand"]),
        anchor_pos_w=body_positions[1, 0].unsqueeze(0) + env_origin,
        robot_anchor_pos_w=torch.tensor([[100.0, 200.0, 300.0]]),
        # Robot base is yawed +90 degrees, so a +world-X displacement
        # appears along -Y in the robot-base frame.
        robot_anchor_quat_w=torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]]),
    )

    def sample_body_window(time_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return body_positions[time_steps], _identity_quaternions(1, time_steps.shape[1], 2)

    command.sample_body_window = sample_body_window
    env = SimpleNamespace(
        num_envs=1,
        scene=SimpleNamespace(env_origins=env_origin),
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )

    pose = mdp.motion_body_pose_reference_anchor_window_by_entity_xz(env, "motion", [0, 1])
    flat_pose = mdp.motion_body_pose_reference_anchor_window_xz_flat(env, "motion", [0, 1])
    positions = pose.reshape(1, 2, 2, 9)[..., :3]

    assert flat_pose.shape == (1, 36)
    torch.testing.assert_close(flat_pose, pose.flatten(start_dim=1))
    torch.testing.assert_close(positions[0, 0, 0], torch.zeros(3))
    torch.testing.assert_close(positions[0, 0, 1], torch.tensor([0.0, -0.3, 0.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(positions[0, 1, 0], torch.tensor([0.0, -1.0, 0.5]), atol=1e-6, rtol=0.0)

    zeroed_pose = mdp.motion_body_pose_reference_anchor_window_by_entity_xz(
        env,
        "motion",
        [0, 2],
        zero_invalid_offsets=True,
    ).reshape(1, 2, 2, 9)
    torch.testing.assert_close(zeroed_pose[:, :, 1], torch.zeros(1, 2, 9))


def test_v24_has_fourteen_links_without_head_and_sets_requested_action_rate_weight() -> None:
    v23 = T800FlatWoStateEstimationEnvCfgV23Scale()
    v24 = T800FlatWoStateEstimationEnvCfgV24Scale()

    # V5 already reduced this lineage to 14 links. The transformer therefore
    # sees 14 command tokens plus its readout token, not 15 body tokens.
    assert len(v23.commands.motion.body_names) == 14
    assert "LINK_HEAD_YAW" not in v23.commands.motion.body_names
    assert len(v24.commands.motion.body_names) == 14
    assert "LINK_HEAD_YAW" not in v24.commands.motion.body_names
    assert v24.observations.command.link_pose_b.func is mdp.motion_body_pose_reference_anchor_window_by_entity_xz
    assert v24.observations.critic.command.func is mdp.motion_body_pose_and_error_b_window_flat
    assert v24.observations.critic.local_command.func is mdp.motion_body_pose_reference_anchor_window_xz_flat
    assert v24.observations.critic.local_command.params["frame_offsets"] == [0]
    assert v23.rewards.action_rate_l2.weight == -0.05
    assert v24.rewards.action_rate_l2.weight == -0.1


def test_v24_registration_and_runner() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v24-scale")
    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v24:T800FlatWoStateEstimationEnvCfgV24Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".rsl_rl_ppo_cfg_v24:T800FlatV24ScalePPORunnerCfg")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"

    runner = T800FlatV24ScalePPORunnerCfg()
    v23_runner = T800FlatV23ScalePPORunnerCfg()
    assert runner.run_name == "v24-scale"
    assert runner.algorithm.class_name == "PPO"
    assert runner.algorithm.to_dict() == v23_runner.algorithm.to_dict()
    assert runner.actor.to_dict() == v23_runner.actor.to_dict()
    assert runner.critic.to_dict() == v23_runner.critic.to_dict()
