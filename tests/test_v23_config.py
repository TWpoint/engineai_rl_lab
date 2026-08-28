from __future__ import annotations

import engineai_rl_lab.tasks  # noqa: F401
import engineai_rl_lab.tasks.tracking.mdp as mdp
import gymnasium as gym
import torch
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v22 import (
    T800FlatV22ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v23 import (
    T800FlatV23ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v22 import (
    T800FlatWoStateEstimationEnvCfgV22Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v23 import (
    T800_V23_MOTION_CATALOG_CACHE,
    T800_V23_TERMINATION_END_EFFECTOR_BODY_NAMES,
    T800FlatWoStateEstimationEnvCfgV23Scale,
)
from engineai_rl_lab.tasks.tracking.robots.t800 import T800_POLICY_JOINT_NAMES


def _joint_tracking_env():
    command = type(
        "Command",
        (),
        {
            "robot": type("Robot", (), {"joint_names": ["j0", "head", "j1"], "num_joints": 3})(),
            "joint_pos": torch.tensor([[0.5, 9.0, -0.5]]),
            "robot_joint_pos": torch.tensor([[0.0, 0.0, 0.5]]),
            "joint_vel": torch.tensor([[1.0, 9.0, -2.0]]),
            "robot_joint_vel": torch.tensor([[0.0, 0.0, 1.0]]),
        },
    )()
    command_manager = type("CommandManager", (), {"get_term": lambda self, name: command})()
    return type("Env", (), {"command_manager": command_manager})()


def test_v23_joint_tracking_functions_select_policy_joints() -> None:
    env = _joint_tracking_env()
    joint_names = ["j0", "j1"]

    pos_obs = mdp.motion_joint_position_target_and_error(env, "motion", joint_names)
    vel_obs = mdp.motion_joint_velocity_target_and_error(env, "motion", joint_names)
    assert torch.equal(pos_obs, torch.tensor([[0.5, -0.5, 0.5, -1.0]]))
    assert torch.equal(vel_obs, torch.tensor([[1.0, -2.0, 1.0, -3.0]]))

    expected_pos_reward = torch.exp(torch.tensor(-0.625 / 0.5**2))
    expected_vel_reward = torch.exp(torch.tensor(-5.0 / 3.0**2))
    assert torch.allclose(
        mdp.motion_joint_position_error_exp(env, "motion", std=0.5, joint_names=joint_names),
        expected_pos_reward.reshape(1),
    )
    assert torch.allclose(
        mdp.motion_joint_velocity_error_exp(env, "motion", std=3.0, joint_names=joint_names),
        expected_vel_reward.reshape(1),
    )


def test_v23_adds_policy_joint_tracking_rewards_without_mutating_v22() -> None:
    v22 = T800FlatWoStateEstimationEnvCfgV22Scale()
    v23 = T800FlatWoStateEstimationEnvCfgV23Scale()

    assert not hasattr(v22.rewards, "motion_joint_pos")
    assert not hasattr(v22.rewards, "motion_joint_vel")

    joint_pos = v23.rewards.motion_joint_pos
    assert joint_pos.func is mdp.motion_joint_position_error_exp
    assert joint_pos.weight == 0.5
    assert joint_pos.params == {
        "command_name": "motion",
        "std": 0.5,
        "joint_names": T800_POLICY_JOINT_NAMES,
    }

    joint_vel = v23.rewards.motion_joint_vel
    assert joint_vel.func is mdp.motion_joint_velocity_error_exp
    assert joint_vel.weight == 0.25
    assert joint_vel.params == {
        "command_name": "motion",
        "std": 3.0,
        "joint_names": T800_POLICY_JOINT_NAMES,
    }


def test_v23_enables_context_and_reward_aligned_adaptive_sampling_without_mutating_v22() -> None:
    v22 = T800FlatWoStateEstimationEnvCfgV22Scale()
    v23 = T800FlatWoStateEstimationEnvCfgV23Scale()
    sampler22 = v22.commands.motion.adaptive_sampling
    sampler23 = v23.commands.motion.adaptive_sampling

    assert sampler22.pre_failure_window == 0
    assert sampler22.deduplicate_failure_events is False
    assert sampler22.global_tracking_error_weight == 1.0
    assert sampler22.relative_position_error_weight == 0.0
    assert sampler22.relative_orientation_error_weight == 0.0
    assert sampler22.local_position_error_weight == 0.0

    assert sampler23.pre_failure_window == 50
    assert sampler23.deduplicate_failure_events is True
    assert sampler23.tracking_error_scale == 0.40
    assert sampler23.global_tracking_error_weight == 0.40
    assert sampler23.relative_position_error_weight == 0.20
    assert sampler23.relative_position_error_scale == 0.40
    assert sampler23.relative_orientation_error_weight == 0.20
    assert sampler23.relative_orientation_error_scale == 0.60
    assert sampler23.local_position_error_weight == 0.20
    assert sampler23.local_position_error_scale == 0.20
    assert sampler23.local_position_body_names == [
        "LINK_BASE",
        "LINK_WRIST_END_L",
        "LINK_WRIST_END_R",
        "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_R",
    ]


def test_v23_alone_enables_cached_deterministic_rank_sharding() -> None:
    v22 = T800FlatWoStateEstimationEnvCfgV22Scale()
    v23 = T800FlatWoStateEstimationEnvCfgV23Scale()

    assert v22.commands.motion.motion_catalog_cache is None
    assert v22.commands.motion.motion_shard_across_ranks is False
    assert v23.commands.motion.motion_catalog_cache == T800_V23_MOTION_CATALOG_CACHE
    assert v23.commands.motion.motion_shard_across_ranks is True
    assert v23.commands.motion.motion_load_workers == 4


def test_v23_adds_looser_immediate_tracking_terminations_without_mutating_v22() -> None:
    v22 = T800FlatWoStateEstimationEnvCfgV22Scale()
    v23 = T800FlatWoStateEstimationEnvCfgV23Scale()

    assert not hasattr(v22.terminations, "anchor_height")
    assert not hasattr(v22.terminations, "anchor_orientation")
    assert not hasattr(v22.terminations, "end_effector_height")

    anchor_height = v23.terminations.anchor_height
    assert anchor_height.func is mdp.bad_anchor_pos_z_only
    assert anchor_height.params == {"command_name": "motion", "threshold": 0.30}

    anchor_orientation = v23.terminations.anchor_orientation
    assert anchor_orientation.func is mdp.bad_anchor_ori
    assert anchor_orientation.params["asset_cfg"].name == "robot"
    assert anchor_orientation.params["command_name"] == "motion"
    assert anchor_orientation.params["threshold"] == 0.90

    end_effector_height = v23.terminations.end_effector_height
    assert end_effector_height.func is mdp.bad_motion_body_pos_z_only
    assert end_effector_height.params == {
        "command_name": "motion",
        "threshold": 0.30,
        "body_names": T800_V23_TERMINATION_END_EFFECTOR_BODY_NAMES,
    }
    assert T800_V23_TERMINATION_END_EFFECTOR_BODY_NAMES == [
        "LINK_WRIST_END_L",
        "LINK_WRIST_END_R",
        "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_R",
    ]


def test_v23_adds_joint_targets_and_errors_to_critic_only() -> None:
    v22 = T800FlatWoStateEstimationEnvCfgV22Scale()
    v23 = T800FlatWoStateEstimationEnvCfgV23Scale()

    assert not hasattr(v22.observations.critic, "motion_joint_pos")
    assert not hasattr(v22.observations.critic, "motion_joint_vel")
    assert not hasattr(v23.observations.proprioception, "motion_joint_pos")
    assert not hasattr(v23.observations.proprioception, "motion_joint_vel")

    joint_pos = v23.observations.critic.motion_joint_pos
    assert joint_pos.func is mdp.motion_joint_position_target_and_error
    assert joint_pos.params == {
        "command_name": "motion",
        "joint_names": T800_POLICY_JOINT_NAMES,
    }

    joint_vel = v23.observations.critic.motion_joint_vel
    assert joint_vel.func is mdp.motion_joint_velocity_target_and_error
    assert joint_vel.params == {
        "command_name": "motion",
        "joint_names": T800_POLICY_JOINT_NAMES,
    }


def test_v23_registration_and_critic_shape() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v23-scale")
    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v23:T800FlatWoStateEstimationEnvCfgV23Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".rsl_rl_ppo_cfg_v23:T800FlatV23ScalePPORunnerCfg")

    v22 = T800FlatV22ScalePPORunnerCfg()
    v23 = T800FlatV23ScalePPORunnerCfg()
    assert v22.critic.hidden_dims == [2048, 1024, 512]
    assert v22.actor.distribution_cfg.std_range == (0.001, 0.5)
    assert v23.run_name == "v23-scale"
    assert v23.actor.distribution_cfg.std_range == (1.0e-6, 1.0e6)
    assert v23.critic.hidden_dims == [1024, 1024, 512, 512]
