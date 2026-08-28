from __future__ import annotations

from pathlib import Path

import engineai_rl_lab.tasks.tracking.mdp as mdp
import gymnasium as gym
import pytest
import torch
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v19 import (
    T800FlatV19ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v20 import (
    T800FlatV20ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v19 import (
    T800FlatWoStateEstimationEnvCfgV19Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v20 import (
    T800FlatWoStateEstimationEnvCfgV20Scale,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import MotionCommandV1, MotionCommandV1Cfg
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand
from rsl_rl.models import ModelGraph
from tensordict import TensorDict


def test_v20_alone_selects_the_new_motion_command() -> None:
    v19_motion = T800FlatWoStateEstimationEnvCfgV19Scale().commands.motion
    v20_motion = T800FlatWoStateEstimationEnvCfgV20Scale().commands.motion

    assert v19_motion.class_type is MotionCommand
    assert isinstance(v20_motion, MotionCommandV1Cfg)
    assert v20_motion.class_type is MotionCommandV1
    assert not hasattr(v20_motion, "curriculum_sampling_enabled")


def test_v20_preserves_task_settings_and_declares_its_sampling_budget() -> None:
    v19 = T800FlatWoStateEstimationEnvCfgV19Scale()
    v20 = T800FlatWoStateEstimationEnvCfgV20Scale()
    old_motion = v19.commands.motion
    new_motion = v20.commands.motion

    common_fields = (
        "asset_name",
        "resampling_time_range",
        "debug_vis",
        "motion_file",
        "motion_data_device",
        "max_num_load_motions",
        "working_set_replacement",
        "motion_load_workers",
        "motion_chunk_frames",
        "anchor_body_name",
        "body_names",
        "pose_range",
        "velocity_range",
        "joint_position_range",
        "resample_at_motion_end",
        "start_at_motion_beginning",
        "playback_start_frame",
    )
    for name in common_fields:
        assert getattr(new_motion, name) == getattr(old_motion, name)

    sampler = new_motion.adaptive_sampling
    assert sampler.coverage_fraction == 0.20
    assert sampler.fast_half_life == 32.0
    assert sampler.slow_half_life == 64.0
    assert sampler.probability_cap_ratio == 200.0
    assert (1.0 - sampler.coverage_fraction) * sampler.max_learnable_fraction == pytest.approx(0.60)
    assert (1.0 - sampler.coverage_fraction) * (1.0 - sampler.max_learnable_fraction) == pytest.approx(0.20)
    assert v20.rewards.joint_acc_l2 is None


def test_v20_actor_command_removes_error_but_critic_keeps_it() -> None:
    v19 = T800FlatWoStateEstimationEnvCfgV19Scale()
    v20 = T800FlatWoStateEstimationEnvCfgV20Scale()

    assert v19.observations.command.link_pose_b.func is mdp.motion_body_pose_and_error_b_window_by_entity
    assert v20.observations.command.link_pose_b.func is mdp.motion_body_pose_b_window_by_entity_xz
    assert v20.observations.command.link_pose_b.params == v19.observations.command.link_pose_b.params
    assert v20.observations.command.link_pose_b.params["zero_invalid_offsets"] is True
    assert v20.observations.critic.command.func is mdp.motion_body_pose_and_error_b_window_flat


def test_v20_scales_only_actor_joint_velocity() -> None:
    v20 = T800FlatWoStateEstimationEnvCfgV20Scale()
    v19 = T800FlatWoStateEstimationEnvCfgV19Scale()

    assert v20.observations.proprioception.joint_vel.scale == pytest.approx(0.05)
    assert v19.observations.proprioception.joint_vel.scale is None
    assert v20.observations.critic.joint_vel.scale is None


def test_v20_runner_uses_scratch_hyperparameters() -> None:
    v19 = T800FlatV19ScalePPORunnerCfg().to_dict()
    v20 = T800FlatV20ScalePPORunnerCfg().to_dict()

    assert v20["run_name"] == "v20-scale"
    assert v20["num_steps_per_env"] == 32
    assert v20["algorithm"]["num_learning_epochs"] == 2
    assert v20["algorithm"]["num_mini_batches"] == 16
    assert v20["max_iterations"] is None
    assert v20["algorithm"]["schedule"] == "adaptive"
    assert v20["algorithm"]["critic_learning_rate"] == 2.0e-5
    assert v20["algorithm"]["optimizer_fused"] is False
    assert v20["algorithm"]["learning_rate_min"] == 1.0e-5
    assert v20["algorithm"]["learning_rate_max"] == 2.0e-4
    assert v20["actor"]["distribution_cfg"]["init_std"] == 0.05
    assert v20["actor"]["distribution_cfg"]["std_range"] == (0.001, 0.5)
    assert v20["critic"]["hidden_dims"] == [2048, 1024, 512]

    v19["run_name"] = "v20-scale"
    v19["num_steps_per_env"] = 32
    v19["algorithm"]["num_learning_epochs"] = 2
    v19["algorithm"]["num_mini_batches"] = 16
    v19["algorithm"]["schedule"] = "adaptive"
    v19["algorithm"]["critic_learning_rate"] = 2.0e-5
    v19["algorithm"]["optimizer_fused"] = False
    v19["algorithm"]["learning_rate_min"] = 1.0e-5
    v19["algorithm"]["learning_rate_max"] = 2.0e-4
    v19["algorithm"]["shared_kl_adaptation"] = False
    v19["actor"]["distribution_cfg"]["init_std"] = 0.05
    v19["actor"]["distribution_cfg"]["std_range"] = (0.001, 0.5)
    v19["critic"]["hidden_dims"] = [2048, 1024, 512]
    for projection_name in ("proprioception_projection", "action_projection"):
        v19["actor"]["nodes"][projection_name]["cell"] = {
            "class_name": "TokenProjectionCell",
            "output_dim": 256,
        }
    v19["actor"]["nodes"]["attention_blocks"]["cell"].update(
        command_ffn_dim=344,
        command_ffn_type="swiglu",
        ffn_dim=344,
        ffn_type="swiglu",
        causal=True,
    )
    assert v20 == v19


def test_v20_doubles_critic_widths_without_mutating_v19() -> None:
    v19 = T800FlatV19ScalePPORunnerCfg()
    v20 = T800FlatV20ScalePPORunnerCfg()

    assert v19.critic.hidden_dims == [1024, 512, 256]
    assert v20.critic.hidden_dims == [2048, 1024, 512]


def test_v20_actor_uses_linear_history_projections_and_command_mlp() -> None:
    runner_cfg = T800FlatV20ScalePPORunnerCfg()
    observations = TensorDict(
        {
            "proprioception": torch.randn(2, 5, 56),
            "action": torch.randn(2, 4, 25),
            "command": torch.randn(2, 14, 99),
        },
        batch_size=[2],
    )
    actor_cfg = runner_cfg.actor.to_dict()
    actor_cfg.pop("class_name")

    actor = ModelGraph(observations, runner_cfg.obs_groups, "actor", 25, **actor_cfg)

    assert actor.nodes["proprioception_projection"].projection.in_features == 56
    assert actor.nodes["proprioception_projection"].projection.out_features == 256
    assert actor.nodes["action_projection"].projection.in_features == 25
    assert actor.nodes["action_projection"].projection.out_features == 256
    assert actor.nodes["command_projection"].mlp[0].in_features == 99
    for block in actor.nodes["attention_blocks"].blocks:
        assert block.temporal_self_attention.causal
        assert block.command_self_attention.ffn.__class__.__name__ == "SwiGLU"
        assert block.command_self_attention.ffn.gate.out_features == 344
        assert block.cross_attention.ffn.__class__.__name__ == "SwiGLU"
        assert block.cross_attention.ffn.gate.out_features == 344
    assert actor(observations).shape == (2, 25)


def test_v20_task_registration_points_to_v20_configs() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v20-scale")

    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v20:T800FlatWoStateEstimationEnvCfgV20Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".rsl_rl_ppo_cfg_v20:T800FlatV20ScalePPORunnerCfg")


def test_command_v1_uses_only_behavioral_names() -> None:
    source = Path(__file__).parents[1] / ("source/engineai_rl_lab/engineai_rl_lab/tasks/tracking/mdp/command_v1.py")

    assert "sonic" not in source.read_text().lower()
