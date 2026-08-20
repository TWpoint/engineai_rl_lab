from __future__ import annotations

import engineai_rl_lab.tasks.tracking.mdp as mdp
import torch
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v13 import (
    T800FlatV13ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v12 import T800_V0_MOTION_MANIFEST
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v13 import (
    T800_BEYONDMINIC_FRAME_OFFSETS,
    T800FlatWoStateEstimationEnvCfgV13Scale,
)


def test_v13_inherits_v12_scale_dataset_and_keeps_command_shape_contract() -> None:
    cfg = T800FlatWoStateEstimationEnvCfgV13Scale()

    assert cfg.commands.motion.motion_file == T800_V0_MOTION_MANIFEST
    assert cfg.commands.motion.max_num_load_motions == 200_000
    assert cfg.commands.motion.motion_load_workers == 8
    assert cfg.commands.motion.working_set_replacement is False
    assert cfg.commands.motion.sequence_length_agnostic is False
    assert cfg.commands.motion.adaptive_sampling_alpha == 0.001
    assert cfg.commands.motion.adaptive_kernel_size == 5
    assert cfg.commands.motion.adaptive_kernel_lambda == 0.8
    assert cfg.observations.proprioception.history_length == 5
    assert not cfg.observations.proprioception.flatten_history_dim
    assert cfg.observations.action.history_length == 4
    assert not cfg.observations.action.flatten_history_dim
    assert cfg.observations.command.link_pose_b.func is mdp.motion_body_pose_and_error_b_window_by_entity
    assert cfg.observations.command.link_pose_b.params["frame_offsets"] == T800_BEYONDMINIC_FRAME_OFFSETS
    assert cfg.observations.command.link_pose_b.params["zero_invalid_offsets"] is True
    assert T800_BEYONDMINIC_FRAME_OFFSETS == [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    assert cfg.observations.critic.command.func is mdp.motion_body_pose_and_error_b_window_flat
    assert cfg.observations.critic.command.params["frame_offsets"] == [0]
    assert cfg.observations.critic.command.params["zero_invalid_offsets"] is True
    assert cfg.observations.critic.target_link_pos_b is None
    assert cfg.observations.critic.target_link_ori_b is None
    assert cfg.observations.critic.body_lin_vel.func is mdp.robot_body_lin_vel_b
    assert cfg.observations.critic.body_ang_vel.func is mdp.robot_body_ang_vel_b
    assert not hasattr(cfg.commands, "token_mask")
    assert cfg.commands.motion.uniform_sampling_rate == 0.2
    assert cfg.commands.motion.adp_samp_failure_rate_max_over_mean == 30.0
    assert cfg.commands.motion.pre_failure_sample_window == 0


def test_v13_uses_short_rollouts_and_fewer_optimizer_batches() -> None:
    runner = T800FlatV13ScalePPORunnerCfg()

    assert runner.num_steps_per_env == 64
    assert runner.save_interval == 250
    assert runner.sync_adaptive_sampling_all_gpus_freq == 10
    assert runner.motion_resample_frequency == 0
    assert runner.stagger_motion_working_set_refresh is False
    assert runner.algorithm.num_mini_batches == 64
    assert runner.algorithm.num_learning_epochs == 2
    assert runner.algorithm.actor_learning_rate == 2.0e-5
    assert runner.algorithm.critic_learning_rate == 1.0e-3
    assert runner.algorithm.schedule == "fixed"
    assert runner.algorithm.max_grad_norm == 1.0
    assert len(runner.actor.nodes["topology_projection"]["cell"]["topology"]) == 14
    attention = runner.actor.nodes["attention_blocks"]["cell"]
    assert attention["command_ffn_dim"] == 512
    assert attention["ffn_dim"] == 512
    assert attention["ffn_type"] == "mlp"
    assert attention["activation"] == "gelu"
    assert runner.actor.nodes["action_decoder"]["cell"]["zero_init_output"] is True
    assert runner.actor.distribution_cfg.std_range == (0.05, 2.0)


def test_v13_action_decoder_zero_initializes_only_its_output_layer() -> None:
    from rsl_rl.models.graph.cells import LinearCell

    decoder_cfg = T800FlatV13ScalePPORunnerCfg().actor.nodes["action_decoder"]["cell"].copy()
    decoder_cfg.pop("class_name")
    decoder = LinearCell(input_dim=256, output_dim=25, **decoder_cfg)

    assert torch.count_nonzero(decoder.linear.weight) == 0
    assert torch.count_nonzero(decoder.linear.bias) == 0


def test_v13_uses_beyondminic_global_tracking_rewards() -> None:
    rewards = T800FlatWoStateEstimationEnvCfgV13Scale().rewards

    assert rewards.alive.weight == 1.0
    assert rewards.motion_global_root_height.params["std"] == 0.3
    assert rewards.motion_body_pos.func is mdp.motion_global_body_position_error_exp
    assert rewards.motion_body_pos.params["std"] == 0.3
    assert rewards.motion_body_ori.func is mdp.motion_global_body_orientation_error_exp
    assert rewards.action_rate_l2.weight == -0.05
    assert rewards.joint_acc_l2.weight == -2.5e-7
    assert not hasattr(rewards, "joint_torques_l2")


def test_v13_uses_beyondminic_global_body_termination() -> None:
    terms = T800FlatWoStateEstimationEnvCfgV13Scale().terminations

    assert not hasattr(terms, "time_out")
    assert terms.motion_time_out.time_out
    assert terms.invalid_robot_state is not None
    assert terms.body_pos.func is mdp.bad_global_motion_body_pos
    assert terms.body_pos.params == {"command_name": "motion", "threshold": 0.5}
    assert not hasattr(terms, "anchor_pos")
    assert not hasattr(terms, "anchor_ori")
    assert not hasattr(terms, "ee_body_pos")
