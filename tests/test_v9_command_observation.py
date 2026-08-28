from __future__ import annotations

from types import SimpleNamespace

import engineai_rl_lab.tasks.tracking.mdp as mdp
import torch
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v9 import (
    T800FlatV9PPORunnerCfg,
    T800FlatV9ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v9_n1 import (
    T800FlatV9N1PPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v10 import (
    T800FlatV10PPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v11 import (
    T800FlatV11PPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v11_n1 import (
    T800FlatV11N1PPORunnerCfg,
    T800FlatV11N1ScaleLafanPPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v11_n2 import (
    T800FlatV11N2PPORunnerCfg,
    T800FlatV11N2ScaleLafanPPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v12 import (
    V12_COMMAND_TOPOLOGY,
    T800FlatV12PPORunnerCfg,
    T800FlatV12ScaleLafanPPORunnerCfg,
    T800FlatV12ScaleLafanSonicPPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v8 import T800FlatWoStateEstimationEnvCfgV8Scale
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v9 import (
    V9_FRAME_OFFSETS,
    T800FlatWoStateEstimationEnvCfgV9Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v9_n1 import (
    T800FlatWoStateEstimationEnvCfgV9N1,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v10 import (
    T800FlatWoStateEstimationEnvCfgV10,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v11 import (
    T800FlatWoStateEstimationEnvCfgV11,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v11_n1 import (
    T800FlatWoStateEstimationEnvCfgV11N1,
    T800FlatWoStateEstimationEnvCfgV11N1ScaleLafan,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v11_n2 import (
    T800FlatWoStateEstimationEnvCfgV11N2,
    T800FlatWoStateEstimationEnvCfgV11N2ScaleLafan,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v12 import (
    T800FlatWoStateEstimationEnvCfgV12,
    T800FlatWoStateEstimationEnvCfgV12ScaleLafan,
    T800FlatWoStateEstimationEnvCfgV12ScaleLafanSonic,
)
from engineai_rl_lab.tasks.tracking.mdp.observations import (
    _pose_by_entity_xz,
    motion_body_pose_and_error_b_window_by_entity,
    motion_body_pose_b_window_by_entity,
    motion_body_pose_b_window_by_entity_xz,
)
from rsl_rl.models import MLPModel, ModelGraph
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


def test_v9_command_doubles_last_dimension_and_preserves_v8_target_positions() -> None:
    env, command = _fake_env()
    frame_offsets = [-1, 0, 1]

    target = motion_body_pose_b_window_by_entity(env, "motion", frame_offsets)
    target_and_error = motion_body_pose_and_error_b_window_by_entity(env, "motion", frame_offsets)

    assert target.shape == (1, 2, 27)
    assert target_and_error.shape == (1, 2, 54)
    v8_target = target.reshape(1, 2, 3, 9)
    v9_target = target_and_error[..., :27].reshape(1, 2, 3, 9)
    torch.testing.assert_close(v9_target[..., :3], v8_target[..., :3])
    assert command.sample_calls == 2


def test_v9_current_frame_error_is_zero_position_and_identity_rotation() -> None:
    env, _ = _fake_env()

    target_and_error = motion_body_pose_and_error_b_window_by_entity(env, "motion", [-1, 0, 1])
    error = target_and_error[..., 27:].reshape(1, 2, 3, 9)

    torch.testing.assert_close(error[:, :, 1, :3], torch.zeros(1, 2, 3), atol=1.0e-6, rtol=0.0)
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(1, 2, -1)
    torch.testing.assert_close(error[:, :, 1, 3:], identity_6d, atol=1.0e-6, rtol=0.0)


def test_v9_rotation_uses_contiguous_rotated_x_and_z_axes() -> None:
    positions = torch.zeros(1, 1, 2, 3)
    orientations = torch.zeros(1, 1, 2, 4)
    orientations[..., -1] = 1.0

    pose = _pose_by_entity_xz(positions, orientations)
    expected_xz = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(1, 2, -1)
    torch.testing.assert_close(pose[..., 3:9], expected_xz, atol=1.0e-6, rtol=0.0)


def test_v11_command_is_exactly_v9_target_half() -> None:
    env, _ = _fake_env()
    frame_offsets = [-1, 0, 1]

    target_only = motion_body_pose_b_window_by_entity_xz(env, "motion", frame_offsets)
    target_and_error = motion_body_pose_and_error_b_window_by_entity(env, "motion", frame_offsets)

    assert target_only.shape == (1, 2, 27)
    torch.testing.assert_close(target_only, target_and_error[..., :27])


def test_target_only_command_preserves_beyondminic_invalid_offset_mask() -> None:
    env, _ = _fake_env()
    frame_offsets = [-2, 0, 2]

    target_only = motion_body_pose_b_window_by_entity_xz(env, "motion", frame_offsets, zero_invalid_offsets=True)
    target_and_error = motion_body_pose_and_error_b_window_by_entity(
        env, "motion", frame_offsets, zero_invalid_offsets=True
    )

    assert target_only.shape == (1, 2, 27)
    torch.testing.assert_close(target_only, target_and_error[..., :27])
    target_frames = target_only.reshape(1, 2, 3, 9)
    assert torch.count_nonzero(target_frames[:, :, 0]) == 0
    assert torch.count_nonzero(target_frames[:, :, 1]) > 0
    assert torch.count_nonzero(target_frames[:, :, 2]) == 0


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
    assert env_cfg.terminations.body_pos.params["threshold"] == 0.7
    assert env_cfg.commands.motion.max_num_load_motions is None
    assert runner_cfg.run_name == "v9_scale_lafan"
    assert runner_cfg.num_steps_per_env == 64
    assert runner_cfg.save_interval == 500
    assert runner_cfg.actor.nodes["attention_blocks"]["cell"]["ffn_dim"] == 384
    assert runner_cfg.critic.hidden_dims == [1024, 512, 256]


def test_v9_single_motion_overfit_keeps_mdp_but_removes_randomization() -> None:
    env_cfg = T800FlatWoStateEstimationEnvCfgV9Scale()
    rewards = env_cfg.rewards
    terminations = env_cfg.terminations
    observations = env_cfg.observations

    env_cfg.enable_single_motion_overfit()

    assert env_cfg.rewards is rewards
    assert env_cfg.terminations is terminations
    assert env_cfg.observations is observations
    assert env_cfg.commands.motion.max_num_load_motions is None
    assert env_cfg.commands.motion.motion_load_workers == 1
    assert env_cfg.commands.motion.adp_samp_failure_rate_max_over_mean is None
    assert env_cfg.commands.motion.pre_failure_sample_window == 0
    assert env_cfg.commands.motion.pose_range == {}
    assert env_cfg.commands.motion.velocity_range == {}
    assert env_cfg.commands.motion.joint_position_range == (0.0, 0.0)
    assert not env_cfg.observations.policy.enable_corruption
    assert env_cfg.events.physics_material is None
    assert env_cfg.events.add_joint_default_pos is None
    assert env_cfg.events.base_com is None
    assert env_cfg.events.push_robot is None


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


def test_v9_n1_flattens_every_actor_group_into_large_mlp() -> None:
    env_cfg = T800FlatWoStateEstimationEnvCfgV9N1()
    runner_cfg = T800FlatV9N1PPORunnerCfg()
    observations = TensorDict(
        {
            "policy": torch.randn(2, 5 * 53),
            "action": torch.randn(2, 4 * 25),
            "command": torch.randn(2, 14 * 162),
            "critic": torch.randn(2, 392),
        },
        batch_size=[2],
    )
    actor_cfg = runner_cfg.actor.to_dict()
    actor_cfg.pop("class_name")
    for legacy_key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
        actor_cfg.pop(legacy_key, None)
    actor = MLPModel(observations, runner_cfg.obs_groups, "actor", 25, **actor_cfg)

    assert env_cfg.observations.policy.flatten_history_dim
    assert env_cfg.observations.action.flatten_history_dim
    assert env_cfg.observations.command.link_pose_b.func is mdp.motion_body_pose_and_error_b_window_flat
    assert runner_cfg.actor.hidden_dims == [1024, 512, 256]
    assert runner_cfg.critic.hidden_dims == [1024, 512, 256]
    assert actor.obs_dim == 5 * 53 + 4 * 25 + 14 * 162
    assert actor(observations).shape == (2, 25)


def test_v10_only_changes_requested_adaptive_sampling_parameters() -> None:
    baseline_cfg = T800FlatWoStateEstimationEnvCfgV9N1()
    baseline_runner_cfg = T800FlatV9N1PPORunnerCfg()
    env_cfg = T800FlatWoStateEstimationEnvCfgV10()
    runner_cfg = T800FlatV10PPORunnerCfg()

    assert env_cfg.commands.motion.uniform_sampling_rate == 0.1
    assert env_cfg.commands.motion.adp_samp_failure_rate_max_over_mean == 200.0
    assert env_cfg.commands.motion.pre_failure_sample_window == 200

    baseline_motion_cfg = baseline_cfg.commands.motion.to_dict()
    v10_motion_cfg = env_cfg.commands.motion.to_dict()
    for field_name in (
        "uniform_sampling_rate",
        "adp_samp_failure_rate_max_over_mean",
        "pre_failure_sample_window",
    ):
        baseline_motion_cfg.pop(field_name)
        v10_motion_cfg.pop(field_name)
    assert v10_motion_cfg == baseline_motion_cfg

    assert env_cfg.observations.to_dict() == baseline_cfg.observations.to_dict()
    assert env_cfg.rewards.to_dict() == baseline_cfg.rewards.to_dict()
    assert env_cfg.terminations.to_dict() == baseline_cfg.terminations.to_dict()
    assert env_cfg.events.to_dict() == baseline_cfg.events.to_dict()

    baseline_runner = baseline_runner_cfg.to_dict()
    v10_runner = runner_cfg.to_dict()
    baseline_runner.pop("run_name")
    v10_runner.pop("run_name")
    assert v10_runner == baseline_runner
    assert runner_cfg.run_name == "v10"
    assert runner_cfg.actor.hidden_dims == [1024, 512, 256]
    assert runner_cfg.critic.hidden_dims == [1024, 512, 256]


def test_v11_inherits_v10_and_only_removes_command_error_features() -> None:
    v10_env_cfg = T800FlatWoStateEstimationEnvCfgV10()
    v10_runner_cfg = T800FlatV10PPORunnerCfg()
    env_cfg = T800FlatWoStateEstimationEnvCfgV11()
    runner_cfg = T800FlatV11PPORunnerCfg()
    observations = TensorDict(
        {
            "policy": torch.randn(2, 5 * 53),
            "action": torch.randn(2, 4 * 25),
            "command": torch.randn(2, 14 * 81),
            "critic": torch.randn(2, 392),
        },
        batch_size=[2],
    )
    actor_cfg = runner_cfg.actor.to_dict()
    actor_cfg.pop("class_name")
    for legacy_key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
        actor_cfg.pop(legacy_key, None)
    actor = MLPModel(observations, runner_cfg.obs_groups, "actor", 25, **actor_cfg)

    command_term = env_cfg.observations.command.link_pose_b
    assert command_term.func is mdp.motion_body_pose_b_window_xz_flat
    assert command_term.params["frame_offsets"] == V9_FRAME_OFFSETS
    assert env_cfg.commands.motion.to_dict() == v10_env_cfg.commands.motion.to_dict()
    assert env_cfg.commands.motion.uniform_sampling_rate == 0.1
    assert env_cfg.commands.motion.adp_samp_failure_rate_max_over_mean == 200.0
    assert env_cfg.commands.motion.pre_failure_sample_window == 200
    assert env_cfg.rewards.action_rate_l2.weight == -0.075
    assert env_cfg.terminations.body_pos.params["threshold"] == 0.7
    assert env_cfg.observations.policy.flatten_history_dim
    assert env_cfg.observations.action.flatten_history_dim
    v10_runner = v10_runner_cfg.to_dict()
    v11_runner = runner_cfg.to_dict()
    v10_runner.pop("run_name")
    v11_runner.pop("run_name")
    assert v11_runner == v10_runner
    assert runner_cfg.run_name == "v11"
    assert runner_cfg.actor.hidden_dims == [1024, 512, 256]
    assert runner_cfg.critic.hidden_dims == [1024, 512, 256]
    assert actor.obs_dim == 5 * 53 + 4 * 25 + 14 * 81
    assert actor(observations).shape == (2, 25)


def test_v11_n1_uses_v9_actor_with_structured_target_only_tokens() -> None:
    v11_env_cfg = T800FlatWoStateEstimationEnvCfgV11()
    env_cfg = T800FlatWoStateEstimationEnvCfgV11N1()
    v9_runner_cfg = T800FlatV9PPORunnerCfg()
    runner_cfg = T800FlatV11N1PPORunnerCfg()
    observations = TensorDict(
        {
            "policy": torch.randn(2, 5, 53),
            "action": torch.randn(2, 4, 25),
            "command": torch.randn(2, 14, 81),
            "critic": torch.randn(2, 392),
        },
        batch_size=[2],
    )
    actor_cfg = runner_cfg.actor.to_dict()
    actor_cfg.pop("class_name")
    actor = ModelGraph(observations, runner_cfg.obs_groups, "actor", 25, **actor_cfg)

    command_term = env_cfg.observations.command.link_pose_b
    assert command_term.func is mdp.motion_body_pose_b_window_by_entity_xz
    assert command_term.params["frame_offsets"] == V9_FRAME_OFFSETS
    assert not env_cfg.observations.policy.flatten_history_dim
    assert not env_cfg.observations.action.flatten_history_dim
    assert env_cfg.commands.motion.to_dict() == v11_env_cfg.commands.motion.to_dict()
    assert env_cfg.commands.motion.uniform_sampling_rate == 0.1
    assert env_cfg.commands.motion.adp_samp_failure_rate_max_over_mean == 200.0
    assert env_cfg.commands.motion.pre_failure_sample_window == 200
    assert env_cfg.rewards.to_dict() == v11_env_cfg.rewards.to_dict()
    assert env_cfg.terminations.to_dict() == v11_env_cfg.terminations.to_dict()

    v9_runner = v9_runner_cfg.to_dict()
    v11_n1_runner = runner_cfg.to_dict()
    v9_runner.pop("run_name")
    v11_n1_runner.pop("run_name")
    assert v11_n1_runner == v9_runner
    assert runner_cfg.run_name == "v11-n1"
    assert runner_cfg.actor.nodes["attention_blocks"]["cell"]["num_blocks"] == 2
    assert runner_cfg.actor.nodes["attention_blocks"]["cell"]["ffn_dim"] == 384
    assert runner_cfg.algorithm.num_mini_batches == 32
    assert runner_cfg.algorithm.actor_learning_rate == 5.0e-5
    assert actor.nodes["command_projection"].mlp[0].in_features == 81
    assert actor(observations).shape == (2, 25)


def test_v11_n1_scale_lafan_only_changes_the_motion_pool_and_run_name() -> None:
    baseline_env_cfg = T800FlatWoStateEstimationEnvCfgV11N1()
    env_cfg = T800FlatWoStateEstimationEnvCfgV11N1ScaleLafan()
    runner_cfg = T800FlatV11N1ScaleLafanPPORunnerCfg()

    baseline_motion_cfg = baseline_env_cfg.commands.motion.to_dict()
    scale_motion_cfg = env_cfg.commands.motion.to_dict()
    for field_name in ("motion_file", "motion_chunk_frames", "motion_load_workers"):
        baseline_motion_cfg.pop(field_name)
        scale_motion_cfg.pop(field_name)
    assert scale_motion_cfg == baseline_motion_cfg
    assert env_cfg.commands.motion.motion_file.endswith("lafan_and_slow2x_v0.yaml")
    assert env_cfg.commands.motion.motion_chunk_frames == 8_388_608
    assert runner_cfg.run_name == "v11-n1-scale-lafan"


def test_v11_n2_only_changes_marmotlab_aligned_actor_optimization() -> None:
    n1_env_cfg = T800FlatWoStateEstimationEnvCfgV11N1()
    env_cfg = T800FlatWoStateEstimationEnvCfgV11N2()
    n1_runner_cfg = T800FlatV11N1PPORunnerCfg()
    runner_cfg = T800FlatV11N2PPORunnerCfg()

    assert env_cfg.to_dict() == n1_env_cfg.to_dict()
    assert runner_cfg.run_name == "v11-n2"
    assert runner_cfg.actor.nodes["attention_blocks"]["cell"]["num_blocks"] == 2
    assert runner_cfg.actor.nodes["attention_blocks"]["cell"]["ffn_dim"] == 384
    assert runner_cfg.algorithm.num_mini_batches == 64
    assert runner_cfg.algorithm.actor_learning_rate == 2.0e-5
    assert runner_cfg.algorithm.critic_learning_rate == 5.0e-4

    n1_runner = n1_runner_cfg.to_dict()
    n2_runner = runner_cfg.to_dict()
    n1_runner.pop("run_name")
    n2_runner.pop("run_name")
    n1_runner["algorithm"]["num_mini_batches"] = 64
    n1_runner["algorithm"]["actor_learning_rate"] = 2.0e-5
    assert n2_runner == n1_runner


def test_v11_n2_scale_lafan_only_changes_the_motion_pool_and_run_name() -> None:
    baseline_env_cfg = T800FlatWoStateEstimationEnvCfgV11N2()
    env_cfg = T800FlatWoStateEstimationEnvCfgV11N2ScaleLafan()
    runner_cfg = T800FlatV11N2ScaleLafanPPORunnerCfg()

    baseline_motion_cfg = baseline_env_cfg.commands.motion.to_dict()
    scale_motion_cfg = env_cfg.commands.motion.to_dict()
    for field_name in ("motion_file", "motion_chunk_frames", "motion_load_workers"):
        baseline_motion_cfg.pop(field_name)
        scale_motion_cfg.pop(field_name)
    assert scale_motion_cfg == baseline_motion_cfg
    assert env_cfg.commands.motion.motion_file.endswith("lafan_and_slow2x_v0.yaml")
    assert env_cfg.commands.motion.motion_chunk_frames == 8_388_608
    assert runner_cfg.run_name == "v11-n2-scale-lafan"


def test_v12_adds_named_proprioception_with_gravity_and_gelu_projections() -> None:
    v11_env_cfg = T800FlatWoStateEstimationEnvCfgV11N2()
    env_cfg = T800FlatWoStateEstimationEnvCfgV12()
    v11_runner_cfg = T800FlatV11N2PPORunnerCfg()
    runner_cfg = T800FlatV12PPORunnerCfg()
    observations = TensorDict(
        {
            "proprioception": torch.randn(2, 5, 56),
            "action": torch.randn(2, 4, 25),
            "command": torch.randn(2, 14, 81),
            "critic": torch.randn(2, 392),
        },
        batch_size=[2],
    )
    actor_cfg = runner_cfg.actor.to_dict()
    actor_cfg.pop("class_name")
    actor = ModelGraph(observations, runner_cfg.obs_groups, "actor", 25, **actor_cfg)

    assert env_cfg.observations.policy is None
    assert env_cfg.observations.proprioception.history_length == 5
    assert not env_cfg.observations.proprioception.flatten_history_dim
    assert env_cfg.observations.proprioception.actions is None
    assert env_cfg.observations.proprioception.projected_gravity.func is mdp.projected_gravity
    assert env_cfg.observations.proprioception.projected_gravity.noise.n_min == -0.05
    assert env_cfg.observations.proprioception.projected_gravity.noise.n_max == 0.05
    assert env_cfg.observations.action.to_dict() == v11_env_cfg.observations.action.to_dict()
    v11_policy_cfg = v11_env_cfg.observations.policy.to_dict()
    v12_proprioception_cfg = env_cfg.observations.proprioception.to_dict()
    v12_proprioception_cfg.pop("projected_gravity")
    assert v12_proprioception_cfg == v11_policy_cfg
    assert env_cfg.commands.to_dict() == v11_env_cfg.commands.to_dict()
    assert env_cfg.rewards.to_dict() == v11_env_cfg.rewards.to_dict()
    assert env_cfg.terminations.to_dict() == v11_env_cfg.terminations.to_dict()

    assert runner_cfg.run_name == "v12"
    assert runner_cfg.obs_groups["actor"] == ["proprioception", "action", "command"]
    assert runner_cfg.obs_groups["critic"] == v11_runner_cfg.obs_groups["critic"]
    assert "policy_projection" not in runner_cfg.actor.nodes
    assert "proprioception_projection" in runner_cfg.actor.nodes
    assert "action_projection" in runner_cfg.actor.nodes
    assert "token_interleaver" in runner_cfg.actor.nodes
    for projection_name in ("proprioception_projection", "action_projection", "command_projection"):
        assert runner_cfg.actor.nodes[projection_name]["cell"]["activation"] == "gelu"
    topology_cfg = runner_cfg.actor.nodes["topology_projection"]["cell"]
    assert topology_cfg["topology"] == V12_COMMAND_TOPOLOGY
    assert topology_cfg["hidden_dims"] == [64]
    assert topology_cfg["activation"] == "gelu"
    topology = torch.tensor(V12_COMMAND_TOPOLOGY)
    assert topology.shape == (14, 9)
    assert torch.all(topology[:, :4].sum(dim=1) == 1)
    assert torch.all(topology[:, 4:7].sum(dim=1) == 1)
    assert torch.all((topology[:, 7] >= 0) & (topology[:, 7] <= 1))
    assert set(topology[:, 8].tolist()) == {0.0, 1.0}
    topology_projection = actor.nodes["topology_projection"].projection
    assert topology_projection[0].weight.shape == (64, 9)
    assert isinstance(topology_projection[1], torch.nn.GELU)
    assert topology_projection[2].weight.shape == (256, 64)
    v11_actor_cfg = v11_runner_cfg.actor.to_dict()
    v12_actor_cfg = runner_cfg.actor.to_dict()
    v11_actor_cfg["nodes"]["proprioception_projection"] = v11_actor_cfg["nodes"].pop("policy_projection")
    for route in v11_actor_cfg["routes"]:
        if route["source"] == "inputs.policy":
            route["source"] = "inputs.proprioception"
        for endpoint in ("source", "target"):
            route[endpoint] = route[endpoint].replace(
                "nodes.policy_projection.",
                "nodes.proprioception_projection.",
            )
    for projection_name in ("proprioception_projection", "action_projection", "command_projection"):
        v11_actor_cfg["nodes"][projection_name]["cell"]["activation"] = "gelu"
    v11_actor_cfg["nodes"]["topology_projection"]["cell"].update(
        topology=V12_COMMAND_TOPOLOGY,
        hidden_dims=[64],
        activation="gelu",
    )
    assert v12_actor_cfg == v11_actor_cfg
    assert runner_cfg.actor.nodes["attention_blocks"]["cell"]["num_blocks"] == 2
    assert runner_cfg.actor.nodes["attention_blocks"]["cell"]["ffn_type"] == "swiglu"
    assert runner_cfg.algorithm.to_dict() == v11_runner_cfg.algorithm.to_dict()
    assert actor(observations).shape == (2, 25)


def test_v12_scale_lafan_only_changes_the_motion_pool_and_run_name() -> None:
    baseline_env_cfg = T800FlatWoStateEstimationEnvCfgV12()
    env_cfg = T800FlatWoStateEstimationEnvCfgV12ScaleLafan()
    runner_cfg = T800FlatV12ScaleLafanPPORunnerCfg()

    baseline_motion_cfg = baseline_env_cfg.commands.motion.to_dict()
    scale_motion_cfg = env_cfg.commands.motion.to_dict()
    for field_name in ("motion_file", "motion_chunk_frames", "motion_load_workers"):
        baseline_motion_cfg.pop(field_name)
        scale_motion_cfg.pop(field_name)
    assert scale_motion_cfg == baseline_motion_cfg
    assert env_cfg.commands.motion.motion_file.endswith("lafan_and_slow2x_v0.yaml")
    assert env_cfg.commands.motion.motion_chunk_frames == 8_388_608
    assert runner_cfg.run_name == "v12-scale-lafan"


def test_v12_scale_lafan_sonic_only_extends_the_motion_pool_and_run_name() -> None:
    lafan_env_cfg = T800FlatWoStateEstimationEnvCfgV12ScaleLafan()
    env_cfg = T800FlatWoStateEstimationEnvCfgV12ScaleLafanSonic()
    lafan_runner_cfg = T800FlatV12ScaleLafanPPORunnerCfg()
    runner_cfg = T800FlatV12ScaleLafanSonicPPORunnerCfg()

    lafan_motion_cfg = lafan_env_cfg.commands.motion.to_dict()
    sonic_motion_cfg = env_cfg.commands.motion.to_dict()
    lafan_motion_cfg.pop("motion_file")
    sonic_motion_cfg.pop("motion_file")
    assert sonic_motion_cfg == lafan_motion_cfg
    assert env_cfg.commands.motion.motion_file.endswith("lafan_slow2x_and_sonic_v0.yaml")
    assert runner_cfg.run_name == "v12-scale-lafan-sonic"

    runner_cfg_dict = runner_cfg.to_dict()
    lafan_runner_cfg_dict = lafan_runner_cfg.to_dict()
    runner_cfg_dict.pop("run_name")
    lafan_runner_cfg_dict.pop("run_name")
    assert runner_cfg_dict == lafan_runner_cfg_dict
