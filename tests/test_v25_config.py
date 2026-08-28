from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import engineai_rl_lab.tasks  # noqa: F401
import engineai_rl_lab.tasks.tracking.mdp as mdp
import gymnasium as gym
import pytest
import torch
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v24 import (
    T800FlatV24ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v25 import (
    V25_ACTOR_NUM_BLOCKS,
    V25_REWARD_GROUPS,
    T800FlatV25ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v24 import (
    T800FlatWoStateEstimationEnvCfgV24Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v25 import (
    T800_V25_FRAME_OFFSETS,
    T800FlatWoStateEstimationEnvCfgV25Scale,
)
from engineai_rl_lab.utils.v25_compact_multi_critic_ppo import V25CompactMultiCriticPPO
from engineai_rl_lab.utils.v25_multi_critic_ppo import (
    V25MultiCriticPPO,
    V25RolloutStorage,
    aggregate_grouped_rewards,
    bootstrap_grouped_timeouts,
    compute_grouped_gae,
    resolve_reward_groups,
)
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


@pytest.fixture(scope="module")
def env_cfgs() -> tuple[T800FlatWoStateEstimationEnvCfgV24Scale, T800FlatWoStateEstimationEnvCfgV25Scale]:
    return T800FlatWoStateEstimationEnvCfgV24Scale(), T800FlatWoStateEstimationEnvCfgV25Scale()


def _active_reward_terms(env_cfg: T800FlatWoStateEstimationEnvCfgV25Scale) -> tuple[str, ...]:
    return tuple(name for name, term_cfg in env_cfg.rewards.to_dict().items() if term_cfg is not None)


def _make_storage() -> tuple[V25RolloutStorage, TensorDict]:
    observations = TensorDict(
        {
            "actor": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "critic": torch.arange(10, dtype=torch.float32).reshape(2, 5),
        },
        batch_size=[2],
    )
    storage = V25RolloutStorage(
        "rl",
        num_envs=2,
        num_transitions_per_env=3,
        obs=observations,
        actions_shape=[2],
        num_reward_groups=3,
        device="cpu",
    )
    return storage, observations


def _make_transition(observations: TensorDict) -> RolloutStorage.Transition:
    transition = RolloutStorage.Transition()
    transition.observations = observations.clone()
    transition.actions = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    transition.rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    transition.dones = torch.tensor([[False], [True]])
    transition.values = torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
    transition.actions_log_prob = torch.tensor([[-0.1], [-0.2]])
    transition.distribution_params = (
        torch.tensor([[0.0, 0.1], [0.2, 0.3]]),
        torch.tensor([[0.5, 0.5], [0.6, 0.6]]),
    )
    return transition


def test_v25_environment_changes_only_action_rate_and_actor_offsets_from_v24(
    env_cfgs: tuple[T800FlatWoStateEstimationEnvCfgV24Scale, T800FlatWoStateEstimationEnvCfgV25Scale],
) -> None:
    v24, v25 = env_cfgs

    # V25 changes exactly the action-rate weight and actor command offsets.
    v25_as_v24 = v25.to_dict()
    v25_as_v24["rewards"]["action_rate_l2"]["weight"] = v24.rewards.action_rate_l2.weight
    v25_as_v24["observations"]["command"]["link_pose_b"]["params"]["frame_offsets"] = (
        v24.observations.command.link_pose_b.params["frame_offsets"]
    )
    assert v25_as_v24 == v24.to_dict()
    assert v25.observations.command.link_pose_b.func is mdp.motion_body_pose_reference_anchor_window_by_entity_xz
    assert v25.observations.command.link_pose_b.params["frame_offsets"] == list(range(-5, 6))
    assert len(v25.observations.command.link_pose_b.params["frame_offsets"]) * 9 == 99
    assert v25.observations.critic.local_command.func is mdp.motion_body_pose_reference_anchor_window_xz_flat
    assert v24.rewards.action_rate_l2.weight == -0.1
    assert v25.rewards.action_rate_l2.weight == -0.05
    assert len(v25.commands.motion.body_names) == 14
    assert "LINK_HEAD_YAW" not in v25.commands.motion.body_names
    for tracking_cfg in (v25.commands.to_dict(), v25.rewards.to_dict(), v25.terminations.to_dict()):
        assert "LINK_HEAD_YAW" not in repr(tracking_cfg)


def test_v25_action_rate_override_does_not_mutate_v24_or_config_defaults() -> None:
    v24_before = T800FlatWoStateEstimationEnvCfgV24Scale()
    v25_first = T800FlatWoStateEstimationEnvCfgV25Scale()

    assert v25_first.rewards is not v24_before.rewards
    assert v25_first.rewards.action_rate_l2 is not v24_before.rewards.action_rate_l2
    assert v24_before.rewards.action_rate_l2.weight == -0.1
    assert v25_first.rewards.action_rate_l2.weight == -0.05
    assert v25_first.observations.command.link_pose_b.params["frame_offsets"] == T800_V25_FRAME_OFFSETS
    assert v24_before.observations.command.link_pose_b.params["frame_offsets"] != T800_V25_FRAME_OFFSETS

    # Mutating one constructed V25 config must not leak into either class's
    # inherited mutable config tree or a subsequently constructed instance.
    v25_first.rewards.action_rate_l2.weight = -99.0
    v25_first.observations.command.link_pose_b.params["frame_offsets"][0] = -99
    v24_after = T800FlatWoStateEstimationEnvCfgV24Scale()
    v25_after = T800FlatWoStateEstimationEnvCfgV25Scale()
    assert v24_after.rewards.action_rate_l2.weight == -0.1
    assert v25_after.rewards.action_rate_l2.weight == -0.05
    assert v25_after.observations.command.link_pose_b.params["frame_offsets"] == T800_V25_FRAME_OFFSETS


def test_v25_registration_and_runner_only_change_actor_depth_and_multi_critic_fields() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v25-scale")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v25:T800FlatWoStateEstimationEnvCfgV25Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".rsl_rl_ppo_cfg_v25:T800FlatV25ScalePPORunnerCfg")

    v24 = T800FlatV24ScalePPORunnerCfg().to_dict()
    v25 = T800FlatV25ScalePPORunnerCfg().to_dict()
    assert v25["run_name"] == "v25-scale"
    for key in v24.keys() - {"run_name", "actor", "algorithm"}:
        assert v25[key] == v24[key], f"Unexpected V25 runner difference in {key!r}"

    v24_actor = deepcopy(v24["actor"])
    v25_actor = deepcopy(v25["actor"])
    assert v24_actor["nodes"]["attention_blocks"]["cell"]["num_blocks"] == 2
    assert v25_actor["nodes"]["attention_blocks"]["cell"]["num_blocks"] == V25_ACTOR_NUM_BLOCKS == 3
    v25_actor["nodes"]["attention_blocks"]["cell"]["num_blocks"] = 2
    assert v25_actor == v24_actor

    v24_algorithm = dict(v24["algorithm"])
    v25_algorithm = dict(v25["algorithm"])
    assert v24_algorithm.pop("class_name") == "PPO"
    assert v25_algorithm.pop("class_name") == (
        "engineai_rl_lab.utils.v25_compact_multi_critic_ppo:V25CompactMultiCriticPPO"
    )
    assert v25_algorithm.pop("reward_groups") == V25_REWARD_GROUPS
    assert v25_algorithm.pop("value_loss_reduction") == "mean"
    assert v25_algorithm == v24_algorithm


def test_v25_actor_depth_override_does_not_mutate_v24_or_future_instances() -> None:
    v24_before = T800FlatV24ScalePPORunnerCfg()
    v25_first = T800FlatV25ScalePPORunnerCfg()

    assert v24_before.actor.nodes["attention_blocks"]["cell"]["num_blocks"] == 2
    assert v25_first.actor.nodes["attention_blocks"]["cell"]["num_blocks"] == 3
    v25_first.actor.nodes["attention_blocks"]["cell"]["num_blocks"] = 99

    assert T800FlatV24ScalePPORunnerCfg().actor.nodes["attention_blocks"]["cell"]["num_blocks"] == 2
    assert T800FlatV25ScalePPORunnerCfg().actor.nodes["attention_blocks"]["cell"]["num_blocks"] == 3


def test_v25_reward_groups_are_a_unique_exhaustive_partition_of_all_thirteen_rewards(
    env_cfgs: tuple[T800FlatWoStateEstimationEnvCfgV24Scale, T800FlatWoStateEstimationEnvCfgV25Scale],
) -> None:
    _, v25 = env_cfgs
    active_terms = _active_reward_terms(v25)
    flattened_terms = tuple(term for terms in V25_REWARD_GROUPS.values() for term in terms)
    expected_schema = {
        "global": [
            "motion_global_root_height",
            "motion_body_pos",
            "motion_body_ori",
            "motion_body_lin_vel",
            "motion_body_ang_vel",
            "motion_joint_pos",
            "motion_joint_vel",
        ],
        "local": [
            "motion_local_end_effector_pos",
            "motion_relative_body_pos",
            "motion_relative_body_ori",
        ],
        "regularization": ["alive", "action_rate_l2", "joint_limit"],
    }

    assert len(active_terms) == 13
    # Preserve the semantic schema recorded by the original p29526
    # multi-critic checkpoints; a merely exhaustive partition is not enough.
    assert expected_schema == V25_REWARD_GROUPS
    assert tuple(V25_REWARD_GROUPS) == ("global", "local", "regularization")
    assert len(flattened_terms) == len(set(flattened_terms)) == 13
    assert set(flattened_terms) == set(active_terms)

    group_names, resolved_groups = resolve_reward_groups(active_terms, V25_REWARD_GROUPS)
    assert group_names == ("global", "local", "regularization")
    assert resolved_groups == {name: tuple(terms) for name, terms in V25_REWARD_GROUPS.items()}


@pytest.mark.parametrize(
    ("groups", "error_pattern"),
    [
        ({"global": ["reward_a", "not_active"], "local": ["reward_b"]}, "unknown terms"),
        ({"global": ["reward_a"], "local": ["reward_a", "reward_b"]}, "duplicates"),
        ({"global": ["reward_a"]}, "do not cover"),
    ],
)
def test_resolve_reward_groups_rejects_unknown_duplicate_and_uncovered_terms(
    groups: Mapping[str, Sequence[str]],
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        resolve_reward_groups(("reward_a", "reward_b"), groups)


def test_aggregate_grouped_rewards_preserves_each_group_and_the_scalar_sum() -> None:
    reward_terms = {
        "height": torch.tensor([1.0, 2.0]),
        "pose": torch.tensor([0.5, 1.5]),
        "local": torch.tensor([3.0, 4.0]),
        "action_rate": torch.tensor([-0.1, -0.2]),
    }
    group_names = ("global", "local", "regularization")
    groups = {
        "global": ("height", "pose"),
        "local": ("local",),
        "regularization": ("action_rate",),
    }

    grouped = aggregate_grouped_rewards(reward_terms, group_names, groups, num_envs=2, device="cpu")
    expected = torch.tensor([[1.5, 3.0, -0.1], [3.5, 4.0, -0.2]])
    scalar_reward = torch.stack(tuple(reward_terms.values()), dim=-1).sum(dim=-1)

    assert grouped.shape == (2, 3)
    torch.testing.assert_close(grouped, expected)
    torch.testing.assert_close(grouped.sum(dim=-1), scalar_reward)
    with pytest.raises(RuntimeError, match="missing configured terms"):
        aggregate_grouped_rewards(
            {"height": reward_terms["height"]},
            group_names,
            groups,
            num_envs=2,
            device="cpu",
        )


def test_v25_rollout_storage_keeps_three_value_heads_but_one_policy_advantage() -> None:
    storage, observations = _make_storage()
    transition = _make_transition(observations)
    storage.add_transition(transition)

    assert storage.rewards.shape == (3, 2, 3)
    assert storage.values.shape == (3, 2, 3)
    assert storage.returns.shape == (3, 2, 3)
    assert storage.group_advantages.shape == (3, 2, 3)
    assert storage.advantages.shape == (3, 2, 1)
    torch.testing.assert_close(storage.rewards[0], transition.rewards)
    torch.testing.assert_close(storage.values[0], transition.values)

    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    assert batch.values.shape == (6, 3)
    assert batch.returns.shape == (6, 3)
    assert batch.advantages.shape == (6, 1)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("rewards", torch.zeros(2, 1)),
        ("values", torch.zeros(2, 1)),
    ],
)
def test_v25_rollout_storage_rejects_collapsed_reward_or_value_heads(
    field: str,
    bad_value: torch.Tensor,
) -> None:
    storage, observations = _make_storage()
    transition = _make_transition(observations)
    setattr(transition, field, bad_value)

    with pytest.raises(ValueError, match=rf"Transition {field} must have shape \(2, 3\)"):
        storage.add_transition(transition)
    assert storage.step == 0


def test_compute_grouped_gae_computes_each_head_and_returns_only_one_policy_advantage() -> None:
    rewards = torch.tensor([[[1.0, 10.0]], [[2.0, 20.0]], [[4.0, 40.0]]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[[False]], [[True]], [[False]]])
    last_values = torch.tensor([[8.0, 80.0]])

    returns, group_advantages, policy_advantages = compute_grouped_gae(
        rewards,
        values,
        dones,
        last_values,
        gamma=0.5,
        lam=1.0,
    )
    expected_group_advantages = torch.tensor([[[2.0, 20.0]], [[2.0, 20.0]], [[8.0, 80.0]]])

    torch.testing.assert_close(returns, expected_group_advantages)
    torch.testing.assert_close(group_advantages, expected_group_advantages)
    torch.testing.assert_close(policy_advantages, expected_group_advantages.sum(dim=-1, keepdim=True))
    assert group_advantages.shape == (3, 1, 2)
    assert policy_advantages.shape == (3, 1, 1)


def test_sum_of_grouped_gae_is_identical_to_gae_of_the_original_scalar_reward() -> None:
    rewards = torch.tensor(
        [
            [[1.0, 0.2, -0.1], [0.5, 0.7, -0.3]],
            [[0.2, 1.3, -0.4], [1.1, -0.1, -0.2]],
            [[0.8, 0.4, -0.2], [0.3, 0.9, -0.1]],
            [[0.6, 0.5, -0.5], [0.4, 0.2, -0.2]],
        ]
    )
    values = torch.tensor(
        [
            [[0.3, 0.1, -0.2], [0.4, 0.2, -0.1]],
            [[0.2, 0.5, -0.1], [0.7, 0.1, -0.2]],
            [[0.6, 0.2, -0.3], [0.2, 0.4, -0.1]],
            [[0.4, 0.3, -0.2], [0.3, 0.1, -0.2]],
        ]
    )
    dones = torch.tensor([[[False], [False]], [[False], [True]], [[True], [False]], [[False], [False]]])
    last_values = torch.tensor([[0.5, 0.2, -0.1], [0.3, 0.4, -0.2]])

    grouped_returns, grouped_advantages, policy_advantages = compute_grouped_gae(
        rewards,
        values,
        dones,
        last_values,
        gamma=0.99,
        lam=0.95,
    )
    scalar_returns, scalar_advantages, scalar_policy_advantages = compute_grouped_gae(
        rewards.sum(dim=-1, keepdim=True),
        values.sum(dim=-1, keepdim=True),
        dones,
        last_values.sum(dim=-1, keepdim=True),
        gamma=0.99,
        lam=0.95,
    )

    torch.testing.assert_close(grouped_returns.sum(dim=-1, keepdim=True), scalar_returns)
    torch.testing.assert_close(grouped_advantages.sum(dim=-1, keepdim=True), scalar_advantages)
    torch.testing.assert_close(policy_advantages, scalar_policy_advantages)
    assert policy_advantages.shape == (*rewards.shape[:2], 1)


def test_actor_sums_raw_head_advantages_before_the_single_ppo_clip() -> None:
    """Opposing heads must cancel before clipping, not become separate objectives."""
    algorithm = object.__new__(V25MultiCriticPPO)
    algorithm.clip_param = 0.2
    head_advantages = torch.tensor([[2.0, -2.0]])
    policy_advantage = head_advantages.sum(dim=-1, keepdim=True)
    ratio = torch.tensor([1.5])

    correct_loss = algorithm._compute_surrogate_loss(policy_advantage, ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - algorithm.clip_param, 1.0 + algorithm.clip_param)
    wrong_per_head_loss = (
        torch.maximum(
            -head_advantages * ratio.unsqueeze(-1),
            -head_advantages * clipped_ratio.unsqueeze(-1),
        )
        .sum(dim=-1)
        .mean()
    )

    torch.testing.assert_close(correct_loss, torch.zeros_like(correct_loss))
    assert wrong_per_head_loss.abs().item() > 0.5


def test_v25_clips_actor_and_critic_gradients_independently() -> None:
    algorithm = object.__new__(V25MultiCriticPPO)
    algorithm.actor = torch.nn.Linear(1, 1, bias=False)
    algorithm.critic = torch.nn.Linear(1, 1, bias=False)
    algorithm.max_grad_norm = 1.0
    algorithm.actor.weight.grad = torch.tensor([[3.0]])
    algorithm.critic.weight.grad = torch.tensor([[4.0]])

    returned_norm = algorithm._clip_gradients()

    torch.testing.assert_close(algorithm.actor.weight.grad, torch.tensor([[1.0]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(algorithm.critic.weight.grad, torch.tensor([[1.0]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(returned_norm, torch.tensor(4.0))


def test_compact_logger_records_dynamic_effective_value_loss_heads() -> None:
    algorithm = object.__new__(V25CompactMultiCriticPPO)
    algorithm.num_reward_groups = 4
    algorithm.device = "cpu"
    algorithm.is_multi_gpu = False
    algorithm._head_value_loss_sum = torch.zeros(4, dtype=torch.float64)
    algorithm._head_value_loss_count = torch.zeros((), dtype=torch.float64)

    effective_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]], requires_grad=True)
    algorithm._record_value_loss(effective_loss)
    head_losses, total_loss = algorithm._global_head_value_losses()

    torch.testing.assert_close(head_losses, torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64))
    torch.testing.assert_close(total_loss, torch.tensor(3.5, dtype=torch.float64))
    assert algorithm._head_value_loss_sum.grad_fn is None


def test_compact_logger_uses_cross_rank_weighted_value_loss_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    algorithm = object.__new__(V25CompactMultiCriticPPO)
    algorithm.is_multi_gpu = True
    algorithm._head_value_loss_sum = torch.tensor([2.0, 6.0, 10.0], dtype=torch.float64)
    algorithm._head_value_loss_count = torch.tensor(2.0, dtype=torch.float64)
    remote_statistics = torch.tensor([6.0, 10.0, 14.0, 2.0], dtype=torch.float64)

    def add_remote_rank(statistics: torch.Tensor, op: torch.distributed.ReduceOp) -> None:
        assert op is torch.distributed.ReduceOp.SUM
        statistics.add_(remote_statistics)

    monkeypatch.setattr(torch.distributed, "all_reduce", add_remote_rank)
    head_losses, total_loss = algorithm._global_head_value_losses()

    torch.testing.assert_close(head_losses, torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64))
    torch.testing.assert_close(total_loss, torch.tensor(4.0, dtype=torch.float64))


def test_compact_logger_keeps_reward_consistency_guard_without_logging_it() -> None:
    algorithm = object.__new__(V25CompactMultiCriticPPO)
    algorithm.is_multi_gpu = False
    algorithm.reward_consistency_atol = 2.0e-6
    algorithm.reward_consistency_rtol = 1.0e-5
    algorithm._reward_sum_error_max = torch.tensor(1.0e-3, dtype=torch.float64)
    algorithm._reward_sum_error_ratio_max = torch.tensor(1.01, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="Grouped rewards do not reproduce"):
        algorithm._validate_reward_consistency()


def test_timeout_bootstrap_is_applied_independently_to_every_value_head() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    values = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    time_outs = torch.tensor([[True], [False]])

    bootstrapped = bootstrap_grouped_timeouts(rewards, values, time_outs, gamma=0.9)
    expected = torch.tensor([[10.0, 20.0, 30.0], [4.0, 5.0, 6.0]])
    torch.testing.assert_close(bootstrapped, expected)

    with pytest.raises(ValueError, match="reward/value shapes must match"):
        bootstrap_grouped_timeouts(rewards, values[:, :1], time_outs, gamma=0.9)
    with pytest.raises(ValueError, match=r"Timeout shape must be \(2, 1\)"):
        bootstrap_grouped_timeouts(rewards, values, time_outs.squeeze(-1), gamma=0.9)


def test_process_env_step_mirrors_finite_reset_for_terminal_nonfinite_group_rewards() -> None:
    class NoOpModel:
        def update_normalization(self, obs: TensorDict) -> None:
            del obs

        def reset(self, dones: torch.Tensor) -> None:
            del dones

    class CapturingStorage:
        num_envs = 3

        def __init__(self) -> None:
            self.rewards: torch.Tensor | None = None

        def add_transition(self, transition: RolloutStorage.Transition) -> None:
            self.rewards = transition.rewards.clone()

    reward_manager = SimpleNamespace(
        _term_names=("global", "local", "regularization"),
        _step_reward=torch.tensor([[float("nan"), 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )
    algorithm = object.__new__(V25MultiCriticPPO)
    algorithm.actor = NoOpModel()
    algorithm.critic = NoOpModel()
    algorithm.rnd = None
    algorithm.device = "cpu"
    algorithm.env = SimpleNamespace(reward_manager=reward_manager, unwrapped=SimpleNamespace(step_dt=1.0))
    algorithm.reward_group_names = ("global", "local", "regularization")
    algorithm.reward_groups = {name: (name,) for name in algorithm.reward_group_names}
    algorithm.storage = CapturingStorage()
    algorithm.transition = RolloutStorage.Transition()
    algorithm.gamma = 0.99
    algorithm.reward_consistency_atol = 2.0e-6
    algorithm.reward_consistency_rtol = 1.0e-5
    algorithm._pure_reward_sum = torch.zeros(3, dtype=torch.float64)
    algorithm._pure_reward_count = torch.zeros((), dtype=torch.float64)
    algorithm._reward_sum_error_max = torch.zeros((), dtype=torch.float64)
    algorithm._reward_sum_error_ratio_max = torch.zeros((), dtype=torch.float64)

    observations = TensorDict({"actor": torch.zeros(3, 1)}, batch_size=[3])
    algorithm.process_env_step(
        observations,
        rewards=torch.tensor([0.0, 15.0, 24.0]),
        dones=torch.tensor([True, True, False]),
        extras={},
    )

    assert algorithm.storage.rewards is not None
    torch.testing.assert_close(
        algorithm.storage.rewards,
        torch.tensor([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )


def test_v25_real_model_contract_runs_rollout_gae_and_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the production actor/critic constructors and base PPO update together."""
    torch.manual_seed(7)
    num_envs = 8
    num_steps = 2
    observations = TensorDict(
        {
            "proprioception": torch.randn(num_envs, 5, 52),
            "action": torch.randn(num_envs, 4, 23),
            "command": torch.randn(num_envs, 14, 99),
            "critic": torch.randn(num_envs, 764),
        },
        batch_size=[num_envs],
    )
    reward_term_names = tuple(term for terms in V25_REWARD_GROUPS.values() for term in terms)
    reward_manager = SimpleNamespace(
        _term_names=reward_term_names,
        _step_reward=torch.zeros(num_envs, len(reward_term_names)),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        num_actions=23,
        reward_manager=reward_manager,
        step_dt=0.02,
    )
    env.unwrapped = env

    cfg = deepcopy(T800FlatV25ScalePPORunnerCfg().to_dict())
    # Match train.py's legacy-config sanitizer before the runner constructs
    # current RSL-RL model classes.
    for model_name in ("actor", "critic"):
        for legacy_key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            cfg[model_name].pop(legacy_key, None)
    cfg["multi_gpu"] = None
    cfg["num_steps_per_env"] = num_steps
    cfg["torch_compile_mode"] = None
    cfg["algorithm"]["num_learning_epochs"] = 1
    cfg["algorithm"]["num_mini_batches"] = 2
    original_cfg = deepcopy(cfg)

    algorithm = V25CompactMultiCriticPPO.construct_algorithm(observations, env, cfg, "cpu")
    assert isinstance(algorithm, V25CompactMultiCriticPPO)
    assert cfg == original_cfg
    assert algorithm.actor.nodes["attention_blocks"].num_blocks == V25_ACTOR_NUM_BLOCKS
    assert len(algorithm.actor.nodes["attention_blocks"].blocks) == V25_ACTOR_NUM_BLOCKS
    assert algorithm.actor.nodes["attention_blocks"].residual_scale == pytest.approx(1.0 / (6.0**0.5))
    assert algorithm.critic(observations).shape == (num_envs, 3)

    # Match MotionOnPolicyRunner exactly: rollout, process_env_step and return
    # computation all happen inside one torch.inference_mode() context.
    with torch.inference_mode():
        for _ in range(num_steps):
            algorithm.act(observations)
            reward_manager._step_reward = torch.randn(num_envs, len(reward_term_names)) * 0.1
            scalar_rewards = reward_manager._step_reward.sum(dim=-1) * env.step_dt
            algorithm.process_env_step(observations, scalar_rewards, torch.zeros(num_envs), {})
        algorithm.compute_returns(observations)

    assert algorithm.storage.rewards.shape == (num_steps, num_envs, 3)
    assert algorithm.storage.values.shape == (num_steps, num_envs, 3)
    assert algorithm.storage.returns.shape == (num_steps, num_envs, 3)
    assert algorithm.storage.group_advantages.shape == (num_steps, num_envs, 3)
    assert algorithm.storage.advantages.shape == (num_steps, num_envs, 1)
    assert not algorithm.storage.advantages.is_inference()
    assert not algorithm._reward_sum_error_max.is_inference()
    assert not algorithm._reward_sum_error_ratio_max.is_inference()

    metrics = algorithm.update()
    assert torch.isfinite(torch.tensor(tuple(metrics.values()))).all()
    expected_metric_names = {"value", "surrogate", "entropy"} | {
        f"value/{group_name}" for group_name in V25_REWARD_GROUPS
    }
    assert set(metrics) == expected_metric_names
    head_value_losses = torch.tensor([metrics[f"value/{name}"] for name in V25_REWARD_GROUPS])
    torch.testing.assert_close(torch.tensor(metrics["value"]), head_value_losses.mean())

    def fail_on_checkpoint_source_read(self: Path, *args, **kwargs) -> str:
        del self, args, kwargs
        raise AssertionError("checkpoint save must use import-time frozen sources")

    monkeypatch.setattr(Path, "read_text", fail_on_checkpoint_source_read)
    checkpoint = algorithm.save()
    assert checkpoint["v25_multi_critic"]["format_version"] == 2
    assert len(checkpoint["v25_multi_critic"]["implementation_sha256"]) == 64
    assert len(checkpoint["v25_multi_critic"]["base_implementation_sha256"]) == 64
    assert len(checkpoint["v25_multi_critic"]["compact_implementation_sha256"]) == 64
    assert len(checkpoint["v25_multi_critic"]["rsl_ppo_implementation_sha256"]) == 64
    assert "class V25MultiCriticPPO" in checkpoint["v25_multi_critic_source"]
    assert "class V25CompactMultiCriticPPO" in checkpoint["v25_compact_multi_critic_source"]
    assert "class PPO" in checkpoint["v25_rsl_rl_ppo_source"]
    assert checkpoint["v25_multi_critic"]["logging_contract"] == ("v24_plus_dynamic_per_head_effective_value_loss")
    assert checkpoint["v25_multi_critic"]["reward_group_names"] == ("global", "local", "regularization")
    assert checkpoint["v25_multi_critic"]["reward_groups"] == {
        name: tuple(terms) for name, terms in V25_REWARD_GROUPS.items()
    }
    assert algorithm.load(checkpoint, load_cfg=None, strict=True) is True

    incompatible_checkpoint = deepcopy(checkpoint)
    incompatible_checkpoint["v25_multi_critic"] = {
        **checkpoint["v25_multi_critic"],
        "reward_group_names": ("local", "global", "regularization"),
    }
    with pytest.raises(ValueError, match="implementation or reward contract does not match"):
        algorithm.load(incompatible_checkpoint, load_cfg={"critic": True}, strict=True)

    source_incompatible_checkpoint = deepcopy(checkpoint)
    source_incompatible_checkpoint["v25_compact_multi_critic_source"] += "\n# tampered"
    with pytest.raises(ValueError, match="frozen base/compact/RSL sources do not match"):
        algorithm.load(source_incompatible_checkpoint, load_cfg={"critic": True}, strict=True)

    rsl_source_incompatible_checkpoint = deepcopy(checkpoint)
    rsl_source_incompatible_checkpoint["v25_rsl_rl_ppo_source"] += "\n# tampered"
    with pytest.raises(ValueError, match="frozen base/compact/RSL sources do not match"):
        algorithm.load(rsl_source_incompatible_checkpoint, load_cfg={"critic": True}, strict=True)

    actor_only_checkpoint = {"actor_state_dict": checkpoint["actor_state_dict"]}
    assert algorithm.load(actor_only_checkpoint, load_cfg={"actor": True}, strict=True) is False
