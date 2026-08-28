from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import engineai_rl_lab.tasks  # noqa: F401
import gymnasium as gym
import pytest
import torch
import torch.nn as nn
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v25 import (
    T800FlatV25ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v26 import (
    T800FlatV26ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v25 import (
    T800FlatWoStateEstimationEnvCfgV25Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v26 import (
    T800FlatWoStateEstimationEnvCfgV26Scale,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import AdaptiveSamplerV1, AdaptiveSamplerV1Cfg, MotionCommandV1
from engineai_rl_lab.utils.v25_compact_multi_critic_ppo import V25CompactMultiCriticPPO
from engineai_rl_lab.utils.v26_star_multi_critic_ppo import (
    V26StarMultiCriticPPO,
    V26StarRolloutStorage,
    _validate_star_parameters,
    difficulty_conditioned_normalize,
)
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

_REWARD_GROUPS = {
    "global": (
        "motion_global_root_height",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "motion_joint_pos",
        "motion_joint_vel",
    ),
    "local": (
        "motion_local_end_effector_pos",
        "motion_relative_body_pos",
        "motion_relative_body_ori",
    ),
    "regularization": ("alive", "action_rate_l2", "joint_limit"),
}


def _make_storage(
    *,
    num_envs: int = 2,
    num_steps: int = 4,
    priority_fraction: float = 0.125,
    top_fraction: float = 0.05,
    reuse_cap: int = 2,
) -> tuple[V26StarRolloutStorage, TensorDict]:
    observations = TensorDict(
        {
            "actor": torch.zeros(num_envs, 2),
            "critic": torch.zeros(num_envs, 3),
        },
        batch_size=[num_envs],
    )
    storage = V26StarRolloutStorage(
        "rl",
        num_envs=num_envs,
        num_transitions_per_env=num_steps,
        obs=observations,
        actions_shape=[2],
        num_reward_groups=3,
        star_priority_fraction=priority_fraction,
        star_high_difficulty_threshold=1.0,
        star_top_fraction=top_fraction,
        star_difficulty_boundaries=(1.5, 2.0, 4.0),
        star_reuse_cap=reuse_cap,
        star_priority_weight_cap=8.0,
        device="cpu",
    )
    return storage, observations


def _make_transition(observations: TensorDict, bin_ids: torch.Tensor, ratios: torch.Tensor):
    transition = RolloutStorage.Transition()
    num_envs = observations.batch_size[0]
    transition.observations = observations.clone()
    transition.actions = torch.zeros(num_envs, 2)
    transition.rewards = torch.zeros(num_envs, 3)
    transition.dones = torch.zeros(num_envs, 1, dtype=torch.bool)
    transition.values = torch.zeros(num_envs, 3)
    transition.actions_log_prob = torch.zeros(num_envs, 1)
    transition.distribution_params = (torch.zeros(num_envs, 2), torch.ones(num_envs, 2))
    transition.star_bin_ids = bin_ids
    transition.star_probability_ratios = ratios
    return transition


def _fake_algorithm(seed: int) -> V26StarMultiCriticPPO:
    torch.manual_seed(seed)
    algorithm = object.__new__(V26StarMultiCriticPPO)
    algorithm._raw_actor = nn.Linear(3, 2)
    algorithm._raw_critic = nn.Linear(3, 3)
    algorithm.optimizer = torch.optim.Adam(
        [
            {"params": algorithm._raw_actor.parameters(), "lr": 2.0e-5},
            {"params": algorithm._raw_critic.parameters(), "lr": 1.0e-3},
        ]
    )
    algorithm.rnd = None
    algorithm.shared_kl_adaptation = False
    algorithm.reward_group_names = tuple(_REWARD_GROUPS)
    algorithm.reward_groups = deepcopy(_REWARD_GROUPS)
    algorithm.value_loss_reduction = "mean"
    algorithm.star_priority_fraction = 0.125
    algorithm.star_high_difficulty_threshold = 1.0
    algorithm.star_top_fraction = 0.05
    algorithm.star_difficulty_boundaries = (1.5, 2.0, 4.0)
    algorithm.star_reuse_cap = 2
    algorithm.star_priority_weight_cap = 8.0
    return algorithm


def _optimizer_step(algorithm: V26StarMultiCriticPPO) -> None:
    loss = sum(parameter.square().sum() for parameter in algorithm._raw_actor.parameters())
    loss += sum(parameter.square().sum() for parameter in algorithm._raw_critic.parameters())
    loss.backward()
    algorithm.optimizer.step()
    algorithm.optimizer.zero_grad()


def test_v26_environment_matches_v25_except_for_runner_registration() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v26-scale")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(
        ".flat_env_cfg_v26:T800FlatWoStateEstimationEnvCfgV26Scale"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        ".rsl_rl_ppo_cfg_v26:T800FlatV26ScalePPORunnerCfg"
    )
    expected = T800FlatWoStateEstimationEnvCfgV25Scale().to_dict()
    assert T800FlatWoStateEstimationEnvCfgV26Scale().to_dict() == expected
    assert T800FlatWoStateEstimationEnvCfgV26Scale().observations.command.link_pose_b.params[
        "frame_offsets"
    ] == list(range(-5, 6))
    motion_cfg = T800FlatWoStateEstimationEnvCfgV26Scale().commands.motion
    assert motion_cfg.adaptive_sampling.equal_motion_weighting is False
    assert motion_cfg.adaptive_sampling.learnability_full_scale == 0.05
    assert motion_cfg.start_at_motion_beginning is False


def test_v26_runner_matches_v25_except_for_star_sampling_fields() -> None:
    v25 = T800FlatV25ScalePPORunnerCfg().to_dict()
    v26 = T800FlatV26ScalePPORunnerCfg().to_dict()
    assert v26["run_name"] == "v26-scale"
    assert v26["actor"]["nodes"]["attention_blocks"]["cell"]["num_blocks"] == 3
    for key in v25.keys() - {"run_name", "algorithm"}:
        assert v26[key] == v25[key], f"Unexpected V26 runner difference in {key!r}"

    v25_algorithm = dict(v25["algorithm"])
    v26_algorithm = dict(v26["algorithm"])
    assert v25_algorithm.pop("class_name").endswith(":V25CompactMultiCriticPPO")
    assert v26_algorithm.pop("class_name").endswith(":V26StarMultiCriticPPO")
    assert v26_algorithm["reward_groups"] == {
        name: list(terms) for name, terms in _REWARD_GROUPS.items()
    }
    assert v26_algorithm["value_loss_reduction"] == "mean"
    assert v26_algorithm.pop("star_priority_fraction") == 0.125
    assert v26_algorithm.pop("star_high_difficulty_threshold") == 1.0
    assert v26_algorithm.pop("star_top_fraction") == 0.05
    assert v26_algorithm.pop("star_difficulty_boundaries") == (1.5, 2.0, 4.0)
    assert v26_algorithm.pop("star_reuse_cap") == 2
    assert v26_algorithm.pop("star_priority_weight_cap") == 8.0
    assert v26_algorithm == v25_algorithm


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"priority_fraction": 1.0}, "priority_fraction"),
        ({"high_difficulty_threshold": 0.0}, "high_difficulty_threshold"),
        ({"top_fraction": 0.0}, "top_fraction"),
        ({"difficulty_boundaries": (2.0, 1.5)}, "strictly increasing"),
        ({"reuse_cap": 0}, "reuse_cap"),
        ({"priority_weight_cap": 0.5}, "priority_weight_cap"),
    ],
)
def test_star_parameter_validation_rejects_unsafe_values(overrides: dict, message: str) -> None:
    parameters = {
        "priority_fraction": 0.125,
        "high_difficulty_threshold": 1.0,
        "top_fraction": 0.05,
        "difficulty_boundaries": (1.5, 2.0, 4.0),
        "reuse_cap": 2,
        "priority_weight_cap": 8.0,
    }
    parameters.update(overrides)
    with pytest.raises(ValueError, match=message):
        _validate_star_parameters(**parameters)


def test_h_and_remaining_advantages_are_normalized_independently() -> None:
    raw = torch.tensor([[[-1.0], [1.0], [1.0]], [[3.0], [3.0], [5.0]]])
    ratios = torch.tensor([[[0.5], [1.0], [1.2]], [[0.8], [2.0], [4.0]]])

    normalized, diagnostics = difficulty_conditioned_normalize(
        raw,
        ratios,
        high_difficulty_threshold=1.0,
        distributed=False,
    )

    high = ratios > 1.0
    remaining = ~high
    assert normalized[high].mean().item() == pytest.approx(0.0, abs=1.0e-6)
    assert normalized[high].std().item() == pytest.approx(1.0, abs=1.0e-6)
    assert normalized[remaining].mean().item() == pytest.approx(0.0, abs=1.0e-6)
    assert normalized[remaining].std().item() == pytest.approx(1.0, abs=1.0e-6)
    assert diagnostics["high_count"].item() == 3
    assert diagnostics["remaining_count"].item() == 3


def test_neutral_cold_rollout_matches_baseline_advantage_normalization() -> None:
    raw = torch.tensor([[[-2.0], [0.5]], [[1.0], [4.0]]])

    normalized, _ = difficulty_conditioned_normalize(
        raw,
        torch.ones_like(raw),
        high_difficulty_threshold=1.0,
        distributed=False,
    )

    expected = (raw - raw.mean()) / (raw.std() + 1.0e-8)
    torch.testing.assert_close(normalized, expected)


def test_sampling_context_uses_rank_conditioned_probability_ratio_and_actual_bin() -> None:
    sampler = AdaptiveSamplerV1(torch.tensor([100, 100, 100]), AdaptiveSamplerV1Cfg(), device="cpu")
    command = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        _has_sampled=torch.ones(2, dtype=torch.bool),
        time_steps=torch.tensor([0, 50]),
        motion_lengths=torch.tensor([100, 100]),
        motion_ids=torch.tensor([0, 1]),
        motion=SimpleNamespace(global_ids=torch.tensor([0, 2])),
        adaptive_sampler=sampler,
        global_sampling_probabilities=torch.tensor([0.05, 0.15, 0.10, 0.10, 0.20, 0.40]),
        _active_adaptive_probability_mass=torch.tensor(0.80, dtype=torch.float64),
        _active_coverage_probability_mass=torch.tensor(200.0, dtype=torch.float64),
    )
    command._bucket_ids = lambda motion_ids, time_steps: sampler.bucket_ids(
        command.motion.global_ids[motion_ids], time_steps
    )

    bin_ids, ratios = MotionCommandV1.snapshot_transition_sampling_context(command)

    torch.testing.assert_close(bin_ids, torch.tensor([0, 5]))
    torch.testing.assert_close(ratios, torch.tensor([0.25, 2.0]))


def test_sampling_context_maps_post_update_motion_timeout_to_terminal_bin() -> None:
    sampler = AdaptiveSamplerV1(torch.tensor([8]), AdaptiveSamplerV1Cfg(), device="cpu")
    global_probabilities = sampler.distribution()
    command = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        _has_sampled=torch.ones(1, dtype=torch.bool),
        time_steps=torch.tensor([8]),
        motion_lengths=torch.tensor([8]),
        motion_ids=torch.tensor([0]),
        motion=SimpleNamespace(global_ids=torch.tensor([0])),
        adaptive_sampler=sampler,
        global_sampling_probabilities=global_probabilities,
        _active_adaptive_probability_mass=global_probabilities.double().sum(),
        _active_coverage_probability_mass=sampler.base_weights.sum(),
    )
    command._bucket_ids = lambda motion_ids, time_steps: sampler.bucket_ids(
        command.motion.global_ids[motion_ids], time_steps
    )

    bin_ids, ratios = MotionCommandV1.snapshot_transition_sampling_context(command)

    torch.testing.assert_close(bin_ids, torch.tensor([0]))
    torch.testing.assert_close(ratios, torch.ones(1))


def test_sampling_context_rejects_reference_beyond_motion_timeout_boundary() -> None:
    command = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        _has_sampled=torch.ones(1, dtype=torch.bool),
        time_steps=torch.tensor([9]),
        motion_lengths=torch.tensor([8]),
    )
    with pytest.raises(RuntimeError, match="outside the current reference motion"):
        MotionCommandV1.snapshot_transition_sampling_context(command)


def test_sampling_context_handles_duplicate_working_set_entries_and_partial_final_bins() -> None:
    sampler = AdaptiveSamplerV1(torch.tensor([75, 100]), AdaptiveSamplerV1Cfg(), device="cpu")
    active_bins, _ = sampler.active_mapping(torch.tensor([0, 0, 1]))
    global_probabilities = torch.tensor([0.2, 0.1, 0.3, 0.4])
    command = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        _has_sampled=torch.ones(2, dtype=torch.bool),
        time_steps=torch.tensor([74, 99]),
        motion_lengths=torch.tensor([75, 100]),
        motion_ids=torch.tensor([1, 2]),
        motion=SimpleNamespace(global_ids=torch.tensor([0, 0, 1])),
        adaptive_sampler=sampler,
        global_sampling_probabilities=global_probabilities,
        _active_adaptive_probability_mass=global_probabilities[active_bins].double().sum(),
        _active_coverage_probability_mass=sampler.base_weights[active_bins].sum(),
    )
    command._bucket_ids = lambda motion_ids, time_steps: sampler.bucket_ids(
        command.motion.global_ids[motion_ids], time_steps
    )

    bin_ids, ratios = MotionCommandV1.snapshot_transition_sampling_context(command)

    torch.testing.assert_close(bin_ids, torch.tensor([1, 3]))
    torch.testing.assert_close(ratios, torch.tensor([10 / 13, 20 / 13]))


def test_cold_coverage_roundoff_stays_in_the_neutral_difficulty_group() -> None:
    sampler = AdaptiveSamplerV1(torch.tensor([75, 100]), AdaptiveSamplerV1Cfg(), device="cpu")
    active_bins, _ = sampler.active_mapping(torch.tensor([0, 1]))
    global_probabilities = sampler.distribution()
    command = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        _has_sampled=torch.ones(2, dtype=torch.bool),
        time_steps=torch.tensor([74, 99]),
        motion_lengths=torch.tensor([75, 100]),
        motion_ids=torch.tensor([0, 1]),
        motion=SimpleNamespace(global_ids=torch.tensor([0, 1])),
        adaptive_sampler=sampler,
        global_sampling_probabilities=global_probabilities,
        _active_adaptive_probability_mass=global_probabilities[active_bins].double().sum(),
        _active_coverage_probability_mass=sampler.base_weights[active_bins].sum(),
    )
    command._bucket_ids = lambda motion_ids, time_steps: sampler.bucket_ids(
        command.motion.global_ids[motion_ids], time_steps
    )

    _, ratios = MotionCommandV1.snapshot_transition_sampling_context(command)

    torch.testing.assert_close(ratios, torch.ones(2), rtol=0.0, atol=0.0)


def test_act_clones_pre_step_sampling_context(monkeypatch: pytest.MonkeyPatch) -> None:
    bin_ids = torch.tensor([7], dtype=torch.long)
    ratios = torch.tensor([2.5])
    command = SimpleNamespace(
        snapshot_transition_sampling_context=lambda: (bin_ids, ratios),
    )
    manager = SimpleNamespace(get_term=lambda name: command)
    algorithm = object.__new__(V26StarMultiCriticPPO)
    algorithm.env = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=manager))
    algorithm.device = "cpu"
    algorithm.transition = SimpleNamespace()

    def fake_parent_act(self, obs):
        bin_ids.fill_(99)
        ratios.fill_(99.0)
        return torch.zeros(1, 2)

    monkeypatch.setattr(V25CompactMultiCriticPPO, "act", fake_parent_act)
    actions = V26StarMultiCriticPPO.act(algorithm, TensorDict({}, batch_size=[1]))

    torch.testing.assert_close(actions, torch.zeros(1, 2))
    torch.testing.assert_close(algorithm.transition.star_bin_ids, torch.tensor([7]))
    torch.testing.assert_close(algorithm.transition.star_probability_ratios, torch.tensor([2.5]))


def test_v26_storage_requires_and_copies_transition_context() -> None:
    storage, observations = _make_storage()
    transition = _make_transition(observations, torch.tensor([3, 7]), torch.tensor([0.5, 2.0]))
    storage.add_transition(transition)

    torch.testing.assert_close(storage.star_bin_ids[0, :, 0], torch.tensor([3, 7]))
    torch.testing.assert_close(storage.star_probability_ratios[0, :, 0], torch.tensor([0.5, 2.0]))
    transition.star_bin_ids[0] = 99
    transition.star_probability_ratios[1] = 99.0
    torch.testing.assert_close(storage.star_bin_ids[0, :, 0], torch.tensor([3, 7]))
    torch.testing.assert_close(storage.star_probability_ratios[0, :, 0], torch.tensor([0.5, 2.0]))

    invalid, invalid_observations = _make_storage()
    bad_transition = _make_transition(
        invalid_observations,
        torch.tensor([-1, 0]),
        torch.ones(2),
    )
    with pytest.raises(RuntimeError, match="non-negative int64"):
        invalid.add_transition(bad_transition)


def test_star_selects_per_band_fragment_pairs_then_pools_and_weights_full_fragments() -> None:
    storage, _ = _make_storage(top_fraction=0.5)
    storage.star_bin_ids.fill_(1)
    storage.dones.zero_()
    storage.dones[1, 0, 0] = True
    storage.star_probability_ratios[..., 0].copy_(
        torch.tensor(
            [
                [1.2, 1.1],
                [0.5, 4.2],
                [4.5, 0.8],
                [0.5, 0.8],
            ]
        )
    )
    storage.raw_policy_advantages[..., 0].copy_(
        torch.tensor(
            [
                [4.0, 1.0],
                [-2.0, 2.0],
                [5.0, 0.0],
                [-20.0, 0.0],
            ]
        )
    )

    storage.prepare_star_pool()

    # Env 0's first fragment wins band 0; its second fragment wins band 3
    # despite a negative whole-fragment mean. Every transition in both retained
    # fragments is pooled, while env 1's lower-scoring full fragment is not.
    torch.testing.assert_close(storage._priority_pool_indices, torch.tensor([0, 2, 4, 6]))
    torch.testing.assert_close(storage._priority_pool_weights, torch.tensor([0.85, 0.85, 2.5, 2.5]))
    assert storage._star_prepare_statistics[6].item() == 2  # selected pairs
    assert storage._star_prepare_statistics[7].item() == 2  # selected unique fragments
    assert storage._star_prepare_statistics[17].item() == 1  # band (1, 1.5]
    assert storage._star_prepare_statistics[20].item() == 1  # band (4, inf]


def test_priority_generator_keeps_batch_shape_and_caps_extra_reuse_over_the_update() -> None:
    storage, _ = _make_storage(
        num_envs=4,
        num_steps=4,
        priority_fraction=0.25,
        reuse_cap=2,
    )
    flat_ids = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1)
    storage.observations["actor"][..., :1].copy_(flat_ids)
    storage.observations["critic"][..., :1].copy_(flat_ids)
    storage.distribution_params = (torch.zeros(4, 4, 2), torch.ones(4, 4, 2))
    storage._priority_pool_indices = torch.tensor([0, 1, 2, 3])
    storage._priority_pool_weights = torch.ones(4)

    batches = list(storage.mini_batch_generator(num_mini_batches=2, num_epochs=2))

    assert len(batches) == 4
    assert all(batch.actions.shape == (8, 2) for batch in batches)
    torch.testing.assert_close(
        storage._star_generator_statistics,
        torch.tensor([8.0, 8.0, 4.0, 2.0], dtype=torch.float64),
    )


def test_priority_generator_fills_with_uniform_samples_when_pool_capacity_is_small() -> None:
    storage, _ = _make_storage(
        num_envs=4,
        num_steps=4,
        priority_fraction=0.25,
        reuse_cap=1,
    )
    storage.distribution_params = (torch.zeros(4, 4, 2), torch.ones(4, 4, 2))
    storage._priority_pool_indices = torch.tensor([0, 1])
    storage._priority_pool_weights = torch.ones(2)

    batches = list(storage.mini_batch_generator(num_mini_batches=2, num_epochs=2))

    assert len(batches) == 4
    assert all(batch.actions.shape[0] == 8 for batch in batches)
    torch.testing.assert_close(
        storage._star_generator_statistics,
        torch.tensor([8.0, 2.0, 2.0, 1.0], dtype=torch.float64),
    )


def test_v26_real_models_run_pre_step_context_grouped_gae_and_star_update() -> None:
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
    reward_term_names = tuple(term for terms in _REWARD_GROUPS.values() for term in terms)
    reward_manager = SimpleNamespace(
        _term_names=reward_term_names,
        _step_reward=torch.zeros(num_envs, len(reward_term_names)),
    )
    sampling_command = SimpleNamespace(
        snapshot_transition_sampling_context=lambda: (
            torch.arange(num_envs, dtype=torch.long),
            torch.tensor([0.5, 0.8, 1.0, 0.9, 1.2, 1.5, 2.0, 4.5]),
        )
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        num_actions=23,
        reward_manager=reward_manager,
        command_manager=SimpleNamespace(get_term=lambda name: sampling_command),
        step_dt=0.02,
    )
    env.unwrapped = env

    cfg = deepcopy(T800FlatV26ScalePPORunnerCfg().to_dict())
    for model_name in ("actor", "critic"):
        for legacy_key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            cfg[model_name].pop(legacy_key, None)
    cfg["multi_gpu"] = None
    cfg["num_steps_per_env"] = num_steps
    cfg["torch_compile_mode"] = None
    cfg["algorithm"]["num_learning_epochs"] = 1
    cfg["algorithm"]["num_mini_batches"] = 2
    original_cfg = deepcopy(cfg)

    algorithm = V26StarMultiCriticPPO.construct_algorithm(observations, env, cfg, "cpu")
    assert cfg == original_cfg
    assert isinstance(algorithm.storage, V26StarRolloutStorage)

    with torch.inference_mode():
        for step in range(num_steps):
            algorithm.act(observations)
            reward_manager._step_reward = torch.randn(num_envs, len(reward_term_names)) * 0.1
            scalar_rewards = reward_manager._step_reward.sum(dim=-1) * env.step_dt
            dones = torch.zeros(num_envs)
            if step == 0:
                dones[0] = 1.0
            algorithm.process_env_step(observations, scalar_rewards, dones, {})
        algorithm.compute_returns(observations)

    torch.testing.assert_close(
        algorithm.storage.star_probability_ratios[0, :, 0],
        torch.tensor([0.5, 0.8, 1.0, 0.9, 1.2, 1.5, 2.0, 4.5]),
    )
    assert not algorithm.storage.advantages.is_inference()
    assert not algorithm.storage.raw_policy_advantages.is_inference()
    assert torch.isfinite(algorithm.storage.advantages).all()

    metrics = algorithm.update()
    assert torch.isfinite(torch.tensor(tuple(metrics.values()))).all()
    assert {"value", "surrogate", "entropy"}.issubset(metrics)
    assert {f"value/{name}" for name in _REWARD_GROUPS}.issubset(metrics)
    assert {name for name in metrics if name.startswith("star/")} == {
        "star/high_difficulty_transition_fraction",
        "star/actual_priority_fraction",
        "star/max_priority_reuse",
    }


def test_v26_exact_resume_validates_all_sources_and_advances_iteration() -> None:
    source = _fake_algorithm(seed=3)
    _optimizer_step(source)
    checkpoint = source.save()
    checkpoint["iter"] = 11
    target = _fake_algorithm(seed=4)

    assert target.load(checkpoint, load_cfg=None, strict=True)
    assert checkpoint["iter"] == 12

    tampered_v26 = source.save()
    tampered_v26["iter"] = 11
    tampered_v26["v26_star_source"] += "\n# tampered"
    with pytest.raises(ValueError, match="V26 STAR checkpoint source"):
        target.load(tampered_v26, load_cfg=None, strict=True)

    tampered_dependency = source.save()
    tampered_dependency["iter"] = 11
    tampered_dependency["v25_rsl_rl_ppo_source"] += "\n# tampered"
    with pytest.raises(ValueError, match="base/compact/RSL"):
        target.load(tampered_dependency, load_cfg=None, strict=True)

    not_v26 = source.save()
    not_v26["iter"] = 11
    del not_v26["v26_star"]
    del not_v26["v26_star_source"]
    with pytest.raises(ValueError, match="independent scratch experiment"):
        target.load(not_v26, load_cfg=None, strict=True)

    missing_optimizer = source.save()
    missing_optimizer["iter"] = 11
    del missing_optimizer["optimizer_state_dict"]
    with pytest.raises(ValueError, match="optimizer state"):
        target.load(missing_optimizer, load_cfg=None, strict=True)


def test_actor_only_load_does_not_require_metadata_or_advance_iteration() -> None:
    source = _fake_algorithm(seed=5)
    checkpoint = {"actor_state_dict": source._raw_actor.state_dict(), "iter": 7}
    target = _fake_algorithm(seed=6)

    assert not target.load(
        checkpoint,
        load_cfg={"actor": True, "iteration": False},
        strict=True,
    )
    assert checkpoint["iter"] == 7
