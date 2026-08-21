from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand


def _curriculum_command() -> SimpleNamespace:
    command = SimpleNamespace()
    command.device = torch.device("cpu")
    command.num_envs = 2
    command.cfg = SimpleNamespace(
        body_names=["base", "foot"],
        bin_size=4,
        sequence_length_agnostic=False,
        init_num_failures=1.0,
        uniform_sampling_rate=0.2,
        pre_failure_sample_window=0,
        start_at_motion_beginning=False,
        use_failure_rate_decay=False,
        decay_gamma=0.8,
        adaptive_sampling_alpha=0.5,
        adaptive_kernel_size=1,
        adaptive_kernel_lambda=0.8,
        adp_samp_failure_rate_max_over_mean=30.0,
        failure_counts_multiplier=1.0,
        max_prob_per_bin=None,
        max_prob_per_motion=None,
        curriculum_sampling_enabled=True,
        curriculum_shadow_iterations=100,
        curriculum_blend_iterations=50,
        curriculum_state_update_interval=1,
        curriculum_min_window_trials=8,
        curriculum_min_known_trials=16,
        curriculum_min_quarantine_trials=32,
        curriculum_min_exit_probe_trials=8,
        curriculum_min_known_fraction=0.2,
        curriculum_beta_prior_alpha=1.0,
        curriculum_beta_prior_beta=1.0,
        curriculum_mastered_enter_threshold=0.10,
        curriculum_mastered_exit_threshold=0.15,
        curriculum_stalled_enter_threshold=0.80,
        curriculum_stalled_exit_threshold=0.75,
        curriculum_quarantine_enter_threshold=0.90,
        curriculum_quarantine_exit_threshold=0.80,
        curriculum_no_progress_threshold=0.02,
        curriculum_improvement_threshold=0.05,
        curriculum_start_horizon_frames=None,
        curriculum_exclude_invalid_failures=False,
        curriculum_motion_length_exponent=1.0,
        curriculum_min_terminal_visits=0,
        curriculum_quarantine_terminal_hazard_threshold=0.0,
        curriculum_terminal_hazard_window_bins=1,
        curriculum_preserve_absent_state_budgets=False,
        curriculum_probability_smoothing_alpha=1.0,
        curriculum_detailed_metrics=True,
        curriculum_state_sampling_weights={
            "unknown": 0.20,
            "mastered": 0.10,
            "frontier": 0.55,
            "stalled": 0.10,
            "quarantine": 0.05,
        },
    )
    command.global_num_motions = 1
    command.bin_count = 5
    command.motion_bin_counts = torch.tensor([5], dtype=torch.long)
    command.motion_bin_offsets = torch.tensor([0], dtype=torch.long)
    command.bin_motion_ids = torch.zeros(5, dtype=torch.long)
    command.bin_starts = torch.arange(0, 20, 4, dtype=torch.long)
    command.bin_ends = torch.arange(4, 24, 4, dtype=torch.long)
    command.bin_weights = torch.ones(5)
    command.adp_samp_num_episodes = torch.ones(5)
    command.adp_samp_num_failures = torch.ones(5)
    command._current_adp_samp_num_episodes = torch.zeros(5)
    command._current_adp_samp_num_failures = torch.zeros(5)
    command.metrics = {
        "sampling_entropy": torch.zeros(2),
        "sampling_top1_prob": torch.zeros(2),
    }
    command.motion = SimpleNamespace(
        global_num_motions=1,
        world_size=1,
        manifest_fingerprint_words=(11, 22, 33, 44),
        global_ids=torch.tensor([0], dtype=torch.long),
        num_motions=1,
        lengths=lambda motion_ids: torch.full_like(motion_ids, 20),
    )
    command._active_bin_ids = torch.arange(5, dtype=torch.long)
    command._active_local_motion_ids = torch.zeros(5, dtype=torch.long)
    command.motion_ids = torch.zeros(2, dtype=torch.long)
    command._fixed_motion_ids = None
    command.time_steps = torch.zeros(2, dtype=torch.long)
    command.motion_lengths = torch.full((2,), 20, dtype=torch.long)
    command._has_sampled = torch.zeros(2, dtype=torch.bool)
    body_pos = torch.zeros(2, dtype=torch.bool)
    termination_manager = SimpleNamespace(
        terminated=torch.zeros(2, dtype=torch.bool),
        body_pos=body_pos,
    )

    def get_term(name: str) -> torch.Tensor:
        if name != "body_pos":
            raise KeyError(name)
        return termination_manager.body_pos

    termination_manager.get_term = get_term
    command._env = SimpleNamespace(termination_manager=termination_manager)
    command._adaptive_layout_checked = True
    for constant_name in (
        "_CURRICULUM_STATE_NAMES",
        "_CURRICULUM_UNKNOWN",
        "_CURRICULUM_MASTERED",
        "_CURRICULUM_FRONTIER",
        "_CURRICULUM_STALLED",
        "_CURRICULUM_QUARANTINE",
        "_CURRICULUM_STATE_SCHEMA_VERSION",
        "_CURRICULUM_BIASED_FIXED_HORIZON_SCHEMA_VERSION",
        "_CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION",
    ):
        setattr(command, constant_name, getattr(MotionCommand, constant_name))
    for method_name in (
        "_curriculum_sampling_enabled",
        "_curriculum_fixed_horizon_enabled",
        "_curriculum_checkpoint_schema_version",
        "_initialize_curriculum_sampling",
        "_bucket_ids",
        "_compute_failure_rate",
        "_smooth_failure_rate",
        "_marmot_probabilities",
        "_clip_failure_rate",
        "_configured_probability_cap",
        "_apply_probability_caps",
        "_curriculum_blend_factor",
        "_curriculum_state_budget_probabilities",
        "_curriculum_shadow_distribution",
        "_blend_curriculum_distribution",
        "_smooth_curriculum_distribution",
        "_update_curriculum_metrics",
        "_curriculum_known_mask",
        "_curriculum_forward_terminal_hard_mask",
        "_update_curriculum_states",
        "_flush_curriculum_statistics",
        "_advance_curriculum_sampling",
        "_legacy_sampling_probabilities",
        "_rebuild_global_sampling_distribution",
        "_rebuild_sampling_distribution",
        "_record_curriculum_start_outcomes",
        "_record_curriculum_start_censored",
        "_update_adaptive_exposure",
        "_reset_curriculum_sampling_state",
        "get_adaptive_sampling_state",
        "_checkpoint_curriculum_distribution",
        "load_adaptive_sampling_state",
    ):
        setattr(command, method_name, MethodType(getattr(MotionCommand, method_name), command))
    command._cap_probabilities_at_uniform_ratio = MotionCommand._cap_probabilities_at_uniform_ratio
    command._initialize_curriculum_sampling()
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()
    return command


def _sample_first_bin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda _probabilities, num_samples, replacement: torch.zeros(num_samples, dtype=torch.long),
    )


def _record_window(command: SimpleNamespace, bin_id: int, *, trials: int, failures: int) -> None:
    command._curriculum_window_start_trials[bin_id] = float(trials)
    command._curriculum_window_start_failures[bin_id] = float(failures)
    command.curriculum_start_trials[bin_id] += float(trials)
    command.curriculum_start_failures[bin_id] += float(failures)
    command._update_curriculum_states()


def test_failure_is_attributed_to_start_a_and_terminal_c(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _curriculum_command()
    _sample_first_bin(monkeypatch)
    command._has_sampled[0] = True
    command._episode_start_bins[0] = 0
    command._episode_start_frames[0] = 1
    command._episode_last_visited_bins[0] = 1
    command._episode_curriculum_steps[0] = 9
    command.time_steps[0] = 10
    command._env.termination_manager.terminated[0] = True
    command._env.termination_manager.body_pos[0] = True
    command.last_global_body_pos_errors = torch.tensor([[float("nan"), 0.7], [0.0, 0.0]])
    command.last_global_body_pos_error_names = ["base", "foot"]

    MotionCommand._adaptive_sampling(command, torch.tensor([0]))

    assert command._current_curriculum_start_trials[0].item() == 1.0
    assert command._current_curriculum_start_failures[0].item() == 1.0
    assert command._current_curriculum_terminal_visits[2].item() == 1.0
    assert command._current_curriculum_terminal_failures[2].item() == 1.0
    torch.testing.assert_close(command._current_curriculum_terminal_body_failures, torch.tensor([0.0, 1.0]))
    assert command._current_curriculum_start_survival_steps[0].item() == 9.0


def test_timeout_counts_a_start_trial_but_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _curriculum_command()
    _sample_first_bin(monkeypatch)
    command._has_sampled[0] = True
    command._episode_start_bins[0] = 1
    command._episode_start_frames[0] = 4
    command._episode_last_visited_bins[0] = 4
    command._episode_curriculum_steps[0] = 16
    command.time_steps[0] = 19
    command._env.termination_manager.terminated[0] = False

    MotionCommand._adaptive_sampling(command, torch.tensor([0]))

    assert command._current_curriculum_start_trials[1].item() == 1.0
    assert command._current_curriculum_start_failures.sum().item() == 0.0
    assert command._current_curriculum_terminal_failures.sum().item() == 0.0


def test_terminal_visit_counts_each_bin_only_once_per_episode() -> None:
    command = _curriculum_command()
    command._has_sampled[0] = True
    command.time_steps[0] = 2

    command._update_adaptive_exposure()
    command._update_adaptive_exposure()
    command.time_steps[0] = 5
    command._update_adaptive_exposure()

    torch.testing.assert_close(
        command._current_curriculum_terminal_visits,
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
    )


def test_unknown_frontier_mastered_and_mastered_hysteresis() -> None:
    command = _curriculum_command()
    bin_id = 0

    _record_window(command, bin_id, trials=8, failures=4)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_UNKNOWN
    _record_window(command, bin_id, trials=8, failures=4)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_FRONTIER

    _record_window(command, bin_id, trials=20, failures=0)
    _record_window(command, bin_id, trials=20, failures=0)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_MASTERED

    _record_window(command, bin_id, trials=20, failures=2)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_MASTERED
    _record_window(command, bin_id, trials=20, failures=3)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_FRONTIER


def test_sparse_window_evidence_accumulates_until_it_is_valid() -> None:
    command = _curriculum_command()
    bin_id = 0

    command._curriculum_window_start_trials[bin_id] = 4.0
    command._curriculum_window_start_failures[bin_id] = 2.0
    command.curriculum_start_trials[bin_id] = 4.0
    command._update_curriculum_states()

    assert command._curriculum_window_start_trials[bin_id].item() == 4.0
    assert torch.isnan(command._curriculum_failure_rate_history[:, bin_id]).all()

    command._curriculum_window_start_trials[bin_id] += 4.0
    command._curriculum_window_start_failures[bin_id] += 2.0
    command.curriculum_start_trials[bin_id] += 4.0
    command._update_curriculum_states()

    assert command._curriculum_failure_rate_history[-1, bin_id].item() == pytest.approx(0.5)
    assert command._curriculum_window_start_trials[bin_id].item() == 0.0
    assert command._curriculum_window_start_failures[bin_id].item() == 0.0


def test_persistently_hard_frontier_becomes_stalled_then_quarantined() -> None:
    command = _curriculum_command()
    bin_id = 1

    _record_window(command, bin_id, trials=10, failures=10)
    _record_window(command, bin_id, trials=10, failures=10)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_FRONTIER
    _record_window(command, bin_id, trials=10, failures=10)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_STALLED
    _record_window(command, bin_id, trials=10, failures=10)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_QUARANTINE
    assert command._curriculum_state_entry_trials[bin_id].item() == 40.0


def test_improving_stalled_bin_returns_to_frontier() -> None:
    command = _curriculum_command()
    bin_id = 2
    command._curriculum_states[bin_id] = command._CURRICULUM_STALLED
    command._curriculum_failure_rate_history[:, bin_id] = torch.tensor([0.95, 0.95, 0.95])
    command.curriculum_start_trials[bin_id] = 40.0

    _record_window(command, bin_id, trials=10, failures=7)

    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_FRONTIER


def test_high_failure_but_improving_bin_stays_frontier() -> None:
    command = _curriculum_command()
    bin_id = 2
    command._curriculum_states[bin_id] = command._CURRICULUM_FRONTIER
    command._curriculum_failure_rate_history[:, bin_id] = torch.tensor([0.99, 0.96, 0.96])
    command.curriculum_start_trials[bin_id] = 100.0

    _record_window(command, bin_id, trials=100, failures=89)

    assert command._curriculum_failure_rate_history[-1, bin_id].item() < 0.90
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_FRONTIER


def test_quarantine_requires_enough_probe_trials_before_exit() -> None:
    command = _curriculum_command()
    command.cfg.curriculum_min_exit_probe_trials = 16
    bin_id = 3
    command._curriculum_states[bin_id] = command._CURRICULUM_QUARANTINE
    command._curriculum_failure_rate_history[:, bin_id] = torch.tensor([0.95, 0.95, 0.95])
    command.curriculum_start_trials[bin_id] = 40.0
    command._curriculum_state_entry_trials[bin_id] = 40.0

    _record_window(command, bin_id, trials=10, failures=7)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_QUARANTINE
    _record_window(command, bin_id, trials=10, failures=7)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_FRONTIER


def test_distributed_curriculum_statistics_use_raw_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _curriculum_command()
    command._current_curriculum_start_trials[0] = 2.0
    command._current_curriculum_start_failures[0] = 1.0
    command._current_curriculum_terminal_visits[2] = 3.0
    command._current_curriculum_terminal_failures[2] = 1.0
    command._current_curriculum_terminal_body_failures[1] = 1.0
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def _all_reduce(values: torch.Tensor, **_kwargs) -> None:
        values.mul_(2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)

    command._flush_curriculum_statistics()

    assert command.curriculum_start_trials[0].item() == 4.0
    assert command.curriculum_start_failures[0].item() == 2.0
    assert command.curriculum_terminal_visits[2].item() == 6.0
    assert command.curriculum_terminal_failures[2].item() == 2.0
    assert command.curriculum_terminal_body_failures[1].item() == 2.0


def test_state_budgets_are_redistributed_across_nonempty_states() -> None:
    command = _curriculum_command()
    command._curriculum_states[:] = torch.tensor(
        [
            command._CURRICULUM_UNKNOWN,
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_FRONTIER,
            command._CURRICULUM_FRONTIER,
            command._CURRICULUM_QUARANTINE,
        ],
        dtype=torch.uint8,
    )

    probabilities = command._curriculum_state_budget_probabilities(command._active_bin_ids, dtype=torch.float64)

    present_weights = torch.tensor([0.20, 0.10, 0.55, 0.05], dtype=torch.float64)
    present_weights /= present_weights.sum()
    expected = torch.tensor(
        [
            present_weights[0],
            present_weights[1],
            present_weights[2] / 2.0,
            present_weights[2] / 2.0,
            present_weights[3],
        ]
    )
    torch.testing.assert_close(probabilities, expected)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_shadow_mode_and_blend_schedule_preserve_then_mix_legacy_distribution() -> None:
    command = _curriculum_command()
    command.cfg.adp_samp_failure_rate_max_over_mean = None
    command.cfg.curriculum_blend_iterations = 10
    command._curriculum_states[:] = torch.tensor([0, 1, 2, 2, 4], dtype=torch.uint8)
    legacy = torch.tensor([0.50, 0.20, 0.10, 0.10, 0.10], dtype=torch.float64)
    curriculum = command._curriculum_state_budget_probabilities(command._active_bin_ids, dtype=legacy.dtype)

    torch.testing.assert_close(command._blend_curriculum_distribution(legacy, command._active_bin_ids), legacy)
    command._curriculum_blend_start_iteration = 100
    command._curriculum_iteration = 100
    torch.testing.assert_close(command._blend_curriculum_distribution(legacy, command._active_bin_ids), legacy)
    command._curriculum_iteration = 105
    torch.testing.assert_close(
        command._blend_curriculum_distribution(legacy, command._active_bin_ids),
        legacy.lerp(curriculum, 0.5),
    )
    command._curriculum_iteration = 110
    torch.testing.assert_close(command._blend_curriculum_distribution(legacy, command._active_bin_ids), curriculum)


def test_probability_cap_is_reapplied_after_full_curriculum_blend() -> None:
    command = _curriculum_command()
    command.cfg.adp_samp_failure_rate_max_over_mean = 2.0
    command.cfg.curriculum_blend_iterations = 0
    command._curriculum_blend_start_iteration = 1
    command._curriculum_iteration = 1
    command._curriculum_states[:] = torch.tensor([2, 1, 1, 1, 1], dtype=torch.uint8)

    probabilities = command._blend_curriculum_distribution(
        torch.full((5,), 0.2, dtype=torch.float64), command._active_bin_ids
    )

    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert probabilities.max().item() <= 0.4 + 1.0e-12


def test_loading_legacy_state_preserves_v13_stats_and_zeros_curriculum() -> None:
    command = _curriculum_command()
    command.curriculum_start_trials.fill_(9.0)
    command.curriculum_start_failures.fill_(4.0)
    command.curriculum_terminal_visits.fill_(7.0)
    command.curriculum_terminal_failures.fill_(3.0)
    command._curriculum_states.fill_(command._CURRICULUM_QUARANTINE)
    command._curriculum_failure_rate_history.fill_(0.9)
    command._curriculum_iteration = 77
    legacy = {
        "adp_samp_num_episodes": torch.arange(1, 6, dtype=torch.float32),
        "adp_samp_num_failures": torch.arange(6, 11, dtype=torch.float32),
    }

    assert command.load_adaptive_sampling_state(legacy)

    torch.testing.assert_close(command.adp_samp_num_episodes, legacy["adp_samp_num_episodes"])
    torch.testing.assert_close(command.adp_samp_num_failures, legacy["adp_samp_num_failures"])
    assert command.curriculum_start_trials.count_nonzero().item() == 0
    assert command.curriculum_start_failures.count_nonzero().item() == 0
    assert command.curriculum_terminal_visits.count_nonzero().item() == 0
    assert command.curriculum_terminal_failures.count_nonzero().item() == 0
    assert torch.all(command._curriculum_states == command._CURRICULUM_UNKNOWN)
    assert torch.isnan(command._curriculum_failure_rate_history).all()
    assert command._curriculum_iteration == 0
    assert command._curriculum_blend_start_iteration == -1


def test_v14_curriculum_state_round_trip() -> None:
    source = _curriculum_command()
    source.adp_samp_num_episodes[:] = torch.arange(1, 6, dtype=torch.float32)
    source.adp_samp_num_failures[:] = torch.arange(6, 11, dtype=torch.float32)
    source.curriculum_start_trials[:] = torch.tensor([16.0, 32.0, 48.0, 64.0, 80.0])
    source.curriculum_start_failures[:] = torch.tensor([1.0, 4.0, 20.0, 60.0, 78.0])
    source.curriculum_start_survival_steps[:] = torch.arange(5, dtype=torch.float32) + 10.0
    source.curriculum_start_completion_fraction[:] = torch.arange(5, dtype=torch.float32) / 10.0
    source.curriculum_terminal_visits[:] = torch.arange(5, dtype=torch.float32) + 20.0
    source.curriculum_terminal_failures[:] = torch.arange(5, dtype=torch.float32)
    source.curriculum_terminal_body_failures[:] = torch.tensor([3.0, 7.0])
    source._curriculum_window_start_trials[:] = torch.arange(5, dtype=torch.float32) + 1.0
    source._curriculum_window_start_failures[:] = torch.arange(5, dtype=torch.float32)
    source._curriculum_failure_rate_history[:] = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 20.0
    source._curriculum_states[:] = torch.arange(5, dtype=torch.uint8)
    source._curriculum_state_entry_trials[:] = torch.arange(5, dtype=torch.float32) + 30.0
    source._curriculum_iteration = 123
    source._curriculum_last_state_update_iteration = 120
    source._curriculum_blend_start_iteration = 100
    state = source.get_adaptive_sampling_state()
    restored = _curriculum_command()

    assert restored.load_adaptive_sampling_state(state)

    restored_state = restored.get_adaptive_sampling_state()
    assert restored_state.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(restored_state[name], state[name])
