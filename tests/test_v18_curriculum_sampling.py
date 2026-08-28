from __future__ import annotations

from types import MethodType

import pytest
import torch
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand
from test_v17_curriculum_sampling import _v17_command

_V18_METHOD_NAMES = (
    "_curriculum_unified_window_enabled",
    "_initialize_curriculum_unified_geometry",
    "_initialize_curriculum_unified_episode",
    "_reset_curriculum_window_quality",
    "_record_curriculum_unified_window_quality",
    "_accumulate_curriculum_terminal_quality",
    "_record_curriculum_unified_window_outcomes",
    "_record_curriculum_unified_window_censored",
    "_advance_curriculum_unified_windows",
    "_update_curriculum_unified_windows",
    "_enter_curriculum_forward_bins",
    "_record_curriculum_forward_outcomes",
    "_curriculum_quality_evidence_mask",
    "_curriculum_unified_need_probabilities",
    "_update_curriculum_unified_states",
    "_finalize_curriculum_unified_episodes",
    "_curriculum_unified_checkpoint_targets",
    "_curriculum_unified_checkpoint_state_valid",
    "_load_curriculum_unified_checkpoint",
)


def _v18_command(*, lengths: tuple[int, ...] = (11,)):
    """Build the CPU-only V17 fixture with V18's unified credit path."""

    command = _v17_command(lengths=lengths)
    command.cfg.curriculum_unified_window_enabled = True
    command.cfg.curriculum_start_eligibility_enabled = False
    command.cfg.curriculum_terminal_replay_fraction = 0.0
    command.cfg.curriculum_coverage_aware_enabled = False
    command.cfg.pre_failure_sample_window = 0
    command.cfg.curriculum_need_uniform_fraction = 0.20
    command.cfg.curriculum_forward_risk_horizon_bins = 5
    command.cfg.curriculum_forward_risk_decay = 0.80
    command.cfg.curriculum_forward_risk_good_threshold = 0.05
    command.cfg.curriculum_forward_risk_bad_threshold = 0.60
    command.cfg.curriculum_quality_good_threshold = 0.08
    command.cfg.curriculum_quality_bad_threshold = 0.30
    command.cfg.curriculum_quarantine_need_scale = 0.25
    # Production uses 32. Keeping that value here also makes the uncertainty
    # mapping's 0/16/32 boundary explicit in the sampler tests.
    command.cfg.curriculum_min_known_trials = 32

    command._CURRICULUM_UNIFIED_WINDOW_SCHEMA_VERSION = (
        MotionCommand._CURRICULUM_UNIFIED_WINDOW_SCHEMA_VERSION
    )
    for name in _V18_METHOD_NAMES:
        setattr(command, name, MethodType(getattr(MotionCommand, name), command))
    command._normalize_curriculum_need = MotionCommand._normalize_curriculum_need
    command._shift_curriculum_history = MotionCommand._shift_curriculum_history

    termination_manager = command._env.termination_manager
    termination_manager.invalid_robot_state = torch.zeros(command.num_envs, dtype=torch.bool)
    termination_manager.motion_time_out = torch.zeros(command.num_envs, dtype=torch.bool)

    def get_term(name: str) -> torch.Tensor:
        try:
            return getattr(termination_manager, name)
        except AttributeError as exc:
            raise KeyError(name) from exc

    termination_manager.get_term = get_term

    # _v17_command initialized the V16 eligible/tail geometry first. Replace
    # it with V18's all-frame geometry, then allocate the new tensors once.
    command._initialize_curriculum_unified_geometry()
    command._initialize_curriculum_sampling()
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()
    return command


def _begin_unified_episode(command, *, env_id: int = 0, motion_id: int = 0, start_frame: int) -> None:
    env_ids = torch.tensor([env_id], dtype=torch.long)
    command.motion_ids[env_id] = motion_id
    command.motion_lengths[env_id] = command.motion.lengths(torch.tensor([motion_id]))[0]
    command.time_steps[env_id] = start_frame
    command._has_sampled[env_id] = True
    command._episode_last_visited_bins[env_id] = -1
    command._initialize_curriculum_unified_episode(env_ids)


def _observe_unified_frame(command, *, env_id: int = 0, frame: int, error: float | None = None) -> None:
    env_ids = torch.tensor([env_id], dtype=torch.long)
    command.time_steps[env_id] = frame
    if error is not None:
        errors = torch.zeros((command.num_envs, len(command.cfg.body_names)), dtype=torch.float32)
        errors[env_id] = error
        command.last_global_body_pos_errors = errors
        command.last_global_body_pos_error_time_steps = command.time_steps.clone()
        command._accumulate_curriculum_quality_error()
    command._update_curriculum_unified_windows(env_ids)


def _finalize_unified_episode(
    command,
    *,
    env_id: int = 0,
    failed: bool = False,
    invalid: bool = False,
    motion_completed: bool = False,
) -> None:
    env_ids = torch.tensor([env_id], dtype=torch.long)
    terminal_bins = command._bucket_ids(command.motion_ids[env_ids], command.time_steps[env_ids])
    command._finalize_curriculum_unified_episodes(
        env_ids,
        terminal_bins,
        torch.tensor([failed]),
        torch.tensor([invalid]),
        torch.tensor([motion_completed]),
    )


def test_unified_chain_credits_start_transit_and_short_tail_identically() -> None:
    # B=H=4, L=11, start=1 gives [1,5), [5,9), and the two-frame [9,11).
    command = _v18_command(lengths=(11,))
    _begin_unified_episode(command, start_frame=1)

    for frame in range(1, 11):
        _observe_unified_frame(command, frame=frame)

    torch.testing.assert_close(command._current_curriculum_start_trials, torch.ones(3))
    torch.testing.assert_close(command._current_curriculum_start_failures, torch.zeros(3))
    torch.testing.assert_close(command._current_curriculum_start_survival_steps, torch.tensor([4.0, 4.0, 2.0]))
    torch.testing.assert_close(command._current_curriculum_start_completion_fraction, torch.ones(3))
    assert not command._episode_unified_window_active[0]


def test_later_failure_keeps_completed_successes_and_failure_wins_at_the_boundary() -> None:
    command = _v18_command(lengths=(20,))
    _begin_unified_episode(command, start_frame=1)

    # Two complete windows are already durable successes.
    for frame in range(1, 9):
        _observe_unified_frame(command, frame=frame, error=0.3)

    # The fourth and nominally completing observation of the third window is
    # a terminating frame. It must make that window a failure, not a success.
    for frame in range(9, 12):
        _observe_unified_frame(command, frame=frame, error=0.3)
    command.time_steps[0] = 12
    _finalize_unified_episode(command, failed=True, motion_completed=True)

    torch.testing.assert_close(command._current_curriculum_start_trials[:3], torch.ones(3))
    torch.testing.assert_close(
        command._current_curriculum_start_failures[:3], torch.tensor([0.0, 0.0, 1.0])
    )
    # Failed windows never contribute quality; the first two successes remain.
    torch.testing.assert_close(command._current_curriculum_quality_trials[:3], torch.tensor([1.0, 1.0, 0.0]))


@pytest.mark.parametrize(
    ("start_frame", "target"),
    ((7, 4), (8, 3), (10, 1)),
)
def test_natural_motion_end_turns_every_remaining_length_into_an_ordinary_success(
    start_frame: int,
    target: int,
) -> None:
    command = _v18_command(lengths=(11,))
    _begin_unified_episode(command, start_frame=start_frame)

    # The final reference frame is observed by the termination manager and is
    # deliberately finalized outside _update_curriculum_unified_windows().
    for frame in range(start_frame, 10):
        _observe_unified_frame(command, frame=frame)
    command.time_steps[0] = 10
    _finalize_unified_episode(command, motion_completed=True)

    bin_id = start_frame // command.cfg.bin_size
    assert command._current_curriculum_start_trials[bin_id].item() == 1.0
    assert command._current_curriculum_start_failures[bin_id].item() == 0.0
    assert command._current_curriculum_start_censored[bin_id].item() == 0.0
    assert command._current_curriculum_start_survival_steps[bin_id].item() == float(target)
    assert command._current_curriculum_start_completion_fraction[bin_id].item() == 1.0


def test_manual_or_invalid_interruption_is_censored_but_motion_end_failure_is_not() -> None:
    manual = _v18_command(lengths=(11,))
    _begin_unified_episode(manual, start_frame=8)
    _observe_unified_frame(manual, frame=8)
    _finalize_unified_episode(manual)
    assert manual._current_curriculum_start_trials.sum().item() == 0.0
    assert manual._current_curriculum_start_censored[2].item() == 1.0
    assert manual._current_curriculum_forward_trials.sum().item() == 0.0

    invalid = _v18_command(lengths=(11,))
    _begin_unified_episode(invalid, start_frame=8)
    _observe_unified_frame(invalid, frame=8)
    _finalize_unified_episode(invalid, failed=True, invalid=True)
    assert invalid._current_curriculum_start_trials.sum().item() == 0.0
    assert invalid._current_curriculum_start_censored[2].item() == 1.0
    assert invalid._current_curriculum_forward_trials.sum().item() == 0.0

    failed = _v18_command(lengths=(11,))
    _begin_unified_episode(failed, start_frame=8)
    _observe_unified_frame(failed, frame=8)
    failed.time_steps[0] = 10
    _finalize_unified_episode(failed, failed=True, motion_completed=True)
    assert failed._current_curriculum_start_trials[2].item() == 1.0
    assert failed._current_curriculum_start_failures[2].item() == 1.0
    assert failed._current_curriculum_start_censored.sum().item() == 0.0


def test_quality_is_per_window_and_uses_the_actual_short_tail_observation_count() -> None:
    command = _v18_command(lengths=(11,))
    _begin_unified_episode(command, start_frame=1)

    for frame in range(1, 5):
        _observe_unified_frame(command, frame=frame, error=0.3)
    for frame in range(5, 9):
        _observe_unified_frame(command, frame=frame, error=0.6)
    for frame in range(9, 11):
        _observe_unified_frame(command, frame=frame, error=0.3)

    expected_low = 1.0 - torch.exp(torch.tensor(-1.0))
    expected_high = 1.0 - torch.exp(torch.tensor(-4.0))
    torch.testing.assert_close(command._current_curriculum_quality_trials, torch.ones(3))
    torch.testing.assert_close(
        command._current_curriculum_quality_error_sum,
        torch.stack((expected_low, expected_high, expected_low)),
    )


def test_nonfinite_quality_invalidates_only_its_current_window() -> None:
    command = _v18_command(lengths=(12,))
    _begin_unified_episode(command, start_frame=0)

    _observe_unified_frame(command, frame=0, error=float("nan"))
    for frame in range(1, 4):
        _observe_unified_frame(command, frame=frame, error=0.3)
    for frame in range(4, 8):
        _observe_unified_frame(command, frame=frame, error=0.3)

    # Survival credit is unaffected, but the poisoned first quality window is
    # dropped and the next window begins with a fresh validity flag.
    torch.testing.assert_close(command._current_curriculum_start_trials[:2], torch.ones(2))
    torch.testing.assert_close(command._current_curriculum_quality_trials[:2], torch.tensor([0.0, 1.0]))
    assert torch.all(torch.isfinite(command._current_curriculum_quality_error_sum))


def test_forward_failure_credits_only_the_latest_five_visited_bins_with_decay() -> None:
    command = _v18_command(lengths=(28,))
    env_ids = torch.tensor([0])

    for bin_id in range(7):
        command._enter_curriculum_forward_bins(env_ids, torch.tensor([bin_id]))
    command._record_curriculum_forward_outcomes(env_ids, failed=True)

    # Bins 0 and 1 safely aged out before the failure. D-4..D get one
    # non-cancelling failure-risk outcome with lambda**distance.
    torch.testing.assert_close(command._current_curriculum_forward_trials, torch.ones(7))
    torch.testing.assert_close(
        command._current_curriculum_forward_risk_sum,
        torch.tensor([0.0, 0.0, 0.8**4, 0.8**3, 0.8**2, 0.8, 1.0]),
    )
    assert torch.all(command._episode_forward_risk_bins[0] == -1)


def test_forward_credit_never_reaches_unvisited_or_previous_motion_bins() -> None:
    # Motion 1 starts at global bin 3. A failure in its first visited bin must
    # not leak backward into motion 0's final bins.
    command = _v18_command(lengths=(12, 12))
    second_motion_first_bin = int(command.motion_bin_offsets[1].item())
    command._enter_curriculum_forward_bins(
        torch.tensor([0]), torch.tensor([second_motion_first_bin])
    )
    command._record_curriculum_forward_outcomes(torch.tensor([0]), failed=True)

    expected_trials = torch.zeros(command.bin_count)
    expected_trials[second_motion_first_bin] = 1.0
    torch.testing.assert_close(command._current_curriculum_forward_trials, expected_trials)
    torch.testing.assert_close(command._current_curriculum_forward_risk_sum, expected_trials)


def test_safe_forward_history_resolves_to_zero_at_horizon_or_motion_end() -> None:
    command = _v18_command(lengths=(24,))
    env_ids = torch.tensor([0])
    for bin_id in range(6):
        command._enter_curriculum_forward_bins(env_ids, torch.tensor([bin_id]))

    # Bin 0 has already aged out safely; a successful end resolves bins 1..5.
    command._record_curriculum_forward_outcomes(env_ids, failed=False)
    torch.testing.assert_close(command._current_curriculum_forward_trials, torch.ones(6))
    torch.testing.assert_close(command._current_curriculum_forward_risk_sum, torch.zeros(6))


def test_need_components_use_calibrated_ranges_and_max_not_addition() -> None:
    command = _v18_command(lengths=(16,))
    ids = torch.arange(4)

    # Beta(1,1) gives exact local rates .1, .5, and .9 for these counts.
    command.curriculum_start_trials[:3] = 8.0
    command.curriculum_start_failures[:3] = torch.tensor([0.0, 4.0, 8.0])
    command.curriculum_start_trials[3] = 32.0
    command.curriculum_start_failures[3] = 16.0
    command.curriculum_forward_trials[:] = 32.0
    command._curriculum_forward_risk_ema[:] = torch.tensor([0.05, 0.325, 0.60, 0.325])
    command.curriculum_quality_trials[:] = float(command.cfg.curriculum_quality_min_trials)
    command._curriculum_quality_observed_windows[:] = int(command.cfg.curriculum_quality_min_windows)
    command._curriculum_quality_error_ema[:] = torch.tensor([0.08, 0.19, 0.30, 0.19])

    command._curriculum_unified_need_probabilities(ids, dtype=torch.float64)

    torch.testing.assert_close(command._curriculum_last_survival_need[:3], torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(command._curriculum_last_forward_need[:3], torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(command._curriculum_last_quality_need[:3], torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(
        command._curriculum_last_uncertainty_need,
        torch.tensor([0.75, 0.75, 0.75, 0.0]),
    )
    # Bin 3 has three .5 diagnostics; max keeps its need at .5 rather than 1.5.
    assert command._curriculum_last_combined_need[3].item() == pytest.approx(0.5)


def test_survival_need_uses_all_unified_outcomes_not_a_small_recent_window() -> None:
    command = _v18_command(lengths=(8,))
    ids = torch.arange(2)

    # The cumulative Beta posterior is exactly .10, hence zero survival need.
    # A contradictory eight-sample decision window is only state evidence and
    # must not replace the cumulative sampling estimate.
    command.curriculum_start_trials[:] = 98.0
    command.curriculum_start_failures[:] = 9.0
    command._curriculum_failure_rate_history[2] = 0.90
    command._curriculum_window_start_trials[:] = 8.0
    command._curriculum_window_start_failures[:] = 8.0

    command._curriculum_unified_need_probabilities(ids, dtype=torch.float64)

    torch.testing.assert_close(command._curriculum_last_survival_need, torch.zeros(2))


def test_unified_states_cannot_advance_twice_without_fresh_window_evidence() -> None:
    command = _v18_command(lengths=(8,))
    command.curriculum_start_trials[0] = 128.0
    command.curriculum_start_failures[0] = 120.0
    command.curriculum_terminal_visits[0] = 128.0
    command.curriculum_terminal_failures[0] = 128.0
    command._curriculum_failure_rate_history[:, 0] = 0.95
    command._curriculum_states[0] = command._CURRICULUM_MASTERED

    # One fresh hard window is allowed to move MASTERED -> FRONTIER.
    command._curriculum_window_start_trials[0] = 8.0
    command._curriculum_window_start_failures[0] = 8.0
    command._update_curriculum_unified_states()
    assert command._curriculum_states[0].item() == command._CURRICULUM_FRONTIER

    # The window was consumed. Persistent history alone cannot immediately
    # reuse the same evidence to move FRONTIER -> STALLED.
    command._update_curriculum_unified_states()
    assert command._curriculum_states[0].item() == command._CURRICULUM_FRONTIER

    # The same rule prevents a stalled label from entering quarantine on an
    # empty update tick, even when its retained history and terminal hazard
    # are severe.
    command._curriculum_states[0] = command._CURRICULUM_STALLED
    command._update_curriculum_unified_states()
    assert command._curriculum_states[0].item() == command._CURRICULUM_STALLED


def test_need_sampler_is_twenty_percent_all_frame_uniform_and_eighty_percent_frame_weighted_need() -> None:
    command = _v18_command(lengths=(11,))
    ids = torch.arange(3)

    # Produce exact needs [0, 1, .5]. The last bin has only three frames.
    command.curriculum_start_trials[:] = torch.tensor([32.0, 8.0, 32.0])
    command.curriculum_start_failures[:] = torch.tensor([0.0, 8.0, 16.0])
    probabilities = command._curriculum_unified_need_probabilities(ids, dtype=torch.float64)

    expected = torch.tensor([0.8 / 11.0, 7.2 / 11.0, 3.0 / 11.0], dtype=torch.float64)
    torch.testing.assert_close(probabilities, expected)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_empty_need_falls_back_to_all_frame_uniform_without_state_budgets() -> None:
    command = _v18_command(lengths=(11,))
    ids = torch.arange(3)
    command.curriculum_start_trials[:] = 32.0
    command.curriculum_start_failures.zero_()
    command.curriculum_forward_trials[:] = 32.0
    command._curriculum_forward_risk_ema.zero_()
    command.curriculum_quality_trials[:] = float(command.cfg.curriculum_quality_min_trials)
    command._curriculum_quality_observed_windows[:] = int(command.cfg.curriculum_quality_min_windows)
    command._curriculum_quality_error_ema.zero_()

    command._curriculum_states[:] = command._CURRICULUM_MASTERED
    mastered = command._curriculum_unified_need_probabilities(ids, dtype=torch.float64)
    command._curriculum_states[:] = torch.tensor(
        [command._CURRICULUM_UNKNOWN, command._CURRICULUM_FRONTIER, command._CURRICULUM_STALLED],
        dtype=torch.uint8,
    )
    relabeled = command._curriculum_unified_need_probabilities(ids, dtype=torch.float64)

    expected = torch.tensor([4.0 / 11.0, 4.0 / 11.0, 3.0 / 11.0], dtype=torch.float64)
    torch.testing.assert_close(mastered, expected)
    torch.testing.assert_close(relabeled, expected)


def test_mastered_quality_or_forward_problem_receives_need_without_fixed_state_mass() -> None:
    command = _v18_command(lengths=(12,))
    ids = torch.arange(3)
    command.curriculum_start_trials[:] = 32.0
    command.curriculum_start_failures.zero_()
    command.curriculum_forward_trials[:] = 32.0
    command._curriculum_forward_risk_ema[:] = torch.tensor([0.0, 0.60, 0.0])
    command.curriculum_quality_trials[:] = float(command.cfg.curriculum_quality_min_trials)
    command._curriculum_quality_observed_windows[:] = int(command.cfg.curriculum_quality_min_windows)
    command._curriculum_quality_error_ema[:] = torch.tensor([0.0, 0.0, 0.30])
    command._curriculum_states[:] = command._CURRICULUM_MASTERED

    probabilities = command._curriculum_unified_need_probabilities(ids, dtype=torch.float64)

    torch.testing.assert_close(command._curriculum_last_combined_need, torch.tensor([0.0, 1.0, 1.0]))
    assert probabilities[1].item() == pytest.approx(probabilities[2].item())
    assert probabilities[1].item() > probabilities[0].item()


def test_schema7_round_trip_restores_unified_and_forward_statistics_exactly() -> None:
    source = _v18_command(lengths=(20,))
    source.curriculum_start_trials[:3] = torch.tensor([40.0, 48.0, 56.0])
    source.curriculum_start_failures[:3] = torch.tensor([2.0, 8.0, 40.0])
    source.curriculum_forward_trials[:3] = torch.tensor([32.0, 40.0, 48.0])
    source.curriculum_forward_risk_sum[:3] = torch.tensor([1.0, 8.0, 24.0])
    source._curriculum_window_forward_trials[:3] = 8.0
    source._curriculum_window_forward_risk_sum[:3] = torch.tensor([0.4, 1.6, 4.0])
    source._curriculum_forward_risk_history[:, :3] = torch.tensor(
        [[0.02, 0.20, 0.50], [0.03, 0.18, 0.55], [0.04, 0.16, 0.60]]
    )
    source._curriculum_forward_risk_ema[:3] = torch.tensor([0.03, 0.17, 0.55])
    source._curriculum_forward_observed_windows[:3] = 2
    source._curriculum_states[:3] = torch.tensor(
        [source._CURRICULUM_MASTERED, source._CURRICULUM_FRONTIER, source._CURRICULUM_STALLED],
        dtype=torch.uint8,
    )
    source._curriculum_iteration = 120
    source._curriculum_last_state_update_iteration = 100
    source._curriculum_blend_start_iteration = 50
    source._rebuild_global_sampling_distribution()
    source._rebuild_sampling_distribution()
    state = source.get_adaptive_sampling_state()
    restored = _v18_command(lengths=(20,))

    assert state["curriculum_state_schema_version"].item() == 7
    assert "curriculum_smoothed_eligible_probabilities" not in state
    assert restored.load_adaptive_sampling_state(state)

    restored_state = restored.get_adaptive_sampling_state()
    assert restored_state.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(restored_state[name], state[name], equal_nan=True)
    torch.testing.assert_close(restored.adp_sampling_prob, source.adp_sampling_prob)


def test_schema6_to_schema7_keeps_v17_evidence_and_sampler_as_shadow() -> None:
    source = _v17_command(lengths=(20,))
    source.curriculum_start_trials[:3] = torch.tensor([32.0, 48.0, 64.0])
    source.curriculum_start_failures[:3] = torch.tensor([1.0, 12.0, 50.0])
    source._curriculum_states[:3] = torch.tensor(
        [source._CURRICULUM_MASTERED, source._CURRICULUM_FRONTIER, source._CURRICULUM_STALLED],
        dtype=torch.uint8,
    )
    source.curriculum_quality_trials[:3] = 8.0
    source._curriculum_quality_observed_windows[:3] = 2
    source._curriculum_quality_error_ema[:3] = torch.tensor([0.1, 0.3, 0.7])
    source._curriculum_iteration = 200
    source._curriculum_last_state_update_iteration = 200
    source._curriculum_blend_start_iteration = 0
    source._curriculum_known_fraction = 1.0
    source._rebuild_global_sampling_distribution()
    source._rebuild_sampling_distribution()
    state = source.get_adaptive_sampling_state()
    checkpoint_probabilities = state["curriculum_smoothed_probabilities"].clone()
    restored = _v18_command(lengths=(20,))

    assert state["curriculum_state_schema_version"].item() == 6
    assert restored.load_adaptive_sampling_state(state)

    torch.testing.assert_close(restored.curriculum_start_trials, source.curriculum_start_trials)
    torch.testing.assert_close(restored.curriculum_start_failures, source.curriculum_start_failures)
    torch.testing.assert_close(restored._curriculum_states, source._curriculum_states)
    torch.testing.assert_close(restored.curriculum_quality_trials, source.curriculum_quality_trials)
    torch.testing.assert_close(
        restored._curriculum_quality_error_ema,
        source._curriculum_quality_error_ema,
        equal_nan=True,
    )
    assert restored.curriculum_forward_trials.count_nonzero().item() == 0
    assert restored.curriculum_forward_risk_sum.count_nonzero().item() == 0
    assert torch.isnan(restored._curriculum_forward_risk_ema).all()
    assert restored._curriculum_iteration == 0
    assert restored._curriculum_blend_start_iteration == -1
    assert restored._curriculum_has_shadow_distribution
    torch.testing.assert_close(restored._curriculum_shadow_probabilities, checkpoint_probabilities)
    torch.testing.assert_close(restored.adp_sampling_prob, checkpoint_probabilities)
