from __future__ import annotations

from types import MethodType

import pytest
import torch
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand
from test_v16_curriculum_sampling import _v16_command

_V17_METHOD_NAMES = (
    "_curriculum_quality_learning_enabled",
    "_initialize_curriculum_quality_learning",
    "_accumulate_curriculum_quality_error",
    "_record_curriculum_start_quality",
    "_update_curriculum_quality_statistics",
    "_refresh_curriculum_quality_learning_pool",
    "_curriculum_quality_known_mask",
)


def _v17_command(*, lengths: tuple[int, ...] = (84,)):
    """Build the CPU-only V16 fixture with the opt-in V17 quality layer."""

    command = _v16_command(lengths=lengths)
    command.cfg.curriculum_quality_learning_enabled = True
    command.cfg.curriculum_learning_pool_min_fraction = 0.15
    command.cfg.curriculum_quality_filler_weight = 0.5
    command.cfg.curriculum_quality_min_trials = 4
    command.cfg.curriculum_quality_min_window_trials = 2
    command.cfg.curriculum_quality_min_windows = 2
    command.cfg.curriculum_quality_error_ema_alpha = 0.5
    command.cfg.curriculum_quality_body_pos_std = 0.3
    command._env.reset_buf = torch.zeros(command.num_envs, dtype=torch.bool)
    command._env.reset_terminated = torch.zeros(command.num_envs, dtype=torch.bool)
    command._env.reset_time_outs = torch.zeros(command.num_envs, dtype=torch.bool)
    command._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION = MotionCommand._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION
    for name in _V17_METHOD_NAMES:
        setattr(command, name, MethodType(getattr(MotionCommand, name), command))
    command._select_deterministic_topk_ids = MotionCommand._select_deterministic_topk_ids
    # The V16 helper initialized before the opt-in flag existed. Reinitialize
    # once so the quality tensors are allocated through the production path.
    command._initialize_curriculum_sampling()
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()
    return command


def test_global_body_position_quality_records_only_a_complete_successful_horizon() -> None:
    command = _v17_command(lengths=(20,))
    command._has_sampled[:] = True
    command._episode_start_bins[:] = torch.tensor([1, 2])
    command._episode_start_label_enabled[:] = True
    command._episode_start_outcome_recorded[:] = False
    command._episode_curriculum_quality_valid[:] = True
    command.last_global_body_pos_errors = torch.tensor([[0.3, 0.3], [0.6, 0.6]])

    for step in range(command.cfg.curriculum_start_horizon_frames):
        command.time_steps[:] = step
        command.last_global_body_pos_error_time_steps = command.time_steps.clone()
        command._accumulate_curriculum_quality_error()

    # Only env 0 is a complete successful fixed-H outcome. Env 1 models the
    # failure/censor path, which must remain exclusively in the survival lane.
    command._record_curriculum_start_quality(torch.tensor([0]))

    expected = 1.0 - torch.exp(torch.tensor(-1.0))
    assert command._current_curriculum_quality_trials[1].item() == 1.0
    torch.testing.assert_close(command._current_curriculum_quality_error_sum[1], expected)
    assert command._current_curriculum_quality_trials[2].item() == 0.0
    assert command._current_curriculum_quality_error_sum[2].item() == 0.0
    assert command._episode_curriculum_quality_observation_count[0].item() == 4.0
    # Attribution is to the episode's start bin, not its current/terminal bin.
    assert command._current_curriculum_quality_trials.sum().item() == 1.0


def test_nonfinite_quality_invalidates_the_episode_without_polluting_statistics() -> None:
    command = _v17_command(lengths=(20,))
    command._has_sampled[0] = True
    command._episode_start_bins[0] = 1
    command._episode_start_label_enabled[0] = True
    command._episode_start_outcome_recorded[0] = False
    command._episode_curriculum_quality_valid[0] = True
    command.last_global_body_pos_errors = torch.tensor([[float("nan"), 0.0], [0.0, 0.0]])
    command.last_global_body_pos_error_time_steps = command.time_steps.clone()

    command._accumulate_curriculum_quality_error()
    command._record_curriculum_start_quality(torch.tensor([0]))

    assert not command._episode_curriculum_quality_valid[0]
    assert torch.all(torch.isfinite(command._episode_curriculum_quality_error_sum))
    assert command._current_curriculum_quality_trials.count_nonzero().item() == 0
    assert command._current_curriculum_quality_error_sum.count_nonzero().item() == 0


def test_quality_episode_accumulators_reset_when_a_new_start_is_sampled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _v17_command(lengths=(20,))
    command._episode_curriculum_quality_error_sum[:] = torch.tensor([3.0, 4.0])
    command._episode_curriculum_quality_observation_count[:] = torch.tensor([2.0, 3.0])
    command._episode_curriculum_quality_valid[:] = False
    command.adp_sampling_active_prob.zero_()
    command.adp_sampling_active_prob[0] = 1.0
    command._curriculum_active_eligible_component_probabilities.zero_()
    command._curriculum_active_eligible_component_probabilities[0] = 1.0
    command._curriculum_active_terminal_component_probabilities.zero_()
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda _probabilities, num_samples, replacement: torch.zeros(num_samples, dtype=torch.long),
    )

    MotionCommand._adaptive_sampling(command, torch.tensor([0, 1]))

    torch.testing.assert_close(command._episode_curriculum_quality_error_sum, torch.zeros(2))
    torch.testing.assert_close(command._episode_curriculum_quality_observation_count, torch.zeros(2))
    assert torch.all(command._episode_curriculum_quality_valid)


def test_same_step_autoreset_cache_is_not_consumed_by_the_new_episode() -> None:
    command = _v17_command(lengths=(20,))
    command._has_sampled[:] = True
    command._episode_start_bins[:] = torch.tensor([0, 1])
    command._episode_start_label_enabled[:] = True
    command._episode_start_outcome_recorded[:] = False
    command._episode_curriculum_quality_valid[:] = True
    command.time_steps[:] = torch.tensor([0, 5])
    # Env 0 was reset in this same simulator step. Its cached large error came
    # from the just-ended episode and must not leak into the new start at t=0.
    command._env.reset_buf[:] = torch.tensor([True, False])
    command.last_global_body_pos_error_time_steps = torch.tensor([0, 5])
    command.last_global_body_pos_errors = torch.tensor([[1.0, 1.0], [0.3, 0.3]])

    command._accumulate_curriculum_quality_error()

    assert command._episode_curriculum_quality_observation_count[0].item() == 0.0
    assert command._episode_curriculum_quality_error_sum[0].item() == 0.0
    assert command._episode_curriculum_quality_observation_count[1].item() == 1.0
    assert command._episode_curriculum_quality_error_sum[1].item() == pytest.approx(0.3**2)


def test_manual_reset_cache_is_not_consumed_when_reset_buf_is_not_set() -> None:
    command = _v17_command(lengths=(20,))
    command._has_sampled[0] = True
    command._episode_start_label_enabled[0] = True
    command._episode_curriculum_quality_valid[0] = True
    command._env.reset_terminated[0] = True
    command.last_global_body_pos_error_time_steps = command.time_steps.clone()
    command.last_global_body_pos_errors = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    command._accumulate_curriculum_quality_error()

    assert command._episode_curriculum_quality_observation_count[0].item() == 0.0
    assert command._episode_curriculum_quality_error_sum[0].item() == 0.0


def test_stale_termination_cache_with_a_different_time_step_is_not_consumed() -> None:
    command = _v17_command(lengths=(20,))
    command._has_sampled[0] = True
    command._episode_start_bins[0] = 0
    command._episode_start_label_enabled[0] = True
    command._episode_curriculum_quality_valid[0] = True
    command.time_steps[0] = 3
    command.last_global_body_pos_error_time_steps = torch.tensor([2, 0])
    command.last_global_body_pos_errors = torch.tensor([[0.3, 0.3], [0.0, 0.0]])

    command._accumulate_curriculum_quality_error()

    assert command._episode_curriculum_quality_observation_count[0].item() == 0.0
    assert command._episode_curriculum_quality_error_sum[0].item() == 0.0


@pytest.mark.parametrize("episode_failed", [False, True])
def test_failed_or_censored_pre_horizon_start_never_records_quality(
    monkeypatch: pytest.MonkeyPatch,
    episode_failed: bool,
) -> None:
    command = _v17_command(lengths=(20,))
    command._has_sampled[0] = True
    command._episode_start_bins[0] = 1
    command._episode_start_frames[0] = 4
    command._episode_start_label_enabled[0] = True
    command._episode_start_outcome_recorded[0] = False
    command._episode_curriculum_steps[0] = 2
    command._episode_curriculum_quality_valid[0] = True
    command._episode_curriculum_quality_error_sum[0] = 1.5
    command.time_steps[0] = 6
    command.motion_lengths[0] = 20
    command._env.termination_manager.terminated[0] = episode_failed
    command._env.termination_manager.body_pos[0] = episode_failed
    command.adp_sampling_active_prob.zero_()
    command.adp_sampling_active_prob[0] = 1.0
    command._curriculum_active_eligible_component_probabilities.zero_()
    command._curriculum_active_eligible_component_probabilities[0] = 1.0
    command._curriculum_active_terminal_component_probabilities.zero_()
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda _probabilities, num_samples, replacement: torch.zeros(num_samples, dtype=torch.long),
    )

    MotionCommand._adaptive_sampling(command, torch.tensor([0]))

    assert command._current_curriculum_quality_trials.count_nonzero().item() == 0
    assert command._current_curriculum_quality_error_sum.count_nonzero().item() == 0


def _make_quality_known(command, bin_ids: torch.Tensor, scores: torch.Tensor) -> None:
    command.curriculum_quality_trials[bin_ids] = float(command.cfg.curriculum_quality_min_trials)
    command._curriculum_quality_observed_windows[bin_ids] = int(command.cfg.curriculum_quality_min_windows)
    command._curriculum_quality_error_ema[bin_ids] = scores


def test_learning_pool_keeps_all_frontier_and_uses_high_error_mastered_only_as_filler() -> None:
    # L=84, B=H=4: exactly 20 eligible bins and one tail-only bin.
    command = _v17_command(lengths=(84,))
    assert command._curriculum_eligible_bin_count == 20
    assert command._curriculum_learning_pool_target_count == 3
    command._curriculum_states[command._curriculum_eligible_bins] = command._CURRICULUM_MASTERED
    command._curriculum_states[:2] = command._CURRICULUM_FRONTIER
    candidate_ids = torch.tensor([2, 3, 4, 5])
    _make_quality_known(command, candidate_ids, torch.tensor([0.2, 0.9, 0.7, 0.4]))

    command._refresh_curriculum_quality_learning_pool()

    torch.testing.assert_close(torch.where(command._curriculum_quality_filler_mask)[0], torch.tensor([3]))
    assert command._curriculum_quality_filler_count == 1
    learning_pool = command._curriculum_eligible_bins & (
        (command._curriculum_states == command._CURRICULUM_FRONTIER) | command._curriculum_quality_filler_mask
    )
    assert learning_pool.sum().item() == 3
    assert command._curriculum_quality_error_cutoff == pytest.approx(0.9)

    # Once the native survival frontier itself exceeds the floor, quality does
    # not displace it or enlarge the pool further.
    command._curriculum_states[:4] = command._CURRICULUM_FRONTIER
    command._refresh_curriculum_quality_learning_pool()
    assert command._curriculum_quality_filler_count == 0
    assert command._curriculum_quality_filler_mask.count_nonzero().item() == 0


def test_learning_pool_never_fakes_the_floor_when_quality_evidence_is_insufficient() -> None:
    command = _v17_command(lengths=(84,))
    command._curriculum_states[command._curriculum_eligible_bins] = command._CURRICULUM_MASTERED
    # Only one of the 20 eligible MASTERED bins has enough successful windows.
    _make_quality_known(command, torch.tensor([7]), torch.tensor([0.8]))
    # Tail-only, UNKNOWN, STALLED, and QUARANTINE bins must not be promoted by
    # their score, even if they otherwise appear to have sufficient evidence.
    disallowed = torch.tensor([1, 2, command.bin_count - 1])
    command._curriculum_states[1] = command._CURRICULUM_STALLED
    command._curriculum_states[2] = command._CURRICULUM_QUARANTINE
    _make_quality_known(command, disallowed, torch.tensor([1.0, 1.0, 1.0]))

    command._refresh_curriculum_quality_learning_pool()

    torch.testing.assert_close(torch.where(command._curriculum_quality_filler_mask)[0], torch.tensor([7]))
    assert command._curriculum_quality_filler_count == 1
    assert command._curriculum_quality_filler_count < command._curriculum_learning_pool_target_count


def test_quality_windows_update_ema_only_when_complete_and_gate_known_status() -> None:
    command = _v17_command(lengths=(20,))
    bin_id = 1
    command._curriculum_states[bin_id] = command._CURRICULUM_MASTERED
    command.curriculum_quality_trials[bin_id] = 4.0
    command._curriculum_quality_window_trials[bin_id] = 1.0
    command._curriculum_quality_window_error_sum[bin_id] = 0.9

    command._update_curriculum_quality_statistics()

    assert torch.isnan(command._curriculum_quality_error_ema[bin_id])
    assert command._curriculum_quality_window_trials[bin_id].item() == 1.0
    assert not command._curriculum_quality_known_mask()[bin_id]

    command._curriculum_quality_window_trials[bin_id] += 1.0
    command._curriculum_quality_window_error_sum[bin_id] += 0.1
    command._update_curriculum_quality_statistics()
    assert command._curriculum_quality_error_ema[bin_id].item() == pytest.approx(0.5)
    assert command._curriculum_quality_observed_windows[bin_id].item() == 1
    assert not command._curriculum_quality_known_mask()[bin_id]

    command._curriculum_quality_window_trials[bin_id] = 2.0
    command._curriculum_quality_window_error_sum[bin_id] = 0.4
    command._update_curriculum_quality_statistics()
    # alpha=.5: lerp(.5, .2, .5) = .35
    assert command._curriculum_quality_error_ema[bin_id].item() == pytest.approx(0.35)
    assert command._curriculum_quality_observed_windows[bin_id].item() == 2
    assert command._curriculum_quality_known_mask()[bin_id]


def test_distributed_quality_statistics_are_summed_as_raw_sufficient_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _v17_command(lengths=(20,))
    command._current_curriculum_quality_trials[1] = 2.0
    command._current_curriculum_quality_error_sum[1] = 0.6
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def _all_reduce(values: torch.Tensor, **_kwargs) -> None:
        values.mul_(2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)

    command._flush_curriculum_statistics()

    assert command.curriculum_quality_trials[1].item() == 4.0
    assert command._curriculum_quality_window_trials[1].item() == 4.0
    assert command._curriculum_quality_window_error_sum[1].item() == pytest.approx(1.2)
    assert command._current_curriculum_quality_trials.count_nonzero().item() == 0
    assert command._current_curriculum_quality_error_sum.count_nonzero().item() == 0


def test_quality_topk_is_deterministic_and_breaks_cutoff_ties_by_global_bin_id() -> None:
    candidate_ids = torch.tensor([9, 3, 7, 5, 11])
    candidate_scores = torch.tensor([0.8, 0.9, 0.9, 0.9, 0.1])

    selected_a, cutoff_a = MotionCommand._select_deterministic_topk_ids(candidate_ids, candidate_scores, 2)
    selected_b, cutoff_b = MotionCommand._select_deterministic_topk_ids(candidate_ids, candidate_scores, 2)

    torch.testing.assert_close(selected_a.sort().values, torch.tensor([3, 5]))
    torch.testing.assert_close(selected_b, selected_a)
    torch.testing.assert_close(cutoff_b, cutoff_a)
    assert cutoff_a.item() == pytest.approx(0.9)


def test_frontier_to_quality_filler_weight_is_two_to_one_without_losing_frame_weighting() -> None:
    # L=83 leaves the last eligible bin with 3 eligible frames. Compare a full
    # four-frame frontier bin to that partial quality filler: 4 / (3 * .5).
    command = _v17_command(lengths=(83,))
    command.cfg.curriculum_preserve_absent_state_budgets = False
    command.cfg.curriculum_unconfirmed_min_coverage_ratio = 0.0
    command._curriculum_known_fraction = 1.0
    command._curriculum_states[command._curriculum_eligible_bins] = command._CURRICULUM_MASTERED
    frontier_bin = 0
    filler_bin = 19
    command._curriculum_states[frontier_bin] = command._CURRICULUM_FRONTIER
    command._curriculum_quality_filler_mask[filler_bin] = True
    command._curriculum_quality_filler_count = 1

    total, eligible, terminal = command._curriculum_target_distribution_components(
        command._active_bin_ids, dtype=torch.float64
    )

    assert command._curriculum_eligible_start_counts[frontier_bin].item() == 4
    assert command._curriculum_eligible_start_counts[filler_bin].item() == 3
    assert eligible[frontier_bin] / eligible[filler_bin] == pytest.approx(4.0 / (3.0 * 0.5))
    torch.testing.assert_close(total.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.all(total >= 0.0)
    assert torch.all(torch.isfinite(total))
    assert torch.all(eligible + terminal <= total + 1.0e-7)
    assert torch.all(eligible[~command._curriculum_eligible_bins] == 0.0)
    assert torch.all(terminal[command._curriculum_terminal_start_counts == 0] == 0.0)


def test_schema6_round_trip_restores_quality_statistics_and_rebuilds_the_filler_mask() -> None:
    source = _v17_command(lengths=(20,))
    source.curriculum_start_trials[:4] = 32.0
    source._curriculum_states[:4] = torch.tensor(
        [
            source._CURRICULUM_FRONTIER,
            source._CURRICULUM_MASTERED,
            source._CURRICULUM_MASTERED,
            source._CURRICULUM_MASTERED,
        ],
        dtype=torch.uint8,
    )
    source.curriculum_quality_trials[:4] = torch.tensor([0.0, 8.0, 7.0, 6.0])
    source._curriculum_quality_window_trials[:4] = torch.tensor([0.0, 1.0, 0.0, 2.0])
    source._curriculum_quality_window_error_sum[:4] = torch.tensor([0.0, 0.2, 0.0, 0.8])
    source._curriculum_quality_error_ema[:4] = torch.tensor([float("nan"), 0.2, 0.8, 0.5])
    source._curriculum_quality_observed_windows[:4] = torch.tensor([0, 2, 3, 2], dtype=torch.uint8)
    source._curriculum_iteration = 75
    source._curriculum_last_state_update_iteration = 50
    source._curriculum_blend_start_iteration = 0
    source._curriculum_known_fraction = 1.0
    source._refresh_curriculum_quality_learning_pool()
    source._rebuild_global_sampling_distribution()
    source._rebuild_sampling_distribution()
    state = source.get_adaptive_sampling_state()
    restored = _v17_command(lengths=(20,))

    assert state["curriculum_state_schema_version"].item() == 6
    assert restored.load_adaptive_sampling_state(state)

    restored_state = restored.get_adaptive_sampling_state()
    assert restored_state.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(restored_state[name], state[name], equal_nan=True)
    torch.testing.assert_close(restored.adp_sampling_prob, source.adp_sampling_prob)
    torch.testing.assert_close(
        restored._curriculum_quality_filler_mask,
        source._curriculum_quality_filler_mask,
    )


def test_schema5_to_schema6_preserves_the_exact_v16_sampler_and_cold_starts_quality() -> None:
    source = _v16_command(lengths=(20,))
    source.curriculum_start_trials[:4] = torch.tensor([32.0, 24.0, 16.0, 8.0])
    source.curriculum_start_failures[:4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    source._curriculum_states[:4] = torch.tensor(
        [
            source._CURRICULUM_MASTERED,
            source._CURRICULUM_FRONTIER,
            source._CURRICULUM_STALLED,
            source._CURRICULUM_QUARANTINE,
        ],
        dtype=torch.uint8,
    )
    source._curriculum_iteration = 100
    source._curriculum_last_state_update_iteration = 100
    source._curriculum_blend_start_iteration = 0
    source._curriculum_known_fraction = 1.0
    source._rebuild_global_sampling_distribution()
    source._rebuild_sampling_distribution()
    source_probabilities = source.adp_sampling_prob.clone()
    state = source.get_adaptive_sampling_state()
    restored = _v17_command(lengths=(20,))

    assert state["curriculum_state_schema_version"].item() == 5
    assert not any(name.startswith("curriculum_quality") for name in state)
    assert restored.load_adaptive_sampling_state(state)

    torch.testing.assert_close(restored.adp_sampling_prob, source_probabilities)
    torch.testing.assert_close(restored.curriculum_start_trials, source.curriculum_start_trials)
    torch.testing.assert_close(restored.curriculum_start_failures, source.curriculum_start_failures)
    torch.testing.assert_close(restored._curriculum_states, source._curriculum_states)
    assert restored._curriculum_iteration == source._curriculum_iteration
    assert restored.curriculum_quality_trials.count_nonzero().item() == 0
    assert restored._curriculum_quality_window_trials.count_nonzero().item() == 0
    assert restored._curriculum_quality_window_error_sum.count_nonzero().item() == 0
    assert torch.isnan(restored._curriculum_quality_error_ema).all()
    assert restored._curriculum_quality_observed_windows.count_nonzero().item() == 0
    assert restored._curriculum_quality_filler_mask.count_nonzero().item() == 0


def test_invalid_schema6_quality_state_cold_starts_quality_without_losing_the_base_sampler() -> None:
    source = _v17_command(lengths=(20,))
    source.curriculum_start_trials[:4] = 32.0
    source._curriculum_states[:4] = torch.tensor(
        [
            source._CURRICULUM_MASTERED,
            source._CURRICULUM_FRONTIER,
            source._CURRICULUM_STALLED,
            source._CURRICULUM_QUARANTINE,
        ],
        dtype=torch.uint8,
    )
    source._curriculum_iteration = 50
    source._curriculum_last_state_update_iteration = 50
    source._curriculum_blend_start_iteration = 0
    source._curriculum_known_fraction = 1.0
    source._rebuild_global_sampling_distribution()
    source._rebuild_sampling_distribution()
    state = source.get_adaptive_sampling_state()
    state["curriculum_quality_trials"][1] = 8.0
    state["curriculum_quality_observed_windows"][1] = 2
    state["curriculum_quality_error_ema"][1] = float("inf")
    checkpoint_probabilities = state["curriculum_smoothed_probabilities"].clone()
    restored = _v17_command(lengths=(20,))

    assert restored.load_adaptive_sampling_state(state)

    torch.testing.assert_close(restored.adp_sampling_prob, checkpoint_probabilities)
    torch.testing.assert_close(restored._curriculum_states, source._curriculum_states)
    assert restored.curriculum_quality_trials.count_nonzero().item() == 0
    assert torch.isnan(restored._curriculum_quality_error_ema).all()
    assert restored._curriculum_quality_observed_windows.count_nonzero().item() == 0
    assert restored._curriculum_quality_filler_mask.count_nonzero().item() == 0


def test_v16_disabled_path_keeps_schema5_and_has_no_quality_allocation_or_keys() -> None:
    command = _v16_command(lengths=(20,))
    state = command.get_adaptive_sampling_state()

    assert command._curriculum_checkpoint_schema_version() == 5
    assert not hasattr(command, "curriculum_quality_trials")
    assert not any(name.startswith("curriculum_quality") for name in state)


def test_quality_layer_persistent_storage_is_linear_and_bounded_per_bin() -> None:
    command = _v17_command(lengths=(84,))
    per_bin_tensors = (
        command.curriculum_quality_trials,
        command._current_curriculum_quality_trials,
        command._current_curriculum_quality_error_sum,
        command._curriculum_quality_window_trials,
        command._curriculum_quality_window_error_sum,
        command._curriculum_quality_error_ema,
        command._curriculum_quality_observed_windows,
        command._curriculum_quality_filler_mask,
    )

    assert all(tensor.shape == (command.bin_count,) for tensor in per_bin_tensors)
    allocated_bytes = sum(tensor.numel() * tensor.element_size() for tensor in per_bin_tensors)
    # Six float32 arrays plus uint8/bool masks: 26 bytes per bin. This guards
    # against accidentally adding an H x bin_count history tensor later.
    assert allocated_bytes <= 26 * command.bin_count
