from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand
from test_v14_curriculum_sampling import _curriculum_command


def _v16_command(*, lengths: tuple[int, ...] = (11,), terminal_replay_fraction: float = 0.20):
    """Build a small CPU-only command with mixed eligible/tail bins."""

    command = _curriculum_command()
    command.cfg.curriculum_start_horizon_frames = 4
    command.cfg.curriculum_exclude_invalid_failures = True
    command.cfg.curriculum_start_eligibility_enabled = True
    command.cfg.curriculum_terminal_replay_fraction = terminal_replay_fraction
    command.cfg.curriculum_coverage_aware_enabled = True
    command.cfg.curriculum_provisional_known_trials = 8
    command.cfg.curriculum_min_provisional_known_fraction = 0.25
    command.cfg.curriculum_min_provisional_motion_fraction = 0.50
    command.cfg.curriculum_provisional_motion_bin_fraction = 0.10
    command.cfg.curriculum_state_focus_start_fraction = 0.10
    command.cfg.curriculum_state_focus_end_fraction = 0.40
    command.cfg.curriculum_unconfirmed_min_coverage_ratio = 0.50
    command.cfg.curriculum_motion_length_exponent = 1.0
    command.cfg.curriculum_preserve_absent_state_budgets = True
    command.cfg.curriculum_probability_smoothing_alpha = 1.0
    command.cfg.adp_samp_failure_rate_max_over_mean = None
    command.cfg.resample_at_motion_end = False

    motion_lengths = torch.tensor(lengths, dtype=torch.long)
    command._global_time_totals = motion_lengths
    command.global_num_motions = len(lengths)
    command.motion_bin_counts = torch.div(
        motion_lengths + command.cfg.bin_size - 1,
        command.cfg.bin_size,
        rounding_mode="floor",
    )
    command.bin_count = int(command.motion_bin_counts.sum().item())
    command.motion_bin_offsets = torch.zeros(len(lengths), dtype=torch.long)
    if len(lengths) > 1:
        command.motion_bin_offsets[1:] = torch.cumsum(command.motion_bin_counts[:-1], dim=0)
    command.bin_motion_ids = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.long), command.motion_bin_counts
    )
    bin_ids = torch.arange(command.bin_count, dtype=torch.long)
    local_bin_ids = bin_ids - command.motion_bin_offsets[command.bin_motion_ids]
    command.bin_starts = local_bin_ids * command.cfg.bin_size
    command.bin_ends = torch.minimum(
        command.bin_starts + command.cfg.bin_size,
        motion_lengths[command.bin_motion_ids],
    )
    command.bin_weights = (command.bin_ends - command.bin_starts).float()
    command.bin_weights /= command.bin_weights.mean()
    command.adp_samp_num_episodes = torch.ones(command.bin_count)
    command.adp_samp_num_failures = torch.ones(command.bin_count)
    command._current_adp_samp_num_episodes = torch.zeros(command.bin_count)
    command._current_adp_samp_num_failures = torch.zeros(command.bin_count)

    command.motion = SimpleNamespace(
        global_num_motions=len(lengths),
        world_size=1,
        manifest_fingerprint_words=(11, 22, 33, 44),
        global_ids=torch.arange(len(lengths), dtype=torch.long),
        num_motions=len(lengths),
        lengths=lambda motion_ids: motion_lengths[motion_ids],
    )
    command._active_bin_ids = torch.arange(command.bin_count, dtype=torch.long)
    command._active_local_motion_ids = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.long), command.motion_bin_counts
    )
    command.motion_ids.zero_()
    command.time_steps.zero_()
    command.motion_lengths.fill_(lengths[0])
    command._has_sampled.zero_()

    command._initialize_curriculum_start_geometry()
    command._initialize_curriculum_sampling()
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()
    return command


def _v15_command_for_migration(*, lengths: tuple[int, ...] = (11,)):
    command = _v16_command(lengths=lengths)
    command.cfg.curriculum_start_eligibility_enabled = False
    command.cfg.curriculum_terminal_replay_fraction = 0.0
    command.cfg.curriculum_coverage_aware_enabled = False
    command._initialize_curriculum_sampling()
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()
    return command


def _force_component(command, *, bin_id: int, component: str) -> None:
    command.adp_sampling_active_prob.zero_()
    command.adp_sampling_active_prob[bin_id] = 1.0
    command._curriculum_active_eligible_component_probabilities.zero_()
    command._curriculum_active_terminal_component_probabilities.zero_()
    if component == "eligible":
        command._curriculum_active_eligible_component_probabilities[bin_id] = 1.0
    elif component == "terminal":
        command._curriculum_active_terminal_component_probabilities[bin_id] = 1.0
    else:
        raise ValueError(component)


def test_fixed_horizon_geometry_uses_the_exclusive_l_minus_h_boundary() -> None:
    command = _v16_command(lengths=(11,))

    # L=11, B=H=4 gives bins [0,4), [4,8), [8,11). The last
    # labelable start is L-H-1=6; frame 7 already belongs to tail replay.
    torch.testing.assert_close(command.bin_starts, torch.tensor([0, 4, 8]))
    torch.testing.assert_close(command.bin_ends, torch.tensor([4, 8, 11]))
    torch.testing.assert_close(command._curriculum_eligible_start_ends, torch.tensor([4, 7, 8]))
    torch.testing.assert_close(command._curriculum_eligible_start_counts, torch.tensor([4, 3, 0]))
    torch.testing.assert_close(command._curriculum_terminal_start_counts, torch.tensor([0, 1, 3]))
    torch.testing.assert_close(command._curriculum_eligible_bins, torch.tensor([True, True, False]))

    mixed_bin = 1
    assert command._curriculum_eligible_start_ends[mixed_bin] > 6
    assert not (command._curriculum_eligible_start_ends[mixed_bin] > 7)
    torch.testing.assert_close(
        command._curriculum_eligible_start_counts + command._curriculum_terminal_start_counts,
        command.bin_ends - command.bin_starts,
    )


def test_target_components_preserve_q_r_mass_and_weight_partial_bins_by_frames() -> None:
    command = _v16_command(lengths=(11,), terminal_replay_fraction=0.20)
    command._curriculum_states.fill_(command._CURRICULUM_UNKNOWN)

    total, eligible, terminal = command._curriculum_target_distribution_components(
        command._active_bin_ids, dtype=torch.float64
    )

    expected_eligible = torch.tensor([0.8 * 4.0 / 7.0, 0.8 * 3.0 / 7.0, 0.0], dtype=torch.float64)
    expected_terminal = torch.tensor([0.0, 0.2 * 1.0 / 4.0, 0.2 * 3.0 / 4.0], dtype=torch.float64)
    torch.testing.assert_close(eligible, expected_eligible)
    torch.testing.assert_close(terminal, expected_terminal)
    torch.testing.assert_close(total, expected_eligible + expected_terminal)
    torch.testing.assert_close(eligible.sum(), torch.tensor(0.8, dtype=torch.float64))
    torch.testing.assert_close(terminal.sum(), torch.tensor(0.2, dtype=torch.float64))
    torch.testing.assert_close(total.sum(), torch.tensor(1.0, dtype=torch.float64))

    eligible_counts = command._curriculum_eligible_start_counts[:2].double()
    terminal_counts = command._curriculum_terminal_start_counts[1:].double()
    torch.testing.assert_close(eligible[:2] / eligible_counts, torch.full((2,), 0.8 / 7.0, dtype=torch.float64))
    torch.testing.assert_close(terminal[1:] / terminal_counts, torch.full((2,), 0.2 / 4.0, dtype=torch.float64))


def test_mixed_bin_components_sample_disjoint_intervals_and_terminal_never_labels_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixed_bin = 1
    command = _v16_command(lengths=(11,))
    command.cfg.pre_failure_sample_window = 200
    _force_component(command, bin_id=mixed_bin, component="eligible")
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda _probabilities, num_samples, replacement: torch.full((num_samples,), mixed_bin, dtype=torch.long),
    )

    MotionCommand._adaptive_sampling(command, torch.tensor([0]))

    assert 4 <= command.time_steps[0].item() < 7
    assert command._episode_start_bins[0].item() == mixed_bin
    assert command._episode_start_label_enabled[0].item()

    terminal = _v16_command(lengths=(11,))
    terminal.cfg.pre_failure_sample_window = 200
    _force_component(terminal, bin_id=mixed_bin, component="terminal")

    MotionCommand._adaptive_sampling(terminal, torch.tensor([0]))

    # The mixed bin has exactly one tail frame. It must not be shifted earlier
    # by pre_failure_sample_window and must never enter fixed-H start stats.
    assert terminal.time_steps[0].item() == 7
    assert terminal._episode_start_bins[0].item() == -1
    assert not terminal._episode_start_label_enabled[0].item()
    terminal._env.termination_manager.terminated[0] = True
    terminal._env.termination_manager.body_pos[0] = True

    MotionCommand._adaptive_sampling(terminal, torch.tensor([0]))

    assert terminal._current_curriculum_start_trials.count_nonzero().item() == 0
    assert terminal._current_curriculum_start_failures.count_nonzero().item() == 0
    assert terminal._current_curriculum_start_censored.count_nonzero().item() == 0
    assert terminal._current_curriculum_terminal_failures[mixed_bin].item() == 1.0


def test_coverage_focus_and_blend_gate_use_eligible_bins_and_provisional_motion_coverage() -> None:
    command = _v16_command(lengths=(11, 11))
    command.cfg.curriculum_shadow_iterations = 0
    command.cfg.curriculum_state_update_interval = 1000
    command.cfg.curriculum_min_known_fraction = 0.25
    command.cfg.curriculum_min_provisional_known_fraction = 0.25
    command.cfg.curriculum_min_provisional_motion_fraction = 1.0

    # Four of six bins are eligible. One confirmed bin meets the 25% bin
    # threshold, but only one of the two motions has provisional evidence.
    command.curriculum_start_trials[0] = float(command.cfg.curriculum_min_known_trials)
    command._advance_curriculum_sampling(sync_across_ranks=False)

    assert command._curriculum_known_fraction == pytest.approx(0.25)
    assert command._curriculum_provisional_known_fraction == pytest.approx(0.25)
    assert command._curriculum_provisional_motion_fraction == pytest.approx(0.50)
    assert command._curriculum_blend_start_iteration == -1

    # Provisional evidence in motion 1 opens the motion-coverage gate without
    # pretending that its bin is already confirmed known.
    second_motion_first_bin = int(command.motion_bin_offsets[1].item())
    command.curriculum_start_trials[second_motion_first_bin] = float(command.cfg.curriculum_provisional_known_trials)
    command._advance_curriculum_sampling(sync_across_ranks=False)

    assert command._curriculum_known_fraction == pytest.approx(0.25)
    assert command._curriculum_provisional_known_fraction == pytest.approx(0.50)
    assert command._curriculum_provisional_motion_fraction == pytest.approx(1.0)
    assert command._curriculum_blend_start_iteration == 2
    # With start=10%, end=40%, confirmed coverage=25% is the midpoint of
    # smoothstep and therefore exactly 50% state focus.
    assert command._curriculum_state_focus_factor() == pytest.approx(0.5)


def test_unconfirmed_bins_keep_half_of_their_frame_weighted_coverage_mass() -> None:
    command = _v16_command(lengths=(25,), terminal_replay_fraction=0.20)
    # Six bins are eligible. Make every state present so absent-state replay
    # cannot itself hide starvation of the lone unconfirmed quarantine bin.
    command._curriculum_states[:] = torch.tensor(
        [
            command._CURRICULUM_UNKNOWN,
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_FRONTIER,
            command._CURRICULUM_STALLED,
            command._CURRICULUM_QUARANTINE,
            command._CURRICULUM_FRONTIER,
            command._CURRICULUM_UNKNOWN,
        ],
        dtype=torch.uint8,
    )
    command.curriculum_start_trials[:6] = float(command.cfg.curriculum_min_known_trials)
    unconfirmed_bin = 4
    command.curriculum_start_trials[unconfirmed_bin] = 0.0
    command._curriculum_known_fraction = 5.0 / 6.0

    _, eligible, _ = command._curriculum_target_distribution_components(command._active_bin_ids, dtype=torch.float64)

    eligible_conditional = eligible / eligible.sum()
    coverage_mass = (
        command._curriculum_eligible_start_counts[unconfirmed_bin].double()
        / command._curriculum_eligible_start_counts.sum().double()
    )
    torch.testing.assert_close(
        eligible_conditional[unconfirmed_bin],
        coverage_mass * command.cfg.curriculum_unconfirmed_min_coverage_ratio,
    )
    assert command._curriculum_effective_state_focus < command._curriculum_state_focus_factor()


def test_schema5_round_trip_restores_component_attribution_exactly() -> None:
    source = _v16_command(lengths=(11,))
    source.curriculum_start_trials[:] = torch.tensor([16.0, 8.0, 0.0])
    source.curriculum_start_failures[:] = torch.tensor([1.0, 2.0, 0.0])
    source.curriculum_start_censored[:] = torch.tensor([0.0, 1.0, 0.0])
    source.curriculum_terminal_visits[:] = torch.tensor([5.0, 6.0, 7.0])
    source.curriculum_terminal_failures[:] = torch.tensor([0.0, 1.0, 2.0])
    source._curriculum_states[:] = torch.tensor(
        [source._CURRICULUM_MASTERED, source._CURRICULUM_FRONTIER, source._CURRICULUM_UNKNOWN],
        dtype=torch.uint8,
    )
    source._curriculum_iteration = 75
    source._curriculum_last_state_update_iteration = 50
    source._curriculum_blend_start_iteration = 50
    source._curriculum_shadow_probabilities[:] = torch.tensor([0.40, 0.35, 0.25])
    source._curriculum_smoothed_probabilities[:] = torch.tensor([0.20, 0.50, 0.30])
    source._curriculum_smoothed_eligible_probabilities[:] = torch.tensor([0.10, 0.30, 0.00])
    source._curriculum_smoothed_terminal_probabilities[:] = torch.tensor([0.00, 0.05, 0.20])
    source._curriculum_smoothed_probabilities_initialized = True
    source._curriculum_last_probability_smoothing_iteration = 75
    state = source.get_adaptive_sampling_state()
    restored = _v16_command(lengths=(11,))

    assert state["curriculum_state_schema_version"].item() == 5
    assert restored.load_adaptive_sampling_state(state)

    restored_state = restored.get_adaptive_sampling_state()
    assert restored_state.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(restored_state[name], state[name], equal_nan=True)
    torch.testing.assert_close(restored.adp_sampling_prob, source._curriculum_smoothed_probabilities)
    torch.testing.assert_close(
        restored._curriculum_smoothed_eligible_probabilities,
        source._curriculum_smoothed_eligible_probabilities,
    )
    torch.testing.assert_close(
        restored._curriculum_smoothed_terminal_probabilities,
        source._curriculum_smoothed_terminal_probabilities,
    )


def test_schema5_replay_fraction_change_uses_shadow_migration_instead_of_exact_restore() -> None:
    source = _v16_command(lengths=(11,), terminal_replay_fraction=0.20)
    source.curriculum_start_trials.fill_(32.0)
    source.curriculum_start_failures.fill_(8.0)
    source.curriculum_terminal_visits[:] = torch.tensor([10.0, 20.0, 30.0])
    source.curriculum_terminal_failures[:] = torch.tensor([1.0, 2.0, 3.0])
    source._curriculum_iteration = 100
    source._curriculum_blend_start_iteration = 0
    source._rebuild_global_sampling_distribution()
    source_distribution = source.adp_sampling_prob.clone()
    state = source.get_adaptive_sampling_state()
    restored = _v16_command(lengths=(11,), terminal_replay_fraction=0.10)

    assert restored.load_adaptive_sampling_state(state)

    torch.testing.assert_close(restored.adp_sampling_prob, source_distribution)
    torch.testing.assert_close(restored.curriculum_terminal_visits, source.curriculum_terminal_visits)
    torch.testing.assert_close(restored.curriculum_terminal_failures, source.curriculum_terminal_failures)
    assert restored.curriculum_start_trials.count_nonzero().item() == 0
    assert restored.curriculum_start_failures.count_nonzero().item() == 0
    assert restored._curriculum_smoothed_eligible_probabilities.count_nonzero().item() == 0
    assert restored._curriculum_smoothed_terminal_probabilities.count_nonzero().item() == 0
    assert restored._curriculum_has_shadow_distribution
    assert restored._curriculum_iteration == 0
    assert restored._curriculum_blend_start_iteration == -1


def test_schema5_invalid_component_support_uses_safe_shadow_migration() -> None:
    source = _v16_command(lengths=(11,))
    source.curriculum_start_trials[0] = 32.0
    source.curriculum_terminal_visits[:] = torch.tensor([1.0, 2.0, 3.0])
    state = source.get_adaptive_sampling_state()
    tail_only_bin = 2
    state["curriculum_smoothed_eligible_probabilities"][tail_only_bin] = 0.01
    restored = _v16_command(lengths=(11,))

    assert restored.load_adaptive_sampling_state(state)

    # The total checkpoint sampler is still finite and remains the frozen
    # source, but invalid component attribution must not exact-restore start
    # classifier state into V16.
    torch.testing.assert_close(restored.adp_sampling_prob, state["curriculum_smoothed_probabilities"])
    torch.testing.assert_close(restored.curriculum_terminal_visits, source.curriculum_terminal_visits)
    assert restored.curriculum_start_trials.count_nonzero().item() == 0
    assert restored._curriculum_smoothed_eligible_probabilities.count_nonzero().item() == 0
    assert restored._curriculum_smoothed_terminal_probabilities.count_nonzero().item() == 0
    assert restored._curriculum_has_shadow_distribution
    assert restored._curriculum_iteration == 0
    assert restored._curriculum_blend_start_iteration == -1


def test_schema4_to_schema5_keeps_sampler_and_only_fully_eligible_start_state() -> None:
    source = _v15_command_for_migration(lengths=(11,))
    source.cfg.adp_samp_failure_rate_max_over_mean = None
    source.curriculum_start_trials.fill_(64.0)
    source.curriculum_start_failures.fill_(32.0)
    source.curriculum_start_censored.fill_(4.0)
    source._curriculum_window_start_trials[:] = torch.tensor([3.0, 4.0, 5.0])
    source._curriculum_window_start_failures[:] = torch.tensor([1.0, 2.0, 3.0])
    source._curriculum_failure_rate_history[:] = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
    source._curriculum_state_entry_trials[:] = torch.tensor([8.0, 16.0, 24.0])
    source.curriculum_terminal_visits[:] = torch.tensor([30.0, 40.0, 50.0])
    source.curriculum_terminal_failures[:] = torch.tensor([1.0, 2.0, 3.0])
    source.curriculum_terminal_body_failures[:] = torch.tensor([3.0, 7.0])
    source._curriculum_states[:] = torch.tensor(
        [source._CURRICULUM_MASTERED, source._CURRICULUM_FRONTIER, source._CURRICULUM_QUARANTINE],
        dtype=torch.uint8,
    )
    source._curriculum_iteration = 200
    source._curriculum_blend_start_iteration = 100
    source._rebuild_global_sampling_distribution()
    source_distribution = source.adp_sampling_prob.clone()
    state = source.get_adaptive_sampling_state()
    restored = _v16_command(lengths=(11,))

    assert state["curriculum_state_schema_version"].item() == 4
    assert restored.load_adaptive_sampling_state(state)

    torch.testing.assert_close(restored.adp_sampling_prob, source_distribution)
    torch.testing.assert_close(restored.curriculum_terminal_visits, source.curriculum_terminal_visits)
    torch.testing.assert_close(restored.curriculum_terminal_failures, source.curriculum_terminal_failures)
    torch.testing.assert_close(
        restored.curriculum_terminal_body_failures,
        source.curriculum_terminal_body_failures,
    )
    # Bin 0 contains only eligible starts and is safe to reuse. Bin 1 mixes
    # eligible/tail starts and bin 2 is tail-only, so both restart cleanly.
    torch.testing.assert_close(restored.curriculum_start_trials, torch.tensor([64.0, 0.0, 0.0]))
    torch.testing.assert_close(restored.curriculum_start_failures, torch.tensor([32.0, 0.0, 0.0]))
    torch.testing.assert_close(restored.curriculum_start_censored, torch.tensor([4.0, 0.0, 0.0]))
    torch.testing.assert_close(
        restored._curriculum_states,
        torch.tensor(
            [restored._CURRICULUM_MASTERED, restored._CURRICULUM_UNKNOWN, restored._CURRICULUM_UNKNOWN],
            dtype=torch.uint8,
        ),
    )
    torch.testing.assert_close(restored._curriculum_window_start_trials, torch.tensor([3.0, 0.0, 0.0]))
    torch.testing.assert_close(restored._curriculum_window_start_failures, torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(restored._curriculum_failure_rate_history[:, 0], torch.tensor([0.1, 0.2, 0.3]))
    assert torch.isnan(restored._curriculum_failure_rate_history[:, 1:]).all()
    torch.testing.assert_close(restored._curriculum_state_entry_trials, torch.tensor([8.0, 0.0, 0.0]))
    assert restored._curriculum_smoothed_eligible_probabilities.count_nonzero().item() == 0
    assert restored._curriculum_smoothed_terminal_probabilities.count_nonzero().item() == 0
    assert restored._curriculum_has_shadow_distribution
    assert restored._curriculum_iteration == 0
    assert restored._curriculum_blend_start_iteration == -1


def test_smoothing_cap_and_active_conditioning_preserve_component_conservation() -> None:
    command = _v16_command(lengths=(11,), terminal_replay_fraction=0.20)
    command.cfg.curriculum_probability_smoothing_alpha = 0.20
    command.cfg.adp_samp_failure_rate_max_over_mean = 1.20
    command.cfg.curriculum_blend_iterations = 0
    command._curriculum_blend_start_iteration = 0
    command._curriculum_iteration = 1
    command._curriculum_known_fraction = 1.0
    command.curriculum_start_trials[:2] = float(command.cfg.curriculum_min_known_trials)
    command._curriculum_states[:] = torch.tensor(
        [command._CURRICULUM_FRONTIER, command._CURRICULUM_QUARANTINE, command._CURRICULUM_UNKNOWN],
        dtype=torch.uint8,
    )

    command._rebuild_global_sampling_distribution()

    total = command.adp_sampling_prob
    eligible = command._curriculum_smoothed_eligible_probabilities
    terminal = command._curriculum_smoothed_terminal_probabilities
    torch.testing.assert_close(total.sum(), torch.tensor(1.0))
    assert torch.all(eligible >= 0.0)
    assert torch.all(terminal >= 0.0)
    assert torch.all(eligible + terminal <= total + 1.0e-7)
    assert torch.all(eligible[~command._curriculum_eligible_bins] == 0.0)
    assert torch.all(terminal[command._curriculum_terminal_start_counts == 0] == 0.0)
    assert total.max().item() <= 1.20 / command.bin_count + 1.0e-7

    # Condition onto a mixed+tail-only working set. The second cap and its
    # renormalization must scale every component with the same per-bin factor.
    command._active_bin_ids = torch.tensor([1, 2], dtype=torch.long)
    command._active_local_motion_ids = torch.zeros(2, dtype=torch.long)
    command._rebuild_sampling_distribution()

    active_total = command.adp_sampling_active_prob
    active_eligible = command._curriculum_active_eligible_component_probabilities
    active_terminal = command._curriculum_active_terminal_component_probabilities
    torch.testing.assert_close(active_total.sum(), torch.tensor(1.0))
    assert torch.all(active_eligible >= 0.0)
    assert torch.all(active_terminal >= 0.0)
    assert torch.all(active_eligible + active_terminal <= active_total + 1.0e-7)
    assert active_eligible[1].item() == 0.0
    assert active_total.max().item() <= 1.20 / len(active_total) + 1.0e-7
