from __future__ import annotations

import pytest
import torch
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand
from test_v14_curriculum_sampling import _curriculum_command, _record_window, _sample_first_bin


def _v15_command():
    command = _curriculum_command()
    command.cfg.curriculum_start_horizon_frames = 4
    command.cfg.curriculum_exclude_invalid_failures = True
    command.cfg.curriculum_motion_length_exponent = 1.0
    command.cfg.curriculum_min_terminal_visits = 8
    command.cfg.curriculum_quarantine_terminal_hazard_threshold = 0.10
    command.cfg.curriculum_terminal_hazard_window_bins = 2
    command.cfg.curriculum_preserve_absent_state_budgets = True
    command.cfg.curriculum_probability_smoothing_alpha = 0.2
    command.cfg.curriculum_detailed_metrics = False
    command._reset_curriculum_sampling_state()
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()
    return command


def _set_active_episode(command, *, steps: int) -> None:
    command._has_sampled[0] = True
    command._episode_start_bins[0] = 0
    command._episode_start_frames[0] = 0
    command._episode_last_visited_bins[0] = 0
    command._episode_curriculum_steps[0] = steps
    command.motion_lengths[0] = 20
    command.time_steps[0] = min(steps, 19)


def test_failure_before_horizon_is_a_single_failed_start_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _v15_command()
    _sample_first_bin(monkeypatch)
    _set_active_episode(command, steps=3)
    command._env.termination_manager.terminated[0] = True
    command._env.termination_manager.body_pos[0] = True

    MotionCommand._adaptive_sampling(command, torch.tensor([0]))

    assert command._current_curriculum_start_trials[0].item() == 1.0
    assert command._current_curriculum_start_failures[0].item() == 1.0
    assert command._current_curriculum_start_censored.sum().item() == 0.0


def test_reaching_horizon_records_success_without_ending_episode_and_later_failure_does_not_backpropagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _v15_command()
    _sample_first_bin(monkeypatch)
    _set_active_episode(command, steps=3)

    command._update_adaptive_exposure()

    assert command._current_curriculum_start_trials[0].item() == 1.0
    assert command._current_curriculum_start_failures.sum().item() == 0.0
    assert command._episode_start_outcome_recorded[0].item()
    assert command._has_sampled[0].item()
    assert command._episode_start_bins[0].item() == 0
    assert command.time_steps[0].item() == 3

    # The same episode continues past the horizon. A later exposure must not
    # create another start outcome or resample the motion.
    command.time_steps[0] = 4
    command._update_adaptive_exposure()
    assert command._current_curriculum_start_trials[0].item() == 1.0
    assert command._has_sampled[0].item()
    assert command._episode_start_bins[0].item() == 0

    command._env.termination_manager.terminated[0] = True
    command._env.termination_manager.body_pos[0] = True
    MotionCommand._adaptive_sampling(command, torch.tensor([0]))

    assert command._current_curriculum_start_trials[0].item() == 1.0
    assert command._current_curriculum_start_failures.sum().item() == 0.0
    assert command._current_curriculum_start_censored.sum().item() == 0.0


def test_motion_completion_before_horizon_is_censored_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _v15_command()
    _sample_first_bin(monkeypatch)
    _set_active_episode(command, steps=3)
    command._env.termination_manager.terminated[0] = False

    MotionCommand._adaptive_sampling(command, torch.tensor([0]))

    assert command._current_curriculum_start_trials.sum().item() == 0.0
    assert command._current_curriculum_start_failures.sum().item() == 0.0
    assert command._current_curriculum_start_censored[0].item() == 1.0
    assert command._current_curriculum_start_survival_steps[0].item() == 3.0
    assert command._current_curriculum_start_completion_fraction[0].item() == pytest.approx(3.0 / 20.0)


def test_invalid_failure_is_censored_and_excluded_from_local_failure_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _v15_command()
    _sample_first_bin(monkeypatch)
    _set_active_episode(command, steps=3)
    command._env.termination_manager.terminated[0] = True
    invalid = torch.tensor([True, False])

    def get_term(name: str) -> torch.Tensor:
        if name == "invalid_robot_state":
            return invalid
        if name == "body_pos":
            return command._env.termination_manager.body_pos
        raise KeyError(name)

    command._env.termination_manager.get_term = get_term

    MotionCommand._adaptive_sampling(command, torch.tensor([0]))

    assert command._current_curriculum_start_trials.sum().item() == 0.0
    assert command._current_curriculum_start_failures.sum().item() == 0.0
    assert command._current_curriculum_start_censored[0].item() == 1.0
    assert command._current_curriculum_terminal_failures.sum().item() == 0.0
    assert command._current_curriculum_terminal_body_failures.sum().item() == 0.0


def test_terminal_only_evidence_keeps_fixed_horizon_start_unknown() -> None:
    command = _v15_command()
    bin_id = 4
    command.curriculum_terminal_visits[bin_id] = 128.0
    command.curriculum_terminal_failures[bin_id] = 0.0

    for _ in range(4):
        command._update_curriculum_states()

    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_UNKNOWN
    assert not command._curriculum_known_mask()[bin_id].item()
    assert torch.isnan(command._curriculum_failure_rate_history[:, bin_id]).all()


def test_terminal_only_evidence_does_not_start_the_fixed_horizon_blend() -> None:
    command = _v15_command()
    command.cfg.curriculum_shadow_iterations = 0
    command.cfg.curriculum_min_known_fraction = 0.01
    command.curriculum_terminal_visits.fill_(128.0)

    command._advance_curriculum_sampling(sync_across_ranks=False)

    assert command._curriculum_known_fraction == 0.0
    assert command._curriculum_blend_start_iteration == -1


def test_terminal_evidence_does_not_override_subthreshold_fixed_horizon_evidence() -> None:
    command = _v15_command()
    command.cfg.curriculum_min_known_trials = 32
    bin_id = 2
    command.curriculum_terminal_visits[bin_id] = 32.0

    _record_window(command, bin_id, trials=8, failures=8)

    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_UNKNOWN
    assert command._curriculum_window_start_trials[bin_id].item() == 8.0

    command._update_curriculum_states()

    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_UNKNOWN
    assert command._curriculum_window_start_trials[bin_id].item() == 8.0

    _record_window(command, bin_id, trials=24, failures=24)

    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_FRONTIER
    assert command._curriculum_window_start_trials[bin_id].item() == 0.0


def test_missing_unknown_budget_becomes_uniform_replay_instead_of_frontier_mass() -> None:
    command = _v15_command()
    command._curriculum_states[:] = torch.tensor(
        [
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_FRONTIER,
            command._CURRICULUM_STALLED,
            command._CURRICULUM_QUARANTINE,
            command._CURRICULUM_QUARANTINE,
        ],
        dtype=torch.uint8,
    )

    probabilities = command._curriculum_state_budget_probabilities(command._active_bin_ids, dtype=torch.float64)

    # The absent unknown state's 20% budget is baseline replay over every bin.
    # It must not be renormalized into the present states (which would give the
    # single frontier bin 55 / 80 = 68.75% mass).
    expected = torch.tensor([0.14, 0.59, 0.14, 0.065, 0.065], dtype=torch.float64)
    torch.testing.assert_close(probabilities, expected)
    assert probabilities[1].item() < 0.60


def test_state_refresh_probability_change_is_smoothed_once_per_iteration() -> None:
    command = _v15_command()
    command.cfg.adp_samp_failure_rate_max_over_mean = None
    command.cfg.curriculum_blend_iterations = 0
    command._curriculum_blend_start_iteration = 0
    command._curriculum_iteration = 1
    before = command.adp_sampling_prob.clone()
    command._curriculum_states[:] = torch.tensor(
        [
            command._CURRICULUM_UNKNOWN,
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_FRONTIER,
            command._CURRICULUM_STALLED,
            command._CURRICULUM_QUARANTINE,
        ],
        dtype=torch.uint8,
    )
    raw_target = command._curriculum_state_budget_probabilities(command._active_bin_ids, dtype=torch.float64).float()

    command._rebuild_global_sampling_distribution()

    expected = before.lerp(raw_target, command.cfg.curriculum_probability_smoothing_alpha)
    torch.testing.assert_close(command.adp_sampling_prob, expected)
    assert not torch.allclose(command.adp_sampling_prob, raw_target)

    # Rebuilding a working set in the same PPO iteration must not apply the EMA
    # repeatedly and silently accelerate the transition.
    after_first_rebuild = command.adp_sampling_prob.clone()
    command._rebuild_global_sampling_distribution()
    torch.testing.assert_close(command.adp_sampling_prob, after_first_rebuild)

    command._curriculum_iteration += 1
    command._rebuild_global_sampling_distribution()
    torch.testing.assert_close(
        command.adp_sampling_prob,
        after_first_rebuild.lerp(raw_target, command.cfg.curriculum_probability_smoothing_alpha),
    )


def test_quarantine_requires_forward_local_terminal_hazard() -> None:
    command = _v15_command()
    bin_id = 1

    for _ in range(3):
        _record_window(command, bin_id, trials=50, failures=50)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_STALLED

    _record_window(command, bin_id, trials=50, failures=50)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_STALLED

    command.curriculum_terminal_visits[bin_id + 1] = 32.0
    command.curriculum_terminal_failures[bin_id + 1] = 8.0
    _record_window(command, bin_id, trials=50, failures=50)
    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_QUARANTINE


def test_state_motion_bin_sampling_preserves_linear_bin_count_exposure() -> None:
    command = _v15_command()
    command.global_num_motions = 2
    command.bin_motion_ids[:] = torch.tensor([0, 0, 0, 0, 1])
    command._curriculum_states.fill_(command._CURRICULUM_FRONTIER)

    probabilities = command._curriculum_state_budget_probabilities(command._active_bin_ids, dtype=torch.float64)

    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    # The first motion has four times as many bins, so it receives four times
    # the total mass while every bin keeps the same base exposure. The
    # absent-state replay is also uniform per bin and preserves the ratio.
    expected_first_motion_mass = 4.0 / 5.0
    expected_second_motion_mass = 1.0 / 5.0
    torch.testing.assert_close(probabilities[:4].sum(), torch.tensor(expected_first_motion_mass, dtype=torch.float64))
    torch.testing.assert_close(probabilities[4], torch.tensor(expected_second_motion_mass, dtype=torch.float64))
    torch.testing.assert_close(
        probabilities[:4], torch.full((4,), expected_first_motion_mass / 4.0, dtype=torch.float64)
    )


def test_v14_schema_migration_preserves_the_actual_v14_shadow_distribution() -> None:
    v14 = _curriculum_command()
    v14.cfg.adp_samp_failure_rate_max_over_mean = None
    v14.curriculum_start_trials.fill_(64.0)
    v14.curriculum_start_failures.fill_(60.0)
    v14.curriculum_terminal_visits[:] = torch.arange(5, dtype=torch.float32) + 30.0
    v14.curriculum_terminal_failures[:] = torch.arange(5, dtype=torch.float32)
    v14.curriculum_terminal_body_failures[:] = torch.tensor([3.0, 7.0])
    v14._curriculum_states[:] = torch.tensor(
        [
            v14._CURRICULUM_MASTERED,
            v14._CURRICULUM_FRONTIER,
            v14._CURRICULUM_FRONTIER,
            v14._CURRICULUM_STALLED,
            v14._CURRICULUM_QUARANTINE,
        ],
        dtype=torch.uint8,
    )
    v14._curriculum_iteration = 200
    v14._curriculum_blend_start_iteration = 100
    v14._rebuild_global_sampling_distribution()
    v14_distribution = v14.adp_sampling_prob.clone()
    assert not torch.allclose(v14_distribution, torch.full_like(v14_distribution, 0.2))
    state = v14.get_adaptive_sampling_state()
    v15 = _v15_command()

    assert v15.load_adaptive_sampling_state(state)

    # The fixed-horizon classifier starts clean, but its shadow sampler must be
    # the distribution that was actually active in V14, not a reconstructed
    # V13/Marmot distribution from the legacy counters.
    torch.testing.assert_close(v15.adp_sampling_prob, v14_distribution)
    assert v15.curriculum_start_trials.count_nonzero().item() == 0
    assert v15.curriculum_start_failures.count_nonzero().item() == 0
    assert torch.all(v15._curriculum_states == v15._CURRICULUM_UNKNOWN)
    assert v15._curriculum_iteration == 0
    assert v15._curriculum_blend_start_iteration == -1
    torch.testing.assert_close(v15.curriculum_terminal_visits, v14.curriculum_terminal_visits)
    torch.testing.assert_close(v15.curriculum_terminal_failures, v14.curriculum_terminal_failures)
    torch.testing.assert_close(v15.curriculum_terminal_body_failures, v14.curriculum_terminal_body_failures)

    # A working-set distribution must be recomputed from the frozen V14
    # state-budget recipe.  Slicing the global distribution changes the state
    # budgets when the subset contains only part of a state's bins.
    active_bin_ids = torch.tensor([0, 1], dtype=torch.long)
    active_legacy = torch.tensor([0.8, 0.2], dtype=torch.float64)
    expected_active = v14._blend_curriculum_distribution(active_legacy, active_bin_ids)
    sliced_global = v14_distribution[active_bin_ids].double()
    sliced_global /= sliced_global.sum()
    assert not torch.allclose(expected_active, sliced_global)
    actual_active = MotionCommand._curriculum_shadow_distribution(v15, active_legacy, active_bin_ids)
    torch.testing.assert_close(actual_active, expected_active)

    # A checkpoint taken partway through migration shadow must restore the
    # frozen endpoint and schedule for both global and working-set rebuilds.
    v15._curriculum_iteration = 50
    v15._rebuild_global_sampling_distribution()
    torch.testing.assert_close(v15.adp_sampling_prob, v14_distribution)
    mid_shadow_state = v15.get_adaptive_sampling_state()
    resumed = _v15_command()
    assert resumed.load_adaptive_sampling_state(mid_shadow_state)
    assert resumed._curriculum_iteration == 50
    assert resumed._curriculum_blend_start_iteration == -1
    torch.testing.assert_close(resumed.adp_sampling_prob, v15.adp_sampling_prob)
    resumed_active = MotionCommand._curriculum_shadow_distribution(resumed, active_legacy, active_bin_ids)
    torch.testing.assert_close(resumed_active, expected_active)


def test_v15_fixed_horizon_state_round_trip() -> None:
    source = _v15_command()
    source.curriculum_start_trials[:] = torch.arange(5, dtype=torch.float32) + 10.0
    source.curriculum_start_failures[:] = torch.arange(5, dtype=torch.float32)
    source.curriculum_start_censored[:] = torch.arange(5, dtype=torch.float32) + 2.0
    source._curriculum_states[:] = torch.arange(5, dtype=torch.uint8)
    source._curriculum_iteration = 75
    source._curriculum_last_state_update_iteration = 50
    source._curriculum_blend_start_iteration = 50
    source._curriculum_shadow_probabilities = torch.tensor([0.35, 0.25, 0.20, 0.15, 0.05])
    source._curriculum_smoothed_probabilities = torch.tensor([0.10, 0.15, 0.20, 0.25, 0.30])
    source._curriculum_last_probability_smoothing_iteration = 75
    state = source.get_adaptive_sampling_state()
    restored = _v15_command()

    assert state["curriculum_state_schema_version"].item() == 4
    assert {
        "curriculum_shadow_probabilities",
        "curriculum_smoothed_probabilities",
        "curriculum_last_probability_smoothing_iteration",
    } <= state.keys()
    assert restored.load_adaptive_sampling_state(state)

    restored_state = restored.get_adaptive_sampling_state()
    assert restored_state.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(restored_state[name], state[name], equal_nan=True)


def test_schema3_upgrade_preserves_sufficient_statistics_but_resets_the_classifier() -> None:
    source = _v15_command()
    # Schema 3 was produced by the original V15 configuration, which used
    # square-root motion weighting. Migration must reconstruct that historical
    # sampler even though the corrected V15 target is now linear in length.
    source.cfg.curriculum_motion_length_exponent = 0.5
    source.cfg.curriculum_preserve_absent_state_budgets = False
    source.cfg.adp_samp_failure_rate_max_over_mean = None
    source.cfg.curriculum_probability_smoothing_alpha = 1.0
    source.curriculum_start_trials[:] = torch.arange(5, dtype=torch.float32) + 32.0
    source.curriculum_start_failures[:] = torch.arange(5, dtype=torch.float32) + 4.0
    source.curriculum_start_censored[:] = torch.arange(5, dtype=torch.float32) + 2.0
    source.curriculum_start_survival_steps[:] = torch.arange(5, dtype=torch.float32) + 100.0
    source.curriculum_start_completion_fraction[:] = torch.arange(5, dtype=torch.float32) / 10.0
    source.curriculum_terminal_visits[:] = torch.arange(5, dtype=torch.float32) + 64.0
    source.curriculum_terminal_failures[:] = torch.arange(5, dtype=torch.float32)
    source.curriculum_terminal_body_failures[:] = torch.tensor([3.0, 7.0])
    source.global_num_motions = 2
    source.bin_motion_ids[:] = torch.tensor([0, 0, 0, 0, 1])
    source._curriculum_states.fill_(source._CURRICULUM_FRONTIER)
    source._curriculum_failure_rate_history.fill_(0.9)
    source._curriculum_iteration = 200
    source._curriculum_blend_start_iteration = 100
    source._rebuild_global_sampling_distribution()
    source_distribution = source.adp_sampling_prob.clone()
    state = source.get_adaptive_sampling_state()
    schema3_state = {
        name: value
        for name, value in state.items()
        if not name.startswith("curriculum_shadow_")
        and name
        not in {
            "curriculum_smoothed_probabilities",
            "curriculum_last_probability_smoothing_iteration",
        }
    }
    schema3_state["curriculum_state_schema_version"] = torch.tensor(3)
    restored = _v15_command()
    restored.global_num_motions = 2
    restored.bin_motion_ids[:] = source.bin_motion_ids

    assert restored.load_adaptive_sampling_state(schema3_state)

    # Fixed-H sufficient statistics are valid and expensive to recollect, but
    # schema3 states/history were contaminated by terminal-only fallback.
    for name in (
        "curriculum_start_trials",
        "curriculum_start_failures",
        "curriculum_start_censored",
        "curriculum_start_survival_steps",
        "curriculum_start_completion_fraction",
        "curriculum_terminal_visits",
        "curriculum_terminal_failures",
        "curriculum_terminal_body_failures",
    ):
        torch.testing.assert_close(getattr(restored, name), getattr(source, name))
    assert torch.all(restored._curriculum_states == restored._CURRICULUM_UNKNOWN)
    assert torch.isnan(restored._curriculum_failure_rate_history).all()
    assert restored._curriculum_window_start_trials.count_nonzero().item() == 0
    assert restored._curriculum_window_start_failures.count_nonzero().item() == 0
    assert restored._curriculum_iteration == 0
    assert restored._curriculum_blend_start_iteration == -1
    # Despite resetting the classifier, migration shadow reconstructs the
    # sampler that was actually active at the schema3 checkpoint.
    torch.testing.assert_close(restored.adp_sampling_prob, source_distribution)


def test_v15_horizon_change_preserves_only_terminal_evidence() -> None:
    source = _v15_command()
    source.curriculum_start_trials.fill_(40.0)
    source.curriculum_start_failures.fill_(20.0)
    source.curriculum_terminal_visits.fill_(64.0)
    source.curriculum_terminal_failures.fill_(4.0)
    source._curriculum_states.fill_(source._CURRICULUM_FRONTIER)
    source._curriculum_iteration = 75
    state = source.get_adaptive_sampling_state()
    restored = _v15_command()
    restored.cfg.curriculum_start_horizon_frames = 5

    assert restored.load_adaptive_sampling_state(state)

    assert restored.curriculum_start_trials.count_nonzero().item() == 0
    assert restored.curriculum_start_failures.count_nonzero().item() == 0
    assert torch.all(restored._curriculum_states == restored._CURRICULUM_UNKNOWN)
    assert restored._curriculum_iteration == 0
    torch.testing.assert_close(restored.curriculum_terminal_visits, source.curriculum_terminal_visits)
    torch.testing.assert_close(restored.curriculum_terminal_failures, source.curriculum_terminal_failures)
