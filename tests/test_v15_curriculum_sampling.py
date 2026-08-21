from __future__ import annotations

import pytest
import torch
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand
from test_v14_curriculum_sampling import _curriculum_command, _record_window, _sample_first_bin


def _v15_command():
    command = _curriculum_command()
    command.cfg.curriculum_start_horizon_frames = 4
    command.cfg.curriculum_exclude_invalid_failures = True
    command.cfg.curriculum_motion_length_exponent = 0.5
    command.cfg.curriculum_min_terminal_visits = 8
    command.cfg.curriculum_quarantine_terminal_hazard_threshold = 0.10
    command.cfg.curriculum_terminal_hazard_window_bins = 2
    command.cfg.curriculum_detailed_metrics = False
    command._reset_curriculum_sampling_state()
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


def test_reaching_horizon_records_success_and_later_failure_does_not_backpropagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _v15_command()
    _sample_first_bin(monkeypatch)
    _set_active_episode(command, steps=3)

    command._update_adaptive_exposure()

    assert command._current_curriculum_start_trials[0].item() == 1.0
    assert command._current_curriculum_start_failures.sum().item() == 0.0
    assert command._episode_start_outcome_recorded[0].item()

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


def test_terminal_only_fallback_can_master_but_never_stall_or_quarantine() -> None:
    command = _v15_command()
    bin_id = 4
    command.curriculum_terminal_visits[bin_id] = 8.0

    for _ in range(4):
        command._update_curriculum_states()

    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_MASTERED

    command = _v15_command()
    command.curriculum_terminal_visits[bin_id] = 32.0
    command.curriculum_terminal_failures[bin_id] = 32.0
    for _ in range(5):
        command._update_curriculum_states()

    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_FRONTIER


def test_terminal_fallback_does_not_consume_subthreshold_fixed_horizon_evidence() -> None:
    command = _v15_command()
    command.cfg.curriculum_min_known_trials = 32
    bin_id = 2
    command.curriculum_terminal_visits[bin_id] = 32.0

    _record_window(command, bin_id, trials=8, failures=8)

    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_FRONTIER
    assert command._curriculum_window_start_trials[bin_id].item() == 8.0

    command._update_curriculum_states()

    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_MASTERED
    assert command._curriculum_window_start_trials[bin_id].item() == 8.0

    _record_window(command, bin_id, trials=24, failures=24)

    assert command._curriculum_states[bin_id].item() == command._CURRICULUM_FRONTIER
    assert command._curriculum_window_start_trials[bin_id].item() == 0.0


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


def test_state_motion_bin_sampling_tempers_length_by_square_root() -> None:
    command = _v15_command()
    command.global_num_motions = 2
    command.bin_motion_ids[:] = torch.tensor([0, 0, 0, 0, 1])
    command._curriculum_states.fill_(command._CURRICULUM_FRONTIER)

    probabilities = command._curriculum_state_budget_probabilities(command._active_bin_ids, dtype=torch.float64)

    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(probabilities[:4].sum(), torch.tensor(2.0 / 3.0, dtype=torch.float64))
    torch.testing.assert_close(probabilities[4], torch.tensor(1.0 / 3.0, dtype=torch.float64))
    torch.testing.assert_close(probabilities[:4], torch.full((4,), 1.0 / 6.0, dtype=torch.float64))


def test_v14_schema_migrates_only_terminal_evidence_into_v15() -> None:
    v14 = _curriculum_command()
    v14.curriculum_start_trials.fill_(64.0)
    v14.curriculum_start_failures.fill_(60.0)
    v14.curriculum_terminal_visits[:] = torch.arange(5, dtype=torch.float32) + 30.0
    v14.curriculum_terminal_failures[:] = torch.arange(5, dtype=torch.float32)
    v14.curriculum_terminal_body_failures[:] = torch.tensor([3.0, 7.0])
    v14._curriculum_states.fill_(v14._CURRICULUM_QUARANTINE)
    v14._curriculum_iteration = 200
    state = v14.get_adaptive_sampling_state()
    v15 = _v15_command()

    assert v15.load_adaptive_sampling_state(state)

    assert v15.curriculum_start_trials.count_nonzero().item() == 0
    assert v15.curriculum_start_failures.count_nonzero().item() == 0
    assert torch.all(v15._curriculum_states == v15._CURRICULUM_UNKNOWN)
    assert v15._curriculum_iteration == 0
    assert v15._curriculum_blend_start_iteration == -1
    torch.testing.assert_close(v15.curriculum_terminal_visits, v14.curriculum_terminal_visits)
    torch.testing.assert_close(v15.curriculum_terminal_failures, v14.curriculum_terminal_failures)
    torch.testing.assert_close(v15.curriculum_terminal_body_failures, v14.curriculum_terminal_body_failures)


def test_v15_fixed_horizon_state_round_trip() -> None:
    source = _v15_command()
    source.curriculum_start_trials[:] = torch.arange(5, dtype=torch.float32) + 10.0
    source.curriculum_start_failures[:] = torch.arange(5, dtype=torch.float32)
    source.curriculum_start_censored[:] = torch.arange(5, dtype=torch.float32) + 2.0
    source._curriculum_states[:] = torch.arange(5, dtype=torch.uint8)
    source._curriculum_iteration = 75
    source._curriculum_last_state_update_iteration = 50
    source._curriculum_blend_start_iteration = 50
    state = source.get_adaptive_sampling_state()
    restored = _v15_command()

    assert state["curriculum_state_schema_version"].item() == 3
    assert restored.load_adaptive_sampling_state(state)

    restored_state = restored.get_adaptive_sampling_state()
    assert restored_state.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(restored_state[name], state[name], equal_nan=True)


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
