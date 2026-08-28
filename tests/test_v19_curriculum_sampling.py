from __future__ import annotations

from types import MethodType

import pytest
import torch
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand
from test_v17_curriculum_sampling import _v17_command
from test_v18_curriculum_sampling import _V18_METHOD_NAMES

_PERSISTED_EVIDENCE_NAMES = (
    "adp_samp_num_episodes",
    "adp_samp_num_failures",
    "curriculum_start_trials",
    "curriculum_start_failures",
    "curriculum_start_censored",
    "curriculum_start_survival_steps",
    "curriculum_start_completion_fraction",
    "curriculum_terminal_visits",
    "curriculum_terminal_failures",
    "curriculum_terminal_body_failures",
    "curriculum_window_start_trials",
    "curriculum_window_start_failures",
    "curriculum_failure_rate_history",
    "curriculum_states",
    "curriculum_state_entry_trials",
    "curriculum_forward_trials",
    "curriculum_forward_risk_sum",
    "curriculum_window_forward_trials",
    "curriculum_window_forward_risk_sum",
    "curriculum_forward_risk_history",
    "curriculum_forward_risk_ema",
    "curriculum_forward_observed_windows",
    "curriculum_quality_trials",
    "curriculum_quality_window_trials",
    "curriculum_quality_window_error_sum",
    "curriculum_quality_error_ema",
    "curriculum_quality_observed_windows",
)


def _bind_motion_method(command, name: str) -> None:
    descriptor = MotionCommand.__dict__[name]
    if isinstance(descriptor, staticmethod):
        setattr(command, name, descriptor.__func__)
    else:
        setattr(command, name, MethodType(descriptor, command))


def _bind_schema9_contract_methods(command) -> None:
    """Extend the lightweight V18 fixture with the production schema-9 path."""

    names = {
        "_curriculum_sampling_strategy",
        "_curriculum_learnability_sampling_enabled",
        "_curriculum_learnability_recipe",
        "_curriculum_checkpoint_schema_version",
        "_curriculum_normalized_entropy",
        "_curriculum_recent_progress_need",
        "_curriculum_quality_base_coverage",
        "_curriculum_learnability_probabilities",
        "_curriculum_unified_need_probabilities",
        "_curriculum_unified_checkpoint_state_valid",
        "_curriculum_unified_checkpoint_recipe_valid",
        "_load_curriculum_unified_checkpoint",
        "get_adaptive_sampling_state",
        "load_adaptive_sampling_state",
    }
    # Checkpoint helpers may be split without making these behavior tests care
    # about their exact private names.
    names.update(name for name in MotionCommand.__dict__ if "learnability" in name)
    for name in names:
        if name in MotionCommand.__dict__:
            _bind_motion_method(command, name)

    command._CURRICULUM_LEARNABILITY_SCHEMA_VERSION = MotionCommand._CURRICULUM_LEARNABILITY_SCHEMA_VERSION


def _unified_command(*, lengths: tuple[int, ...]):
    """Build the V18 fixture while binding schema-9 feature gates first."""

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
    command.cfg.curriculum_min_known_trials = 32
    command.cfg.curriculum_sampling_strategy = "need"

    command._CURRICULUM_UNIFIED_WINDOW_SCHEMA_VERSION = MotionCommand._CURRICULUM_UNIFIED_WINDOW_SCHEMA_VERSION
    _bind_schema9_contract_methods(command)
    for name in _V18_METHOD_NAMES:
        _bind_motion_method(command, name)
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
    command._initialize_curriculum_unified_geometry()
    command._initialize_curriculum_sampling()
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()
    return command


def _schema7_command(*, lengths: tuple[int, ...] = (20,)):
    command = _unified_command(lengths=lengths)
    command.cfg.curriculum_sampling_strategy = "need"
    _bind_schema9_contract_methods(command)
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()
    return command


def _v19_command(*, lengths: tuple[int, ...] = (24,)):
    command = _unified_command(lengths=lengths)
    command.cfg.curriculum_sampling_strategy = "learnability"
    _bind_schema9_contract_methods(command)

    # V18 allocated its diagnostics before the schema-9 opt-in existed.
    command._initialize_curriculum_sampling()
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()
    return command


def _probabilities(command) -> torch.Tensor:
    probabilities = command._curriculum_unified_need_probabilities(
        torch.arange(command.bin_count),
        dtype=torch.float64,
    )
    _assert_distribution(probabilities)
    return probabilities


def _assert_distribution(probabilities: torch.Tensor) -> None:
    assert torch.all(torch.isfinite(probabilities))
    assert torch.all(probabilities >= 0.0)
    torch.testing.assert_close(
        probabilities.sum(),
        torch.tensor(1.0, dtype=probabilities.dtype),
    )


def _set_known_safe(command) -> None:
    """Give every lane enough flat, easy evidence to make D=U=P=0."""

    command.curriculum_start_trials.fill_(float(command.cfg.curriculum_min_known_trials))
    command.curriculum_start_failures.zero_()
    command._curriculum_failure_rate_history.fill_(0.05)
    command.curriculum_forward_trials.fill_(float(command.cfg.curriculum_min_known_trials))
    command.curriculum_forward_risk_sum.zero_()
    command._curriculum_window_forward_trials.zero_()
    command._curriculum_window_forward_risk_sum.zero_()
    command._curriculum_forward_risk_history.fill_(0.05)
    command._curriculum_forward_risk_ema.zero_()


def _mark_quality(command, scores: torch.Tensor) -> None:
    assert scores.shape == (command.bin_count,)
    command.curriculum_quality_trials.fill_(float(command.cfg.curriculum_quality_min_trials))
    command._curriculum_quality_observed_windows.fill_(int(command.cfg.curriculum_quality_min_windows))
    command._curriculum_quality_error_ema.copy_(scores)


def _set_strong_recent_progress(command, bin_id: int, *, forward: bool = False) -> None:
    history = command._curriculum_forward_risk_history if forward else command._curriculum_failure_rate_history
    history[:, bin_id] = torch.tensor([0.90, 0.80, 0.70])


def _populate_checkpoint_evidence(command) -> None:
    """Fill every persisted unified-window evidence lane nontrivially."""

    assert command.bin_count == 5
    command.adp_samp_num_episodes[:] = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    command.adp_samp_num_failures[:] = torch.tensor([1.0, 4.0, 9.0, 20.0, 40.0])
    command.curriculum_start_trials[:] = torch.tensor([40.0, 48.0, 56.0, 64.0, 72.0])
    command.curriculum_start_failures[:] = torch.tensor([2.0, 8.0, 24.0, 40.0, 60.0])
    command.curriculum_start_censored[:] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    command.curriculum_start_survival_steps[:] = torch.tensor([40.0, 38.0, 30.0, 20.0, 8.0])
    command.curriculum_start_completion_fraction[:] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6])
    command.curriculum_terminal_visits[:] = torch.tensor([8.0, 12.0, 16.0, 20.0, 24.0])
    command.curriculum_terminal_failures[:] = torch.tensor([0.0, 1.0, 4.0, 8.0, 16.0])
    command.curriculum_terminal_body_failures[:] = torch.tensor([3.0, 7.0])
    command._curriculum_window_start_trials[:] = torch.tensor([0.0, 8.0, 8.0, 8.0, 8.0])
    command._curriculum_window_start_failures[:] = torch.tensor([0.0, 1.0, 3.0, 5.0, 8.0])
    command._curriculum_failure_rate_history[:] = torch.tensor(
        [
            [0.02, 0.10, 0.30, 0.60, 0.90],
            [0.03, 0.12, 0.35, 0.65, 0.92],
            [0.04, 0.14, 0.40, 0.70, 0.94],
        ]
    )
    command._curriculum_states[:] = torch.tensor(
        [
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_FRONTIER,
            command._CURRICULUM_STALLED,
            command._CURRICULUM_QUARANTINE,
            command._CURRICULUM_UNKNOWN,
        ],
        dtype=torch.uint8,
    )
    command._curriculum_state_entry_trials[:] = torch.tensor([0.0, 8.0, 16.0, 24.0, 0.0])

    command.curriculum_forward_trials[:] = torch.tensor([32.0, 40.0, 48.0, 56.0, 64.0])
    command.curriculum_forward_risk_sum[:] = torch.tensor([1.0, 8.0, 18.0, 28.0, 40.0])
    command._curriculum_window_forward_trials[:] = torch.tensor([0.0, 8.0, 8.0, 8.0, 8.0])
    command._curriculum_window_forward_risk_sum[:] = torch.tensor([0.0, 0.8, 2.0, 4.0, 6.0])
    command._curriculum_forward_risk_history[:] = torch.tensor(
        [
            [0.02, 0.10, 0.30, 0.50, 0.70],
            [0.03, 0.12, 0.35, 0.55, 0.75],
            [0.04, 0.14, 0.40, 0.60, 0.80],
        ]
    )
    command._curriculum_forward_risk_ema[:] = torch.tensor([0.03, 0.13, 0.35, 0.55, 0.75])
    command._curriculum_forward_observed_windows[:] = torch.tensor([2, 3, 4, 5, 6], dtype=torch.uint8)

    command.curriculum_quality_trials[:] = torch.tensor([32.0, 40.0, 48.0, 56.0, 64.0])
    command._curriculum_quality_window_trials[:] = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0])
    command._curriculum_quality_window_error_sum[:] = torch.tensor([0.0, 0.2, 1.0, 2.0, 4.0])
    command._curriculum_quality_error_ema[:] = torch.tensor([0.10, 0.20, 0.30, 0.40, 0.50])
    command._curriculum_quality_observed_windows[:] = torch.tensor([2, 3, 4, 5, 6], dtype=torch.uint8)

    command._curriculum_iteration = 137
    command._curriculum_last_state_update_iteration = 125
    command._curriculum_blend_start_iteration = 20
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()


def test_cold_start_all_unknown_is_naturally_uniform_by_actual_frames() -> None:
    command = _v19_command(lengths=(7,))

    assert torch.all(command._curriculum_states == command._CURRICULUM_UNKNOWN)
    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([4.0 / 7.0, 3.0 / 7.0], dtype=torch.float64),
    )


def test_learnability_boost_requires_difficulty_and_uncertainty_or_positive_progress() -> None:
    command = _v19_command(lengths=(16,))
    _set_known_safe(command)
    command._curriculum_states.fill_(command._CURRICULUM_FRONTIER)

    # Bin 1 has difficulty alone; bin 2 uncertainty alone. Only bin 3 has
    # D>0 and recent positive progress, so A=D*max(U,P) is nonzero only there.
    command.curriculum_start_failures[1] = command.curriculum_start_trials[1]
    command.curriculum_start_trials[2] = 16.0
    command.curriculum_start_failures[2] = 0.0
    command.curriculum_start_failures[3] = command.curriculum_start_trials[3]
    _set_strong_recent_progress(command, 3)

    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([0.2, 0.2, 0.2, 0.4], dtype=torch.float64),
    )


def test_uncertainty_and_recent_progress_are_equivalent_learnability_routes() -> None:
    command = _v19_command(lengths=(8,))
    _set_known_safe(command)
    command._curriculum_states.fill_(command._CURRICULUM_FRONTIER)

    # A0=1*.75 through uncertainty; A1=1*1 through recent progress.
    command.curriculum_start_trials[0] = 8.0
    command.curriculum_start_failures[0] = 8.0
    command.curriculum_start_failures[1] = command.curriculum_start_trials[1]
    _set_strong_recent_progress(command, 1)

    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([13.0 / 28.0, 15.0 / 28.0], dtype=torch.float64),
    )


def test_learnability_is_a_self_normalized_product_and_is_scale_invariant() -> None:
    command = _v19_command(lengths=(12,))
    _set_known_safe(command)
    command._curriculum_states.fill_(command._CURRICULUM_FRONTIER)
    command.curriculum_start_trials[:] = torch.tensor([8.0, 16.0, 24.0])
    command.curriculum_start_failures.copy_(command.curriculum_start_trials)

    # D=1 and U=(.75,.50,.25), whose positive frame-weighted mean is .50.
    expected = torch.tensor([5.0 / 12.0, 1.0 / 3.0, 1.0 / 4.0], dtype=torch.float64)
    torch.testing.assert_close(_probabilities(command), expected)

    # Halving every positive A leaves A/mean_positive(A) unchanged.
    scaled = _v19_command(lengths=(12,))
    _set_known_safe(scaled)
    scaled._curriculum_states.fill_(scaled._CURRICULUM_FRONTIER)
    scaled.curriculum_start_trials[:] = torch.tensor([20.0, 24.0, 28.0])
    scaled.curriculum_start_failures.copy_(scaled.curriculum_start_trials)
    torch.testing.assert_close(_probabilities(scaled), expected)


def test_survival_and_forward_are_equivalent_sources_of_difficulty() -> None:
    command = _v19_command(lengths=(8,))
    _set_known_safe(command)
    command._curriculum_states.fill_(command._CURRICULUM_FRONTIER)

    command.curriculum_start_failures[0] = command.curriculum_start_trials[0]
    _set_strong_recent_progress(command, 0)
    command._curriculum_forward_risk_ema[1] = float(command.cfg.curriculum_forward_risk_bad_threshold)
    _set_strong_recent_progress(command, 1, forward=True)

    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )


def test_base_coverage_keeps_every_state_on_the_same_actual_frame_floor() -> None:
    command = _v19_command(lengths=(20,))
    _set_known_safe(command)
    command._curriculum_states[:] = torch.tensor(
        [
            command._CURRICULUM_STALLED,
            command._CURRICULUM_QUARANTINE,
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_FRONTIER,
            command._CURRICULUM_UNKNOWN,
        ],
        dtype=torch.uint8,
    )
    command.curriculum_start_failures[:2] = command.curriculum_start_trials[:2]
    _mark_quality(command, torch.zeros(5))

    torch.testing.assert_close(
        _probabilities(command),
        torch.full((5,), 0.20, dtype=torch.float64),
    )


def test_quarantine_base_has_no_probe_debt_or_entry_trial_gate() -> None:
    command = _v19_command(lengths=(8,))
    _set_known_safe(command)
    command._curriculum_states.fill_(command._CURRICULUM_QUARANTINE)
    command.curriculum_start_trials.fill_(64.0)
    command.curriculum_start_failures.copy_(command.curriculum_start_trials)
    command._curriculum_state_entry_trials[:] = torch.tensor([64.0, 0.0])

    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )


def test_high_error_mastered_reorders_only_inside_mastered_and_preserves_its_mass() -> None:
    command = _v19_command(lengths=(12,))
    _set_known_safe(command)
    command._curriculum_states[:] = torch.tensor(
        [
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_STALLED,
        ],
        dtype=torch.uint8,
    )
    command.curriculum_start_failures[2] = command.curriculum_start_trials[2]
    good = float(command.cfg.curriculum_quality_good_threshold)
    bad = float(command.cfg.curriculum_quality_bad_threshold)
    _mark_quality(command, torch.tensor([good, bad, bad]))

    first = _probabilities(command)
    torch.testing.assert_close(
        first,
        torch.tensor([2.0 / 9.0, 4.0 / 9.0, 1.0 / 3.0], dtype=torch.float64),
    )

    _mark_quality(command, torch.tensor([bad, good, bad]))
    second = _probabilities(command)
    torch.testing.assert_close(
        second,
        torch.tensor([4.0 / 9.0, 2.0 / 9.0, 1.0 / 3.0], dtype=torch.float64),
    )
    assert first[:2].sum().item() == pytest.approx(2.0 / 3.0)
    assert second[:2].sum().item() == pytest.approx(2.0 / 3.0)

    # Learnability multiplies the already quality-shaped base. Doubling the
    # first MASTERED bin turns [2,4,3]/9 into [4,4,3]/11.
    _mark_quality(command, torch.tensor([good, bad, bad]))
    command.curriculum_start_failures[0] = command.curriculum_start_trials[0]
    _set_strong_recent_progress(command, 0)
    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([4.0 / 11.0, 4.0 / 11.0, 3.0 / 11.0], dtype=torch.float64),
    )


def test_learnability_strategy_ignores_state_counts_and_legacy_quality_fillers() -> None:
    command = _v19_command(lengths=(20,))
    _set_known_safe(command)
    command._curriculum_states[:] = torch.tensor(
        [
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_STALLED,
            command._CURRICULUM_FRONTIER,
            command._CURRICULUM_FRONTIER,
        ],
        dtype=torch.uint8,
    )
    command.curriculum_start_failures[2] = command.curriculum_start_trials[2]
    _mark_quality(
        command,
        torch.tensor(
            [
                command.cfg.curriculum_quality_good_threshold,
                command.cfg.curriculum_quality_bad_threshold,
                command.cfg.curriculum_quality_bad_threshold,
                0.0,
                0.0,
            ]
        ),
    )
    before = _probabilities(command)

    command._curriculum_states[4] = command._CURRICULUM_UNKNOWN
    command._curriculum_quality_filler_mask[:] = torch.tensor([False, True, False, True, True])

    torch.testing.assert_close(_probabilities(command), before)


def test_single_positive_learnability_need_doubles_its_base_before_normalization() -> None:
    command = _v19_command(lengths=(16,))
    _set_known_safe(command)
    command._curriculum_states[:] = torch.tensor(
        [
            command._CURRICULUM_FRONTIER,
            command._CURRICULUM_MASTERED,
            command._CURRICULUM_STALLED,
            command._CURRICULUM_QUARANTINE,
        ],
        dtype=torch.uint8,
    )
    command.curriculum_start_failures[:] = command.curriculum_start_trials
    _set_strong_recent_progress(command, 0)
    _mark_quality(command, torch.zeros(4))

    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([0.4, 0.2, 0.2, 0.2], dtype=torch.float64),
    )


def test_zero_learnability_need_returns_quality_shaped_base_exactly() -> None:
    command = _v19_command(lengths=(16,))
    _set_known_safe(command)
    command._curriculum_states.fill_(command._CURRICULUM_FRONTIER)
    command._curriculum_states[:2] = command._CURRICULUM_MASTERED
    _mark_quality(
        command,
        torch.tensor(
            [
                command.cfg.curriculum_quality_bad_threshold,
                command.cfg.curriculum_quality_good_threshold,
                0.0,
                0.0,
            ]
        ),
    )

    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([1.0 / 3.0, 1.0 / 6.0, 0.25, 0.25], dtype=torch.float64),
    )


def test_positive_mean_and_priority_are_weighted_by_partial_tail_frames() -> None:
    command = _v19_command(lengths=(7,))
    _set_known_safe(command)
    command._curriculum_states.fill_(command._CURRICULUM_FRONTIER)
    command.curriculum_start_trials[:] = torch.tensor([8.0, 24.0])
    command.curriculum_start_failures.copy_(command.curriculum_start_trials)

    # A=(.75,.25), frame weights=(4,3), so mean_positive(A)=15/28.
    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([24.0 / 35.0, 11.0 / 35.0], dtype=torch.float64),
    )


def test_zero_learnability_need_without_quality_is_actual_frame_uniform() -> None:
    command = _v19_command(lengths=(7,))
    _set_known_safe(command)
    command._curriculum_states.fill_(command._CURRICULUM_FRONTIER)

    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([4.0 / 7.0, 3.0 / 7.0], dtype=torch.float64),
    )


def test_single_bin_distribution_is_finite_and_normalized() -> None:
    command = _v19_command(lengths=(7,))

    probabilities = command._curriculum_unified_need_probabilities(
        torch.tensor([0]),
        dtype=torch.float64,
    )
    torch.testing.assert_close(probabilities, torch.ones(1, dtype=torch.float64))


def test_need_strategy_reproduces_v18_exactly() -> None:
    command = _v19_command(lengths=(11,))
    command.cfg.curriculum_sampling_strategy = "need"
    command.curriculum_start_trials[:] = torch.tensor([32.0, 8.0, 32.0])
    command.curriculum_start_failures[:] = torch.tensor([0.0, 8.0, 16.0])

    torch.testing.assert_close(
        _probabilities(command),
        torch.tensor([0.8 / 11.0, 7.2 / 11.0, 3.0 / 11.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "source_schema",
    (7, 8),
    ids=("schema7-v18", "schema8-staged"),
)
def test_schema7_and_schema8_migrate_all_evidence_behind_frozen_sampler(source_schema: int) -> None:
    source = _schema7_command(lengths=(20,))
    _populate_checkpoint_evidence(source)
    state = source.get_adaptive_sampling_state()
    if source_schema == 8:
        # Schema 8 had a staged runtime sampler. The learnability loader only
        # needs its unified evidence and saved policy-facing distribution.
        state["curriculum_state_schema_version"] = torch.tensor(8)
        state["curriculum_staged_sampling_config"] = torch.tensor(
            [0.20, 0.50, 0.10, 0.10, 0.03, 0.07, 0.15, 0.50, 0.25]
        )
    saved_sampler = state["curriculum_smoothed_probabilities"].clone()
    restored = _v19_command(lengths=(20,))

    assert int(state["curriculum_state_schema_version"].item()) == source_schema
    assert "curriculum_learnability_recipe" not in state
    assert restored.load_adaptive_sampling_state(state)

    restored_state = restored.get_adaptive_sampling_state()
    for name in _PERSISTED_EVIDENCE_NAMES:
        torch.testing.assert_close(restored_state[name], state[name], equal_nan=True)
    assert restored_state["curriculum_state_schema_version"].item() == 9
    assert "curriculum_learnability_recipe" in restored_state
    assert not any("frontier_peak" in name for name in restored_state)
    torch.testing.assert_close(restored._curriculum_shadow_probabilities, saved_sampler)
    torch.testing.assert_close(restored._curriculum_smoothed_probabilities, saved_sampler)
    torch.testing.assert_close(restored.adp_sampling_prob, saved_sampler)
    assert restored._curriculum_has_shadow_distribution
    assert not restored._curriculum_has_shadow_state_recipe
    assert restored._curriculum_smoothed_probabilities_initialized
    assert restored._curriculum_iteration == 0
    assert restored._curriculum_last_state_update_iteration == 0
    assert restored._curriculum_blend_start_iteration == -1
    assert restored._curriculum_last_probability_smoothing_iteration == 0


def test_schema9_round_trip_restores_evidence_sampler_and_recipe_exactly() -> None:
    source = _v19_command(lengths=(20,))
    _populate_checkpoint_evidence(source)
    state = source.get_adaptive_sampling_state()
    restored = _v19_command(lengths=(20,))

    assert state["curriculum_state_schema_version"].item() == 9
    assert state["curriculum_learnability_recipe"].shape == (15,)
    assert not any("frontier_peak" in name for name in state)
    assert restored.load_adaptive_sampling_state(state)

    restored_state = restored.get_adaptive_sampling_state()
    assert restored_state.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(restored_state[name], state[name], equal_nan=True)
    torch.testing.assert_close(restored.adp_sampling_prob, source.adp_sampling_prob)


def test_schema9_recipe_mismatch_preserves_evidence_behind_fresh_frozen_sampler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _v19_command(lengths=(20,))
    _populate_checkpoint_evidence(source)
    state = {name: value.clone() for name, value in source.get_adaptive_sampling_state().items()}
    saved_sampler = state["curriculum_smoothed_probabilities"].clone()
    state["curriculum_learnability_recipe"][4] += 0.001
    restored = _v19_command(lengths=(20,))

    assert restored.load_adaptive_sampling_state(state)

    output = capsys.readouterr().out.lower()
    assert "evidence" in output
    assert "shadow" in output or "saved sampler" in output
    restored_state = restored.get_adaptive_sampling_state()
    for name in _PERSISTED_EVIDENCE_NAMES:
        torch.testing.assert_close(restored_state[name], state[name], equal_nan=True)
    assert not torch.equal(
        restored_state["curriculum_learnability_recipe"],
        state["curriculum_learnability_recipe"],
    )
    torch.testing.assert_close(restored._curriculum_shadow_probabilities, saved_sampler)
    torch.testing.assert_close(restored._curriculum_smoothed_probabilities, saved_sampler)
    torch.testing.assert_close(restored.adp_sampling_prob, saved_sampler)
    assert restored._curriculum_has_shadow_distribution
    assert not restored._curriculum_has_shadow_state_recipe
    assert restored._curriculum_smoothed_probabilities_initialized
    assert restored._curriculum_iteration == 0
    assert restored._curriculum_last_state_update_iteration == 0
    assert restored._curriculum_blend_start_iteration == -1
    assert restored._curriculum_last_probability_smoothing_iteration == 0
