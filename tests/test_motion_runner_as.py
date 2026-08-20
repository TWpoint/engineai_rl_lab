from __future__ import annotations

from types import SimpleNamespace

import engineai_rl_lab.utils.my_on_policy_runner as runner_module
import pytest
import torch
from engineai_rl_lab.utils.my_on_policy_runner import MotionOnPolicyRunner
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


def test_runner_recomputes_each_iteration_and_syncs_every_200() -> None:
    sync_flags: list[bool] = []
    command = SimpleNamespace(
        sync_and_compute_adaptive_sampling=lambda *, sync_across_ranks: sync_flags.append(sync_across_ranks)
    )
    runner = SimpleNamespace(
        cfg={"sync_adaptive_sampling_all_gpus_freq": 200, "save_interval": 250},
        _motion_command=lambda: command,
    )

    MotionOnPolicyRunner._sync_adaptive_sampling(runner, 1)
    MotionOnPolicyRunner._sync_adaptive_sampling(runner, 198)
    MotionOnPolicyRunner._sync_adaptive_sampling(runner, 199)

    assert sync_flags == [False, False, True]


def test_runner_syncs_on_save_boundary_and_when_forced() -> None:
    sync_flags: list[bool] = []
    command = SimpleNamespace(
        sync_and_compute_adaptive_sampling=lambda *, sync_across_ranks: sync_flags.append(sync_across_ranks)
    )
    runner = SimpleNamespace(
        cfg={"sync_adaptive_sampling_all_gpus_freq": 200, "save_interval": 250},
        _motion_command=lambda: command,
    )

    MotionOnPolicyRunner._sync_adaptive_sampling(runner, 249)
    MotionOnPolicyRunner._sync_adaptive_sampling(runner, 250)
    MotionOnPolicyRunner._sync_adaptive_sampling(runner, 251, force_sync=True)

    assert sync_flags == [False, True, True]


def _make_learn_runner(sync_calls: list[tuple[int, bool]]) -> SimpleNamespace:
    policy = SimpleNamespace(output_std=torch.tensor([1.0]))
    alg = SimpleNamespace(
        train_mode=lambda: None,
        compute_returns=lambda obs: None,
        update=lambda: {},
        learning_rate=0.001,
        get_policy=lambda: policy,
    )
    logger = SimpleNamespace(
        writer=None,
        init_logging_writer=lambda: None,
        log=lambda **kwargs: None,
    )
    return SimpleNamespace(
        env=SimpleNamespace(get_observations=lambda: torch.tensor([0.0])),
        device="cpu",
        alg=alg,
        is_distributed=False,
        logger=logger,
        cfg={"num_steps_per_env": 0, "algorithm": {"rnd_cfg": None}, "save_interval": 250},
        current_learning_iteration=7,
        _sync_adaptive_sampling=lambda iteration, force_sync=False: sync_calls.append((iteration, force_sync)),
        _resample_motion_working_set=lambda iteration, obs: obs,
    )


def test_finite_learn_forces_only_the_final_iteration_sync() -> None:
    sync_calls: list[tuple[int, bool]] = []
    runner = _make_learn_runner(sync_calls)

    MotionOnPolicyRunner.learn(runner, num_learning_iterations=2)

    assert sync_calls == [(7, False), (8, True)]


def test_unbounded_learn_does_not_mark_iterations_as_final(monkeypatch: pytest.MonkeyPatch) -> None:
    sync_calls: list[tuple[int, bool]] = []
    runner = _make_learn_runner(sync_calls)
    monkeypatch.setattr(runner_module, "count", lambda start: range(start, start + 2))

    MotionOnPolicyRunner.learn(runner, num_learning_iterations=None)

    assert sync_calls == [(7, False), (8, False)]


def test_runner_reloads_and_resets_every_250_iterations() -> None:
    reset_calls = 0

    def _reset():
        nonlocal reset_calls
        reset_calls += 1
        return torch.tensor([2.0]), {}

    command = SimpleNamespace(resample_motion_working_set=lambda: True)
    runner = SimpleNamespace(
        cfg={"motion_resample_frequency": 250},
        device="cpu",
        env=SimpleNamespace(reset=_reset),
        _motion_command=lambda: command,
    )
    original_obs = torch.tensor([1.0])

    unchanged = MotionOnPolicyRunner._resample_motion_working_set(runner, 248, original_obs)
    reloaded = MotionOnPolicyRunner._resample_motion_working_set(runner, 249, original_obs)

    assert unchanged is original_obs
    torch.testing.assert_close(reloaded, torch.tensor([2.0]))
    assert reset_calls == 1


def test_runner_staggers_refresh_by_local_rank_on_each_node(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    refresh_calls = 0

    def _refresh() -> bool:
        nonlocal refresh_calls
        refresh_calls += 1
        return False

    command = SimpleNamespace(resample_motion_working_set=_refresh)
    runner = SimpleNamespace(
        cfg={"motion_resample_frequency": 250, "stagger_motion_working_set_refresh": True},
        gpu_local_rank=3,
        device="cpu",
        env=SimpleNamespace(),
        _motion_command=lambda: command,
    )
    obs = torch.tensor([1.0])

    # Refresh events 0, 1, 2 target local ranks 0, 1, 2.
    for iteration in (249, 499, 749):
        assert MotionOnPolicyRunner._resample_motion_working_set(runner, iteration, obs) is obs
    # Refresh event 3 selects local rank 3 on every node.
    assert MotionOnPolicyRunner._resample_motion_working_set(runner, 999, obs) is obs

    assert refresh_calls == 1


@pytest.mark.parametrize("working_set_changed", [False, True])
def test_runner_resets_once_after_restoring_adaptive_state(
    monkeypatch: pytest.MonkeyPatch,
    working_set_changed: bool,
) -> None:
    reset_calls = 0
    working_set_calls = 0

    def _reset():
        nonlocal reset_calls
        reset_calls += 1
        return torch.tensor([0.0]), {}

    def _resample_motion_working_set() -> bool:
        nonlocal working_set_calls
        working_set_calls += 1
        return working_set_changed

    command = SimpleNamespace(
        load_adaptive_sampling_state=lambda state: state == {"restored": True},
        resample_motion_working_set=_resample_motion_working_set,
    )
    runner = object.__new__(MotionOnPolicyRunner)
    runner.env = SimpleNamespace(reset=_reset)
    runner._motion_command = lambda: command
    checkpoint_infos = {
        "runner_infos": {"tag": "resume"},
        "motion_adaptive_sampling": {"restored": True},
    }
    monkeypatch.setattr(OnPolicyRunner, "load", lambda *args, **kwargs: checkpoint_infos)

    infos = runner.load("model.pt")

    assert infos == {"tag": "resume"}
    assert working_set_calls == 1
    assert reset_calls == 1


def test_runner_does_not_reset_when_adaptive_state_restore_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_calls = 0
    working_set_calls = 0

    def _reset():
        nonlocal reset_calls
        reset_calls += 1
        return torch.tensor([0.0]), {}

    def _resample_motion_working_set() -> bool:
        nonlocal working_set_calls
        working_set_calls += 1
        return False

    command = SimpleNamespace(
        load_adaptive_sampling_state=lambda state: False,
        resample_motion_working_set=_resample_motion_working_set,
    )
    runner = object.__new__(MotionOnPolicyRunner)
    runner.env = SimpleNamespace(reset=_reset)
    runner._motion_command = lambda: command
    checkpoint_infos = {
        "runner_infos": {"tag": "resume"},
        "motion_adaptive_sampling": {"restored": False},
    }
    monkeypatch.setattr(OnPolicyRunner, "load", lambda *args, **kwargs: checkpoint_infos)

    infos = runner.load("model.pt")

    assert infos == {"tag": "resume"}
    assert working_set_calls == 0
    assert reset_calls == 0
