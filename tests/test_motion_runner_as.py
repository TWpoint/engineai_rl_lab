from __future__ import annotations

from types import SimpleNamespace

import torch
from engineai_rl_lab.utils.my_on_policy_runner import MotionOnPolicyRunner


def test_runner_recomputes_each_iteration_and_syncs_every_200() -> None:
    sync_flags: list[bool] = []
    command = SimpleNamespace(
        sync_and_compute_adaptive_sampling=lambda *, sync_across_ranks: sync_flags.append(sync_across_ranks)
    )
    runner = SimpleNamespace(
        cfg={"sync_adaptive_sampling_all_gpus_freq": 200},
        _motion_command=lambda: command,
    )

    MotionOnPolicyRunner._sync_adaptive_sampling(runner, 0)
    MotionOnPolicyRunner._sync_adaptive_sampling(runner, 198)
    MotionOnPolicyRunner._sync_adaptive_sampling(runner, 199)

    assert sync_flags == [False, False, True]


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
