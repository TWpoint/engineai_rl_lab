from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand


def _command(*, sharded: bool = True) -> SimpleNamespace:
    command = SimpleNamespace()
    command.device = torch.device("cpu")
    command.num_envs = 2
    command.cfg = SimpleNamespace(
        adaptive_failure_rate_max_over_mean=None,
        adaptive_uniform_ratio=0.0,
        adaptive_failure_multiplier=1.0,
        adaptive_bin_size=4,
        adaptive_pre_failure_sample_window=0,
    )
    command.bin_count = 2
    command.bin_prior_weights = torch.tensor([0.5, 0.5])
    command.bin_episode_count = torch.tensor([1.0, 1.0])
    command.bin_failed_count = torch.tensor([1.0, 1.0])
    command._current_bin_episodes = torch.zeros(2)
    command._current_bin_failed = torch.zeros(2)
    command.metrics = {
        "sampling_entropy": torch.zeros(2),
        "sampling_top1_prob": torch.zeros(2),
        "sampling_top1_bin": torch.zeros(2),
    }
    lengths = torch.tensor([8, 8])
    command.motion = SimpleNamespace(
        is_distributed_shard=sharded,
        global_num_motions=2,
        world_size=2,
        manifest_fingerprint_words=(1, 2, 3, 4),
        lengths=lambda motion_ids: lengths[motion_ids],
    )
    command._adaptive_granularity = "bin"
    command.motion_bin_offsets = torch.tensor([0, 1])
    command.bin_motion_ids = torch.tensor([0, 1])
    command.bin_starts = torch.tensor([0, 0])
    command.bin_ends = torch.tensor([8, 8])
    command.motion_ids = torch.zeros(2, dtype=torch.long)
    command.time_steps = torch.zeros(2, dtype=torch.long)
    command.motion_lengths = torch.zeros(2, dtype=torch.long)
    command._has_sampled = torch.zeros(2, dtype=torch.bool)
    command._env = SimpleNamespace(termination_manager=SimpleNamespace(terminated=torch.zeros(2, dtype=torch.bool)))
    command._adaptive_layout_checked = True
    command._bucket_ids = MethodType(MotionCommand._bucket_ids, command)
    command._rebuild_sampling_distribution = MethodType(MotionCommand._rebuild_sampling_distribution, command)
    command._check_adaptive_distributed_layout = MethodType(MotionCommand._check_adaptive_distributed_layout, command)
    return command


def test_zero_uniform_never_selects_zero_probability_bin(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command()
    command.bin_prior_weights = torch.tensor([0.0, 1.0])
    command.bin_failed_count = torch.tensor([0.0, 1.0])
    command._rebuild_sampling_distribution()
    random_values = iter((torch.tensor([0.0]), torch.tensor([0.5])))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(random_values))

    MotionCommand._adaptive_sampling(command, torch.tensor([0]))

    assert command.motion_ids[0].item() == 1
    assert command.motion_lengths[0].item() == 8


def test_repeated_previous_bins_are_counted_without_dense_bincount(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command()
    command._has_sampled[:] = True
    command.motion_ids[:] = 0
    command.motion_lengths[:] = 8
    command.time_steps[:] = torch.tensor([2, 3])
    command._rebuild_sampling_distribution()
    random_values = iter((torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2])))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(random_values))

    MotionCommand._adaptive_sampling(command, torch.tensor([0, 1]))

    torch.testing.assert_close(command._current_bin_episodes, torch.tensor([2.0, 0.0]))


def test_sharded_statistics_stay_local(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command(sharded=True)
    command._current_bin_episodes[:] = torch.tensor([2.0, 3.0])
    command._current_bin_failed[:] = torch.tensor([1.0, 0.0])
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("rank-local adaptive arrays must not be all-reduced"),
    )

    MotionCommand._sync_adaptive_stats(command)

    torch.testing.assert_close(command.bin_episode_count, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(command.bin_failed_count, torch.tensor([2.0, 1.0]))
    assert torch.count_nonzero(command._current_bin_episodes) == 0
    assert torch.count_nonzero(command._current_bin_failed) == 0


def test_replicated_statistics_retain_global_all_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command(sharded=False)
    command._current_bin_episodes[:] = torch.tensor([2.0, 3.0])
    command._current_bin_failed[:] = torch.tensor([1.0, 0.0])
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    calls = 0

    def _all_reduce(values: torch.Tensor, **kwargs) -> None:
        nonlocal calls
        calls += 1
        values.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)

    MotionCommand._sync_adaptive_stats(command)

    assert calls == 1
    torch.testing.assert_close(command.bin_episode_count, torch.tensor([5.0, 7.0]))
    torch.testing.assert_close(command.bin_failed_count, torch.tensor([3.0, 1.0]))


def test_layout_check_waits_for_distributed_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command()
    command._adaptive_layout_checked = False
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    MotionCommand._check_adaptive_distributed_layout(command)

    assert not command._adaptive_layout_checked


def test_layout_check_rejects_manifest_fingerprint_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command()
    command._adaptive_layout_checked = False
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def _all_reduce(values: torch.Tensor, *, op: object) -> None:
        if op == torch.distributed.ReduceOp.MAX:
            values[-1] += 1

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)

    with pytest.raises(RuntimeError, match="manifest fingerprint"):
        MotionCommand._check_adaptive_distributed_layout(command)
