from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand


def _command(*, active_global_ids: tuple[int, ...] = (0, 1)) -> SimpleNamespace:
    command = SimpleNamespace()
    command.device = torch.device("cpu")
    command.num_envs = 2
    command.cfg = SimpleNamespace(
        bin_size=4,
        sequence_length_agnostic=True,
        init_num_failures=1.0,
        uniform_sampling_rate=0.1,
        pre_failure_sample_window=0,
        use_failure_rate_decay=False,
        decay_gamma=0.8,
        adp_samp_failure_rate_max_over_mean=200.0,
        failure_counts_multiplier=1.0,
        max_prob_per_bin=None,
        max_prob_per_motion=None,
    )
    command.global_num_motions = 2
    command.bin_count = 2
    command.motion_bin_counts = torch.ones(2, dtype=torch.long)
    command.motion_bin_offsets = torch.arange(2)
    command.bin_motion_ids = torch.arange(2)
    command.bin_starts = torch.zeros(2, dtype=torch.long)
    command.bin_ends = torch.full((2,), 8, dtype=torch.long)
    command.bin_weights = torch.ones(2)
    command.adp_samp_num_episodes = torch.ones(2)
    command.adp_samp_num_failures = torch.ones(2)
    command.metrics = {
        "sampling_entropy": torch.zeros(2),
        "sampling_top1_prob": torch.zeros(2),
    }
    lengths = torch.tensor([8, 8])
    global_ids = torch.tensor(active_global_ids, dtype=torch.long)
    local_lengths = lengths[global_ids]
    command.motion = SimpleNamespace(
        global_num_motions=2,
        world_size=2,
        manifest_fingerprint_words=(1, 2, 3, 4),
        global_ids=global_ids,
        num_motions=len(global_ids),
        lengths=lambda motion_ids: local_lengths[motion_ids],
    )
    command._active_bin_ids = global_ids.clone()
    command._active_local_motion_ids = torch.arange(len(global_ids))
    command.motion_ids = torch.zeros(2, dtype=torch.long)
    command.time_steps = torch.zeros(2, dtype=torch.long)
    command.motion_lengths = torch.zeros(2, dtype=torch.long)
    command._has_sampled = torch.zeros(2, dtype=torch.bool)
    command._env = SimpleNamespace(termination_manager=SimpleNamespace(terminated=torch.zeros(2, dtype=torch.bool)))
    command._adaptive_layout_checked = True
    for method_name in (
        "_bucket_ids",
        "_compute_failure_rate",
        "_clip_failure_rate",
        "_configured_probability_cap",
        "_apply_probability_caps",
        "_rebuild_global_sampling_distribution",
        "_rebuild_sampling_distribution",
        "_update_adaptive_exposure",
        "_check_adaptive_distributed_layout",
        "sync_and_compute_adaptive_sampling",
        "get_adaptive_sampling_state",
        "load_adaptive_sampling_state",
    ):
        setattr(command, method_name, MethodType(getattr(MotionCommand, method_name), command))
    command._rebuild_global_sampling_distribution()
    command._rebuild_sampling_distribution()
    return command


def test_probability_order_exactly_matches_sonic() -> None:
    command = _command()
    command.bin_weights = torch.tensor([1.0, 2.0])
    command.adp_samp_num_episodes[:] = 1.0
    command.adp_samp_num_failures[:] = torch.tensor([1.0, 3.0])

    command._rebuild_sampling_distribution()

    failure_probability = torch.tensor([0.25, 0.75], dtype=torch.float64)
    expected = 0.9 * failure_probability + 0.1 * torch.tensor([0.5, 0.5], dtype=torch.float64)
    expected *= torch.tensor([1.0, 2.0])
    expected /= expected.sum()
    torch.testing.assert_close(command.adp_sampling_active_prob, expected.float())


def test_replacement_sample_maps_duplicate_active_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command(active_global_ids=(1, 1))
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probabilities, num_samples, replacement: torch.tensor([1]),
    )

    MotionCommand._adaptive_sampling(command, torch.tensor([0]))

    assert command.motion_ids[0].item() == 1
    assert command.motion_lengths[0].item() == 8
    assert MotionCommand._bucket_ids(command, torch.tensor([1]), torch.tensor([2])).item() == 1


def test_active_bins_accumulate_sonic_length_normalized_episode_count() -> None:
    command = _command()
    command._has_sampled[:] = True
    command.motion_ids[:] = torch.tensor([0, 1])
    command.motion_lengths[:] = 8
    command.time_steps[:] = torch.tensor([2, 3])

    MotionCommand._update_adaptive_exposure(command)

    torch.testing.assert_close(command.adp_samp_num_episodes, torch.tensor([1.125, 1.125]))


def test_only_terminated_resets_increment_sonic_failure_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command()
    command._has_sampled[:] = True
    command.motion_ids[:] = torch.tensor([0, 1])
    command.motion_lengths[:] = 8
    command.time_steps[:] = torch.tensor([2, 3])
    command._env.termination_manager.terminated[:] = torch.tensor([True, False])
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probabilities, num_samples, replacement: torch.tensor([0, 1]),
    )

    MotionCommand._adaptive_sampling(command, torch.tensor([0, 1]))

    torch.testing.assert_close(command.adp_samp_num_failures, torch.tensor([2.0, 1.0]))


def test_distributed_sync_averages_cumulative_sonic_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command()
    command.adp_samp_num_episodes[:] = torch.tensor([2.0, 4.0])
    command.adp_samp_num_failures[:] = torch.tensor([1.0, 3.0])
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def _all_reduce(values: torch.Tensor, **kwargs) -> None:
        values += torch.tensor([2.0, 2.0, 1.0, 1.0])

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)

    command.sync_and_compute_adaptive_sampling(sync_across_ranks=True)

    torch.testing.assert_close(command.adp_samp_num_episodes, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(command.adp_samp_num_failures, torch.tensor([1.0, 2.0]))


def test_adaptive_sampling_state_round_trip() -> None:
    source = _command()
    source.adp_samp_num_episodes[:] = torch.tensor([5.0, 7.0])
    source.adp_samp_num_failures[:] = torch.tensor([2.0, 3.0])
    restored = _command()

    assert restored.load_adaptive_sampling_state(source.get_adaptive_sampling_state())

    torch.testing.assert_close(restored.adp_samp_num_episodes, source.adp_samp_num_episodes)
    torch.testing.assert_close(restored.adp_samp_num_failures, source.adp_samp_num_failures)


def test_layout_check_waits_for_distributed_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command()
    command._adaptive_layout_checked = False
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    MotionCommand._check_adaptive_distributed_layout(command)

    assert not command._adaptive_layout_checked


def test_layout_check_rejects_manifest_fingerprint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
