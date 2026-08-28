from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import (
    AdaptiveSamplerV1,
    AdaptiveSamplerV1Cfg,
    MotionCommandV1,
    MotionCommandV1Cfg,
)
from engineai_rl_lab.tasks.tracking.robots.actuator import DelayedImplicitActuator


def _sampler(
    lengths: tuple[int, ...],
    **cfg_overrides,
) -> AdaptiveSamplerV1:
    cfg = AdaptiveSamplerV1Cfg(**cfg_overrides)
    return AdaptiveSamplerV1(torch.tensor(lengths), cfg, device="cpu")


def test_cold_start_is_frame_coverage_with_explicit_20_20_60_budget() -> None:
    sampler = _sampler((100, 75))

    probabilities = sampler.distribution()

    torch.testing.assert_close(probabilities, torch.tensor([2 / 7, 2 / 7, 2 / 7, 1 / 7]))
    assert sampler.last_gate == pytest.approx(0.75)
    assert sampler.last_coverage_mass == pytest.approx(0.20)
    assert sampler.last_hard_mass == pytest.approx(0.20)
    assert sampler.last_learnable_mass == pytest.approx(0.60)


def test_sparse_learnability_receives_its_explicit_budget() -> None:
    sampler = _sampler((1000,))
    sampler.total_exposure[:] = sampler.cfg.uncertainty_exposure
    sampler.total_exposure[0] = 0.0

    probabilities = sampler.distribution()

    expected = torch.full((20,), 0.02)
    expected[0] = 0.62
    torch.testing.assert_close(probabilities, expected)
    assert sampler.last_remaining_learnability == pytest.approx(0.05)
    assert sampler.last_learnable_mass == pytest.approx(0.60)


def test_sampler_moves_all_adaptive_mass_to_hard_bins_after_progress_stalls() -> None:
    sampler = _sampler((100,))
    sampler.total_exposure[:] = sampler.cfg.uncertainty_exposure
    sampler.difficulty_fast[:] = torch.tensor([0.2, 0.8])
    sampler.difficulty_slow.copy_(sampler.difficulty_fast)

    probabilities = sampler.distribution()

    torch.testing.assert_close(probabilities, torch.tensor([0.26, 0.74]))
    assert sampler.last_gate == pytest.approx(0.0)
    assert sampler.last_hard_mass == pytest.approx(0.80)
    assert sampler.last_learnable_mass == pytest.approx(0.0)


def test_recent_improvement_can_outweigh_a_stagnant_harder_bin() -> None:
    sampler = _sampler((100,))
    sampler.total_exposure[:] = sampler.cfg.uncertainty_exposure
    sampler.difficulty_fast[:] = torch.tensor([0.4, 0.8])
    sampler.difficulty_slow[:] = torch.tensor([0.8, 0.8])

    probabilities = sampler.distribution()

    assert sampler.last_gate == pytest.approx(0.75)
    assert probabilities[0] > probabilities[1]


def test_first_evidence_initializes_both_difficulty_emas_without_false_progress() -> None:
    sampler = _sampler((50,), fast_half_life=1.0, slow_half_life=10.0)
    bin_ids = torch.zeros(50, dtype=torch.long)
    sampler.record_exposure(bin_ids)
    sampler.record_tracking_error(bin_ids, torch.full((50, 2), 0.30))

    assert sampler.update(sync_across_ranks=False)

    assert sampler.total_exposure.item() == pytest.approx(1.0)
    expected_difficulty = 1.0 - torch.exp(torch.tensor(-1.0)).item()
    assert sampler.difficulty_fast.item() == pytest.approx(expected_difficulty)
    assert sampler.difficulty_slow.item() == pytest.approx(expected_difficulty)

    sampler.record_exposure(bin_ids)
    sampler.record_tracking_error(bin_ids, torch.zeros(50, 2))
    assert sampler.update(sync_across_ranks=False)

    assert 0.0 < sampler.difficulty_fast.item() < sampler.difficulty_slow.item() < expected_difficulty


def test_total_exposure_saturates_without_stopping_difficulty_updates() -> None:
    sampler = _sampler((50,), uncertainty_exposure=2.0)
    sampler.difficulty_fast.fill_(1.0)
    sampler.difficulty_slow.fill_(1.0)
    bin_ids = torch.zeros(150, dtype=torch.long)
    sampler.record_exposure(bin_ids)

    assert sampler.update(sync_across_ranks=False)

    assert sampler.total_exposure.item() == pytest.approx(2.0)
    difficulty_after_saturation = sampler.difficulty_fast.item()

    sampler.record_exposure(torch.zeros(50, dtype=torch.long))
    assert sampler.update(sync_across_ranks=False)

    assert sampler.total_exposure.item() == pytest.approx(2.0)
    assert sampler.difficulty_fast.item() < difficulty_after_saturation


def test_terminal_failure_is_evidence_even_without_a_recorded_frame() -> None:
    sampler = _sampler((50,))
    sampler.record_failures(torch.tensor([0]))

    assert sampler.update(sync_across_ranks=False)

    assert sampler.total_exposure.item() == 0.0
    assert sampler.difficulty_fast.item() == 1.0
    assert sampler.difficulty_slow.item() == 1.0


def test_failure_events_can_be_deduplicated_by_env_and_bin() -> None:
    sampler = _sampler((100,), deduplicate_failure_events=True)

    sampler.record_failures(
        torch.tensor([0, 0, 1]),
        event_ids=torch.tensor([7, 7, 8]),
    )

    torch.testing.assert_close(sampler._failure_delta, torch.tensor([1.0, 1.0]))


def test_tracking_difficulty_blends_configured_reward_aligned_errors() -> None:
    sampler = _sampler(
        (50,),
        tracking_error_scale=0.40,
        global_tracking_error_weight=0.40,
        relative_position_error_weight=0.20,
        relative_position_error_scale=0.40,
        relative_orientation_error_weight=0.20,
        relative_orientation_error_scale=0.60,
        local_position_error_weight=0.20,
        local_position_error_scale=0.20,
    )
    bin_ids = torch.zeros(2, dtype=torch.long)
    global_errors = torch.full((2, 2), 0.40)
    relative_position_errors = torch.full((2, 2), 0.20)
    relative_orientation_errors = torch.full((2, 2), 0.30)
    local_position_errors = torch.full((2, 2), 0.10)

    sampler.record_tracking_error(
        bin_ids,
        global_errors,
        relative_position_errors=relative_position_errors,
        relative_orientation_errors=relative_orientation_errors,
        local_position_errors=local_position_errors,
    )

    expected = (
        0.40 * (1.0 - torch.exp(torch.tensor(-1.0)))
        + 0.20 * (1.0 - torch.exp(torch.tensor(-0.25)))
        + 0.20 * (1.0 - torch.exp(torch.tensor(-0.25)))
        + 0.20 * (1.0 - torch.exp(torch.tensor(-0.25)))
    )
    assert sampler._tracking_error_sum_delta.item() == pytest.approx(2.0 * expected.item())
    assert sampler._tracking_error_count_delta.item() == 2.0


def test_distributed_updates_wait_for_sync_and_consume_deltas_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _sampler((50,))
    bin_ids = torch.zeros(50, dtype=torch.long)
    sampler.record_exposure(bin_ids)
    sampler.record_tracking_error(bin_ids, torch.zeros(50, 2))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op: None)

    assert not sampler.update(sync_across_ranks=False)
    assert sampler.total_exposure.item() == 0.0
    assert sampler._exposure_delta.item() == pytest.approx(1.0)

    assert sampler.update(sync_across_ranks=True)
    assert sampler.total_exposure.item() == pytest.approx(1.0)
    assert sampler._exposure_delta.item() == 0.0

    assert sampler.update(sync_across_ranks=True)
    assert sampler.total_exposure.item() == pytest.approx(1.0)


def test_probability_cap_is_relative_to_frame_coverage() -> None:
    sampler = _sampler((5000,), probability_cap_ratio=2.0)
    sampler.total_exposure[:] = sampler.cfg.uncertainty_exposure
    sampler.difficulty_fast.zero_()
    sampler.difficulty_fast[0] = 1.0
    sampler.difficulty_slow.copy_(sampler.difficulty_fast)

    probabilities = sampler.distribution()
    coverage = torch.full((100,), 0.01)

    assert probabilities.sum().item() == pytest.approx(1.0)
    assert torch.all(probabilities <= 2.0 * coverage + 1.0e-7)
    assert torch.all(probabilities >= sampler.cfg.coverage_fraction * coverage - 1.0e-7)
    assert probabilities.max().item() == pytest.approx(2.0 * coverage[0].item())


def test_active_distribution_conditions_the_global_distribution() -> None:
    sampler = _sampler((200,))
    sampler.total_exposure[:] = sampler.cfg.uncertainty_exposure
    sampler.difficulty_fast[:] = torch.tensor([0.1, 0.3, 0.6, 1.0])
    sampler.difficulty_slow.copy_(sampler.difficulty_fast)
    active_bin_ids = torch.tensor([0, 2])

    global_probabilities = sampler.distribution()
    active_probabilities = sampler.distribution(active_bin_ids)
    expected = global_probabilities[active_bin_ids]
    expected /= expected.sum()

    torch.testing.assert_close(active_probabilities, expected)


def test_equal_motion_weighting_gives_each_motion_equal_coverage() -> None:
    sampler = _sampler((100, 50), equal_motion_weighting=True)

    torch.testing.assert_close(sampler.distribution(), torch.tensor([0.25, 0.25, 0.50]))


def test_active_mapping_is_vectorized_and_preserves_duplicate_motion_order() -> None:
    sampler = _sampler((100, 50))

    active_bins, local_motion_ids = sampler.active_mapping(torch.tensor([1, 1, 0]))

    torch.testing.assert_close(active_bins, torch.tensor([2, 2, 0, 1]))
    torch.testing.assert_close(local_motion_ids, torch.tensor([0, 1, 2, 2]))


def test_resampling_settles_terminal_exposure_and_invalidates_reset_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _sampler((8, 8), bin_size=8)
    lengths = torch.tensor([8, 8])
    command = SimpleNamespace(
        device=torch.device("cpu"),
        adaptive_sampler=sampler,
        motion=SimpleNamespace(global_ids=torch.tensor([0, 1]), lengths=lambda ids: lengths[ids]),
        motion_ids=torch.tensor([0, 1]),
        motion_lengths=lengths.clone(),
        time_steps=torch.tensor([7, 7]),
        _has_sampled=torch.ones(2, dtype=torch.bool),
        _quality_valid=torch.ones(2, dtype=torch.bool),
        _env=SimpleNamespace(
            termination_manager=SimpleNamespace(terminated=torch.tensor([True, False])),
        ),
        _fixed_motion_ids=None,
        active_sampling_probabilities=torch.tensor([0.5, 0.5]),
        _active_bin_ids=torch.tensor([0, 1]),
        _active_local_motion_ids=torch.tensor([0, 1]),
        motion_bin_counts=sampler.motion_bin_counts,
        motion_bin_offsets=sampler.motion_bin_offsets,
        bin_starts=sampler.bin_starts,
        bin_ends=sampler.bin_ends,
        cfg=SimpleNamespace(
            playback_start_frame=None,
            start_at_motion_beginning=True,
            adaptive_sampling=sampler.cfg,
        ),
    )
    command._bucket_ids = lambda motion_ids, time_steps: sampler.bucket_ids(
        command.motion.global_ids[motion_ids], time_steps
    )
    command._restore_sampling_diagnostics = lambda env_ids: None
    monkeypatch.setattr(torch, "multinomial", lambda probabilities, num_samples, replacement: torch.tensor([0, 1]))

    MotionCommandV1._adaptive_sampling(command, torch.tensor([0, 1]))

    torch.testing.assert_close(sampler._exposure_delta, torch.tensor([1 / 8, 1 / 8]))
    torch.testing.assert_close(sampler._failure_delta, torch.tensor([1.0, 0.0]))
    assert not torch.any(command._quality_valid)


def test_checkpoint_is_small_strict_and_round_trips() -> None:
    source = _sampler((100,))
    source.total_exposure[:] = torch.tensor([3.0, 7.0])
    source.difficulty_fast[:] = torch.tensor([0.2, 0.8])
    source.difficulty_slow[:] = torch.tensor([0.4, 0.9])
    expected_probabilities = source.distribution()
    state = source.state_dict()
    assert set(state) == {"version", "recipe", "total_exposure", "difficulty_fast", "difficulty_slow"}

    restored = _sampler((100,))
    assert restored.load_state_dict(state)
    torch.testing.assert_close(restored.distribution(), expected_probabilities)

    invalid = source.state_dict()
    invalid["version"] = torch.tensor(999)
    assert not restored.load_state_dict(invalid)

    changed_recipe = _sampler((100,), coverage_fraction=0.30)
    assert not changed_recipe.load_state_dict(source.state_dict())


def test_unknown_difficulty_sentinel_round_trips() -> None:
    source = _sampler((100,))
    state = source.state_dict()
    restored = _sampler((100,))

    assert restored.load_state_dict(state)
    torch.testing.assert_close(restored.difficulty_fast, torch.full((2,), -1.0))
    torch.testing.assert_close(restored.difficulty_slow, torch.full((2,), -1.0))
    torch.testing.assert_close(restored.distribution(), source.distribution())

    old_version = source.state_dict()
    old_version["version"] = torch.tensor(1)
    assert not restored.load_state_dict(old_version)


def test_command_config_selects_only_the_new_command() -> None:
    cfg = MotionCommandV1Cfg()

    assert cfg.class_type is MotionCommandV1
    assert isinstance(cfg.adaptive_sampling, AdaptiveSamplerV1Cfg)
    assert cfg.motion_catalog_cache is None
    assert cfg.motion_shard_across_ranks is False
    assert not hasattr(cfg, "curriculum_sampling_enabled")


def test_frame_zero_relative_targets_can_refresh_without_advancing_time() -> None:
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    command = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        cfg=SimpleNamespace(body_names=["anchor", "hand"]),
        time_steps=torch.zeros(2, dtype=torch.long),
        anchor_pos_w=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]]),
        anchor_quat_w=identity.repeat(2, 1),
        robot_anchor_pos_w=torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
        robot_anchor_quat_w=identity.repeat(2, 1),
        body_pos_w=torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]],
                [[0.0, 0.0, 3.0], [0.0, 2.0, 4.0]],
            ]
        ),
        body_quat_w=identity.repeat(2, 2, 1),
        body_pos_relative_w=torch.full((2, 2, 3), -99.0),
        body_quat_relative_w=torch.zeros(2, 2, 4),
    )

    MotionCommandV1.refresh_relative_body_targets(command)

    torch.testing.assert_close(
        command.body_pos_relative_w,
        torch.tensor(
            [
                [[5.0, 6.0, 1.0], [6.0, 6.0, 2.0]],
                [[8.0, 9.0, 3.0], [8.0, 11.0, 4.0]],
            ]
        ),
    )
    torch.testing.assert_close(command.body_quat_relative_w, identity.repeat(2, 2, 1))
    torch.testing.assert_close(command.time_steps, torch.zeros(2, dtype=torch.long))


def test_resample_primes_delayed_actuator_target_with_reset_pose() -> None:
    calls: dict[str, torch.Tensor] = {}

    delayed_actuator = DelayedImplicitActuator.__new__(DelayedImplicitActuator)
    delayed_actuator._joint_indices = torch.tensor([1])
    delayed_actuator.positions_delay_buffer = SimpleNamespace(
        seed=lambda value, env_ids: (
            calls.__setitem__("delay_seed", value.clone()),
            calls.__setitem__("delay_seed_env_ids", env_ids.clone()),
        )
    )

    class _Actuators(SimpleNamespace):
        def values(self):
            return (delayed_actuator,)

    class _Robot:
        data = SimpleNamespace(
            soft_joint_pos_limits=SimpleNamespace(
                torch=torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]] * 2),
            )
        )
        actuators = _Actuators(
            target_command=SimpleNamespace(
                set_position_index=lambda *, value, env_ids: (
                    calls.__setitem__("target", value.clone()),
                    calls.__setitem__("target_env_ids", env_ids.clone()),
                )
            )
        )

        def write_joint_state_to_sim_index(self, *, position, velocity, env_ids) -> None:
            calls["position"] = position.clone()
            calls["velocity"] = velocity.clone()
            calls["state_env_ids"] = env_ids.clone()

        def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids) -> None:
            pass

        def write_root_com_velocity_to_sim_index(self, *, root_velocity, env_ids) -> None:
            pass

    command = SimpleNamespace(
        device=torch.device("cpu"),
        cfg=SimpleNamespace(
            pose_range={name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")},
            velocity_range={name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")},
            joint_position_range=(0.0, 0.0),
        ),
        body_pos_w=torch.zeros(2, 1, 3),
        body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * 2),
        body_lin_vel_w=torch.zeros(2, 1, 3),
        body_ang_vel_w=torch.zeros(2, 1, 3),
        joint_pos=torch.tensor([[0.0, 0.0], [2.0, -2.0]]),
        joint_vel=torch.tensor([[0.0, 0.0], [0.3, -0.4]]),
        robot=_Robot(),
        _adaptive_sampling=lambda env_ids: None,
        _refresh_motion_cache=lambda env_ids: None,
        _apply_fixed_joint_state=lambda position, velocity: position[:, 1].fill_(0.25),
    )

    MotionCommandV1._resample_command(command, [1])

    expected_position = torch.tensor([[1.0, 0.25]])
    torch.testing.assert_close(calls["position"], expected_position)
    torch.testing.assert_close(calls["target"], expected_position)
    torch.testing.assert_close(calls["delay_seed"], expected_position[:, 1:2])
    torch.testing.assert_close(calls["velocity"], torch.tensor([[0.3, -0.4]]))
    torch.testing.assert_close(calls["target_env_ids"], torch.tensor([1]))
    torch.testing.assert_close(calls["delay_seed_env_ids"], torch.tensor([1]))
    torch.testing.assert_close(calls["state_env_ids"], torch.tensor([1]))


def test_command_rank_shards_are_deterministic_disjoint_and_complete() -> None:
    motion_files = [f"motion-{index}.npz" for index in range(10)]

    selections = [MotionCommandV1._select_rank_shard(motion_files, world_size=3, rank=rank) for rank in range(3)]

    assert [selection.global_ids for selection in selections] == [
        (0, 3, 6, 9),
        (1, 4, 7),
        (2, 5, 8),
    ]
    assert sorted(global_id for selection in selections for global_id in selection.global_ids) == list(range(10))

    global_rank_16 = MotionCommandV1._select_rank_shard(
        [f"motion-{index}.npz" for index in range(100)],
        world_size=24,
        rank=16,
    )
    assert global_rank_16.global_ids == (16, 40, 64, 88)


def test_command_motion_schema_descriptor_covers_fps_shapes_and_group_widths() -> None:
    command = SimpleNamespace(
        device=torch.device("cpu"),
        motion=SimpleNamespace(
            fps=50.0,
            world_size=24,
            fields=("joint_pos", "body_pos_w"),
            field_shapes={"joint_pos": (25,), "body_pos_w": (14, 3)},
            _group_widths={"state": 25, "pose": 42},
        ),
    )

    descriptor = MotionCommandV1._motion_schema_descriptor(command)

    assert torch.equal(
        descriptor,
        torch.tensor(
            [
                50_000_000,
                24,
                2,
                1,
                25,
                -1,
                2,
                14,
                3,
                2,
                42,
                25,
            ]
        ),
    )


def test_command_rejects_motion_shard_rank_metadata_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    command = SimpleNamespace(
        _adaptive_layout_checked=False,
        device=torch.device("cpu"),
        motion=SimpleNamespace(world_size=24, rank=16),
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 24)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 15)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op: None)

    with pytest.raises(RuntimeError, match="rank/world-size metadata"):
        MotionCommandV1._check_adaptive_distributed_layout(command)
