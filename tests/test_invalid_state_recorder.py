from __future__ import annotations

from types import SimpleNamespace

import torch
from engineai_rl_lab.tasks.tracking.mdp import recorders as recorder_module
from engineai_rl_lab.tasks.tracking.mdp.recorders import (
    InvalidRobotStateRecorderManagerCfg,
    InvalidRobotStateSnapshotRecorder,
    InvalidRobotStateSnapshotRecorderCfg,
)


class _Proxy:
    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


class _Actuators(dict):
    def __init__(self, target_position: torch.Tensor, output_position: torch.Tensor):
        super().__init__()
        self.target_command = SimpleNamespace(position=_Proxy(target_position))
        self.output_command = SimpleNamespace(position=_Proxy(output_position))


def _fake_recorder_env(num_envs: int = 2):
    num_bodies = 2
    num_joints = 3
    root_quat = torch.zeros(num_envs, 4)
    root_quat[:, 0] = 1.0
    body_quat = torch.zeros(num_envs, num_bodies, 4)
    body_quat[..., 0] = 1.0
    robot_data = SimpleNamespace(
        root_pos_w=_Proxy(torch.zeros(num_envs, 3)),
        root_quat_w=_Proxy(root_quat),
        root_lin_vel_w=_Proxy(torch.zeros(num_envs, 3)),
        root_ang_vel_w=_Proxy(torch.zeros(num_envs, 3)),
        body_pos_w=_Proxy(torch.zeros(num_envs, num_bodies, 3)),
        body_quat_w=_Proxy(body_quat),
        body_lin_vel_w=_Proxy(torch.zeros(num_envs, num_bodies, 3)),
        body_ang_vel_w=_Proxy(torch.zeros(num_envs, num_bodies, 3)),
        joint_pos=_Proxy(torch.zeros(num_envs, num_joints)),
        joint_vel=_Proxy(torch.zeros(num_envs, num_joints)),
        body_com_pos_b=_Proxy(torch.arange(num_envs * num_bodies * 3).reshape(num_envs, num_bodies, 3).float()),
        body_mass=_Proxy(torch.arange(num_envs * num_bodies).reshape(num_envs, num_bodies).float() + 1.0),
        body_inertia=_Proxy(torch.arange(num_envs * num_bodies * 9).reshape(num_envs, num_bodies, 9).float()),
        joint_pos_limits=_Proxy(torch.tensor([[[-1.0, 1.0]] * num_joints] * num_envs)),
        joint_vel_limits=_Proxy(torch.full((num_envs, num_joints), 10.0)),
    )
    target_position = torch.tensor([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])
    output_position = torch.tensor([[0.5, 1.5, 2.5], [10.5, 11.5, 12.5]])
    robot = SimpleNamespace(
        data=robot_data,
        joint_names=["j0", "j1", "j2"],
        body_names=["root", "tip"],
        actuators=_Actuators(target_position, output_position),
    )
    contact_sensor = SimpleNamespace(
        data=SimpleNamespace(
            net_forces_w=_Proxy(torch.arange(num_envs * num_bodies * 3).reshape(num_envs, num_bodies, 3).float())
        )
    )
    command = SimpleNamespace(
        motion_ids=torch.tensor([0, 1]),
        time_steps=torch.tensor([3, 7]),
        motion_lengths=torch.tensor([20, 30]),
        motion=SimpleNamespace(global_ids=torch.tensor([0, 1])),
        _motion_files=["motion-0.npz", "motion-1.npz"],
        cfg=SimpleNamespace(body_names=["root", "tip"]),
        joint_pos=torch.zeros(num_envs, num_joints),
        joint_vel=torch.zeros(num_envs, num_joints),
        body_pos_w=torch.zeros(num_envs, num_bodies, 3),
        body_quat_w=body_quat.clone(),
        body_lin_vel_w=torch.zeros(num_envs, num_bodies, 3),
        body_ang_vel_w=torch.zeros(num_envs, num_bodies, 3),
    )
    action_term = SimpleNamespace(
        processed_actions=torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]),
        _joint_names=["j0", "j1", "j2"],
    )
    termination_manager = SimpleNamespace(
        active_terms=["invalid_robot_state"],
        invalid=torch.tensor([False, False]),
    )
    termination_manager.get_term = lambda _name: termination_manager.invalid
    termination_manager.get_term_cfg = lambda _name: SimpleNamespace(params={})
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        scene={"robot": robot, "contact_forces": contact_sensor},
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        action_manager=SimpleNamespace(
            _terms={"joint_pos": action_term},
            action=torch.tensor([[0.01, 0.02, 0.03], [1.01, 1.02, 1.03]]),
            prev_action=torch.zeros(num_envs, num_joints),
        ),
        termination_manager=termination_manager,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        common_step_counter=0,
        _sim_step_counter=0,
        invalid_state_snapshot_context="checkpoint A/unsafe",
    )
    return env


def test_snapshot_recorder_cfg_has_concrete_class_type() -> None:
    cfg = InvalidRobotStateSnapshotRecorderCfg()
    assert cfg.class_type is InvalidRobotStateSnapshotRecorder
    assert not cfg.capture_pre_step_context
    assert not cfg.capture_finite_runaway
    assert cfg.finite_runaway_joint_pos_limit_margin == 1.0
    assert cfg.finite_runaway_joint_vel_limit_scale == 5.0
    assert not InvalidRobotStateRecorderManagerCfg().update_observations_before_recording


def test_first_control_step_snapshot_uses_public_actuator_command_views(tmp_path, monkeypatch) -> None:
    env = _fake_recorder_env()
    cfg = InvalidRobotStateSnapshotRecorderCfg(
        snapshot_dir=str(tmp_path),
        max_snapshots=0,
        capture_first_control_step=True,
    )
    recorder = InvalidRobotStateSnapshotRecorder(cfg, env)

    recorder.record_pre_step()
    env.episode_length_buf[:] = 1
    env.common_step_counter = 1
    env._sim_step_counter = 2
    recorder.record_post_step()
    recorder.record_post_step()

    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 1
    assert "first-control-step-ctx-checkpoint_A_unsafe-rank" in paths[0].name
    payload = torch.load(paths[0], weights_only=False)
    assert payload["snapshot_kind"] == "first-control-step"
    assert payload["context"] == "checkpoint A/unsafe"
    assert payload["pre_step"] is not None
    assert f"rank{payload['rank']:03d}-pid{payload['pid']}" in paths[0].name
    torch.testing.assert_close(
        payload["event_step"]["actuator"]["target_position"],
        torch.tensor([1.0, 2.0, 3.0]),
    )
    torch.testing.assert_close(
        payload["event_step"]["actuator"]["output_position"],
        torch.tensor([0.5, 1.5, 2.5]),
    )
    monkeypatch.setattr(
        recorder_module,
        "_clone_data_fields",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full-batch clone")),
    )
    recorder.record_pre_step()
    assert recorder._pre_step == {}
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_invalid_snapshot_skips_stale_explicit_reset_and_deduplicates(tmp_path, monkeypatch) -> None:
    env = _fake_recorder_env()
    cfg = InvalidRobotStateSnapshotRecorderCfg(snapshot_dir=str(tmp_path), max_snapshots=4)
    recorder = InvalidRobotStateSnapshotRecorder(cfg, env)
    # Event-only is the default: even with an active invalid-snapshot budget,
    # the recorder must not clone the full vectorized state on every step.
    monkeypatch.setattr(
        recorder_module,
        "_clone_data_fields",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full-batch clone")),
    )
    recorder.record_pre_step()
    assert recorder._pre_step == {}
    env.termination_manager.invalid[:] = True
    env.episode_length_buf[:] = torch.tensor([1, 0])
    env.common_step_counter = 8

    # The event-step builder must select a row directly, not clone every vectorized
    # robot tensor once for each invalid environment.
    recorder.record_pre_reset([0, 1])
    recorder.record_pre_reset([0, 1])

    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 1
    payload = torch.load(paths[0], weights_only=False)
    assert payload["env_id"] == 0
    assert payload["pid"] > 0
    assert f"rank{payload['rank']:03d}-pid{payload['pid']}" in paths[0].name
    torch.testing.assert_close(
        payload["event_step"]["robot_state"]["body_com_pos_b"],
        torch.arange(6).reshape(2, 3).float(),
    )
    torch.testing.assert_close(payload["event_step"]["robot_state"]["body_mass"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(payload["event_step"]["raw_action"], torch.tensor([0.01, 0.02, 0.03]))
    torch.testing.assert_close(payload["event_step"]["processed_action"], torch.tensor([0.1, 0.2, 0.3]))
    assert payload["pre_step"] is None
    legacy_payload = dict(payload)
    legacy_payload["post_step"] = legacy_payload.pop("event_step")
    assert recorder_module.diagnose_invalid_state_snapshot(legacy_payload) == payload["diagnosis"]

    # Explicit reset under a new checkpoint context still exposes the old
    # termination tensor, but episode_length==0 proves this is not a live event.
    env.invalid_state_snapshot_context = "checkpoint-B"
    env.episode_length_buf.zero_()
    recorder.record_pre_reset([0, 1])
    assert len(list(tmp_path.glob("*.pt"))) == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_finite_runaway_snapshot_uses_hard_limits_and_ignores_recoverable_velocity(tmp_path) -> None:
    env = _fake_recorder_env()
    cfg = InvalidRobotStateSnapshotRecorderCfg(
        snapshot_dir=str(tmp_path),
        max_snapshots=0,
        capture_pre_step_context=True,
        capture_finite_runaway=True,
        max_finite_runaway_snapshots=2,
    )
    recorder = InvalidRobotStateSnapshotRecorder(cfg, env)

    # Preserve a normal pre-step, then make env 0 exceed the upper hard limit by
    # more than the 1-rad diagnostic margin.  Env 1's 2.2x velocity transient is
    # intentionally below the default 5x runaway threshold.
    recorder.record_pre_step()
    env.scene["robot"].data.joint_pos.torch[0, 1] = 2.1
    env.scene["robot"].data.joint_vel.torch[1, 2] = 22.0
    env.episode_length_buf[:] = 4
    env.common_step_counter = 10
    env._sim_step_counter = 40
    recorder.record_post_step()
    recorder.record_post_step()

    paths = sorted(tmp_path.glob("*.pt"))
    assert len(paths) == 1
    assert paths[0].name.startswith("finite-runaway-")
    position_payload = torch.load(paths[0], weights_only=False)
    assert position_payload["snapshot_kind"] == "finite-runaway"
    assert position_payload["env_id"] == 0
    assert position_payload["diagnosis"]["primary_reason"] == "joint_pos_above_limit"
    assert position_payload["finite_runaway_thresholds"] == {
        "joint_pos_limit_margin": 1.0,
        "joint_vel_limit_scale": 5.0,
    }
    torch.testing.assert_close(position_payload["pre_step"]["robot_state"]["joint_pos"], torch.zeros(3))
    assert position_payload["event_step"]["robot_state"]["joint_pos"][1] == 2.1
    torch.testing.assert_close(
        position_payload["event_step"]["contact"]["net_forces_w"],
        torch.arange(6).reshape(2, 3).float(),
    )
    torch.testing.assert_close(position_payload["pre_step"]["raw_action"], torch.tensor([0.01, 0.02, 0.03]))

    # A later, distinct event crosses the 5x hard velocity limit and consumes the
    # finite-runaway budget independently of the invalid snapshot budget.
    env.scene["robot"].data.joint_pos.torch.zero_()
    env.scene["robot"].data.joint_vel.torch.zero_()
    recorder.record_pre_step()
    env.scene["robot"].data.joint_vel.torch[1, 2] = 50.1
    env.common_step_counter = 11
    env._sim_step_counter = 44
    recorder.record_post_step()

    paths = sorted(tmp_path.glob("*.pt"))
    assert len(paths) == 2
    velocity_payload = next(torch.load(path, weights_only=False) for path in paths if "env00001" in path.name)
    assert velocity_payload["diagnosis"]["primary_reason"] == "joint_velocity_limit"
    assert recorder._finite_runaway_snapshot_count == 2
    assert recorder._invalid_snapshot_count == 0
    assert not list(tmp_path.glob(".*.tmp"))


def test_finite_runaway_requires_every_post_step_state_tensor_to_be_finite(tmp_path) -> None:
    env = _fake_recorder_env()
    cfg = InvalidRobotStateSnapshotRecorderCfg(
        snapshot_dir=str(tmp_path),
        max_snapshots=0,
        capture_finite_runaway=True,
    )
    recorder = InvalidRobotStateSnapshotRecorder(cfg, env)

    recorder.record_pre_step()
    env.scene["robot"].data.joint_pos.torch[0, 0] = 2.1
    env.scene["robot"].data.root_pos_w.torch[0, 0] = torch.nan
    env.scene["robot"].data.joint_vel.torch[1, 0] = 50.1
    env.scene["robot"].data.body_quat_w.torch[1, 0, 0] = torch.inf
    env.episode_length_buf[:] = 2
    env.common_step_counter = 3
    recorder.record_post_step()

    assert not list(tmp_path.glob("*.pt"))
    assert recorder._finite_runaway_snapshot_count == 0


def test_first_finite_runaway_and_invalid_snapshot_budgets_are_independent(tmp_path) -> None:
    env = _fake_recorder_env()
    cfg = InvalidRobotStateSnapshotRecorderCfg(
        snapshot_dir=str(tmp_path),
        max_snapshots=1,
        capture_first_control_step=True,
        capture_finite_runaway=True,
        max_finite_runaway_snapshots=1,
    )
    recorder = InvalidRobotStateSnapshotRecorder(cfg, env)

    recorder.record_pre_step()
    env.scene["robot"].data.joint_pos.torch[0, 0] = 2.1
    env.episode_length_buf[:] = 1
    env.common_step_counter = 1
    recorder.record_post_step()
    recorder.record_post_step()

    env.termination_manager.invalid[0] = True
    recorder.record_pre_reset([0])
    recorder.record_pre_reset([0])

    payloads = [torch.load(path, weights_only=False) for path in tmp_path.glob("*.pt")]
    assert {payload["snapshot_kind"] for payload in payloads} == {
        "first-control-step",
        "finite-runaway",
        "invalid",
    }
    assert len(payloads) == 3
    assert recorder._finite_runaway_snapshot_count == 1
    assert recorder._invalid_snapshot_count == 1


def test_finite_runaway_capture_can_be_disabled(tmp_path, monkeypatch) -> None:
    env = _fake_recorder_env()
    cfg = InvalidRobotStateSnapshotRecorderCfg(
        snapshot_dir=str(tmp_path),
        max_snapshots=0,
        capture_finite_runaway=False,
    )
    recorder = InvalidRobotStateSnapshotRecorder(cfg, env)
    monkeypatch.setattr(
        recorder,
        "_finite_runaway_env_ids",
        lambda: (_ for _ in ()).throw(AssertionError("full-batch finite-state scan")),
    )

    recorder.record_pre_step()
    env.scene["robot"].data.joint_pos.torch[0, 0] = 2.1
    env.episode_length_buf[:] = 1
    env.common_step_counter = 1
    recorder.record_post_step()

    assert recorder._pre_step == {}
    assert not list(tmp_path.glob("*.pt"))
