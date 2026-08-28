from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg import T800FlatPPORunnerCfg
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v25 import T800FlatWoStateEstimationEnvCfgV25Scale
from engineai_rl_lab.tasks.tracking.mdp.rewards import joint_pos_limits_capped
from engineai_rl_lab.tasks.tracking.mdp.terminations import nonfinite_robot_state
from engineai_rl_lab.tasks.tracking.tracking_env_cfg import TrackingPhysicsCfg
from engineai_rl_lab.utils.finite_reset import (
    drop_nonfinite_episode_reward_logs,
)
from engineai_rl_lab.utils.my_on_policy_runner import (
    accumulate_invalid_state_statistics,
    accumulate_rollout_extrema,
    create_rollout_extrema,
    invalid_state_diagnostic_metrics,
    processed_action_tensors,
    resolve_reward_term_names,
    rollout_extrema_metrics,
)
from rsl_rl.utils import check_nan
from tensordict import TensorDict

from isaaclab.managers import RewardManager, RewardTermCfg


class _Proxy:
    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


def _reward_column(env: SimpleNamespace, column: int) -> torch.Tensor:
    return env.reward_values[:, column]


def _fake_robot_env() -> tuple[SimpleNamespace, SimpleNamespace]:
    num_envs = 2
    num_bodies = 3
    num_joints = 2
    root_quat = torch.zeros(num_envs, 4)
    root_quat[:, 0] = 1.0
    body_quat = torch.zeros(num_envs, num_bodies, 4)
    body_quat[..., 0] = 1.0
    data = SimpleNamespace(
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
        joint_pos_limits=_Proxy(torch.tensor([[[-1.0, 1.0]] * num_joints] * num_envs)),
        soft_joint_pos_limits=_Proxy(torch.tensor([[[-0.9, 0.9]] * num_joints] * num_envs)),
        joint_vel_limits=_Proxy(torch.full((num_envs, num_joints), 10.0)),
        soft_joint_vel_limits=_Proxy(torch.full((num_envs, num_joints), 10.0)),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        scene={"robot": SimpleNamespace(data=data)},
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=slice(None))
    return env, asset_cfg


def test_newton_tracking_preset_preserves_validated_contact_settings() -> None:
    newton_cfg = TrackingPhysicsCfg().newton_mjwarp
    assert newton_cfg.num_substeps == 1
    assert newton_cfg.default_shape_cfg.margin == pytest.approx(0.01)
    env_cfg = T800FlatWoStateEstimationEnvCfgV25Scale()
    assert env_cfg.events.physics_material.params["restitution_range"] == (
        0.0,
        0.5,
    )
    assert env_cfg.events.base_com is not None
    assert env_cfg.events.base_com.params["com_range"] == {
        "x": (-0.025, 0.025),
        "y": (-0.05, 0.05),
        "z": (-0.05, 0.05),
    }


def test_tracking_policy_actions_are_not_globally_clipped() -> None:
    # Several arm joints use a 0.05 action scale while valid reference poses
    # exceed 1 rad.  A global clip of 5 would make those poses unreachable.
    assert T800FlatPPORunnerCfg().clip_actions is None


def test_v25_uses_bounded_joint_limit_penalty() -> None:
    cfg = T800FlatWoStateEstimationEnvCfgV25Scale()
    assert cfg.rewards.joint_limit.func is joint_pos_limits_capped
    assert cfg.rewards.joint_limit.params["max_total_error"] == pytest.approx(1.0)
    assert set(cfg.terminations.invalid_robot_state.params) == {"asset_cfg"}


def test_invalid_robot_state_does_not_relabel_large_finite_joint_state() -> None:
    env, asset_cfg = _fake_robot_env()
    env.scene["robot"].data.joint_pos.torch[0, 0] = 1.6
    env.scene["robot"].data.joint_vel.torch[1, 0] = 1000.0

    invalid = nonfinite_robot_state(env, asset_cfg)

    torch.testing.assert_close(invalid, torch.tensor([False, False]))


def test_invalid_robot_state_catches_nonfinite_joint_state() -> None:
    env, asset_cfg = _fake_robot_env()
    env.scene["robot"].data.joint_vel.torch[1, 0] = float("nan")

    invalid = nonfinite_robot_state(env, asset_cfg)

    torch.testing.assert_close(invalid, torch.tensor([False, True]))


def test_invalid_robot_state_still_catches_nonfinite_body_state() -> None:
    env, asset_cfg = _fake_robot_env()
    env.scene["robot"].data.body_ang_vel_w.torch[1, 2, 0] = float("inf")

    invalid = nonfinite_robot_state(env, asset_cfg)

    torch.testing.assert_close(invalid, torch.tensor([False, True]))


def test_joint_limit_penalty_preserves_small_errors_and_caps_total_error() -> None:
    env, asset_cfg = _fake_robot_env()
    env.scene["robot"].data.joint_pos.torch[0] = torch.tensor([1.0, -1.2])
    env.scene["robot"].data.joint_pos.torch[1] = torch.tensor([1.0e9, -1.0e9])

    penalty = joint_pos_limits_capped(env, asset_cfg)

    assert penalty[0].item() == pytest.approx(0.4)
    assert penalty[1].item() == pytest.approx(1.0)


def test_joint_limit_penalty_caps_nan_and_honors_joint_subset() -> None:
    env, asset_cfg = _fake_robot_env()
    env.scene["robot"].data.joint_pos.torch[0] = torch.tensor([float("nan"), 1.0e9])
    env.scene["robot"].data.joint_pos.torch[1] = torch.tensor([0.0, 1.0e9])
    asset_cfg.joint_ids = [0]

    penalty = joint_pos_limits_capped(env, asset_cfg, max_total_error=0.3)

    assert penalty[0].item() == pytest.approx(0.3)
    assert penalty[1].item() == pytest.approx(0.0)


def test_terminal_nonfinite_rewards_never_enter_episode_sums() -> None:
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        sim=SimpleNamespace(is_playing=lambda: True),
        max_episode_length_s=10.0,
        reset_buf=torch.zeros(3, dtype=torch.bool),
        reward_values=torch.ones(3, 2),
    )
    manager = RewardManager(
        {
            "first": RewardTermCfg(func=_reward_column, weight=1.0, params={"column": 0}),
            "second": RewardTermCfg(func=_reward_column, weight=1.0, params={"column": 1}),
        },
        env,
    )

    manager.compute(dt=0.1)
    episode_sum_before_bad_step = {name: value.clone() for name, value in manager._episode_sums.items()}
    env.reset_buf[:] = torch.tensor([True, False, True])
    env.reward_values[:] = torch.tensor(
        [
            [float("nan"), 2.0],
            [float("nan"), 2.0],
            [torch.finfo(torch.float32).max, torch.finfo(torch.float32).max],
        ]
    )

    rewards = manager.compute(dt=1.0)

    assert rewards[0].item() == pytest.approx(0.0)
    assert torch.isnan(rewards[1])
    assert rewards[2].item() == pytest.approx(0.0)
    torch.testing.assert_close(manager._step_reward[[0, 2]], torch.zeros(2, 2))
    for name, episode_sum in manager._episode_sums.items():
        torch.testing.assert_close(episode_sum[[0, 2]], episode_sum_before_bad_step[name][[0, 2]])

    logs = manager.reset(env_ids=torch.tensor([0, 2]))
    assert all(torch.isfinite(value) for value in logs.values())


def test_nonfinite_episode_reward_logs_are_dropped_not_zero_filled() -> None:
    extras = {
        "log": {
            "Episode_Reward/good": torch.tensor(1.25),
            "Episode_Reward/nan": float("nan"),
            "Episode_Reward/inf": torch.tensor([2.0, float("inf")]),
            "Metrics/keep": float("nan"),
        }
    }

    count = drop_nonfinite_episode_reward_logs(extras)

    assert count == 2
    assert extras["log"]["Episode_Reward/good"].item() == pytest.approx(1.25)
    assert "Episode_Reward/nan" not in extras["log"]
    assert "Episode_Reward/inf" not in extras["log"]
    assert extras["log"]["Diagnostics/nonfinite_episode_reward_terms"] == pytest.approx(2.0)
    assert torch.isnan(torch.tensor(extras["log"]["Metrics/keep"]))


def test_invalid_state_diagnostics_report_events_and_normalized_rates() -> None:
    metrics = invalid_state_diagnostic_metrics(torch.tensor([2.0, 8.0, 2.0e6], dtype=torch.float64))

    assert metrics["Diagnostics/invalid_robot_state_events"].item() == pytest.approx(2.0)
    assert metrics["Diagnostics/invalid_robot_state_events_per_million_transitions"].item() == pytest.approx(1.0)
    assert metrics["Diagnostics/invalid_robot_state_fraction_of_terminations"].item() == pytest.approx(0.25)


def test_invalid_state_statistics_count_current_events_not_latched_history() -> None:
    statistics = torch.zeros(3, dtype=torch.float64)
    termination_manager = SimpleNamespace(current=torch.tensor([True, False]))
    termination_manager.get_term = lambda _name: termination_manager.current
    termination_manager._last_episode_dones = torch.tensor([True, False])

    accumulate_invalid_state_statistics(statistics, termination_manager, torch.tensor([True, False]))
    termination_manager.current = torch.tensor([False, False])
    accumulate_invalid_state_statistics(statistics, termination_manager, torch.tensor([False, False]))

    torch.testing.assert_close(statistics, torch.tensor([1.0, 1.0, 4.0], dtype=torch.float64))


def test_rollout_extrema_keep_finite_maxima_and_flag_nonfinite_values() -> None:
    extrema = create_rollout_extrema(("global/reward", "global reward", ""), "cpu")

    with torch.inference_mode():
        accumulate_rollout_extrema(
            extrema,
            reward_rates=torch.tensor([[float("nan"), -2.0, float("inf")], [3.0, -4.0, -5.0]]),
            policy_actions=torch.tensor([[float("nan"), -6.0]]),
            processed_actions=(torch.tensor([[2.0, float("-inf")]]), torch.tensor([[-7.0]])),
        )
        accumulate_rollout_extrema(
            extrema,
            reward_rates=torch.tensor([[-8.0, 1.0, -6.0]]),
            policy_actions=torch.tensor([[5.0]]),
            processed_actions=(torch.tensor([[3.0]]),),
        )

    metrics = rollout_extrema_metrics(extrema)
    assert metrics["Diagnostics/reward_rate_max_abs/global_reward"].item() == pytest.approx(8.0)
    assert metrics["Diagnostics/reward_rate_max_abs/global_reward_1"].item() == pytest.approx(4.0)
    assert metrics["Diagnostics/reward_rate_max_abs/term_2"].item() == pytest.approx(6.0)
    assert metrics["Diagnostics/reward_rate_nonfinite_detected/global_reward"].item() == pytest.approx(1.0)
    assert metrics["Diagnostics/reward_rate_nonfinite_detected/global_reward_1"].item() == pytest.approx(0.0)
    assert metrics["Diagnostics/reward_rate_nonfinite_detected/term_2"].item() == pytest.approx(1.0)
    assert metrics["Diagnostics/policy_action_max_abs"].item() == pytest.approx(6.0)
    assert metrics["Diagnostics/policy_action_nonfinite_detected"].item() == pytest.approx(1.0)
    assert metrics["Diagnostics/processed_action_max_abs"].item() == pytest.approx(7.0)
    assert metrics["Diagnostics/processed_action_nonfinite_detected"].item() == pytest.approx(1.0)


def test_rollout_extrema_resolve_reward_and_processed_action_api_variants() -> None:
    reward_manager = SimpleNamespace(_term_names=("known",), _step_reward=torch.zeros(2, 2))
    assert resolve_reward_term_names(reward_manager) == ("known", "term_1")

    direct_manager = SimpleNamespace(processed_actions=torch.tensor([[1.0, 2.0]]))
    direct_tensors = processed_action_tensors(direct_manager)
    assert len(direct_tensors) == 1
    assert direct_tensors[0] is direct_manager.processed_actions

    terms = {
        "first": SimpleNamespace(processed_actions=torch.tensor([[3.0]])),
        "second": SimpleNamespace(processed_actions=SimpleNamespace(torch=torch.tensor([[4.0]]))),
    }
    term_manager = SimpleNamespace(active_terms=("first", "second"), get_term=terms.__getitem__)
    tensors = processed_action_tensors(term_manager)
    assert len(tensors) == 2
    torch.testing.assert_close(tensors[0], torch.tensor([[3.0]]))
    torch.testing.assert_close(tensors[1], torch.tensor([[4.0]]))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_rsl_output_check_rejects_every_nonfinite_value(bad_value: float) -> None:
    observations = TensorDict({"actor": torch.tensor([[0.0, bad_value]])}, batch_size=[1])
    with pytest.raises(ValueError, match="non-finite"):
        check_nan(observations, torch.zeros(1), torch.zeros(1))
