from __future__ import annotations

import gymnasium as gym
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v17 import (
    T800FlatV17ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v18 import (
    T800FlatV18ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v17 import (
    T800FlatWoStateEstimationEnvCfgV17Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v18 import (
    T800FlatWoStateEstimationEnvCfgV18Scale,
)


def test_v18_adds_only_unified_credit_and_need_sampling_to_v17() -> None:
    v17 = T800FlatWoStateEstimationEnvCfgV17Scale().to_dict()
    v18_cfg = T800FlatWoStateEstimationEnvCfgV18Scale()
    v18 = v18_cfg.to_dict()

    motion = v18_cfg.commands.motion
    expected_overrides = {
        "curriculum_unified_window_enabled": True,
        "curriculum_start_eligibility_enabled": False,
        "curriculum_terminal_replay_fraction": 0.0,
        "curriculum_coverage_aware_enabled": False,
        "curriculum_need_uniform_fraction": 0.20,
        "curriculum_forward_risk_horizon_bins": 5,
        "curriculum_forward_risk_decay": 0.80,
        "curriculum_forward_risk_good_threshold": 0.05,
        "curriculum_forward_risk_bad_threshold": 0.60,
        "curriculum_quality_good_threshold": 0.08,
        "curriculum_quality_bad_threshold": 0.30,
        "curriculum_quarantine_need_scale": 0.25,
        "curriculum_shadow_iterations": 50,
        "curriculum_blend_iterations": 100,
    }
    for key, value in expected_overrides.items():
        assert getattr(motion, key) == value

    # V18 retains V17's successful-window body-position quality channel.
    assert motion.curriculum_quality_learning_enabled is True

    v17_motion = v17["commands"]["motion"]
    v18_motion = v18["commands"]["motion"]
    for key in expected_overrides:
        v17_motion[key] = v18_motion[key]
    assert v18 == v17


def test_v18_runner_only_changes_the_experiment_identity() -> None:
    v17 = T800FlatV17ScalePPORunnerCfg().to_dict()
    v18 = T800FlatV18ScalePPORunnerCfg().to_dict()

    assert v18["run_name"] == "v18-scale"
    v17["run_name"] = "v18-scale"
    assert v18 == v17


def test_v18_task_registration_points_to_v18_configs() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v18-scale")

    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v18:T800FlatWoStateEstimationEnvCfgV18Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".rsl_rl_ppo_cfg_v18:T800FlatV18ScalePPORunnerCfg")
