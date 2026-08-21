from __future__ import annotations

import gymnasium as gym
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v14 import (
    T800FlatV14ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v15 import (
    T800FlatV15ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v14 import (
    T800FlatWoStateEstimationEnvCfgV14Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v15 import (
    T800FlatWoStateEstimationEnvCfgV15Scale,
)


def test_v15_only_changes_the_reviewed_curriculum_controls() -> None:
    v14 = T800FlatWoStateEstimationEnvCfgV14Scale().to_dict()
    v15 = T800FlatWoStateEstimationEnvCfgV15Scale().to_dict()
    expected_changes = {
        "curriculum_start_horizon_frames": 50,
        "curriculum_exclude_invalid_failures": True,
        "curriculum_shadow_iterations": 100,
        "curriculum_blend_iterations": 50,
        "curriculum_min_known_fraction": 0.05,
        "curriculum_motion_length_exponent": 0.5,
        "curriculum_min_terminal_visits": 32,
        "curriculum_quarantine_terminal_hazard_threshold": 0.10,
        "curriculum_terminal_hazard_window_bins": 2,
        "curriculum_preserve_absent_state_budgets": True,
        "curriculum_probability_smoothing_alpha": 0.2,
        "curriculum_detailed_metrics": False,
    }
    for name, value in expected_changes.items():
        assert v15["commands"]["motion"][name] == value
        v14["commands"]["motion"][name] = value
    assert v15 == v14


def test_v15_runner_only_changes_the_run_name() -> None:
    v14 = T800FlatV14ScalePPORunnerCfg().to_dict()
    v15 = T800FlatV15ScalePPORunnerCfg().to_dict()

    assert v15["run_name"] == "v15-scale"
    v14["run_name"] = "v15-scale"
    assert v15 == v14


def test_v15_task_registration_points_to_v15_configs() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v15-scale")

    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v15:T800FlatWoStateEstimationEnvCfgV15Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".rsl_rl_ppo_cfg_v15:T800FlatV15ScalePPORunnerCfg")
