from __future__ import annotations

import gymnasium as gym
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v16 import (
    T800FlatV16ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v17 import (
    T800FlatV17ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v16 import (
    T800FlatWoStateEstimationEnvCfgV16Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v17 import (
    T800FlatWoStateEstimationEnvCfgV17Scale,
)


def test_v17_adds_only_opt_in_quality_learning_to_v16() -> None:
    v16 = T800FlatWoStateEstimationEnvCfgV16Scale().to_dict()
    v17_cfg = T800FlatWoStateEstimationEnvCfgV17Scale()
    v17 = v17_cfg.to_dict()

    motion = v17_cfg.commands.motion
    assert motion.curriculum_quality_learning_enabled is True
    assert motion.curriculum_learning_pool_min_fraction == 0.15
    assert motion.curriculum_quality_filler_weight == 0.50
    assert motion.curriculum_quality_min_trials == 32
    assert motion.curriculum_quality_min_window_trials == 8
    assert motion.curriculum_quality_min_windows == 2
    assert motion.curriculum_quality_error_ema_alpha == 0.50
    assert motion.curriculum_quality_body_pos_std == 0.30

    v16_motion = v16["commands"]["motion"]
    v17_motion = v17["commands"]["motion"]
    for key in (
        "curriculum_quality_learning_enabled",
        "curriculum_learning_pool_min_fraction",
        "curriculum_quality_filler_weight",
        "curriculum_quality_min_trials",
        "curriculum_quality_min_window_trials",
        "curriculum_quality_min_windows",
        "curriculum_quality_error_ema_alpha",
        "curriculum_quality_body_pos_std",
    ):
        v16_motion[key] = v17_motion[key]
    assert v17 == v16


def test_v17_runner_only_changes_the_experiment_identity() -> None:
    v16 = T800FlatV16ScalePPORunnerCfg().to_dict()
    v17 = T800FlatV17ScalePPORunnerCfg().to_dict()

    assert v17["run_name"] == "v17-scale"
    v16["run_name"] = "v17-scale"
    assert v17 == v16


def test_v17_task_registration_points_to_v17_configs() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v17-scale")

    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v17:T800FlatWoStateEstimationEnvCfgV17Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".rsl_rl_ppo_cfg_v17:T800FlatV17ScalePPORunnerCfg")
