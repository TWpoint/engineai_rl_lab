from __future__ import annotations

import gymnasium as gym
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v13 import (
    T800FlatV13ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v14 import (
    T800FlatV14ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v13 import (
    T800FlatWoStateEstimationEnvCfgV13Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v14 import (
    T800FlatWoStateEstimationEnvCfgV14Scale,
)


def test_v14_only_enables_the_five_state_curriculum_over_v13() -> None:
    v13 = T800FlatWoStateEstimationEnvCfgV13Scale().to_dict()
    v14 = T800FlatWoStateEstimationEnvCfgV14Scale().to_dict()

    assert v13["commands"]["motion"]["curriculum_sampling_enabled"] is False
    assert v14["commands"]["motion"]["curriculum_sampling_enabled"] is True
    v13["commands"]["motion"]["curriculum_sampling_enabled"] = True
    assert v14 == v13


def test_v14_uses_the_reviewed_state_thresholds_and_sampling_weights() -> None:
    motion = T800FlatWoStateEstimationEnvCfgV14Scale().commands.motion

    assert motion.curriculum_shadow_iterations == 100
    assert motion.curriculum_blend_iterations == 50
    assert motion.curriculum_state_update_interval == 50
    assert motion.curriculum_min_window_trials == 8
    assert motion.curriculum_min_known_trials == 32
    assert motion.curriculum_min_quarantine_trials == 128
    assert motion.curriculum_min_exit_probe_trials == 32
    assert motion.curriculum_min_known_fraction == 0.10
    assert motion.curriculum_beta_prior_alpha == 1.0
    assert motion.curriculum_beta_prior_beta == 1.0
    assert motion.curriculum_mastered_enter_threshold == 0.10
    assert motion.curriculum_mastered_exit_threshold == 0.15
    assert motion.curriculum_stalled_enter_threshold == 0.80
    assert motion.curriculum_stalled_exit_threshold == 0.75
    assert motion.curriculum_quarantine_enter_threshold == 0.90
    assert motion.curriculum_quarantine_exit_threshold == 0.80
    assert motion.curriculum_no_progress_threshold == 0.02
    assert motion.curriculum_improvement_threshold == 0.05
    assert motion.curriculum_state_sampling_weights == {
        "unknown": 0.20,
        "mastered": 0.10,
        "frontier": 0.55,
        "stalled": 0.10,
        "quarantine": 0.05,
    }
    assert sum(motion.curriculum_state_sampling_weights.values()) == 1.0


def test_v14_runner_only_changes_the_run_name() -> None:
    v13 = T800FlatV13ScalePPORunnerCfg().to_dict()
    v14 = T800FlatV14ScalePPORunnerCfg().to_dict()

    assert v14["run_name"] == "v14-scale"
    v13["run_name"] = "v14-scale"
    assert v14 == v13


def test_v14_task_registration_points_to_v14_configs() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v14-scale")

    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v14:T800FlatWoStateEstimationEnvCfgV14Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".rsl_rl_ppo_cfg_v14:T800FlatV14ScalePPORunnerCfg")
