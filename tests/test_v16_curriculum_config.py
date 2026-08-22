from __future__ import annotations

import gymnasium as gym
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v15 import (
    T800FlatV15ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v16 import (
    T800FlatV16ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v15 import (
    T800FlatWoStateEstimationEnvCfgV15Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v16 import (
    T800FlatWoStateEstimationEnvCfgV16Scale,
)


def test_v16_env_starts_from_the_reviewed_v15_baseline() -> None:
    v16 = T800FlatWoStateEstimationEnvCfgV16Scale()

    assert isinstance(v16, T800FlatWoStateEstimationEnvCfgV15Scale)
    assert v16.commands.motion.curriculum_sampling_enabled is True
    assert v16.commands.motion.curriculum_start_horizon_frames == 50
    assert v16.commands.motion.curriculum_exclude_invalid_failures is True
    assert v16.commands.motion.curriculum_preserve_absent_state_budgets is True
    assert v16.commands.motion.curriculum_start_eligibility_enabled is True
    assert v16.commands.motion.curriculum_terminal_replay_fraction == 0.05
    assert v16.commands.motion.curriculum_coverage_aware_enabled is True
    assert v16.commands.motion.curriculum_provisional_known_trials == 8
    assert v16.commands.motion.curriculum_min_provisional_known_fraction == 0.25
    assert v16.commands.motion.curriculum_min_provisional_motion_fraction == 0.70
    assert v16.commands.motion.curriculum_provisional_motion_bin_fraction == 0.10
    assert v16.commands.motion.curriculum_unconfirmed_min_coverage_ratio == 0.50
    assert v16.commands.motion.curriculum_state_focus_start_fraction == 0.10
    assert v16.commands.motion.curriculum_state_focus_end_fraction == 0.40
    assert v16.commands.motion.curriculum_blend_iterations == 100


def test_v16_runner_only_changes_the_experiment_identity() -> None:
    v15 = T800FlatV15ScalePPORunnerCfg().to_dict()
    v16 = T800FlatV16ScalePPORunnerCfg().to_dict()

    assert v16["run_name"] == "v16-scale"
    v15["run_name"] = "v16-scale"
    assert v16 == v15


def test_v16_task_registration_points_to_v16_configs() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v16-scale")

    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v16:T800FlatWoStateEstimationEnvCfgV16Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".rsl_rl_ppo_cfg_v16:T800FlatV16ScalePPORunnerCfg")
