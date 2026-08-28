from __future__ import annotations

import gymnasium as gym
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v18 import (
    T800FlatV18ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v19 import (
    T800FlatV19ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v18 import (
    T800FlatWoStateEstimationEnvCfgV18Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v19 import (
    T800FlatWoStateEstimationEnvCfgV19Scale,
)


def test_v19_selects_only_learnability_sampling_and_removes_joint_acc_from_v18() -> None:
    v18_cfg = T800FlatWoStateEstimationEnvCfgV18Scale()
    v18 = v18_cfg.to_dict()
    v19_cfg = T800FlatWoStateEstimationEnvCfgV19Scale()
    v19 = v19_cfg.to_dict()

    motion = v19_cfg.commands.motion
    expected_overrides = {"curriculum_sampling_strategy": "learnability"}
    for key, value in expected_overrides.items():
        assert getattr(motion, key) == value
    assert v18_cfg.commands.motion.curriculum_sampling_strategy == "need"
    assert not hasattr(motion, "curriculum_bounded_need_sampling_enabled")
    assert not hasattr(motion, "curriculum_bounded_sampling_entropy_floor")

    assert v18_cfg.rewards.joint_acc_l2 is not None
    assert v18_cfg.rewards.joint_acc_l2.weight == -2.5e-7
    assert v19_cfg.rewards.joint_acc_l2 is None

    v18_motion = v18["commands"]["motion"]
    v19_motion = v19["commands"]["motion"]
    for key in expected_overrides:
        v18_motion[key] = v19_motion[key]
    v18["rewards"]["joint_acc_l2"] = v19["rewards"]["joint_acc_l2"]
    assert v19 == v18


def test_v19_runner_only_changes_the_experiment_identity() -> None:
    v18 = T800FlatV18ScalePPORunnerCfg().to_dict()
    v19 = T800FlatV19ScalePPORunnerCfg().to_dict()

    assert v19["run_name"] == "v19-scale"
    v18["run_name"] = "v19-scale"
    assert v19 == v18


def test_v19_task_registration_points_to_v19_configs() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v19-scale")

    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v19:T800FlatWoStateEstimationEnvCfgV19Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".rsl_rl_ppo_cfg_v19:T800FlatV19ScalePPORunnerCfg")
