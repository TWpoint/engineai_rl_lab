from __future__ import annotations

import engineai_rl_lab.tasks  # noqa: F401
import gymnasium as gym
from engineai_rl_lab.tasks.tracking import mdp
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v22 import (
    T800FlatV22ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v21 import (
    T800FlatV21ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v21 import (
    T800FlatWoStateEstimationEnvCfgV21Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v22 import (
    T800_SONIC_END_EFFECTOR_BODY_NAMES,
    T800_SONIC_END_EFFECTOR_BODY_OFFSETS,
    T800FlatWoStateEstimationEnvCfgV22Scale,
)


def test_v22_adds_sonic_local_tracking_rewards_without_mutating_v21() -> None:
    v21 = T800FlatWoStateEstimationEnvCfgV21Scale()
    v22 = T800FlatWoStateEstimationEnvCfgV22Scale()

    assert not hasattr(v21.rewards, "motion_local_end_effector_pos")
    assert not hasattr(v21.rewards, "motion_relative_body_pos")
    assert not hasattr(v21.rewards, "motion_relative_body_ori")

    end_effector = v22.rewards.motion_local_end_effector_pos
    assert end_effector.func is mdp.motion_local_body_position_error_exp
    assert end_effector.weight == 1.0
    assert end_effector.params == {
        "command_name": "motion",
        "std": 0.1,
        "body_names": T800_SONIC_END_EFFECTOR_BODY_NAMES,
        "body_offsets": T800_SONIC_END_EFFECTOR_BODY_OFFSETS,
    }
    assert T800_SONIC_END_EFFECTOR_BODY_NAMES == [
        "LINK_BASE",
        "LINK_WRIST_END_L",
        "LINK_WRIST_END_R",
        "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_R",
    ]
    assert T800_SONIC_END_EFFECTOR_BODY_OFFSETS == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    body_pos = v22.rewards.motion_relative_body_pos
    assert body_pos.func is mdp.motion_relative_body_position_error_exp
    assert body_pos.weight == 0.5
    assert body_pos.params == {"command_name": "motion", "std": 0.3}

    body_ori = v22.rewards.motion_relative_body_ori
    assert body_ori.func is mdp.motion_relative_body_orientation_error_exp
    assert body_ori.weight == 0.5
    assert body_ori.params == {"command_name": "motion", "std": 0.4}


def test_v22_registration_and_runner() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v22-scale")

    assert spec.kwargs["env_cfg_entry_point"].endswith(
        ".flat_env_cfg_v22:T800FlatWoStateEstimationEnvCfgV22Scale"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        ".rsl_rl_ppo_cfg_v22:T800FlatV22ScalePPORunnerCfg"
    )
    v21 = T800FlatV21ScalePPORunnerCfg()
    v22 = T800FlatV22ScalePPORunnerCfg()

    assert v22.run_name == "v22-scale"
    assert v22.algorithm.entropy_coef == 0.01
    assert v22.algorithm.critic_learning_rate == 1.0e-3
    assert v22.algorithm.shared_kl_adaptation is True
    assert v21.algorithm.entropy_coef == 0.005
    assert v21.algorithm.critic_learning_rate == 2.0e-5
    assert v21.algorithm.shared_kl_adaptation is False
