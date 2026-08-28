from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v21 import (
    T800FlatWoStateEstimationEnvCfgV21Scale,
)

T800_SONIC_END_EFFECTOR_BODY_NAMES = [
    "LINK_BASE",
    "LINK_WRIST_END_L",
    "LINK_WRIST_END_R",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
]
T800_SONIC_END_EFFECTOR_BODY_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]


@configclass
class T800FlatWoStateEstimationEnvCfgV22Scale(T800FlatWoStateEstimationEnvCfgV21Scale):
    """V21 plus SONIC-style local end-effector and relative body-pose rewards."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.motion_local_end_effector_pos = RewTerm(
            func=mdp.motion_local_body_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.1,
                "body_names": T800_SONIC_END_EFFECTOR_BODY_NAMES,
                "body_offsets": T800_SONIC_END_EFFECTOR_BODY_OFFSETS,
            },
        )
        self.rewards.motion_relative_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.3},
        )
        self.rewards.motion_relative_body_ori = RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        )
