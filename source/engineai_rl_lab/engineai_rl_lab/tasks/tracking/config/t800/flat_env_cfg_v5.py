from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v4 import (
    T800FlatObservationsCfgV4,
    T800FlatWoStateEstimationEnvCfgV4,
)


T800_KEY_BODY_NAMES = [
    "LINK_BASE",
    "LINK_HIP_ROLL_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_WAIST_YAW",
    "LINK_SHOULDER_ROLL_L",
    "LINK_ELBOW_YAW_L",
    "LINK_WRIST_END_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_ELBOW_YAW_R",
    "LINK_WRIST_END_R",
]


@configclass
class T800FlatObservationsCfgV5(T800FlatObservationsCfgV4):
    """V4 observations commanded by reference key-body positions and orientations."""

    @configclass
    class CommandCfg(ObsGroup):
        link_pos_b = ObsTerm(func=mdp.motion_body_pos_b, params={"command_name": "motion"})
        link_ori_b = ObsTerm(func=mdp.motion_body_ori_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(T800FlatObservationsCfgV4.PrivilegedCfg):
        target_link_pos_b = ObsTerm(func=mdp.motion_body_pos_b, params={"command_name": "motion"})
        target_link_ori_b = ObsTerm(func=mdp.motion_body_ori_b, params={"command_name": "motion"})

    command: CommandCfg = CommandCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV5(T800FlatWoStateEstimationEnvCfgV4):
    """T800 V5 environment using key-body pose commands instead of joint-state commands."""

    observations: T800FlatObservationsCfgV5 = T800FlatObservationsCfgV5()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.body_names = T800_KEY_BODY_NAMES
