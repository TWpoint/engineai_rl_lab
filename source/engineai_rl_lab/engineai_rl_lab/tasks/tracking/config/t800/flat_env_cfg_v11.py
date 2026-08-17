from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v9 import V9_FRAME_OFFSETS
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v9_n1 import (
    T800FlatObservationsCfgV9N1,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v10 import (
    T800FlatWoStateEstimationEnvCfgV10,
)


@configclass
class T800FlatObservationsCfgV11(T800FlatObservationsCfgV9N1):
    """Flattened V9 command targets with target-to-current errors removed."""

    @configclass
    class CommandCfg(ObsGroup):
        link_pose_b = ObsTerm(
            func=mdp.motion_body_pose_b_window_xz_flat,
            params={"command_name": "motion", "frame_offsets": V9_FRAME_OFFSETS},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    command: CommandCfg = CommandCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV11(T800FlatWoStateEstimationEnvCfgV10):
    """V10 adaptive sampling with a target-only 81D command per body."""

    observations: T800FlatObservationsCfgV11 = T800FlatObservationsCfgV11()
