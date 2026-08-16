from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v7 import T800FlatObservationsCfgV7
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v8 import (
    T800FlatWoStateEstimationEnvCfgV8,
    T800FlatWoStateEstimationEnvCfgV8Scale,
)

V9_FRAME_OFFSETS = [-3, -2, -1, 0, 1, 2, 3, 4, 5]


@configclass
class T800FlatObservationsCfgV9(T800FlatObservationsCfgV7):
    """V8 observations with target-to-current pose errors in every command token."""

    @configclass
    class CommandCfg(ObsGroup):
        link_pose_b = ObsTerm(
            func=mdp.motion_body_pose_and_error_b_window_by_entity,
            params={"command_name": "motion", "frame_offsets": V9_FRAME_OFFSETS},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    command: CommandCfg = CommandCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV9(T800FlatWoStateEstimationEnvCfgV8):
    """V8 MDP with 162-dimensional target-and-error command tokens."""

    observations: T800FlatObservationsCfgV9 = T800FlatObservationsCfgV9()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.075


@configclass
class T800FlatWoStateEstimationEnvCfgV9Scale(T800FlatWoStateEstimationEnvCfgV8Scale):
    """V9 command observations with the rank-sharded LaFAN scale dataset."""

    observations: T800FlatObservationsCfgV9 = T800FlatObservationsCfgV9()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.075
