from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v9 import (
    V9_FRAME_OFFSETS,
    T800FlatObservationsCfgV9,
    T800FlatWoStateEstimationEnvCfgV9,
)


@configclass
class T800FlatObservationsCfgV9N1(T800FlatObservationsCfgV9):
    """V9 observations flattened into vectors for a plain MLP actor."""

    @configclass
    class CommandCfg(ObsGroup):
        link_pose_b = ObsTerm(
            func=mdp.motion_body_pose_and_error_b_window_flat,
            params={"command_name": "motion", "frame_offsets": V9_FRAME_OFFSETS},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    command: CommandCfg = CommandCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV9N1(T800FlatWoStateEstimationEnvCfgV9):
    """V9 MDP with all actor observation groups flattened for an MLP."""

    observations: T800FlatObservationsCfgV9N1 = T800FlatObservationsCfgV9N1()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.flatten_history_dim = True
        self.observations.action.flatten_history_dim = True
