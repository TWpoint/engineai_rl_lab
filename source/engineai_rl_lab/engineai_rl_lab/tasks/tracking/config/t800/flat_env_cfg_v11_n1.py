from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v7_scale import (
    LAFAN_SCALE_MOTION_MANIFEST,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v9 import (
    V9_FRAME_OFFSETS,
    T800FlatObservationsCfgV9,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v10 import (
    T800FlatWoStateEstimationEnvCfgV10,
)


@configclass
class T800FlatObservationsCfgV11N1(T800FlatObservationsCfgV9):
    """V11 target-only command tokens for the structured V9 actor."""

    @configclass
    class CommandCfg(ObsGroup):
        link_pose_b = ObsTerm(
            func=mdp.motion_body_pose_b_window_by_entity_xz,
            params={"command_name": "motion", "frame_offsets": V9_FRAME_OFFSETS},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    command: CommandCfg = CommandCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV11N1(T800FlatWoStateEstimationEnvCfgV10):
    """V11 MDP and adaptive sampling with observations shaped for the V9 actor."""

    observations: T800FlatObservationsCfgV11N1 = T800FlatObservationsCfgV11N1()

    def __post_init__(self):
        super().__post_init__()
        # V10 inherits the V9-N1 MLP ablation, whose post-init flattens these
        # histories.  The V9 attention actor consumes them as temporal tokens.
        self.observations.policy.flatten_history_dim = False
        self.observations.action.flatten_history_dim = False


@configclass
class T800FlatWoStateEstimationEnvCfgV11N1ScaleLafan(T800FlatWoStateEstimationEnvCfgV11N1):
    """V11-N1 trained on the complete regular and 2x-slow LaFAN pool."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = LAFAN_SCALE_MOTION_MANIFEST
        self.commands.motion.motion_load_workers = 4
        self.commands.motion.motion_chunk_frames = 8_388_608
