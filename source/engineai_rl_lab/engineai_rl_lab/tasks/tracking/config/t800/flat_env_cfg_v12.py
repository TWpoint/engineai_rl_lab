from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v7_scale import (
    LAFAN_SCALE_MOTION_MANIFEST,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v11_n1 import (
    T800FlatObservationsCfgV11N1,
    T800FlatWoStateEstimationEnvCfgV11N1,
)

LAFAN_SONIC_MOTION_MANIFEST = "/mnt/data-1/lpz/t800_datasets/lafan_slow2x_and_sonic_v0.yaml"
T800_V0_MOTION_MANIFEST = "/mnt/data-1/lpz/t800_datasets/t800_v0.yaml"


@configclass
class T800FlatObservationsCfgV12(T800FlatObservationsCfgV11N1):
    """V11-N1 observations with an explicitly named proprioception group."""

    @configclass
    class ProprioceptionCfg(T800FlatObservationsCfgV11N1.PolicyCfg):
        base_lin_vel = None
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

    proprioception: ProprioceptionCfg = ProprioceptionCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV12(T800FlatWoStateEstimationEnvCfgV11N1):
    """V11 MDP with projected gravity added to the proprioception group."""

    observations: T800FlatObservationsCfgV12 = T800FlatObservationsCfgV12()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = None


@configclass
class T800FlatWoStateEstimationEnvCfgV12ScaleLafan(T800FlatWoStateEstimationEnvCfgV12):
    """V12 trained on the complete regular and 2x-slow LaFAN pool."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = LAFAN_SCALE_MOTION_MANIFEST
        self.commands.motion.motion_load_workers = 8
        self.commands.motion.motion_chunk_frames = 8_388_608


@configclass
class T800FlatWoStateEstimationEnvCfgV12Scale(T800FlatWoStateEstimationEnvCfgV12):
    """V12 trained on the complete T800 v0 motion manifest."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = T800_V0_MOTION_MANIFEST
        self.commands.motion.max_num_load_motions = 2048
        self.commands.motion.motion_load_workers = 8
        self.commands.motion.motion_chunk_frames = 8_388_608


@configclass
class T800FlatWoStateEstimationEnvCfgV12ScaleLafanSonic(T800FlatWoStateEstimationEnvCfgV12ScaleLafan):
    """V12 trained on the LaFAN regular/2x-slow and SONIC pool."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = LAFAN_SONIC_MOTION_MANIFEST
