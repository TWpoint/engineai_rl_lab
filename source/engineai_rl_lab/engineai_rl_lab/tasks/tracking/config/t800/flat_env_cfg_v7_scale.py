from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v7 import (
    T800FlatWoStateEstimationEnvCfgV7,
)


LAFAN_SCALE_MOTION_MANIFEST = "/mnt/data-1/lpz/t800_datasets/lafan_and_slow2x_v0.yaml"


@configclass
class T800FlatWoStateEstimationEnvCfgV7Scale(T800FlatWoStateEstimationEnvCfgV7):
    """T800 V7 environment trained on regular and 2x-slow LaFAN motions."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = LAFAN_SCALE_MOTION_MANIFEST
