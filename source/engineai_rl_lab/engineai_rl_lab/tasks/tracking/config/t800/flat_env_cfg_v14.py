from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v13 import (
    T800FlatWoStateEstimationEnvCfgV13Scale,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV14Scale(T800FlatWoStateEstimationEnvCfgV13Scale):
    """V13-scale with the five-state adaptive curriculum enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.curriculum_sampling_enabled = True
