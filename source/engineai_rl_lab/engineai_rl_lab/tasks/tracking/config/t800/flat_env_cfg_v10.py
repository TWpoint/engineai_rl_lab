from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v9_n1 import (
    T800FlatWoStateEstimationEnvCfgV9N1,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV10(T800FlatWoStateEstimationEnvCfgV9N1):
    """V9-N1 baseline with SONIC-aligned adaptive-sampling parameters."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.uniform_sampling_rate = 0.1
        self.commands.motion.adp_samp_failure_rate_max_over_mean = 200.0
        self.commands.motion.pre_failure_sample_window = 200
