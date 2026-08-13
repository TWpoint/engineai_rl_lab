from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg import T800FlatWoStateEstimationEnvCfg


@configclass
class T800FlatWoStateEstimationEnvCfgV1(T800FlatWoStateEstimationEnvCfg):
    """T800 flat environment without state estimation and with policy observation history."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10
