from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v2 import (
    T800FlatObservationsCfgV2,
    T800FlatWoStateEstimationEnvCfgV2,
)


@configclass
class T800FlatObservationsCfgV3(T800FlatObservationsCfgV2):
    """V2 observations with the policy history shortened from ten to five frames."""

    @configclass
    class PolicyCfg(T800FlatObservationsCfgV2.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV3(T800FlatWoStateEstimationEnvCfgV2):
    """T800 V2 environment with a five-frame policy observation history."""

    observations: T800FlatObservationsCfgV3 = T800FlatObservationsCfgV3()
