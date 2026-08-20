from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v11_n1 import (
    T800FlatWoStateEstimationEnvCfgV11N1,
    T800FlatWoStateEstimationEnvCfgV11N1ScaleLafan,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV11N2(T800FlatWoStateEstimationEnvCfgV11N1):
    """V11 target-only MDP paired with the N2 runner."""


@configclass
class T800FlatWoStateEstimationEnvCfgV11N2ScaleLafan(T800FlatWoStateEstimationEnvCfgV11N1ScaleLafan):
    """Complete LaFAN pool paired with the V11-N2 runner."""
