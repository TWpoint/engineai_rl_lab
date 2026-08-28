from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v21 import T800FlatV21ScalePPORunnerCfg


@configclass
class T800FlatV22ScalePPORunnerCfg(T800FlatV21ScalePPORunnerCfg):
    """V21 runner paired with the V22 SONIC-style auxiliary rewards."""

    run_name = "v22-scale"
    algorithm = deepcopy(T800FlatV21ScalePPORunnerCfg().algorithm)
    algorithm.entropy_coef = 0.01
    algorithm.critic_learning_rate = 1.0e-3
    algorithm.shared_kl_adaptation = True
