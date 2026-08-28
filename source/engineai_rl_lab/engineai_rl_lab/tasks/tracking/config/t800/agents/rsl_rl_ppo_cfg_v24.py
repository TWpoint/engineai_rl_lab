from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v23 import T800FlatV23ScalePPORunnerCfg


@configclass
class T800FlatV24ScalePPORunnerCfg(T800FlatV23ScalePPORunnerCfg):
    """V23 single-critic PPO runner paired with the V24 environment."""

    run_name = "v24-scale"
    actor = deepcopy(T800FlatV23ScalePPORunnerCfg().actor)
    critic = deepcopy(T800FlatV23ScalePPORunnerCfg().critic)
    algorithm = deepcopy(T800FlatV23ScalePPORunnerCfg().algorithm)
