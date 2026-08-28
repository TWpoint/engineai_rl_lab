from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v22 import T800FlatV22ScalePPORunnerCfg


@configclass
class T800FlatV23ScalePPORunnerCfg(T800FlatV22ScalePPORunnerCfg):
    """V22 runner with a wider critic for joint-reference privileged state."""

    run_name = "v23-scale"
    actor = deepcopy(T800FlatV22ScalePPORunnerCfg().actor)
    # Keep only the Gaussian distribution's numerical positivity guard. The
    # 1e6 upper endpoint is the framework default and is effectively unbounded.
    actor.distribution_cfg.std_range = (1.0e-6, 1.0e6)
    critic = deepcopy(T800FlatV22ScalePPORunnerCfg().critic)
    critic.hidden_dims = [1024, 1024, 512, 512]
