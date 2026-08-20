from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v13 import T800FlatV13ScalePPORunnerCfg


@configclass
class T800FlatV14ScalePPORunnerCfg(T800FlatV13ScalePPORunnerCfg):
    """V13 runner paired with the five-state adaptive curriculum."""

    run_name = "v14-scale"
