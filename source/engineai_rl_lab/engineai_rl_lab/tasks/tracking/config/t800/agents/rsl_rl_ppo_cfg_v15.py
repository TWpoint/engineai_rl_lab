from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v14 import T800FlatV14ScalePPORunnerCfg


@configclass
class T800FlatV15ScalePPORunnerCfg(T800FlatV14ScalePPORunnerCfg):
    """V14-compatible runner for the horizon-corrected curriculum."""

    run_name = "v15-scale"
