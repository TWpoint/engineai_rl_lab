from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v15 import T800FlatV15ScalePPORunnerCfg


@configclass
class T800FlatV16ScalePPORunnerCfg(T800FlatV15ScalePPORunnerCfg):
    """V15-compatible runner with a distinct V16 experiment identity."""

    run_name = "v16-scale"
