from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v16 import T800FlatV16ScalePPORunnerCfg


@configclass
class T800FlatV17ScalePPORunnerCfg(T800FlatV16ScalePPORunnerCfg):
    """V16-compatible runner with a distinct V17 experiment identity."""

    run_name = "v17-scale"
