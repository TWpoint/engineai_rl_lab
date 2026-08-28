from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v17 import T800FlatV17ScalePPORunnerCfg


@configclass
class T800FlatV18ScalePPORunnerCfg(T800FlatV17ScalePPORunnerCfg):
    """V17-compatible runner with a distinct V18 experiment identity."""

    run_name = "v18-scale"
