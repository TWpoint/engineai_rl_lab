from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v18 import T800FlatV18ScalePPORunnerCfg


@configclass
class T800FlatV19ScalePPORunnerCfg(T800FlatV18ScalePPORunnerCfg):
    """V18-compatible runner with a distinct V19 experiment identity."""

    run_name = "v19-scale"
