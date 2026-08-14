from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v7_n2 import T800FlatV7N2PPORunnerCfg


@configclass
class T800FlatV7N2ScalePPORunnerCfg(T800FlatV7N2PPORunnerCfg):
    """V7-N2 runner for the LaFAN regular/slow2x motion scale experiment."""

    run_name = "v7_n2_scale_lafan"
