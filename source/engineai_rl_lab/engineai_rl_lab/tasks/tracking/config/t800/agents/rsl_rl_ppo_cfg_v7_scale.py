from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v7 import T800FlatV7PPORunnerCfg


@configclass
class T800FlatV7ScalePPORunnerCfg(T800FlatV7PPORunnerCfg):
    """V7 runner for the LaFAN regular/slow2x motion scale experiment."""

    run_name = "v7_scale_lafan"
