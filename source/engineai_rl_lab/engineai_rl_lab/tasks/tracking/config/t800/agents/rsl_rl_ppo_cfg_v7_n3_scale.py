from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v7_n3 import T800FlatV7N3PPORunnerCfg


@configclass
class T800FlatV7N3ScalePPORunnerCfg(T800FlatV7N3PPORunnerCfg):
    """V7-N3 runner for the LaFAN regular/slow2x motion scale experiment."""

    run_name = "v7_n3_scale_lafan"
