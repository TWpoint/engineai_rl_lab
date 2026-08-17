from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v9 import T800FlatV9PPORunnerCfg


@configclass
class T800FlatV11N1PPORunnerCfg(T800FlatV9PPORunnerCfg):
    """V9 structured attention actor for the V11 target-only command."""

    run_name = "v11-n1"


@configclass
class T800FlatV11N1ScaleLafanPPORunnerCfg(T800FlatV11N1PPORunnerCfg):
    """V11-N1 runner for the complete regular and 2x-slow LaFAN pool."""

    run_name = "v11-n1-scale-lafan"
