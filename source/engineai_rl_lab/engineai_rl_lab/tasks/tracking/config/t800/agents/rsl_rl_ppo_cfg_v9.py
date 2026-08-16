from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v8_n2 import T800FlatV8N2PPORunnerCfg


@configclass
class T800FlatV9PPORunnerCfg(T800FlatV8N2PPORunnerCfg):
    """V8-N2 policy and PPO settings for the V9 command observation."""

    run_name = "v9"


@configclass
class T800FlatV9ScalePPORunnerCfg(T800FlatV9PPORunnerCfg):
    """V9 runner for the rank-sharded LaFAN regular/slow2x experiment."""

    run_name = "v9_scale_lafan"
