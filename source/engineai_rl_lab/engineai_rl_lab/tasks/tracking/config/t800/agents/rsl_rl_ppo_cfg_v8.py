from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v7_n3 import T800FlatV7N3PPORunnerCfg


@configclass
class T800FlatV8PPORunnerCfg(T800FlatV7N3PPORunnerCfg):
    """V8 runner using the V7-N3 agent architecture."""

    run_name = "v8"
