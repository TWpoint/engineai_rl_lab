from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v4 import T800FlatV4PPORunnerCfg


@configclass
class T800FlatV5PPORunnerCfg(T800FlatV4PPORunnerCfg):
    """V4 model graph trained with key-body pose commands."""

    run_name = "v5"
