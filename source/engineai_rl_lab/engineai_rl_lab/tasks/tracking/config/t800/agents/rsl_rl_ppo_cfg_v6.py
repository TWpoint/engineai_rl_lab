from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v5 import T800FlatV5PPORunnerCfg


@configclass
class T800FlatV6PPORunnerCfg(T800FlatV5PPORunnerCfg):
    """V5 model graph trained with a temporal key-body command window."""

    run_name = "v6"
