from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v9_n1 import T800FlatV9N1PPORunnerCfg


@configclass
class T800FlatV10PPORunnerCfg(T800FlatV9N1PPORunnerCfg):
    """Unchanged V9-N1 MLP and PPO settings for the V10 AS ablation."""

    run_name = "v10"
