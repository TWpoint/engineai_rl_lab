from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v10 import T800FlatV10PPORunnerCfg


@configclass
class T800FlatV11PPORunnerCfg(T800FlatV10PPORunnerCfg):
    """V10 MLP and PPO settings for the target-only V11 command ablation."""

    run_name = "v11"
