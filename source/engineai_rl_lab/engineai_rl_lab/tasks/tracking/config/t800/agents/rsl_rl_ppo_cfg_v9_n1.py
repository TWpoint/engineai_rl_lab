from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .rsl_rl_ppo_cfg_v9 import T800FlatV9PPORunnerCfg


@configclass
class T800FlatV9N1PPORunnerCfg(T800FlatV9PPORunnerCfg):
    """V9 PPO settings with a fully flattened, feed-forward MLP policy."""

    run_name = "v9-n1"
    obs_groups = {"actor": ["policy", "action", "command"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256],
        activation="elu",
        obs_normalization=False,
    )
