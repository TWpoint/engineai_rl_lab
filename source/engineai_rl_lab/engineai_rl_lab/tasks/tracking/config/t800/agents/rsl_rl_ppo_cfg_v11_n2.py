from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v11_n1 import T800FlatV11N1PPORunnerCfg


@configclass
class T800FlatV11N2PPORunnerCfg(T800FlatV11N1PPORunnerCfg):
    """Two-block V11 actor with MarmotLab-aligned actor optimization."""

    run_name = "v11-n2"
    algorithm = deepcopy(T800FlatV11N1PPORunnerCfg().algorithm)
    algorithm.num_mini_batches = 64
    algorithm.actor_learning_rate = 2.0e-5


@configclass
class T800FlatV11N2ScaleLafanPPORunnerCfg(T800FlatV11N2PPORunnerCfg):
    """V11-N2 runner for the complete regular and 2x-slow LaFAN pool."""

    run_name = "v11-n2-scale-lafan"
