from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v20 import T800FlatV20ScalePPORunnerCfg


@configclass
class T800FlatV21ScalePPORunnerCfg(T800FlatV20ScalePPORunnerCfg):
    """V20 scratch runner for the 23-DoF fixed-head task."""

    run_name = "v21-scale"
    actor = deepcopy(T800FlatV20ScalePPORunnerCfg().actor)
    algorithm = deepcopy(T800FlatV20ScalePPORunnerCfg().algorithm)
    algorithm.max_grad_norm = 0.1
