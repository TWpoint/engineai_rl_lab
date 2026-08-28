from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v20 import RslRlV20PpoAlgorithmCfg
from .rsl_rl_ppo_cfg_v24 import T800FlatV24ScalePPORunnerCfg

V25_REWARD_GROUPS = {
    "global": [
        "motion_global_root_height",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "motion_joint_pos",
        "motion_joint_vel",
    ],
    "local": [
        "motion_local_end_effector_pos",
        "motion_relative_body_pos",
        "motion_relative_body_ori",
    ],
    "regularization": [
        "alive",
        "action_rate_l2",
        "joint_limit",
    ],
}

V25_ACTOR_NUM_BLOCKS = 3


@configclass
class RslRlV25MultiCriticPpoAlgorithmCfg(RslRlV20PpoAlgorithmCfg):
    """V24 PPO hyperparameters with partitioned heads and compact value-loss logs."""

    class_name: str = "engineai_rl_lab.utils.v25_compact_multi_critic_ppo:V25CompactMultiCriticPPO"
    reward_groups: dict[str, list[str]] = deepcopy(V25_REWARD_GROUPS)
    value_loss_reduction: str = "mean"


_V25_ALGORITHM_CFG = T800FlatV24ScalePPORunnerCfg().algorithm.to_dict()
_V25_ALGORITHM_CFG.update(
    class_name="engineai_rl_lab.utils.v25_compact_multi_critic_ppo:V25CompactMultiCriticPPO",
    reward_groups=deepcopy(V25_REWARD_GROUPS),
    value_loss_reduction="mean",
)


@configclass
class T800FlatV25ScalePPORunnerCfg(T800FlatV24ScalePPORunnerCfg):
    """V24 settings with three actor attention blocks and a compact multi-head critic."""

    run_name = "v25-scale"
    actor = deepcopy(T800FlatV24ScalePPORunnerCfg().actor)
    actor.nodes["attention_blocks"]["cell"]["num_blocks"] = V25_ACTOR_NUM_BLOCKS
    critic = deepcopy(T800FlatV24ScalePPORunnerCfg().critic)
    algorithm = RslRlV25MultiCriticPpoAlgorithmCfg(**_V25_ALGORITHM_CFG)
