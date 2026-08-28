from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v25 import (
    V25_ACTOR_NUM_BLOCKS,
    V25_REWARD_GROUPS,
    RslRlV25MultiCriticPpoAlgorithmCfg,
    T800FlatV25ScalePPORunnerCfg,
)

V26_REWARD_GROUPS = deepcopy(V25_REWARD_GROUPS)
V26_ACTOR_NUM_BLOCKS = V25_ACTOR_NUM_BLOCKS


@configclass
class RslRlV26StarPpoAlgorithmCfg(RslRlV25MultiCriticPpoAlgorithmCfg):
    """V25 grouped critic with conservative current-rollout STAR reuse."""

    class_name: str = "engineai_rl_lab.utils.v26_star_multi_critic_ppo:V26StarMultiCriticPPO"
    reward_groups: dict[str, list[str]] = deepcopy(V26_REWARD_GROUPS)
    value_loss_reduction: str = "mean"
    star_priority_fraction: float = 0.125
    star_high_difficulty_threshold: float = 1.0
    star_top_fraction: float = 0.05
    star_difficulty_boundaries: tuple[float, ...] = (1.5, 2.0, 4.0)
    star_reuse_cap: int = 2
    star_priority_weight_cap: float = 8.0


_V26_ALGORITHM_CFG = T800FlatV25ScalePPORunnerCfg().algorithm.to_dict()
_V26_ALGORITHM_CFG.update(
    class_name="engineai_rl_lab.utils.v26_star_multi_critic_ppo:V26StarMultiCriticPPO",
    reward_groups=deepcopy(V26_REWARD_GROUPS),
    value_loss_reduction="mean",
    star_priority_fraction=0.125,
    star_high_difficulty_threshold=1.0,
    star_top_fraction=0.05,
    star_difficulty_boundaries=(1.5, 2.0, 4.0),
    star_reuse_cap=2,
    star_priority_weight_cap=8.0,
)


@configclass
class T800FlatV26ScalePPORunnerCfg(T800FlatV25ScalePPORunnerCfg):
    """V25 runner with only the conservative STAR-lite sampling extension."""

    run_name = "v26-scale"
    actor = deepcopy(T800FlatV25ScalePPORunnerCfg().actor)
    critic = deepcopy(T800FlatV25ScalePPORunnerCfg().critic)
    algorithm = RslRlV26StarPpoAlgorithmCfg(**_V26_ALGORITHM_CFG)
