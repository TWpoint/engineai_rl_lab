from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v8_n2 import RslRlSeparateLearningRatePpoAlgorithmCfg
from .rsl_rl_ppo_cfg_v19 import T800FlatV19ScalePPORunnerCfg


@configclass
class RslRlV20PpoAlgorithmCfg(RslRlSeparateLearningRatePpoAlgorithmCfg):
    """Split-rate PPO with configurable KL-adaptive actor learning-rate bounds."""

    optimizer_fused: bool = False
    learning_rate_min: float = 1.0e-5
    learning_rate_max: float = 2.0e-4
    shared_kl_adaptation: bool = False


_V20_ALGORITHM_CFG = T800FlatV19ScalePPORunnerCfg().algorithm.to_dict()
_V20_ALGORITHM_CFG.update(
    num_learning_epochs=2,
    num_mini_batches=16,
    schedule="adaptive",
    critic_learning_rate=2.0e-5,
    optimizer_fused=False,
    learning_rate_min=1.0e-5,
    learning_rate_max=2.0e-4,
)


def _make_v20_actor_cfg():
    actor = deepcopy(T800FlatV19ScalePPORunnerCfg().actor)
    for projection_name in ("proprioception_projection", "action_projection"):
        projection_cfg = actor.nodes[projection_name]["cell"]
        actor.nodes[projection_name]["cell"] = {
            "class_name": "TokenProjectionCell",
            "output_dim": projection_cfg["output_dim"],
        }
    actor.nodes["attention_blocks"]["cell"].update(
        command_ffn_dim=344,
        command_ffn_type="swiglu",
        ffn_dim=344,
        ffn_type="swiglu",
        causal=True,
    )
    return actor


def _make_v20_critic_cfg():
    critic = deepcopy(T800FlatV19ScalePPORunnerCfg().critic)
    critic.hidden_dims = [2048, 1024, 512]
    return critic


@configclass
class T800FlatV20ScalePPORunnerCfg(T800FlatV19ScalePPORunnerCfg):
    """Scratch runner with linear history projections and parameter-matched SwiGLU FFNs."""

    run_name = "v20-scale"
    num_steps_per_env = 32
    actor = _make_v20_actor_cfg()
    actor.distribution_cfg.init_std = 0.05
    actor.distribution_cfg.std_range = (0.001, 0.5)
    critic = _make_v20_critic_cfg()
    algorithm = RslRlV20PpoAlgorithmCfg(**_V20_ALGORITHM_CFG)
