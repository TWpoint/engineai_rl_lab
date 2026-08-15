from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg_v8_n1 import T800FlatV8N1PPORunnerCfg


@configclass
class RslRlSeparateLearningRatePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with optional actor/critic learning-rate overrides."""

    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None


_N2_ALGORITHM_CFG = T800FlatV8N1PPORunnerCfg().algorithm.to_dict()
_N2_ALGORITHM_CFG.update(
    num_learning_epochs=2,
    num_mini_batches=16,
    learning_rate=5.0e-5,
    actor_learning_rate=5.0e-5,
    critic_learning_rate=5.0e-4,
)


@configclass
class T800FlatV8N2PPORunnerCfg(T800FlatV8N1PPORunnerCfg):
    """V8-N1 network with smaller PPO mini-batches and separate actor/critic learning rates."""

    num_steps_per_env = 24
    run_name = "v8-n2"
    algorithm = RslRlSeparateLearningRatePpoAlgorithmCfg(**_N2_ALGORITHM_CFG)


@configclass
class T800FlatV8N2ScalePPORunnerCfg(T800FlatV8N2PPORunnerCfg):
    """V8-N2 runner for the LaFAN regular/slow2x motion scale experiment."""

    run_name = "v8_n2_scale_lafan"
