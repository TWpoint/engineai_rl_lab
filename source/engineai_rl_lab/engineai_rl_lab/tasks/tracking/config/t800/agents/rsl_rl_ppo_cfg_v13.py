from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v12 import T800FlatV12ScalePPORunnerCfg


@configclass
class T800FlatV13ScalePPORunnerCfg(T800FlatV12ScalePPORunnerCfg):
    """T800 BeyondMinic-aligned two-stream runner."""

    run_name = "v13-scale"
    num_steps_per_env = 64
    save_interval = 250
    sync_adaptive_sampling_all_gpus_freq = 10
    motion_resample_frequency = 0
    stagger_motion_working_set_refresh = False
    actor = deepcopy(T800FlatV12ScalePPORunnerCfg().actor)
    actor.nodes["attention_blocks"]["cell"].update(
        command_ffn_dim=512,
        ffn_dim=512,
        ffn_type="mlp",
        activation="gelu",
    )
    actor.nodes["action_decoder"]["cell"]["zero_init_output"] = True
    actor.distribution_cfg.std_range = (0.05, 2.0)
    algorithm = deepcopy(T800FlatV12ScalePPORunnerCfg().algorithm)
    algorithm.num_mini_batches = 64
    algorithm.num_learning_epochs = 2
    algorithm.actor_learning_rate = 2.0e-5
    algorithm.critic_learning_rate = 1.0e-3
    algorithm.schedule = "fixed"
    algorithm.max_grad_norm = 1.0
    critic = deepcopy(T800FlatV12ScalePPORunnerCfg().critic)
    critic.hidden_dims = [1024, 512, 256]
