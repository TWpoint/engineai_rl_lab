from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v7 import T800FlatV7PPORunnerCfg


@configclass
class T800FlatV7N2PPORunnerCfg(T800FlatV7PPORunnerCfg):
    """V7 actor with readout-conditioned command self-attention."""

    run_name = "v7-n2"
    actor = deepcopy(T800FlatV7PPORunnerCfg().actor)
    actor.nodes["command_self_attention"] = {
        "cell": {
            "class_name": "CommandSelfAttentionCell",
            "input_mode": "list",
            "output_dim": 128,
            "num_heads": 4,
            "ffn_dim": 512,
            "activation": "gelu",
            "normalization": "rms_norm",
            "normalization_eps": 1.0e-6,
            "attention_residual": True,
            "ffn_residual": True,
        }
    }
    actor.routes.remove(
        {
            "source": "nodes.command_token_adder.output",
            "target": "nodes.command_encoder.input",
        }
    )
    actor.routes.extend(
        [
            {
                "source": "nodes.temporal_encoder.output",
                "target": "nodes.command_self_attention.input",
            },
            {
                "source": "nodes.command_token_adder.output",
                "target": "nodes.command_self_attention.input",
            },
            {
                "source": "nodes.command_self_attention.output",
                "target": "nodes.command_encoder.input",
            },
        ]
    )
