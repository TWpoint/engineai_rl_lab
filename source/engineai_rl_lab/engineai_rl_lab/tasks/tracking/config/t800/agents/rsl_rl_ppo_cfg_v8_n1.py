from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v8 import T800FlatV8PPORunnerCfg


@configclass
class T800FlatV8N1PPORunnerCfg(T800FlatV8PPORunnerCfg):
    """V8 actor with two scalable full-token tracking attention blocks."""

    run_name = "v8-n1"
    actor = deepcopy(T800FlatV8PPORunnerCfg().actor)

    for node_name in ("temporal_encoder", "command_self_attention", "command_encoder"):
        actor.nodes.pop(node_name)

    actor.routes = [
        route
        for route in actor.routes
        if not any(
            endpoint.startswith(f"nodes.{node_name}.")
            for endpoint in (route["source"], route["target"])
            for node_name in ("temporal_encoder", "command_self_attention", "command_encoder")
        )
    ]
    actor.nodes["attention_blocks"] = {
        "cell": {
            "class_name": "StackedTrackingAttentionCell",
            "input_mode": "list",
            "output_dim": 256,
            "num_blocks": 2,
            "num_heads": 4,
            "ffn_dim": 256,
            "ffn_type": "swiglu",
            "position_embedding": "rope",
            "normalization": "rms_norm",
            "normalization_eps": 1.0e-6,
            "attention_residual": True,
            "ffn_residual": True,
        }
    }
    actor.nodes["action_decoder"] = {"cell": {"class_name": "LinearCell"}}
    actor.routes.extend(
        [
            {
                "source": "nodes.token_interleaver.output",
                "target": "nodes.attention_blocks.input",
            },
            {
                "source": "nodes.command_token_adder.output",
                "target": "nodes.attention_blocks.input",
            },
            {
                "source": "nodes.attention_blocks.output",
                "target": "nodes.action_decoder.input",
            },
        ]
    )
    encoder_name = "attention_blocks"


@configclass
class T800FlatV8N1ScalePPORunnerCfg(T800FlatV8N1PPORunnerCfg):
    """V8-N1 runner for the LaFAN regular/slow2x motion scale experiment."""

    run_name = "v8_n1_scale_lafan"
