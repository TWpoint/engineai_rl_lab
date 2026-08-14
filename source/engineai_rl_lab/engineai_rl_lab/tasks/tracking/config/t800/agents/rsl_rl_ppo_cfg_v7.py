from copy import deepcopy

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .model_graph_cfg import RslRlModelGraphCfg
from .rsl_rl_ppo_cfg_v6 import T800FlatV6PPORunnerCfg


@configclass
class T800FlatV7PPORunnerCfg(T800FlatV6PPORunnerCfg):
    """V6 actor with entity command tokens fused through cross-attention."""

    run_name = "v7"
    actor = RslRlModelGraphCfg(
        nodes={
            "policy_projection": {
                "cell": {
                    "class_name": "TokenProjectionCell",
                    "output_dim": 128,
                }
            },
            "action_projection": {
                "cell": {
                    "class_name": "TokenProjectionCell",
                    "output_dim": 128,
                }
            },
            "token_interleaver": {
                "cell": {
                    "class_name": "TokenMergeCell",
                    "input_mode": "list",
                    "mode": "interleave",
                    "dim": 1,
                }
            },
            "command_projection": {
                "cell": {
                    "class_name": "TokenMLPCell",
                    "hidden_dims": [256],
                    "output_dim": 128,
                    "activation": "elu",
                }
            },
            "topology_projection": {
                "cell": {
                    "class_name": "TopologyProjectionCell",
                    "output_dim": 128,
                    # [side, body part, kinematic-chain depth]
                    "topology": [
                        [0, 0, 0],
                        [1, 2, 1],
                        [1, 2, 2],
                        [1, 2, 3],
                        [-1, 2, 1],
                        [-1, 2, 2],
                        [-1, 2, 3],
                        [0, 0, 1],
                        [1, 1, 1],
                        [1, 1, 2],
                        [1, 1, 3],
                        [-1, 1, 1],
                        [-1, 1, 2],
                        [-1, 1, 3],
                    ],
                }
            },
            "command_token_adder": {
                "cell": {
                    "class_name": "TokenAddCell",
                    "input_mode": "list",
                    "output_dim": 128,
                }
            },
            "temporal_encoder": {
                "cell": {
                    "class_name": "TemporalAttentionCell",
                    "output_dim": 128,
                    "num_heads": 4,
                    "ffn_dim": 512,
                    "activation": "gelu",
                    "position_embedding": "rope",
                    "normalization": "rms_norm",
                    "normalization_eps": 1.0e-6,
                    "attention_residual": True,
                    "ffn_residual": True,
                }
            },
            "command_encoder": {
                "cell": {
                    "class_name": "CrossAttentionCell",
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
            },
            "action_decoder": {
                "cell": {
                    "class_name": "MLPCell",
                    "hidden_dims": [256, 128],
                    "activation": "elu",
                }
            },
        },
        routes=[
            {"source": "inputs.policy", "target": "nodes.policy_projection.input"},
            {"source": "inputs.action", "target": "nodes.action_projection.input"},
            {
                "source": "nodes.policy_projection.output",
                "target": "nodes.token_interleaver.input",
            },
            {
                "source": "nodes.action_projection.output",
                "target": "nodes.token_interleaver.input",
            },
            {
                "source": "nodes.token_interleaver.output",
                "target": "nodes.temporal_encoder.input",
            },
            {"source": "inputs.command", "target": "nodes.command_projection.input"},
            {"source": "inputs.command", "target": "nodes.topology_projection.input"},
            {
                "source": "nodes.command_projection.output",
                "target": "nodes.command_token_adder.input",
            },
            {
                "source": "nodes.topology_projection.output",
                "target": "nodes.command_token_adder.input",
            },
            {
                "source": "nodes.temporal_encoder.output",
                "target": "nodes.command_encoder.input",
            },
            {
                "source": "nodes.command_token_adder.output",
                "target": "nodes.command_encoder.input",
            },
            {"source": "nodes.temporal_encoder.output", "target": "nodes.action_decoder.input"},
            {"source": "nodes.command_encoder.output", "target": "nodes.action_decoder.input"},
        ],
        output="nodes.action_decoder.output",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )


@configclass
class T800FlatV7N1PPORunnerCfg(T800FlatV7PPORunnerCfg):
    """V7 actor without the temporal self-attention FFN."""

    run_name = "v7-n1"
    actor = deepcopy(T800FlatV7PPORunnerCfg().actor)
    actor.nodes["temporal_encoder"]["cell"]["use_ffn"] = False
