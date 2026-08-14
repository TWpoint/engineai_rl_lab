from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .model_graph_cfg import RslRlModelGraphCfg
from .rsl_rl_ppo_cfg_v3 import T800FlatV3PPORunnerCfg


@configclass
class T800FlatV4PPORunnerCfg(T800FlatV3PPORunnerCfg):
    """V4 actor with projected, interleaved observation and action tokens."""

    run_name = "v4"
    obs_groups = {"actor": ["policy", "action", "command"], "critic": ["critic"]}
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
                    "class_name": "MLPCell",
                    "hidden_dims": [256, 128],
                    "output_dim": 128,
                    "activation": "elu",
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
            {"source": "inputs.command", "target": "nodes.command_encoder.input"},
            {"source": "nodes.temporal_encoder.output", "target": "nodes.action_decoder.input"},
            {"source": "nodes.command_encoder.output", "target": "nodes.action_decoder.input"},
        ],
        output="nodes.action_decoder.output",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
