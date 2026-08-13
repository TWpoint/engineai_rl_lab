from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .model_graph_cfg import RslRlModelGraphCfg
from .rsl_rl_ppo_cfg_v1 import T800FlatV1PPORunnerCfg


@configclass
class T800FlatV2PPORunnerCfg(T800FlatV1PPORunnerCfg):
    """V2 actor with temporal attention and a separate command branch."""

    run_name = "v2"
    obs_groups = {"actor": ["policy", "command"], "critic": ["critic"]}
    actor = RslRlModelGraphCfg(
        nodes={
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
            {"source": "inputs.policy", "target": "nodes.temporal_encoder.input"},
            {"source": "inputs.command", "target": "nodes.command_encoder.input"},
            {"source": "nodes.temporal_encoder.output", "target": "nodes.action_decoder.input"},
            {"source": "nodes.command_encoder.output", "target": "nodes.action_decoder.input"},
        ],
        output="nodes.action_decoder.output",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
