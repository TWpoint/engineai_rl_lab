from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .model_graph_cfg import RslRlModelGraphCfg
from .rsl_rl_ppo_cfg import T800FlatPPORunnerCfg


@configclass
class T800FlatV1PPORunnerCfg(T800FlatPPORunnerCfg):
    run_name = "v1"


@configclass
class T800FlatV1N1PPORunnerCfg(T800FlatV1PPORunnerCfg):
    """V1 environment with the first Node/Cell/Route network iteration."""

    run_name = "v1-n1"
    actor = RslRlModelGraphCfg(
        nodes={
            "action_decoder": {
                "cell": {
                    "class_name": "MLPCell",
                    "hidden_dims": [512, 256, 128],
                    "activation": "elu",
                }
            }
        },
        routes=[{"source": "inputs.policy", "target": "nodes.action_decoder.input"}],
        output="nodes.action_decoder.output",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
