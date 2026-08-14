from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v7_n2 import T800FlatV7N2PPORunnerCfg


@configclass
class T800FlatV7N3PPORunnerCfg(T800FlatV7N2PPORunnerCfg):
    """Width-scaled V7-N2 actor with nonlinear per-token input projections."""

    run_name = "v7-n3"
    actor = deepcopy(T800FlatV7N2PPORunnerCfg().actor)

    for projection_name in ("policy_projection", "action_projection"):
        actor.nodes[projection_name]["cell"] = {
            "class_name": "TokenMLPCell",
            "hidden_dims": [256],
            "output_dim": 256,
            "activation": "elu",
        }

    actor.nodes["command_projection"]["cell"]["output_dim"] = 256
    actor.nodes["topology_projection"]["cell"]["output_dim"] = 256
    actor.nodes["command_token_adder"]["cell"]["output_dim"] = 256

    for encoder_name in ("temporal_encoder", "command_self_attention", "command_encoder"):
        actor.nodes[encoder_name]["cell"].update(
            {
                "output_dim": 256,
                "num_heads": 8,
                "ffn_dim": 1024,
            }
        )

    actor.nodes["action_decoder"]["cell"]["hidden_dims"] = [512, 256]
