from copy import deepcopy

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v11_n2 import T800FlatV11N2PPORunnerCfg

# [root, lower_body, torso, upper_body,
#  left, center, right, normalized_chain_depth, is_end_effector]
V12_COMMAND_TOPOLOGY = [
    [1, 0, 0, 0, 0, 1, 0, 0.0, 0],  # base
    [0, 1, 0, 0, 1, 0, 0, 1 / 3, 0],  # left hip
    [0, 1, 0, 0, 1, 0, 0, 2 / 3, 0],  # left knee
    [0, 1, 0, 0, 1, 0, 0, 1.0, 1],  # left ankle
    [0, 1, 0, 0, 0, 0, 1, 1 / 3, 0],  # right hip
    [0, 1, 0, 0, 0, 0, 1, 2 / 3, 0],  # right knee
    [0, 1, 0, 0, 0, 0, 1, 1.0, 1],  # right ankle
    [0, 0, 1, 0, 0, 1, 0, 1 / 3, 0],  # waist
    [0, 0, 0, 1, 1, 0, 0, 1 / 3, 0],  # left shoulder
    [0, 0, 0, 1, 1, 0, 0, 2 / 3, 0],  # left elbow
    [0, 0, 0, 1, 1, 0, 0, 1.0, 1],  # left wrist
    [0, 0, 0, 1, 0, 0, 1, 1 / 3, 0],  # right shoulder
    [0, 0, 0, 1, 0, 0, 1, 2 / 3, 0],  # right elbow
    [0, 0, 0, 1, 0, 0, 1, 1.0, 1],  # right wrist
]


@configclass
class T800FlatV12PPORunnerCfg(T800FlatV11N2PPORunnerCfg):
    """V11-N2 with GELU token projections and projected gravity."""

    run_name = "v12"
    obs_groups = deepcopy(T800FlatV11N2PPORunnerCfg().obs_groups)
    obs_groups["actor"] = ["proprioception", "action", "command"]
    actor = deepcopy(T800FlatV11N2PPORunnerCfg().actor)

    actor.nodes["proprioception_projection"] = actor.nodes.pop("policy_projection")
    for route in actor.routes:
        if route["source"] == "inputs.policy":
            route["source"] = "inputs.proprioception"
        for endpoint in ("source", "target"):
            route[endpoint] = route[endpoint].replace(
                "nodes.policy_projection.",
                "nodes.proprioception_projection.",
            )

    for projection_name in ("proprioception_projection", "action_projection", "command_projection"):
        actor.nodes[projection_name]["cell"]["activation"] = "gelu"

    actor.nodes["topology_projection"]["cell"].update(
        topology=V12_COMMAND_TOPOLOGY,
        hidden_dims=[64],
        activation="gelu",
    )


@configclass
class T800FlatV12ScaleLafanPPORunnerCfg(T800FlatV12PPORunnerCfg):
    """V12 runner for the complete regular and 2x-slow LaFAN pool."""

    run_name = "v12-scale-lafan"


@configclass
class T800FlatV12ScalePPORunnerCfg(T800FlatV12PPORunnerCfg):
    """V12 runner for the complete T800 v0 motion manifest."""

    run_name = "v12-scale"
    torch_compile_mode = "default"
    algorithm = deepcopy(T800FlatV12PPORunnerCfg().algorithm)
    algorithm.optimizer_fused = True


@configclass
class T800FlatV12ScaleLafanSonicPPORunnerCfg(T800FlatV12ScaleLafanPPORunnerCfg):
    """V12 runner for the LaFAN regular/2x-slow and SONIC pool."""

    run_name = "v12-scale-lafan-sonic"
