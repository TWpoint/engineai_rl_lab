# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math
import os

# import numpy as np
import onnx

# import re
import torch
import yaml

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

# from numbers import Number
# from typing import Any, Dict, List, Union


# from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def get_actor_obs_normalizer(policy_or_runner: object) -> object | None:
    """Resolve the actor observation normalizer across RSL-RL versions.

    Older versions exposed the normalizer on the runner as ``obs_normalizer``.
    Newer versions keep it on the policy as ``actor_obs_normalizer``.
    """

    legacy_normalizer = getattr(policy_or_runner, "obs_normalizer", None)
    if legacy_normalizer is not None:
        return legacy_normalizer

    alg = getattr(policy_or_runner, "alg", None)
    policy = alg.get_policy() if hasattr(alg, "get_policy") else getattr(alg, "policy", policy_or_runner)
    return getattr(policy, "actor_obs_normalizer", getattr(policy, "obs_normalizer", None))


def rewrite_silu_for_mnn_2_9_5(input_path: str, output_path: str) -> int:
    """Lower ONNX SiLU patterns so MNN 3.x can target a 2.9.5 runtime.

    MNN 3.6's converter fuses ``x * sigmoid(x)`` into the newer ``SILU``
    unary opcode even when ``--targetVersion 2.9.5`` is supplied. MNN 2.9.5
    cannot create a CPU execution for that opcode. Replacing only Sigmoid
    nodes participating in a SiLU pattern with the exact expression
    ``1 / (1 + exp(-x))`` prevents that fusion while preserving the model.
    """
    model = onnx.load(input_path)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    occupied_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    occupied_names.update(value.name for value in model.graph.input)
    occupied_names.update(value.name for value in model.graph.output)
    occupied_names.update(initializer.name for initializer in model.graph.initializer)

    def unique_name(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in occupied_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        occupied_names.add(candidate)
        return candidate

    rewritten_nodes = []
    rewrite_count = 0
    for node in model.graph.node:
        is_silu_sigmoid = node.op_type == "Sigmoid" and len(node.input) == 1 and len(node.output) == 1
        if is_silu_sigmoid:
            source_name = node.input[0]
            sigmoid_output = node.output[0]
            is_silu_sigmoid = any(
                consumer.op_type == "Mul"
                and source_name in consumer.input
                and sigmoid_output in consumer.input
                for consumer in consumers.get(sigmoid_output, [])
            )
        if not is_silu_sigmoid:
            rewritten_nodes.append(node)
            continue

        rewrite_count += 1
        prefix = unique_name(f"{node.name or 'sigmoid'}_mnn_2_9_5")
        neg_output = unique_name(f"{prefix}_neg")
        exp_output = unique_name(f"{prefix}_exp")
        one_output = unique_name(f"{prefix}_one")
        denominator_output = unique_name(f"{prefix}_denominator")
        source_name = node.input[0]
        sigmoid_output = node.output[0]
        rewritten_nodes.extend(
            [
                onnx.helper.make_node("Neg", [source_name], [neg_output], name=unique_name(f"{prefix}_Neg")),
                onnx.helper.make_node("Exp", [neg_output], [exp_output], name=unique_name(f"{prefix}_Exp")),
                onnx.helper.make_node(
                    "Constant",
                    [],
                    [one_output],
                    name=unique_name(f"{prefix}_Constant"),
                    value=onnx.helper.make_tensor(one_output, onnx.TensorProto.FLOAT, [], [1.0]),
                ),
                onnx.helper.make_node(
                    "Add",
                    [exp_output, one_output],
                    [denominator_output],
                    name=unique_name(f"{prefix}_Add"),
                ),
                onnx.helper.make_node(
                    "Reciprocal",
                    [denominator_output],
                    [sigmoid_output],
                    name=unique_name(f"{prefix}_Reciprocal"),
                ),
            ]
        )

    model.graph.ClearField("node")
    model.graph.node.extend(rewritten_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return rewrite_count


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if hasattr(actor_critic, "as_onnx"):
        policy_exporter = actor_critic.as_onnx(verbose=verbose)
        policy_exporter.to("cpu")
        policy_exporter.eval()
        torch.onnx.export(
            policy_exporter,
            policy_exporter.get_dummy_inputs(),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=policy_exporter.input_names,
            output_names=policy_exporter.output_names,
        )
        return

    # RSL-RL's ModelGraph is itself the actor model (rather than an
    # actor-critic container), so IsaacLab's legacy exporter cannot find an
    # ``actor``/``student`` attribute on it.  Export a tensor-only copy of the
    # graph; this also avoids putting TensorDict operations in the ONNX graph.
    if all(
        hasattr(actor_critic, attr)
        for attr in ("_input_dims", "_incoming", "_execution_order", "nodes", "output_endpoint")
    ):
        policy_exporter = _OnnxModelGraphExporter(env, actor_critic, verbose)
        policy_exporter.export(path, filename)
        return

    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxModelGraphExporter(torch.nn.Module):
    """Tensor-only ONNX adapter for an RSL-RL ModelGraph policy."""

    def __init__(self, env, model, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.input_groups = list(model._input_dims)
        self.input_dims = dict(model._input_dims)
        self.input_shapes = {
            name: self._resolve_input_shape(env.observation_manager.group_obs_dim[name], name)
            for name in self.input_groups
        }
        self.input_normalizers = copy.deepcopy(model.input_normalizers)
        self.nodes = copy.deepcopy(model.nodes)
        self.incoming = copy.deepcopy(model._incoming)
        self.node_input_modes = copy.deepcopy(getattr(model, "_node_input_modes", {}))
        self.execution_order = list(model._execution_order)
        self.output_endpoint = model.output_endpoint
        if model.distribution is not None:
            self.deterministic_output = copy.deepcopy(model.distribution.as_deterministic_output_module())
        else:
            self.deterministic_output = torch.nn.Identity()

    def forward(self, *inputs):
        tensors = {
            f"inputs.{name}": self.input_normalizers[name](value)
            for name, value in zip(self.input_groups, inputs, strict=True)
        }
        for node_name in self.execution_order:
            parts = [tensors[source] for source in self.incoming[node_name]]
            if self.node_input_modes.get(node_name, "concat") == "list":
                node_input = parts
            else:
                node_input = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
            tensors[f"nodes.{node_name}.output"] = self.nodes[node_name](node_input)
        return self.deterministic_output(tensors[self.output_endpoint])

    @staticmethod
    def _resolve_input_shape(group_dim, group_name):
        """Resolve tensor shape for concatenated or single-term non-concatenated groups."""
        if isinstance(group_dim, tuple):
            return group_dim
        if isinstance(group_dim, list) and len(group_dim) == 1:
            return tuple(group_dim[0])
        raise ValueError(
            f"ModelGraph ONNX input group '{group_name}' must be concatenated or contain exactly one term; "
            f"got dimensions {group_dim}."
        )

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        dummy_inputs = tuple(torch.zeros(1, *self.input_shapes[name]) for name in self.input_groups)
        # Keep the historic single-policy input name expected by deployment.
        input_names = ["obs"] if self.input_groups == ["policy"] else self.input_groups
        torch.onnx.export(
            self,
            dummy_inputs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=self.verbose,
            input_names=input_names,
            output_names=["actions"],
        )


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)

    def forward(self, x):
        return (self.actor(self.normalizer(x)),)

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        torch.onnx.export(
            self,
            (obs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=[
                "actions",
            ],
            dynamic_axes={},
        )


# class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
#     def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
#         super().__init__(actor_critic, normalizer, verbose)
#         cmd: MotionCommand = env.command_manager.get_term("motion")

#         self.joint_pos = cmd.motion.joint_pos.to("cpu")
#         self.joint_vel = cmd.motion.joint_vel.to("cpu")
#         self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
#         self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
#         self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
#         self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
#         self.time_step_total = self.joint_pos.shape[0]

#     def forward(self, x, time_step):
#         time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
#         return (
#             self.actor(self.normalizer(x)),
#             self.joint_pos[time_step_clamped],
#             self.joint_vel[time_step_clamped],
#             self.body_pos_w[time_step_clamped],
#             self.body_quat_w[time_step_clamped],
#             self.body_lin_vel_w[time_step_clamped],
#             self.body_ang_vel_w[time_step_clamped],
#         )

#     def export(self, path, filename):
#         self.to("cpu")
#         obs = torch.zeros(1, self.actor[0].in_features)
#         time_step = torch.zeros(1, 1)
#         torch.onnx.export(
#             self,
#             (obs, time_step),
#             os.path.join(path, filename),
#             export_params=True,
#             opset_version=11,
#             verbose=self.verbose,
#             input_names=["obs", "time_step"],
#             output_names=[
#                 "actions",
#                 "joint_pos",
#                 "joint_vel",
#                 "body_pos_w",
#                 "body_quat_w",
#                 "body_lin_vel_w",
#                 "body_ang_vel_w",
#             ],
#             dynamic_axes={},
#         )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x)
        for x in arr  # numbers → format, strings → as-is
    )


def _group_cfg(env: ManagerBasedRLEnv, group_name: str):
    return getattr(env.observation_manager.cfg, group_name)


def _term_cfg(env: ManagerBasedRLEnv, group_name: str, term_name: str):
    """Return the live observation-term config without serializing callables."""
    group_cfg = _group_cfg(env, group_name)
    term_cfg = getattr(group_cfg, term_name, None)
    if term_cfg is None:
        raise ValueError(f"Observation config '{group_name}.{term_name}' is unavailable.")
    return term_cfg


def _plain_metadata_value(value):
    """Convert common Isaac Lab config/runtime values to YAML-safe objects."""
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim > 1:
            value = value[0]
        value = value.tolist()
    if isinstance(value, tuple):
        return [_plain_metadata_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_metadata_value(item) for key, item in value.items()}
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    # NumPy scalars and similar numeric wrappers implement ``item``.
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, (bool, int, float, str)):
            return item
    return value


def _flat_float_list(value, *, name: str) -> list[float]:
    """Read the first environment row of a joint-valued runtime array."""
    value = _plain_metadata_value(value)
    if isinstance(value, (int, float)):
        return [float(value)]
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a scalar or a one-dimensional numeric array, got {type(value).__name__}.")
    if value and isinstance(value[0], list):
        # ``_plain_metadata_value`` already removes the leading environment
        # dimension for tensors. This branch supports list-backed test doubles.
        value = value[0]
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} contains a non-numeric value.") from err


def _resolved_joint_names_from_asset_cfg(asset_cfg, robot_joint_names: list[str]) -> list[str] | None:
    """Resolve a SceneEntityCfg joint selection when its IDs are available."""
    joint_ids = getattr(asset_cfg, "joint_ids", None)
    if hasattr(joint_ids, "torch"):
        joint_ids = joint_ids.torch
    if isinstance(joint_ids, torch.Tensor):
        joint_ids = joint_ids.detach().cpu().reshape(-1).tolist()
    if isinstance(joint_ids, slice):
        joint_ids = list(range(len(robot_joint_names)))[joint_ids]
    if isinstance(joint_ids, (list, tuple)):
        return [robot_joint_names[int(index)] for index in joint_ids]

    configured_names = getattr(asset_cfg, "joint_names", None)
    if configured_names and all(name in robot_joint_names for name in configured_names):
        return list(configured_names)
    return None


def _term_history_length(group_cfg, term_name: str) -> int:
    """Return the effective history length for an observation term."""
    if getattr(group_cfg, "history_length", None) is not None:
        return max(1, int(group_cfg.history_length))
    term_cfg = group_cfg.to_dict()[term_name]
    return max(1, int(term_cfg.get("history_length") or 0))


def _onnx_input_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    """Extract static ONNX input shapes, excluding initializer tensors."""
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    shapes = {}
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                raise ValueError(f"ONNX input '{value.name}' has a dynamic shape; deployment requires static shapes.")
            dims.append(int(dim.dim_value))
        if not dims or dims[0] != 1:
            raise ValueError(f"ONNX input '{value.name}' must have batch dimension 1, got {dims}.")
        shapes[value.name] = dims[1:]
    return shapes


def _resolve_actor_obs_groups(
    env: ManagerBasedRLEnv, actor_obs_groups: list[str] | None, model: onnx.ModelProto
) -> list[str]:
    """Infer named ModelGraph inputs from ONNX while retaining legacy ``obs`` support."""
    if actor_obs_groups is not None:
        groups = list(actor_obs_groups)
    else:
        input_names = list(_onnx_input_shapes(model))
        active_groups = env.observation_manager.active_terms
        if input_names == ["obs"]:
            if "policy" in active_groups:
                groups = ["policy"]
            elif len(active_groups) == 1:
                groups = [next(iter(active_groups))]
            else:
                raise ValueError(
                    "Cannot infer which observation group feeds the legacy ONNX input 'obs'; "
                    f"available groups are {list(active_groups)}. Pass actor_obs_groups explicitly."
                )
        else:
            missing = [name for name in input_names if name not in active_groups]
            if missing:
                raise ValueError(
                    f"ONNX inputs {missing} do not name active observation groups; "
                    f"available groups are {list(active_groups)}."
                )
            groups = input_names

    if not groups:
        raise ValueError("At least one actor observation group is required to export metadata.")
    return groups


def _build_policy_inputs(env: ManagerBasedRLEnv, actor_obs_groups: list[str], model: onnx.ModelProto) -> list[dict]:
    """Build schema-v3 named input descriptions for the actor observation groups."""
    input_shapes = _onnx_input_shapes(model)
    expected_input_names = ["obs"] if actor_obs_groups == ["policy"] else actor_obs_groups
    if list(input_shapes) != expected_input_names:
        raise ValueError(
            "ONNX actor inputs do not match configured actor observation groups: "
            f"expected {expected_input_names}, got {list(input_shapes)}."
        )

    policy_inputs = []
    robot = env.scene["robot"]
    robot_joint_names = list(robot.joint_names)
    term_dims_by_group = getattr(env.observation_manager, "_group_obs_term_dim", {})
    for group_name, tensor_name in zip(actor_obs_groups, expected_input_names, strict=True):
        term_names = list(env.observation_manager.active_terms[group_name])
        group_cfg = _group_cfg(env, group_name)
        term_dims = term_dims_by_group.get(group_name)
        if term_dims is None or len(term_dims) != len(term_names):
            raise ValueError(f"Observation dimensions are unavailable for group '{group_name}'.")

        history_length = max((_term_history_length(group_cfg, name) for name in term_names), default=1)
        term_entries = []
        for term_name, term_dim in zip(term_names, term_dims, strict=True):
            term_size_with_history = math.prod(term_dim)
            term_history_length = _term_history_length(group_cfg, term_name)
            if term_history_length != history_length:
                raise ValueError(
                    f"Observation group '{group_name}' mixes history lengths: term '{term_name}' has "
                    f"{term_history_length}, while the group deploy layout uses {history_length}. "
                    "Split terms with different histories into separate actor observation groups."
                )
            if term_size_with_history % term_history_length != 0:
                raise ValueError(
                    f"Observation term '{group_name}.{term_name}' dimension {term_dim} is not divisible by "
                    f"history length {term_history_length}."
                )
            term_entry = {"name": term_name, "size": term_size_with_history // term_history_length}
            term_cfg = _term_cfg(env, group_name, term_name)
            term_func = getattr(term_cfg, "func", None)
            function_name = None
            if callable(term_func):
                function_name = getattr(term_func, "__name__", type(term_func).__name__)
                term_entry["function"] = function_name
            if function_name == "last_action":
                # Isaac Lab resets the action manager before the first
                # observation and the history buffer backfills that zero
                # sample. Subsequent entries are the unscaled network output,
                # not joint-position targets.
                term_entry["value_semantics"] = "previous_raw_policy_action"
                term_entry["reset_value"] = "zero"
            term_scale = getattr(term_cfg, "scale", None)
            if term_scale is not None:
                term_entry["scale"] = _plain_metadata_value(term_scale)
            term_clip = getattr(term_cfg, "clip", None)
            if term_clip is not None:
                term_entry["clip"] = _plain_metadata_value(term_clip)

            params = getattr(term_cfg, "params", {}) or {}
            asset_cfg = params.get("asset_cfg")
            if asset_cfg is not None:
                selected_joint_names = _resolved_joint_names_from_asset_cfg(asset_cfg, robot_joint_names)
                if selected_joint_names is not None:
                    term_entry["joint_names"] = selected_joint_names
            term_entries.append(term_entry)

        shape = input_shapes[tensor_name]
        if math.prod(shape) != sum(
            term["size"] * _term_history_length(group_cfg, term["name"]) for term in term_entries
        ):
            raise ValueError(
                f"ONNX input '{tensor_name}' shape {shape} does not match observation group '{group_name}'."
            )
        policy_inputs.append(
            {
                "name": group_name,
                "tensor_name": tensor_name,
                "shape": shape,
                "history_length": history_length,
                "flatten_history_dim": bool(getattr(group_cfg, "flatten_history_dim", True)),
                "history_order": "oldest_to_newest",
                "history_initialization": "repeat_first_sample",
                "terms": term_entries,
            }
        )
    return policy_inputs


def _build_legacy_observation_metadata(
    env: ManagerBasedRLEnv, actor_obs_groups: list[str]
) -> tuple[list[str], list[int]]:
    """Build compatibility metadata without assuming an observation group named ``policy``.

    Schema-v3 deployments consume ``policy_inputs`` below.  The flat fields are
    retained for older tooling, so for a multi-input actor they describe the
    first configured actor group instead of incorrectly flattening all named
    inputs into one tensor.
    """
    if not actor_obs_groups:
        raise ValueError("At least one actor observation group is required to export metadata.")

    legacy_group_name = "policy" if "policy" in actor_obs_groups else actor_obs_groups[0]
    try:
        observation_names = list(env.observation_manager.active_terms[legacy_group_name])
    except KeyError as err:
        available_groups = list(env.observation_manager.active_terms)
        raise ValueError(
            f"Actor observation group '{legacy_group_name}' is unavailable; available groups are {available_groups}."
        ) from err

    group_cfg = _group_cfg(env, legacy_group_name)
    observation_history_lengths = [
        _term_history_length(group_cfg, observation_name) for observation_name in observation_names
    ]
    return observation_names, observation_history_lengths


def _joint_value_map(names: list[str], values: list[float], *, field_name: str) -> dict[str, float]:
    if len(names) != len(values):
        raise ValueError(f"{field_name} has {len(values)} values for {len(names)} robot joints.")
    if len(set(names)) != len(names):
        raise ValueError("Robot joint names are not unique; deployment ordering would be ambiguous.")
    return dict(zip(names, values, strict=True))


def _joint_limit_maps(robot, robot_joint_names: list[str], *, field_name: str) -> tuple[dict[str, float], dict[str, float]]:
    """Return lower/upper joint-limit maps from an Isaac Lab ProxyArray/tensor.

    Articulation limits use internal simulator joint order.  Export them by
    name so the deploy file can later be reordered into the frozen SDK order
    without relying on PhysX/Newton articulation ordering.
    """
    value = getattr(robot.data, field_name, None)
    if value is None:
        raise ValueError(f"Robot articulation does not expose {field_name}; safe deployment metadata is incomplete.")
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 3:
            value = value[0]
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)
    if value and isinstance(value[0], list) and len(value) == 1 and len(value[0]) == len(robot_joint_names):
        # List-backed test doubles may retain the leading environment axis.
        value = value[0]
    if not isinstance(value, list) or len(value) != len(robot_joint_names):
        raise ValueError(
            f"{field_name} must have shape [num_joints, 2] (or [num_envs, num_joints, 2])."
        )
    try:
        pairs = [[float(pair[0]), float(pair[1])] for pair in value]
    except (TypeError, ValueError, IndexError) as err:
        raise ValueError(f"{field_name} contains an invalid lower/upper pair.") from err
    if any(len(pair) != 2 or not math.isfinite(pair[0]) or not math.isfinite(pair[1]) or pair[0] >= pair[1] for pair in pairs):
        raise ValueError(f"{field_name} contains an invalid lower/upper pair.")
    lower = _joint_value_map(robot_joint_names, [pair[0] for pair in pairs], field_name=f"{field_name}.lower")
    upper = _joint_value_map(robot_joint_names, [pair[1] for pair in pairs], field_name=f"{field_name}.upper")
    return lower, upper


def _build_joint_metadata(env: ManagerBasedRLEnv) -> dict:
    """Build unambiguous action-order and full-robot-order joint metadata."""
    robot = env.scene["robot"]
    robot_joint_names = list(robot.joint_names)
    action_term = env.action_manager.get_term("joint_pos")
    action_joint_names = list(getattr(action_term, "_joint_names", []))
    action_dim = int(getattr(action_term, "action_dim", len(action_joint_names)))
    if not action_joint_names:
        if action_dim == len(robot_joint_names):
            action_joint_names = robot_joint_names.copy()
        else:
            raise ValueError(
                "The joint_pos action term does not expose resolved joint names, and its action dimension "
                f"({action_dim}) is not the full robot joint count ({len(robot_joint_names)})."
            )
    if len(action_joint_names) != action_dim:
        raise ValueError(
            f"joint_pos resolves {len(action_joint_names)} joint names but reports action_dim={action_dim}."
        )
    unknown_action_joints = [name for name in action_joint_names if name not in robot_joint_names]
    if unknown_action_joints:
        raise ValueError(f"Action joints are absent from the robot articulation: {unknown_action_joints}.")

    nominal_value = getattr(robot.data, "default_joint_pos_nominal", None)
    if nominal_value is None:
        nominal_value = robot.data.default_joint_pos
    nominal_values = _flat_float_list(nominal_value, name="default_joint_pos_nominal")
    stiffness_values = _flat_float_list(robot.data.joint_stiffness, name="joint_stiffness")
    damping_values = _flat_float_list(robot.data.joint_damping, name="joint_damping")
    nominal_by_name = _joint_value_map(robot_joint_names, nominal_values, field_name="default_joint_pos_nominal")
    stiffness_by_name = _joint_value_map(robot_joint_names, stiffness_values, field_name="joint_stiffness")
    damping_by_name = _joint_value_map(robot_joint_names, damping_values, field_name="joint_damping")
    hard_lower_by_name, hard_upper_by_name = _joint_limit_maps(
        robot, robot_joint_names, field_name="joint_pos_limits"
    )
    soft_lower_by_name, soft_upper_by_name = _joint_limit_maps(
        robot, robot_joint_names, field_name="soft_joint_pos_limits"
    )

    action_scale_value = getattr(action_term, "_scale", None)
    if action_scale_value is None:
        raise ValueError("The joint_pos action term does not expose its resolved action scale.")
    action_scale = _flat_float_list(action_scale_value, name="action_scale")
    if len(action_scale) == 1:
        action_scale *= action_dim
    if len(action_scale) != action_dim:
        raise ValueError(f"action_scale has {len(action_scale)} values for action_dim={action_dim}.")

    # Isaac's articulation order is backend-dependent. Prefer the explicit SDK
    # order for full-state arrays when the robot config provides a permutation.
    scene_cfg = getattr(env.scene, "cfg", None)
    robot_cfg = getattr(scene_cfg, "robot", None)
    sdk_joint_names = list(getattr(robot_cfg, "joint_sdk_names", None) or [])
    if len(sdk_joint_names) != len(robot_joint_names) or set(sdk_joint_names) != set(robot_joint_names):
        sdk_joint_names = robot_joint_names.copy()

    fixed_joint_positions = {}
    command_manager = getattr(env, "command_manager", None)
    if command_manager is not None:
        command_names = list(getattr(command_manager, "active_terms", []) or [])
        if "motion" not in command_names:
            command_names.append("motion")
        for command_name in command_names:
            try:
                command = command_manager.get_term(command_name)
            except (KeyError, ValueError):
                continue
            configured_positions = dict(getattr(command.cfg, "fixed_joint_positions", {}) or {})
            for joint_name, position in configured_positions.items():
                if joint_name not in robot_joint_names:
                    raise ValueError(
                        f"Command '{command_name}' fixes unknown robot joint '{joint_name}'."
                    )
                position = float(position)
                if joint_name in fixed_joint_positions and fixed_joint_positions[joint_name] != position:
                    raise ValueError(f"Conflicting fixed positions were configured for joint '{joint_name}'.")
                fixed_joint_positions[joint_name] = position
    fixed_joint_positions = {
        name: fixed_joint_positions[name] for name in sdk_joint_names if name in fixed_joint_positions
    }

    action_default_joint_pos = [nominal_by_name[name] for name in action_joint_names]
    action_joint_stiffness = [stiffness_by_name[name] for name in action_joint_names]
    action_joint_damping = [damping_by_name[name] for name in action_joint_names]
    robot_default_joint_pos = [nominal_by_name[name] for name in sdk_joint_names]
    robot_joint_stiffness = [stiffness_by_name[name] for name in sdk_joint_names]
    robot_joint_damping = [damping_by_name[name] for name in sdk_joint_names]
    robot_joint_position_lower_limit = [hard_lower_by_name[name] for name in sdk_joint_names]
    robot_joint_position_upper_limit = [hard_upper_by_name[name] for name in sdk_joint_names]
    robot_joint_soft_position_lower_limit = [soft_lower_by_name[name] for name in sdk_joint_names]
    robot_joint_soft_position_upper_limit = [soft_upper_by_name[name] for name in sdk_joint_names]

    return {
        # Legacy robot-state fields remain full-sized, but are now explicitly
        # reordered into the robot SDK order. Partial policies pair their
        # shorter scale with ``action_joint_names`` below.
        "default_joint_pos": robot_default_joint_pos.copy(),
        "joint_names": sdk_joint_names.copy(),
        "joint_stiffness": robot_joint_stiffness.copy(),
        "joint_damping": robot_joint_damping.copy(),
        "action_scale": action_scale,
        "action_joint_names": action_joint_names.copy(),
        "action_default_joint_pos": action_default_joint_pos,
        "action_joint_stiffness": action_joint_stiffness,
        "action_joint_damping": action_joint_damping,
        "action_semantics": "joint_position_target = action_default_joint_pos + action * action_scale",
        "action_offset_source": "default_joint_pos_nominal",
        # Explicit full-state order for an SDK boundary. This is separate from
        # policy action order by design (for example, V24 fixes the two head joints).
        "robot_joint_names": sdk_joint_names.copy(),
        "robot_default_joint_pos": robot_default_joint_pos.copy(),
        "robot_joint_stiffness": robot_joint_stiffness.copy(),
        "robot_joint_damping": robot_joint_damping.copy(),
        "robot_joint_position_lower_limit": robot_joint_position_lower_limit,
        "robot_joint_position_upper_limit": robot_joint_position_upper_limit,
        "robot_joint_soft_position_lower_limit": robot_joint_soft_position_lower_limit,
        "robot_joint_soft_position_upper_limit": robot_joint_soft_position_upper_limit,
        "joint_limit_order": "robot_joint_names",
        "joint_limit_unit": "radian",
        "fixed_joint_positions": fixed_joint_positions,
    }


def _control_period_s(env: ManagerBasedRLEnv) -> float:
    step_dt = getattr(env, "step_dt", None)
    if step_dt is not None:
        return float(step_dt)
    cfg = getattr(env, "cfg", None)
    sim_cfg = getattr(cfg, "sim", None)
    sim_dt = getattr(sim_cfg, "dt", None)
    decimation = getattr(cfg, "decimation", None)
    if sim_dt is None or decimation is None:
        raise ValueError("Environment control period is unavailable (expected step_dt or cfg.sim.dt * cfg.decimation).")
    return float(sim_dt) * int(decimation)


def _pose_command_contract(function_name: str) -> dict | None:
    """Describe exact frame and packing conventions for known tracking commands."""
    xz_feature_order = [
        "position_x",
        "position_y",
        "position_z",
        "rotation_column_x_x",
        "rotation_column_x_y",
        "rotation_column_x_z",
        "rotation_column_z_x",
        "rotation_column_z_y",
        "rotation_column_z_z",
    ]
    xy_interleaved_feature_order = [
        "position_x",
        "position_y",
        "position_z",
        "rotation_r00",
        "rotation_r01",
        "rotation_r10",
        "rotation_r11",
        "rotation_r20",
        "rotation_r21",
    ]
    by_entity = function_name.endswith("_by_entity_xz") or function_name.endswith("_by_entity")
    flat = function_name.endswith("_flat")
    if not by_entity and not flat:
        return None

    reference_anchor = function_name.startswith("motion_body_pose_reference_anchor_window")
    target_and_error = function_name.startswith("motion_body_pose_and_error_b_window")
    xz_encoding = function_name.endswith("_xz") or function_name.endswith("_xz_flat") or target_and_error
    if reference_anchor:
        position_origin = "reference_anchor_position_at_current_frame"
        position_frame = "reference_anchor_translation_robot_anchor_axes"
    else:
        position_origin = "robot_anchor_position_at_current_frame"
        position_frame = "robot_anchor_translation_robot_anchor_axes"

    rotation_encoding = (
        "matrix_columns_x_z" if xz_encoding else "matrix_columns_x_y_row_major_interleaved"
    )
    layout_name = "link_offset_feature" if by_entity else "flattened_link_offset_feature"

    contract = {
        "position_frame": position_frame,
        "position_frame_detail": {
            "origin": position_origin,
            "axes": "robot_anchor_orientation_at_current_frame",
        },
        "orientation_frame": "robot_anchor_orientation_at_current_frame",
        "rotation_encoding": rotation_encoding,
        "layout": layout_name,
        "layout_detail": {
            "tensor_axes": (
                ["batch", "body", "frame_feature"] if by_entity else ["batch", "flattened_body_frame_feature"]
            ),
            "entity_order": "body_names",
            "flatten_order": "body_major_then_frame_major_then_feature",
            "features_per_frame": 9,
            "frame_feature_order": xz_feature_order if xz_encoding else xy_interleaved_feature_order,
        },
    }
    if target_and_error:
        contract["layout"] = (
            "link_target_window_then_error_window"
            if by_entity
            else "flattened_link_target_window_then_error_window"
        )
        contract["layout_detail"] = {
            "tensor_axes": (
                ["batch", "body", "target_window_then_error_window"]
                if by_entity
                else ["batch", "flattened_body_target_then_error_window"]
            ),
            "entity_order": "body_names",
            "body_token_sections": ["target_window", "target_minus_current_error_window"],
            "section_order": "frame_major_then_feature",
            "features_per_frame_per_section": 9,
            "frame_feature_order": xz_feature_order,
        }
    return contract


def _build_command_inputs_metadata(
    env: ManagerBasedRLEnv, actor_obs_groups: list[str], policy_inputs: list[dict]
) -> list[dict]:
    """Export the source command, temporal window, frames, and tensor layout."""
    tensor_names = {entry["name"]: entry["tensor_name"] for entry in policy_inputs}
    command_inputs = []
    for group_name in actor_obs_groups:
        for term_name in env.observation_manager.active_terms[group_name]:
            term_cfg = _term_cfg(env, group_name, term_name)
            params = dict(getattr(term_cfg, "params", {}) or {})
            if "command_name" not in params or "frame_offsets" not in params:
                continue
            function = getattr(term_cfg, "func", None)
            function_name = getattr(function, "__name__", type(function).__name__)
            pose_contract = _pose_command_contract(function_name)
            if pose_contract is None:
                # Still expose temporal/source information for custom terms;
                # consumers must not guess an unrecognized pose encoding.
                pose_contract = {
                    "position_frame": "custom_observation_function",
                    "rotation_encoding": "custom_observation_function",
                    "layout": "custom_observation_function",
                }

            command_name = str(params["command_name"])
            try:
                command = env.command_manager.get_term(command_name)
            except (KeyError, ValueError) as err:
                raise ValueError(
                    f"Observation '{group_name}.{term_name}' references missing command '{command_name}'."
                ) from err
            frame_offsets = [int(offset) for offset in params["frame_offsets"]]
            if not frame_offsets:
                raise ValueError(f"Observation '{group_name}.{term_name}' has an empty frame_offsets list.")

            fps_value = getattr(getattr(command, "motion", None), "fps", None)
            source_fps = float(fps_value) if fps_value is not None else None
            entry = {
                "observation_group": group_name,
                "term_name": term_name,
                "tensor_name": tensor_names[group_name],
                "observation_function": function_name,
                "command_name": command_name,
                "body_names": list(getattr(command.cfg, "body_names", []) or []),
                "anchor_body_name": getattr(command.cfg, "anchor_body_name", None),
                "frame_offsets": frame_offsets,
                "source_fps": source_fps,
                "reference_fps": source_fps,
                "quaternion_order": "wxyz",
                "position_unit": "metre",
                "zero_invalid_offsets": bool(params.get("zero_invalid_offsets", False)),
                **pose_contract,
            }
            if source_fps is not None:
                entry["frame_offset_seconds"] = [offset / source_fps for offset in frame_offsets]
                entry["required_history_s"] = max(0, -min(frame_offsets)) / source_fps
                entry["required_lookahead_s"] = max(0, max(frame_offsets)) / source_fps
            command_inputs.append(entry)
    return command_inputs


def attach_onnx_metadata(
    env: ManagerBasedRLEnv,
    run_path: str,
    path: str,
    filename="policy.onnx",
    actor_obs_groups: list[str] | None = None,
) -> None:
    onnx_path = os.path.join(path, filename)
    model = onnx.load(onnx_path)
    actor_obs_groups = _resolve_actor_obs_groups(env, actor_obs_groups, model)
    observation_names, observation_history_lengths = _build_legacy_observation_metadata(env, actor_obs_groups)
    policy_inputs = _build_policy_inputs(env, actor_obs_groups, model)
    command_inputs = _build_command_inputs_metadata(env, actor_obs_groups, policy_inputs)
    control_period_s = _control_period_s(env)
    metadata = {
        "schema_version": 3,
        # "run_path": run_path,
        **_build_joint_metadata(env),
        # "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "control_period_s": control_period_s,
        "control_frequency_hz": 1.0 / control_period_s,
        "mnn_runtime_target_version": "2.9.5",
        "model_input_normalization": "embedded_in_model_graph",
        "external_input_normalization_required": False,
        "policy_inputs": policy_inputs,
        "command_inputs": command_inputs,
    }
    if len(command_inputs) == 1:
        # Convenience alias for the common one-command actor and the native
        # SDK loader. Multi-command actors retain the unambiguous list only.
        metadata["command_contract"] = copy.deepcopy(command_inputs[0])

    # 保存文件
    class CustomListDumper(yaml.SafeDumper):
        def represent_sequence(self, tag, sequence, flow_style=None):
            # Keep legacy scalar arrays compact, but render schema-v3 object lists
            # as readable YAML blocks.
            flow_style = all(not isinstance(item, (dict, list)) for item in sequence)

            # 调用父类方法生成序列节点
            node = yaml.SafeDumper.represent_sequence(self, tag, sequence, flow_style=flow_style)

            # 修改节点中的字符串项，添加引号
            if isinstance(sequence, list):
                for i, item_node in enumerate(node.value):
                    if isinstance(sequence[i], str):
                        # 字符串节点添加双引号
                        item_node.style = '"'

            return node

    # 保存文件
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(os.path.join(path, "deploy_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(
            metadata,
            f,
            Dumper=CustomListDumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
        )
    # ``attach_onnx_metadata`` is intentionally idempotent: replace keys owned
    # by this exporter while retaining third-party ONNX metadata.
    retained_metadata = [(entry.key, entry.value) for entry in model.metadata_props if entry.key not in metadata]
    model.ClearField("metadata_props")
    for key, value in retained_metadata:
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = value
        model.metadata_props.append(entry)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        # Preserve the historic CSV encoding for flat arrays. Structured
        # schema-v3 values use YAML so nested input descriptions are lossless.
        if isinstance(v, list) and all(not isinstance(item, (dict, list)) for item in v):
            entry.value = list_to_csv_str(v)
        elif isinstance(v, (dict, list)):
            entry.value = yaml.safe_dump(v, sort_keys=False).strip()
        else:
            entry.value = str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
