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


def _build_policy_inputs(env: ManagerBasedRLEnv, actor_obs_groups: list[str], model: onnx.ModelProto) -> list[dict]:
    """Build schema-v2 named input descriptions for the actor observation groups."""
    input_shapes = _onnx_input_shapes(model)
    expected_input_names = ["obs"] if actor_obs_groups == ["policy"] else actor_obs_groups
    if list(input_shapes) != expected_input_names:
        raise ValueError(
            "ONNX actor inputs do not match configured actor observation groups: "
            f"expected {expected_input_names}, got {list(input_shapes)}."
        )

    policy_inputs = []
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
            term_entries.append({"name": term_name, "size": term_size_with_history // term_history_length})

        shape = input_shapes[tensor_name]
        if math.prod(shape) != sum(term["size"] * _term_history_length(group_cfg, term["name"]) for term in term_entries):
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
                "terms": term_entries,
            }
        )
    return policy_inputs


def attach_onnx_metadata(
    env: ManagerBasedRLEnv,
    run_path: str,
    path: str,
    filename="policy.onnx",
    actor_obs_groups: list[str] | None = None,
) -> None:
    onnx_path = os.path.join(path, filename)
    robot = env.scene["robot"]

    actor_obs_groups = actor_obs_groups or ["policy"]
    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    default_joint_pos_nominal = getattr(robot.data, "default_joint_pos_nominal", None)
    if default_joint_pos_nominal is None:
        default_joint_pos_nominal = robot.data.default_joint_pos.torch[0]

    action_scale = env.action_manager.get_term("joint_pos")._scale
    if isinstance(action_scale, torch.Tensor):
        if action_scale.ndim > 1:
            action_scale = action_scale[0]
        action_scale = action_scale.cpu().tolist()
    else:
        action_scale = [float(action_scale)] * len(robot.joint_names)

    model = onnx.load(onnx_path)
    metadata = {
        "schema_version": 2,
        # "run_path": run_path,
        "default_joint_pos": default_joint_pos_nominal.cpu().tolist(),
        "joint_names": robot.joint_names,
        "joint_stiffness": robot.data.joint_stiffness.torch[0].cpu().tolist(),
        "joint_damping": robot.data.joint_damping.torch[0].cpu().tolist(),
        # "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": action_scale,
        # "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        # "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }
    metadata["policy_inputs"] = _build_policy_inputs(env, list(actor_obs_groups), model)

    # 保存文件
    class CustomListDumper(yaml.SafeDumper):
        def represent_sequence(self, tag, sequence, flow_style=None):
            # Keep legacy scalar arrays compact, but render schema-v2 object lists
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
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        # Preserve the historic CSV encoding for flat arrays. Structured
        # schema-v2 values use YAML so nested input descriptions are lossless.
        if isinstance(v, list) and all(not isinstance(item, (dict, list)) for item in v):
            entry.value = list_to_csv_str(v)
        elif isinstance(v, (dict, list)):
            entry.value = yaml.safe_dump(v, sort_keys=False).strip()
        else:
            entry.value = str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
