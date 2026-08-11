# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import numpy as np
import os

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

    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


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


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    robot = env.scene["robot"]

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

    metadata = {
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

    # 保存文件
    class CustomListDumper(yaml.SafeDumper):
        def represent_sequence(self, tag, sequence, flow_style=None):
            # 如果是列表，使用流式风格
            flow_style = True  # 强制使用流式风格 [item1, item2, ...]

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
    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
