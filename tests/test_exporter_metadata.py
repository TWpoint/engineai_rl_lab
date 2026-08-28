from __future__ import annotations

import math
from types import SimpleNamespace

import onnx
import torch
import yaml
from onnx import TensorProto, helper

from engineai_rl_lab.utils.exporter import attach_onnx_metadata, rewrite_silu_for_mnn_2_9_5


def motion_body_pose_reference_anchor_window_by_entity_xz():
    """Name-only test double for the observation function."""


def last_action():
    """Name-only test double for Isaac Lab's action observation."""


def _term(func, *, params=None, scale=None, history_length=None):
    return SimpleNamespace(
        func=func,
        params=params or {},
        scale=scale,
        clip=None,
        history_length=history_length,
    )


class _GroupCfg:
    def __init__(self, terms, *, history_length=None, flatten_history_dim=False):
        self._terms = terms
        self.history_length = history_length
        self.flatten_history_dim = flatten_history_dim
        for name, term in terms.items():
            setattr(self, name, term)

    def to_dict(self):
        return {
            name: {"history_length": term.history_length}
            for name, term in self._terms.items()
        }


class _Scene(dict):
    def __init__(self, robot, sdk_joint_names):
        super().__init__(robot=robot)
        self.cfg = SimpleNamespace(robot=SimpleNamespace(joint_sdk_names=sdk_joint_names))


def _make_env():
    internal_joint_names = ["J00", "J02_HEAD", "J01", "J03_HEAD"]
    sdk_joint_names = ["J00", "J01", "J02_HEAD", "J03_HEAD"]
    robot = SimpleNamespace(
        joint_names=internal_joint_names,
        data=SimpleNamespace(
            default_joint_pos_nominal=torch.tensor([[0.0, 2.0, 1.0, 3.0]]),
            joint_stiffness=SimpleNamespace(torch=torch.tensor([[10.0, 12.0, 11.0, 13.0]])),
            joint_damping=SimpleNamespace(torch=torch.tensor([[1.0, 1.2, 1.1, 1.3]])),
            joint_pos_limits=SimpleNamespace(
                torch=torch.tensor([[[-1.0, 1.0], [-3.0, 3.0], [-2.0, 2.0], [-4.0, 4.0]]])
            ),
            soft_joint_pos_limits=SimpleNamespace(
                torch=torch.tensor([[[-0.9, 0.9], [-2.7, 2.7], [-1.8, 1.8], [-3.6, 3.6]]])
            ),
        ),
    )
    action_term = SimpleNamespace(
        _joint_names=["J00", "J01"],
        action_dim=2,
        _scale=torch.tensor([[0.5, 0.2]]),
    )

    selected_joints = SimpleNamespace(joint_ids=[0, 2], joint_names=["J00", "J01"])
    proprioception_terms = {
        "base_ang_vel": _term(lambda: None),
        "joint_pos": _term(lambda: None, params={"asset_cfg": selected_joints}),
        "joint_vel": _term(lambda: None, params={"asset_cfg": selected_joints}, scale=0.05),
    }
    action_terms = {"actions": _term(last_action)}
    command_terms = {
        "link_pose_b": _term(
            motion_body_pose_reference_anchor_window_by_entity_xz,
            params={
                "command_name": "motion",
                "frame_offsets": [0, 1],
                "zero_invalid_offsets": True,
            },
        )
    }
    observation_cfg = SimpleNamespace(
        proprioception=_GroupCfg(proprioception_terms, history_length=5),
        action=_GroupCfg(action_terms, history_length=4),
        command=_GroupCfg(command_terms, flatten_history_dim=True),
    )
    observation_manager = SimpleNamespace(
        cfg=observation_cfg,
        active_terms={
            "proprioception": list(proprioception_terms),
            "action": list(action_terms),
            "command": list(command_terms),
        },
        _group_obs_term_dim={
            "proprioception": [(5, 3), (5, 2), (5, 2)],
            "action": [(4, 2)],
            "command": [(2, 18)],
        },
    )
    motion = SimpleNamespace(
        cfg=SimpleNamespace(
            body_names=["BASE", "HAND"],
            anchor_body_name="BASE",
            fixed_joint_positions={"J02_HEAD": 0.0, "J03_HEAD": 0.0},
        ),
        motion=SimpleNamespace(fps=50.0),
    )
    command_manager = SimpleNamespace(
        active_terms=["motion"],
        get_term=lambda name: motion if name == "motion" else (_ for _ in ()).throw(KeyError(name)),
    )
    return SimpleNamespace(
        scene=_Scene(robot, sdk_joint_names),
        observation_manager=observation_manager,
        action_manager=SimpleNamespace(get_term=lambda name: action_term),
        command_manager=command_manager,
        step_dt=0.02,
    )


def _write_named_input_model(path):
    inputs = [
        helper.make_tensor_value_info("proprioception", TensorProto.FLOAT, [1, 5, 7]),
        helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 4, 2]),
        helper.make_tensor_value_info("command", TensorProto.FLOAT, [1, 2, 18]),
    ]
    output = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 4, 2])
    graph = helper.make_graph([helper.make_node("Identity", ["action"], ["actions"])], "test", inputs, [output])
    onnx.save(helper.make_model(graph), path)


def test_attach_metadata_exports_action_sdk_and_command_orders(tmp_path):
    onnx_path = tmp_path / "policy.onnx"
    _write_named_input_model(onnx_path)

    attach_onnx_metadata(_make_env(), "unused", str(tmp_path))

    metadata = yaml.safe_load((tmp_path / "deploy_config.yaml").read_text())
    assert metadata["schema_version"] == 3
    assert metadata["joint_names"] == ["J00", "J01", "J02_HEAD", "J03_HEAD"]
    assert metadata["action_joint_names"] == ["J00", "J01"]
    assert metadata["default_joint_pos"] == [0.0, 1.0, 2.0, 3.0]
    assert metadata["joint_stiffness"] == [10.0, 11.0, 12.0, 13.0]
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-6)
        for actual, expected in zip(metadata["joint_damping"], [1.0, 1.1, 1.2, 1.3])
    )
    assert metadata["action_default_joint_pos"] == [0.0, 1.0]
    assert metadata["action_joint_stiffness"] == [10.0, 11.0]
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-6)
        for actual, expected in zip(metadata["action_joint_damping"], [1.0, 1.1])
    )
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-6)
        for actual, expected in zip(metadata["action_scale"], [0.5, 0.2])
    )
    assert metadata["robot_joint_names"] == ["J00", "J01", "J02_HEAD", "J03_HEAD"]
    assert metadata["robot_default_joint_pos"] == [0.0, 1.0, 2.0, 3.0]
    assert metadata["robot_joint_position_lower_limit"] == [-1.0, -2.0, -3.0, -4.0]
    assert metadata["robot_joint_position_upper_limit"] == [1.0, 2.0, 3.0, 4.0]
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-6)
        for actual, expected in zip(
            metadata["robot_joint_soft_position_lower_limit"], [-0.9, -1.8, -2.7, -3.6]
        )
    )
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-6)
        for actual, expected in zip(
            metadata["robot_joint_soft_position_upper_limit"], [0.9, 1.8, 2.7, 3.6]
        )
    )
    assert metadata["joint_limit_order"] == "robot_joint_names"
    assert metadata["joint_limit_unit"] == "radian"
    assert metadata["fixed_joint_positions"] == {"J02_HEAD": 0.0, "J03_HEAD": 0.0}
    assert math.isclose(metadata["control_period_s"], 0.02)
    assert metadata["mnn_runtime_target_version"] == "2.9.5"
    assert metadata["model_input_normalization"] == "embedded_in_model_graph"
    assert metadata["external_input_normalization_required"] is False

    proprioception = metadata["policy_inputs"][0]
    assert proprioception["shape"] == [5, 7]
    assert proprioception["history_initialization"] == "repeat_first_sample"
    assert proprioception["terms"][1]["joint_names"] == ["J00", "J01"]
    assert math.isclose(proprioception["terms"][2]["scale"], 0.05, abs_tol=1.0e-6)

    action = metadata["policy_inputs"][1]
    assert action["history_initialization"] == "repeat_first_sample"
    assert action["terms"][0]["value_semantics"] == "previous_raw_policy_action"
    assert action["terms"][0]["reset_value"] == "zero"

    command = metadata["command_inputs"][0]
    assert command["body_names"] == ["BASE", "HAND"]
    assert command["anchor_body_name"] == "BASE"
    assert command["frame_offsets"] == [0, 1]
    assert math.isclose(command["source_fps"], 50.0)
    assert math.isclose(command["reference_fps"], 50.0)
    assert command["quaternion_order"] == "wxyz"
    assert command["position_unit"] == "metre"
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(command["frame_offset_seconds"], [0.0, 0.02])
    )
    assert command["position_frame"] == "reference_anchor_translation_robot_anchor_axes"
    assert command["position_frame_detail"]["origin"] == "reference_anchor_position_at_current_frame"
    assert command["rotation_encoding"] == "matrix_columns_x_z"
    assert command["layout"] == "link_offset_feature"
    assert command["layout_detail"]["flatten_order"] == "body_major_then_frame_major_then_feature"
    assert metadata["command_contract"] == command

    embedded = {entry.key: entry.value for entry in onnx.load(onnx_path).metadata_props}
    assert yaml.safe_load(embedded["command_inputs"])[0]["frame_offsets"] == [0, 1]


def test_mnn_2_9_5_rewrite_only_lowers_sigmoid_used_by_silu(tmp_path):
    input_path = tmp_path / "silu.onnx"
    output_path = tmp_path / "silu_compatible.onnx"
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2]),
        helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2]),
    ]
    outputs = [
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2]),
        helper.make_tensor_value_info("sigmoid_z", TensorProto.FLOAT, [1, 2]),
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Sigmoid", ["x"], ["sigmoid_x"], name="silu_sigmoid"),
            helper.make_node("Mul", ["x", "sigmoid_x"], ["y"], name="silu_mul"),
            helper.make_node("Sigmoid", ["z"], ["sigmoid_z"], name="ordinary_sigmoid"),
        ],
        "silu_test",
        inputs,
        outputs,
    )
    onnx.save(helper.make_model(graph), input_path)

    assert rewrite_silu_for_mnn_2_9_5(str(input_path), str(output_path)) == 1

    compatible = onnx.load(output_path)
    node_types = [node.op_type for node in compatible.graph.node]
    assert node_types.count("Sigmoid") == 1
    assert all(op_type in node_types for op_type in ("Neg", "Exp", "Constant", "Add", "Reciprocal"))
    onnx.checker.check_model(compatible)
