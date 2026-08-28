from __future__ import annotations

import gymnasium as gym
import torch
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v20 import (
    T800FlatV20ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v21 import (
    T800FlatV21ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v20 import (
    T800FlatWoStateEstimationEnvCfgV20Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v21 import (
    T800FlatWoStateEstimationEnvCfgV21Scale,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import MotionCommandV1
from engineai_rl_lab.tasks.tracking.mdp.events import randomize_joint_default_pos
from engineai_rl_lab.tasks.tracking.robots.t800 import (
    T800_CYLINDER_CFG,
    T800_FIXED_HEAD_CFG,
    T800_HEAD_JOINT_POSITIONS,
    T800_POLICY_JOINT_NAMES,
)
from rsl_rl.models import ModelGraph
from tensordict import TensorDict


def test_v21_controls_23_joints_and_holds_the_head_neutral() -> None:
    v20 = T800FlatWoStateEstimationEnvCfgV20Scale()
    v21 = T800FlatWoStateEstimationEnvCfgV21Scale()
    action = v21.actions.joint_pos

    assert v20.actions.joint_pos.joint_names == [".*"]
    assert v20.commands.motion.fixed_joint_positions == {}
    assert action.class_type.endswith(":JointPositionAction")
    assert action.joint_names == T800_POLICY_JOINT_NAMES
    assert action.preserve_order is True
    assert len(action.scale) == 23
    assert v21.commands.motion.fixed_joint_positions == T800_HEAD_JOINT_POSITIONS

    assert "fixed_head" not in T800_CYLINDER_CFG.actuators
    assert T800_FIXED_HEAD_CFG.actuators["heads_and_elbow_yaw"].joint_names_expr == [
        "J17_ELBOW_YAW_L",
        "J22_ELBOW_YAW_R",
    ]
    fixed_head_actuator = T800_FIXED_HEAD_CFG.actuators["fixed_head"]
    assert fixed_head_actuator.joint_names_expr == ["J23_HEAD_PITCH", "J24_HEAD_YAW"]
    assert fixed_head_actuator.fixed_position == 0.0


def test_v21_excludes_head_from_joint_observations_randomization_and_limits() -> None:
    cfg = T800FlatWoStateEstimationEnvCfgV21Scale()

    for group_name in ("proprioception", "critic"):
        group = getattr(cfg.observations, group_name)
        assert group.joint_pos.params["asset_cfg"].joint_names == T800_POLICY_JOINT_NAMES
        assert group.joint_vel.params["asset_cfg"].joint_names == T800_POLICY_JOINT_NAMES
    assert cfg.observations.proprioception.joint_vel.noise.joint_names == T800_POLICY_JOINT_NAMES
    assert cfg.events.add_joint_default_pos.params["asset_cfg"].joint_names == T800_POLICY_JOINT_NAMES
    assert cfg.rewards.joint_limit.params["asset_cfg"].joint_names == T800_POLICY_JOINT_NAMES
    assert "LINK_HEAD_YAW" not in cfg.commands.motion.body_names


def test_motion_command_v1_overrides_fixed_joint_reference_state() -> None:
    command = MotionCommandV1.__new__(MotionCommandV1)
    command._fixed_joint_ids = torch.tensor([1, 3])
    command._fixed_joint_positions = torch.tensor([0.25, -0.5])
    joint_pos = torch.zeros(2, 5)
    joint_vel = torch.ones(2, 5)

    command._apply_fixed_joint_state(joint_pos, joint_vel)

    assert torch.equal(joint_pos[:, [1, 3]], torch.tensor([[0.25, -0.5], [0.25, -0.5]]))
    assert torch.equal(joint_vel[:, [1, 3]], torch.zeros(2, 2))
    assert torch.equal(joint_vel[:, [0, 2, 4]], torch.ones(2, 3))


def test_motion_command_v1_excludes_fixed_joints_from_command() -> None:
    command = MotionCommandV1.__new__(MotionCommandV1)
    command._tracked_joint_ids = torch.tensor([0, 2, 4])
    command._motion_cache = {
        "joint_pos": torch.tensor([[0.0, 10.0, 2.0, 30.0, 4.0]]),
        "joint_vel": torch.tensor([[5.0, 60.0, 7.0, 80.0, 9.0]]),
    }

    assert torch.equal(command.command, torch.tensor([[0.0, 2.0, 4.0, 5.0, 7.0, 9.0]]))


def test_default_position_randomization_updates_subset_action_offsets_by_name(monkeypatch) -> None:
    class Data:
        default_joint_pos = type("Field", (), {"torch": torch.zeros(2, 4)})()

    class Asset:
        device = "cpu"
        joint_names = ["j0", "head_pitch", "j1", "head_yaw"]
        data = Data()

    class Action:
        _joint_names = ["j0", "j1"]
        _offset = torch.zeros(2, 2)

    action = Action()
    env = type(
        "Env",
        (),
        {
            "scene": type("Scene", (), {"num_envs": 2, "__getitem__": lambda self, name: Asset()})(),
            "action_manager": type("ActionManager", (), {"get_term": lambda self, name: action})(),
        },
    )()
    asset_cfg = type("AssetCfg", (), {"name": "robot", "joint_ids": [0, 2]})()
    randomized = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    randomized_full = torch.tensor([[0.1, 0.0, 0.2, 0.0], [0.3, 0.0, 0.4, 0.0]])
    monkeypatch.setattr(
        "engineai_rl_lab.tasks.tracking.mdp.events._randomize_prop_by_op",
        lambda values, *args, **kwargs: randomized_full,
    )

    randomize_joint_default_pos(env, None, asset_cfg, pos_distribution_params=(-0.01, 0.01))

    assert torch.equal(action._offset, randomized)


def test_v21_actor_accepts_23_dof_histories_and_outputs_23_actions() -> None:
    runner_cfg = T800FlatV21ScalePPORunnerCfg()
    observations = TensorDict(
        {
            "proprioception": torch.randn(2, 5, 52),
            "action": torch.randn(2, 4, 23),
            "command": torch.randn(2, 14, 99),
        },
        batch_size=[2],
    )
    actor_cfg = runner_cfg.actor.to_dict()
    actor_cfg.pop("class_name")

    actor = ModelGraph(observations, runner_cfg.obs_groups, "actor", 23, **actor_cfg)

    assert actor.nodes["proprioception_projection"].projection.in_features == 52
    assert actor.nodes["action_projection"].projection.in_features == 23
    assert actor(observations).shape == (2, 23)


def test_v21_uses_tighter_gradient_clipping_without_mutating_v20() -> None:
    v20 = T800FlatV20ScalePPORunnerCfg()
    v21 = T800FlatV21ScalePPORunnerCfg()

    assert v20.algorithm.max_grad_norm == 1.0
    assert v21.algorithm.max_grad_norm == 0.1


def test_v21_registration_points_to_v21_configs() -> None:
    spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v21-scale")

    assert spec.kwargs["env_cfg_entry_point"].endswith(".flat_env_cfg_v21:T800FlatWoStateEstimationEnvCfgV21Scale")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        ".rsl_rl_ppo_cfg_v21:T800FlatV21ScalePPORunnerCfg"
    )
