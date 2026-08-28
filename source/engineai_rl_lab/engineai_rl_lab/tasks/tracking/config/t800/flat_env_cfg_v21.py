from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v20 import (
    T800FlatWoStateEstimationEnvCfgV20Scale,
)
from engineai_rl_lab.tasks.tracking.robots.t800 import (
    T800_FIXED_HEAD_CFG,
    T800_HEAD_JOINT_POSITIONS,
    T800_POLICY_ACTION_SCALE,
    T800_POLICY_JOINT_NAMES,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV21Scale(T800FlatWoStateEstimationEnvCfgV20Scale):
    """V20 task with a 23-DoF policy and both head joints held neutral."""

    def __post_init__(self):
        super().__post_init__()

        policy_joints = list(T800_POLICY_JOINT_NAMES)
        fixed_head = dict(T800_HEAD_JOINT_POSITIONS)
        policy_asset = SceneEntityCfg(
            "robot",
            joint_names=policy_joints,
            preserve_order=True,
        )

        self.scene.robot = T800_FIXED_HEAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=policy_joints,
            preserve_order=True,
            scale=dict(T800_POLICY_ACTION_SCALE),
            use_default_offset=True,
        )

        for group_name in ("proprioception", "critic"):
            group = getattr(self.observations, group_name)
            group.joint_pos.params["asset_cfg"] = policy_asset
            group.joint_vel.params["asset_cfg"] = policy_asset
        self.observations.proprioception.joint_vel.noise.joint_names = policy_joints

        self.events.add_joint_default_pos.params["asset_cfg"] = policy_asset
        self.rewards.joint_limit.params["asset_cfg"] = policy_asset
        self.commands.motion.fixed_joint_positions = fixed_head
