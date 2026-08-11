from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from engineai_rl_lab.tasks.tracking.robots.t800 import T800_ACTION_SCALE, T800_CYLINDER_CFG
from engineai_rl_lab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class T800FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T800_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 3.5
        self.actions.joint_pos.scale = T800_ACTION_SCALE
        self.commands.motion.anchor_body_name = "LINK_BASE"
        self.commands.motion.body_names = [
            "LINK_BASE",
            "LINK_HIP_ROLL_L",
            "LINK_KNEE_PITCH_L",
            "LINK_ANKLE_ROLL_L",
            "LINK_HIP_ROLL_R",
            "LINK_KNEE_PITCH_R",
            "LINK_ANKLE_ROLL_R",
            "LINK_WAIST_YAW",
            "LINK_SHOULDER_ROLL_L",
            "LINK_ELBOW_YAW_L",
            "LINK_WRIST_END_L",
            "LINK_SHOULDER_ROLL_R",
            "LINK_ELBOW_YAW_R",
            "LINK_WRIST_END_R",
            "LINK_HEAD_YAW",
        ]
        self.events.base_com.params["asset_cfg"].body_names = "LINK_WAIST_YAW"
        self.events.base_com.params["com_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)}
        self.rewards.motion_global_anchor_pos.params["std"] = 0.45
        self.rewards.motion_body_pos.params["std"] = 0.45
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!LINK_ANKLE_ROLL_L$)(?!LINK_ANKLE_ROLL_R$)(?!LINK_WRIST_END_L$)(?!LINK_WRIST_END_R$).+$"
        ]
        self.terminations.anchor_pos.params["threshold"] = 0.35
        self.terminations.ee_body_pos.params["threshold"] = 0.35
        self.terminations.ee_body_pos.params["body_names"] = [
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_WRIST_END_L",
            "LINK_WRIST_END_R",
        ]
        self.rewards.action_rate_l2.weight = -0.03


@configclass
class T800FlatWoStateEstimationEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        self.rewards.action_rate_l2.weight = -0.075


@configclass
class T800FlatLowFreqEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
