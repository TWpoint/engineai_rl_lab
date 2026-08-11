from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.pm01.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from engineai_rl_lab.tasks.tracking.robots.pm01 import PM01_ACTION_SCALE, PM01_CYLINDER_CFG
from engineai_rl_lab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class PM01FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PM01_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = PM01_ACTION_SCALE
        self.commands.motion.anchor_body_name = "LINK_BASE"
        self.commands.motion.body_names = [
            "LINK_BASE",
            "LINK_HIP_ROLL_L",
            "LINK_KNEE_PITCH_L",
            "LINK_ANKLE_ROLL_L",
            "LINK_HIP_ROLL_R",
            "LINK_KNEE_PITCH_R",
            "LINK_ANKLE_ROLL_R",
            "LINK_TORSO_YAW",
            "LINK_SHOULDER_ROLL_L",
            "LINK_ELBOW_YAW_L",
            "LINK_SHOULDER_ROLL_R",
            "LINK_ELBOW_YAW_R",
            "LINK_HEAD_YAW",
        ]
        self.events.base_com.params["asset_cfg"].body_names = "LINK_BASE"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!LINK_ANKLE_ROLL_L$)(?!LINK_ANKLE_ROLL_R$)(?!LINK_ELBOW_YAW_L$)(?!LINK_ELBOW_YAW_R$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = [
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_ELBOW_YAW_L",
            "LINK_ELBOW_YAW_R",
        ]


@configclass
class PM01FlatWoStateEstimationEnvCfg(PM01FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class PM01FlatLowFreqEnvCfg(PM01FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
