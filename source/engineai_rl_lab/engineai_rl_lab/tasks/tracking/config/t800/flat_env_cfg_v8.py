from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v7 import (
    T800FlatWoStateEstimationEnvCfgV7,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v7_scale import LAFAN_SCALE_MOTION_MANIFEST


@configclass
class T800ScaleTrackRewardsCfg:
    """ScaleTrack tracking rewards mapped onto the T800 motion bodies."""

    motion_body_height = RewTerm(
        func=mdp.motion_global_anchor_height_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.45},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_global_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.45},
    )
    motion_body_rot = RewTerm(
        func=mdp.motion_global_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits_capped,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "max_error_per_joint": 1.0,
            "max_total_error": 1.0,
        },
    )
    survival = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class T800ScaleTrackTerminationsCfg:
    """ScaleTrack tracking terminations mapped onto MotionCommand."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    invalid_robot_state = DoneTerm(
        func=mdp.nonfinite_robot_state,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    motion_time_out = DoneTerm(
        func=mdp.motion_time_out,
        time_out=True,
        params={"command_name": "motion"},
    )
    body_pos = DoneTerm(
        func=mdp.bad_global_motion_body_pos,
        params={"command_name": "motion", "threshold": 0.6},
    )


@configclass
class T800FlatWoStateEstimationEnvCfgV8(T800FlatWoStateEstimationEnvCfgV7):
    """V7 environment with ScaleTrack rewards and terminations."""

    def __post_init__(self):
        super().__post_init__()
        # Parent post-init hooks tune the legacy V7 terms, so replace the two
        # complete MDP groups only after that inheritance chain has finished.
        self.rewards = T800ScaleTrackRewardsCfg()
        self.terminations = T800ScaleTrackTerminationsCfg()
        self.commands.motion.resample_at_motion_end = False


@configclass
class T800FlatWoStateEstimationEnvCfgV8Scale(T800FlatWoStateEstimationEnvCfgV8):
    """V8 environment trained on regular and 2x-slow LaFAN motions."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = LAFAN_SCALE_MOTION_MANIFEST
        self.commands.motion.motion_load_workers = 4
        # Keep each SONIC-style working set in one contiguous in-memory chunk.
        self.commands.motion.motion_chunk_frames = 8_388_608
        self.commands.motion.adp_samp_failure_rate_max_over_mean = 200.0
        self.commands.motion.pre_failure_sample_window = 200
