from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v12 import (
    T800FlatObservationsCfgV12,
    T800FlatWoStateEstimationEnvCfgV12Scale,
)

T800_BEYONDMINIC_FRAME_OFFSETS = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]


@configclass
class T800FlatObservationsCfgV13(T800FlatObservationsCfgV12):
    """BeyondMinic target-and-error command tokens with V12 proprioception."""

    @configclass
    class CommandCfg(ObsGroup):
        link_pose_b = ObsTerm(
            func=mdp.motion_body_pose_and_error_b_window_by_entity,
            params={
                "command_name": "motion",
                "frame_offsets": T800_BEYONDMINIC_FRAME_OFFSETS,
                "zero_invalid_offsets": True,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class CriticCfg(T800FlatObservationsCfgV12.PrivilegedCfg):
        command = ObsTerm(
            func=mdp.motion_body_pose_and_error_b_window_flat,
            params={"command_name": "motion", "frame_offsets": [0], "zero_invalid_offsets": True},
        )
        target_link_pos_b = None
        target_link_ori_b = None
        body_lin_vel = ObsTerm(func=mdp.robot_body_lin_vel_b, params={"command_name": "motion"})
        body_ang_vel = ObsTerm(func=mdp.robot_body_ang_vel_b, params={"command_name": "motion"})

    command: CommandCfg = CommandCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class T800BeyondMinicRewardsCfg:
    """BeyondMinic global tracking rewards adapted to T800."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    motion_global_root_height = RewTerm(
        func=mdp.motion_global_anchor_height_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_global_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
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
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class T800BeyondMinicTerminationsCfg:
    """BeyondMinic global tracking failure and successful motion completion."""

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
        params={"command_name": "motion", "threshold": 0.5},
    )


@configclass
class T800FlatWoStateEstimationEnvCfgV13Scale(T800FlatWoStateEstimationEnvCfgV12Scale):
    """V12-scale with T800-scaled BeyondMinic commands, rewards, and terminations."""

    observations: T800FlatObservationsCfgV13 = T800FlatObservationsCfgV13()

    def __post_init__(self):
        super().__post_init__()
        # The two training nodes each provide 800 GB RAM, so keep the complete
        # 151k-motion corpus resident and avoid working-set distribution shifts.
        self.commands.motion.max_num_load_motions = 200_000
        self.commands.motion.motion_load_workers = 8
        self.commands.motion.working_set_replacement = False
        self.commands.motion.bin_size = 50
        self.commands.motion.sequence_length_agnostic = False
        self.commands.motion.adaptive_sampling_alpha = 0.001
        self.commands.motion.adaptive_kernel_size = 5
        self.commands.motion.adaptive_kernel_lambda = 0.8
        self.commands.motion.uniform_sampling_rate = 0.2
        self.commands.motion.adp_samp_failure_rate_max_over_mean = 30.0
        self.commands.motion.pre_failure_sample_window = 0
        # Match MarmotLab BeyondMinic's torso COM randomization.  The inherited
        # T800 range (0.1 m on every axis) is substantially more aggressive.
        self.events.base_com.params["com_range"] = {
            "x": (-0.025, 0.025),
            "y": (-0.05, 0.05),
            "z": (-0.05, 0.05),
        }
        self.rewards = T800BeyondMinicRewardsCfg()
        self.terminations = T800BeyondMinicTerminationsCfg()
