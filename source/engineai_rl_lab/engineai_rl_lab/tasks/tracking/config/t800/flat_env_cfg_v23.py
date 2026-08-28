from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.managers import (
    SceneEntityCfg,
)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v13 import T800FlatObservationsCfgV13
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v22 import (
    T800_SONIC_END_EFFECTOR_BODY_NAMES,
    T800_SONIC_END_EFFECTOR_BODY_OFFSETS,
    T800FlatWoStateEstimationEnvCfgV22Scale,
)
from engineai_rl_lab.tasks.tracking.robots.t800 import T800_POLICY_JOINT_NAMES

T800_V23_MOTION_CATALOG_CACHE = "/mnt/data-1/lpz/t800_datasets/t800_v0.motion_catalog.json"

T800_V23_TERMINATION_END_EFFECTOR_BODY_NAMES = [
    "LINK_WRIST_END_L",
    "LINK_WRIST_END_R",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
]


@configclass
class T800FlatObservationsCfgV23(T800FlatObservationsCfgV13):
    """V22 observations plus joint-reference tracking state for the critic."""

    @configclass
    class CriticCfg(T800FlatObservationsCfgV13.CriticCfg):
        motion_joint_pos = ObsTerm(
            func=mdp.motion_joint_position_target_and_error,
            params={
                "command_name": "motion",
                "joint_names": T800_POLICY_JOINT_NAMES,
            },
        )
        motion_joint_vel = ObsTerm(
            func=mdp.motion_joint_velocity_target_and_error,
            params={
                "command_name": "motion",
                "joint_names": T800_POLICY_JOINT_NAMES,
            },
        )

    critic: CriticCfg = CriticCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV23Scale(T800FlatWoStateEstimationEnvCfgV22Scale):
    """V22 plus 23-DoF joint tracking rewards and critic observations."""

    observations: T800FlatObservationsCfgV23 = T800FlatObservationsCfgV23()

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.motion_catalog_cache = T800_V23_MOTION_CATALOG_CACHE
        self.commands.motion.motion_shard_across_ranks = True
        self.commands.motion.motion_load_workers = 4

        sampler = self.commands.motion.adaptive_sampling
        # Start up to one complete 50 Hz bin before a sampled hard point, so
        # the policy observes the context that leads into a failure.
        sampler.pre_failure_window = 50
        # TerminationManager already exposes one aggregate done bit per env;
        # retain that invariant even if reset ids are accidentally repeated.
        sampler.deduplicate_failure_events = True
        # Blend the original global error with the V22 relative/local rewards.
        # Scales are deliberately looser than the reward kernels so moderate
        # pose errors do not make a bin look maximally difficult.
        sampler.tracking_error_scale = 0.40
        sampler.global_tracking_error_weight = 0.40
        sampler.relative_position_error_weight = 0.20
        sampler.relative_position_error_scale = 0.40
        sampler.relative_orientation_error_weight = 0.20
        sampler.relative_orientation_error_scale = 0.60
        sampler.local_position_error_weight = 0.20
        sampler.local_position_error_scale = 0.20
        sampler.local_position_body_names = list(T800_SONIC_END_EFFECTOR_BODY_NAMES)
        sampler.local_position_body_offsets = [list(offset) for offset in T800_SONIC_END_EFFECTOR_BODY_OFFSETS]

        # Immediate recovery/fall boundaries, with slightly looser thresholds
        # than the reference implementation. These supplement the inherited
        # 0.5 m global-body-position termination; there is no delay or grace.
        self.terminations.anchor_height = DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.30},
        )
        self.terminations.anchor_orientation = DoneTerm(
            func=mdp.bad_anchor_ori,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "motion",
                "threshold": 0.90,
            },
        )
        self.terminations.end_effector_height = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.30,
                "body_names": T800_V23_TERMINATION_END_EFFECTOR_BODY_NAMES,
            },
        )

        self.rewards.motion_joint_pos = RewTerm(
            func=mdp.motion_joint_position_error_exp,
            weight=0.5,
            params={
                "command_name": "motion",
                "std": 0.5,
                "joint_names": T800_POLICY_JOINT_NAMES,
            },
        )
        self.rewards.motion_joint_vel = RewTerm(
            func=mdp.motion_joint_velocity_error_exp,
            weight=0.25,
            params={
                "command_name": "motion",
                "std": 3.0,
                "joint_names": T800_POLICY_JOINT_NAMES,
            },
        )
