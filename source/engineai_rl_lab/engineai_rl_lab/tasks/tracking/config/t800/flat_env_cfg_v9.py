from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v7 import T800FlatObservationsCfgV7
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v8 import (
    T800FlatWoStateEstimationEnvCfgV8,
    T800FlatWoStateEstimationEnvCfgV8Scale,
)

V9_FRAME_OFFSETS = [-3, -2, -1, 0, 1, 2, 3, 4, 5]


def _enable_single_motion_overfit(env_cfg) -> None:
    """Remove sources of randomness without changing the learning MDP."""
    env_cfg.commands.motion.motion_load_workers = 1
    env_cfg.commands.motion.adp_samp_failure_rate_max_over_mean = None
    env_cfg.commands.motion.pre_failure_sample_window = 0
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.physics_material = None
    env_cfg.events.add_joint_default_pos = None
    env_cfg.events.base_com = None
    env_cfg.events.push_robot = None


@configclass
class T800FlatObservationsCfgV9(T800FlatObservationsCfgV7):
    """V8 observations with target-to-current pose errors in every command token."""

    @configclass
    class CommandCfg(ObsGroup):
        link_pose_b = ObsTerm(
            func=mdp.motion_body_pose_and_error_b_window_by_entity,
            params={"command_name": "motion", "frame_offsets": V9_FRAME_OFFSETS},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    command: CommandCfg = CommandCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV9(T800FlatWoStateEstimationEnvCfgV8):
    """V8 MDP with 162-dimensional target-and-error command tokens."""

    observations: T800FlatObservationsCfgV9 = T800FlatObservationsCfgV9()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.075
        self.terminations.body_pos.params["threshold"] = 0.7

    def enable_single_motion_overfit(self) -> None:
        """Remove training randomization for a single-motion pipeline check.

        Rewards, terminations, observations, and policy architecture deliberately
        remain unchanged.  This isolates the learning stack from dataset
        diversity, domain randomization, observation noise, and reset noise.
        """
        _enable_single_motion_overfit(self)


@configclass
class T800FlatWoStateEstimationEnvCfgV9Scale(T800FlatWoStateEstimationEnvCfgV8Scale):
    """V9 command observations with the SONIC working-set LaFAN dataset."""

    observations: T800FlatObservationsCfgV9 = T800FlatObservationsCfgV9()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.075
        self.terminations.body_pos.params["threshold"] = 0.7

    def enable_single_motion_overfit(self) -> None:
        """Remove training randomization for a single-motion pipeline check."""
        _enable_single_motion_overfit(self)
