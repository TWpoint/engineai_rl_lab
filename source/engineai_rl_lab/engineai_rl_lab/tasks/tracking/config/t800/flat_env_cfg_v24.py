from copy import deepcopy

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v23 import (
    T800FlatWoStateEstimationEnvCfgV23Scale,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV24Scale(T800FlatWoStateEstimationEnvCfgV23Scale):
    """V23 with ScaleBFM-local commands and no head tracking target."""

    def __post_init__(self):
        # Detach commands before the inherited post-init chain touches them.
        # Isaac Lab's configclass retains inherited mutable defaults while it
        # assembles the dataclass, including during module import.
        self.commands = deepcopy(self.commands)
        super().__post_init__()

        # V13 replaces the reward tree during the base post-init chain, so
        # these two branches must be detached only after that chain completes.
        self.observations = deepcopy(self.observations)
        self.rewards = deepcopy(self.rewards)

        # ScaleBFM local tracking translates every target from the current
        # reference anchor instead of from the robot's world-frame position.
        self.observations.command.link_pose_b.func = mdp.motion_body_pose_reference_anchor_window_by_entity_xz
        # The single critic receives both views: the inherited critic command
        # carries global/current tracking targets and errors, while this term
        # supplies the ScaleBFM-local target in the same privileged group.
        self.observations.critic.local_command = ObsTerm(
            func=mdp.motion_body_pose_reference_anchor_window_xz_flat,
            params={
                "command_name": "motion",
                "frame_offsets": [0],
                "zero_invalid_offsets": True,
            },
        )

        # The fixed head is neither commanded nor included in body-wide
        # tracking rewards/terminations, which consume this selected-body list.
        self.commands.motion.body_names = [
            body_name for body_name in self.commands.motion.body_names if body_name != "LINK_HEAD_YAW"
        ]

        self.rewards.action_rate_l2.weight = -0.1
