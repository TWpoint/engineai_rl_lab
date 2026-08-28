from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v18 import (
    T800FlatWoStateEstimationEnvCfgV18Scale,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV19Scale(T800FlatWoStateEstimationEnvCfgV18Scale):
    """V18 unified evidence with learnability-prioritized sampling."""

    def __post_init__(self):
        super().__post_init__()
        motion = self.commands.motion

        motion.curriculum_sampling_strategy = "learnability"

        # V19 removes the finite-difference joint-acceleration regularizer;
        # V18 and earlier task configurations retain it unchanged.
        self.rewards.joint_acc_l2 = None
