from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v24 import (
    T800FlatWoStateEstimationEnvCfgV24Scale,
)

T800_V25_FRAME_OFFSETS = list(range(-5, 6))


@configclass
class T800FlatWoStateEstimationEnvCfgV25Scale(T800FlatWoStateEstimationEnvCfgV24Scale):
    """V24 task with a denser local command window and the V25 multi-critic runner."""

    def __post_init__(self):
        super().__post_init__()

        # V24 detaches the inherited reward tree before returning from its
        # post-init, so this V25-only override cannot mutate V24 defaults.
        self.rewards.action_rate_l2.weight = -0.05

        # Keep eleven command samples (and therefore the 14 x 99 actor input),
        # but sample every 20 ms over +/-100 ms instead of every 40 ms over
        # +/-200 ms. Assign a fresh list so V24's inherited config is untouched.
        self.observations.command.link_pose_b.params["frame_offsets"] = T800_V25_FRAME_OFFSETS.copy()
