from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v16 import (
    T800FlatWoStateEstimationEnvCfgV16Scale,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV17Scale(T800FlatWoStateEstimationEnvCfgV16Scale):
    """V16 plus a body-position quality channel for successful start bins."""

    def __post_init__(self):
        super().__post_init__()
        motion = self.commands.motion
        # Retain every survival FRONTIER bin.  When that set covers less than
        # 15% of eligible bins, complete the learning pool with the highest
        # global body-position-error MASTERED bins.
        motion.curriculum_quality_learning_enabled = True
        motion.curriculum_learning_pool_min_fraction = 0.15
        # Survival frontier remains the primary signal; successful high-error
        # fillers receive half its per-frame weight inside the frontier budget.
        motion.curriculum_quality_filler_weight = 0.50
        # Only rank successful bins after enough complete, independent windows
        # have made their reward-aligned quality estimate reasonably stable.
        motion.curriculum_quality_min_trials = 32
        motion.curriculum_quality_min_window_trials = 8
        motion.curriculum_quality_min_windows = 2
        motion.curriculum_quality_error_ema_alpha = 0.50
        motion.curriculum_quality_body_pos_std = 0.30
