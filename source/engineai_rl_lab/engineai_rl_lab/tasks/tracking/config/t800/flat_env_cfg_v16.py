from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v15 import (
    T800FlatWoStateEstimationEnvCfgV15Scale,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV16Scale(T800FlatWoStateEstimationEnvCfgV15Scale):
    """V15-scale with fixed-horizon eligibility and explicit tail replay."""

    def __post_init__(self):
        super().__post_init__()
        motion = self.commands.motion
        # Five-state start difficulty is meaningful only where the complete
        # fixed horizon can be observed. Keep censored tail starts out of those
        # states instead of leaving them permanently unknown inside the pool.
        motion.curriculum_start_eligibility_enabled = True
        # Preserve some target mass for valid tail starts so filtering them
        # from the five-state classifier does not erase motion endings.
        motion.curriculum_terminal_replay_fraction = 0.05
        # Do not spend 55% of reset starts on frontier as soon as 5% of bins
        # become known. First keep eligible frames broadly covered, then ramp
        # five-state focus from 10% to 40% confirmed-bin coverage.
        motion.curriculum_coverage_aware_enabled = True
        motion.curriculum_provisional_known_trials = 8
        motion.curriculum_min_provisional_known_fraction = 0.25
        motion.curriculum_min_provisional_motion_fraction = 0.70
        motion.curriculum_provisional_motion_bin_fraction = 0.10
        motion.curriculum_unconfirmed_min_coverage_ratio = 0.50
        motion.curriculum_state_focus_start_fraction = 0.10
        motion.curriculum_state_focus_end_fraction = 0.40
        motion.curriculum_blend_iterations = 100
