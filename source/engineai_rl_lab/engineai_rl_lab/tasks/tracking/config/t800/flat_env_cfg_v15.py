from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v14 import (
    T800FlatWoStateEstimationEnvCfgV14Scale,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV15Scale(T800FlatWoStateEstimationEnvCfgV14Scale):
    """V14-scale with horizon-corrected states and length-tempered motion sampling."""

    def __post_init__(self):
        super().__post_init__()
        motion = self.commands.motion
        # One 50 Hz bin measures local start feasibility.  Later failures are
        # attributed to their terminal bins instead of the original start.
        motion.curriculum_start_horizon_frames = 50
        motion.curriculum_exclude_invalid_failures = True
        # Policy/optimizer state is warm, but the schema-v3 sampler state is
        # new. Keep one full two-window shadow before blending it into use.
        motion.curriculum_shadow_iterations = 100
        motion.curriculum_blend_iterations = 50
        motion.curriculum_min_known_fraction = 0.05
        # Keep state budgets unchanged while reducing aggregate motion exposure
        # from linear in duration to square-root in duration.
        motion.curriculum_motion_length_exponent = 0.5
        # A start bin can be quarantined only when it or the following bin also
        # has enough locally observed terminal risk.
        motion.curriculum_min_terminal_visits = 32
        motion.curriculum_quarantine_terminal_hazard_threshold = 0.10
        motion.curriculum_terminal_hazard_window_bins = 2
        # Missing state budgets become uniform replay rather than being
        # renormalized into the remaining hard states. Smooth state-refresh
        # changes over several PPO iterations to avoid 50-iteration cliffs.
        motion.curriculum_preserve_absent_state_budgets = True
        motion.curriculum_probability_smoothing_alpha = 0.2
        # Keep body-level counts in checkpoints for offline audits without
        # filling W&B with one curve per body and every redundant state mass.
        motion.curriculum_detailed_metrics = False
