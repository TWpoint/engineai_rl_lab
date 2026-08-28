from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v17 import (
    T800FlatWoStateEstimationEnvCfgV17Scale,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV18Scale(T800FlatWoStateEstimationEnvCfgV17Scale):
    """V17 plus unified rollout-window credit and need-based sampling."""

    def __post_init__(self):
        super().__post_init__()
        motion = self.commands.motion

        # Treat start and naturally reached transit windows as the same
        # bin-level survival and tracking-quality evidence.
        motion.curriculum_unified_window_enabled = True

        # Sample every valid motion frame under the unified-window rules.  The
        # old start-eligibility and terminal-replay channels are no longer
        # separate sampling distributions.
        motion.curriculum_start_eligibility_enabled = False
        motion.curriculum_terminal_replay_fraction = 0.0
        motion.curriculum_coverage_aware_enabled = False

        # Keep a uniform all-frame floor; distribute the remaining mass from
        # continuous survival, future-risk, quality, and uncertainty needs.
        motion.curriculum_need_uniform_fraction = 0.20
        motion.curriculum_forward_risk_horizon_bins = 5
        motion.curriculum_forward_risk_decay = 0.80
        motion.curriculum_forward_risk_good_threshold = 0.05
        motion.curriculum_forward_risk_bad_threshold = 0.60
        motion.curriculum_quality_good_threshold = 0.08
        motion.curriculum_quality_bad_threshold = 0.30
        motion.curriculum_quarantine_need_scale = 0.25

        # Collect the new statistics before they influence sampling, then blend
        # smoothly away from the checkpoint's V17 distribution.
        motion.curriculum_shadow_iterations = 50
        motion.curriculum_blend_iterations = 100
