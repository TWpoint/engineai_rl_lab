from __future__ import annotations

import gc
import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .motion_data import MotionCollection, read_motion_lengths, resolve_motion_files
from .motion_data import MotionLoader as MotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cfg_curriculum_sampling_strategy(cfg: MotionCommandCfg) -> str:
    """Read the unified sampler strategy from real or lightweight test config."""

    return str(getattr(cfg, "curriculum_sampling_strategy", "need"))


def _cfg_curriculum_learnability_enabled(cfg: MotionCommandCfg) -> bool:
    return bool(getattr(cfg, "curriculum_unified_window_enabled", False)) and (
        _cfg_curriculum_sampling_strategy(cfg) == "learnability"
    )


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    _CURRICULUM_STATE_NAMES = ("unknown", "mastered", "frontier", "stalled", "quarantine")
    _CURRICULUM_UNKNOWN = 0
    _CURRICULUM_MASTERED = 1
    _CURRICULUM_FRONTIER = 2
    _CURRICULUM_STALLED = 3
    _CURRICULUM_QUARANTINE = 4
    _CURRICULUM_STATE_SCHEMA_VERSION = 2
    _CURRICULUM_BIASED_FIXED_HORIZON_SCHEMA_VERSION = 3
    _CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION = 4
    _CURRICULUM_ELIGIBLE_COVERAGE_SCHEMA_VERSION = 5
    _CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION = 6
    _CURRICULUM_UNIFIED_WINDOW_SCHEMA_VERSION = 7
    # Schema 8 was produced by the retired fixed-quota V19 experiment.  It is
    # retained only as a checkpoint migration source.
    _CURRICULUM_STAGED_SAMPLING_SCHEMA_VERSION = 8
    _CURRICULUM_LEARNABILITY_SCHEMA_VERSION = 9

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self._validate_adaptive_sampling_cfg()
        self._motion_files = resolve_motion_files(self.cfg.motion_file)
        self._motion_storage_device = self.cfg.motion_data_device
        if self._motion_storage_device == "auto":
            self._motion_storage_device = "cpu" if len(self._motion_files) > 1 else self.device
        global_time_totals = read_motion_lengths(self._motion_files, max_workers=self.cfg.motion_load_workers)
        self._global_time_totals = torch.tensor(global_time_totals, dtype=torch.long, device=self.device)
        self.global_num_motions = len(self._motion_files)
        self._initialize_adaptive_sampling()
        self._adaptive_layout_checked = False
        self._rebuild_global_sampling_distribution()

        max_num_load_motions = self.cfg.max_num_load_motions
        if max_num_load_motions is None:
            max_num_load_motions = min(self.num_envs, 1024)
        if max_num_load_motions <= 0:
            raise ValueError("max_num_load_motions must be positive or None")
        self.max_num_load_motions = min(int(max_num_load_motions), self.global_num_motions)
        self.all_motions_loaded = self.max_num_load_motions >= self.global_num_motions
        if self.all_motions_loaded:
            selected_global_ids = torch.arange(self.global_num_motions, dtype=torch.long, device=self.device)
        else:
            selected_global_ids = torch.multinomial(
                self._motion_sampling_probabilities,
                num_samples=self.max_num_load_motions,
                replacement=self.cfg.working_set_replacement,
            )
        self.motion = self._load_motion_collection(selected_global_ids)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fixed_motion_ids: torch.Tensor | None = None
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._has_sampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._motion_cache = {
            field: torch.empty((self.num_envs, *self.motion.field_shapes[field]), device=self.device)
            for field in self.motion._FIELDS
        }
        self._window_time_steps: torch.Tensor | None = None
        self._window_motion_ids: torch.Tensor | None = None
        self._window_body_pos: torch.Tensor | None = None
        self._window_body_quat: torch.Tensor | None = None
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 3] = 1.0

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_motion_prob"] = torch.zeros(self.num_envs, device=self.device)
        if self._curriculum_sampling_enabled():
            self.metrics["curriculum_known_fraction"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["curriculum_start_known_fraction"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["curriculum_blend"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["curriculum_start_failure_rate"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["curriculum_start_censored_fraction"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["curriculum_terminal_hazard"] = torch.zeros(self.num_envs, device=self.device)
            for state_name in self._CURRICULUM_STATE_NAMES:
                self.metrics[f"curriculum_state_fraction_{state_name}"] = torch.zeros(self.num_envs, device=self.device)
            logged_mass_states = (
                self._CURRICULUM_STATE_NAMES
                if getattr(self.cfg, "curriculum_detailed_metrics", True)
                else ("unknown", "frontier", "quarantine")
            )
            for state_name in logged_mass_states:
                self.metrics[f"curriculum_sampling_mass_{state_name}"] = torch.zeros(self.num_envs, device=self.device)
            if self._curriculum_start_eligibility_enabled():
                for metric_name in (
                    "curriculum_eligible_bin_fraction",
                    "curriculum_provisional_known_fraction",
                    "curriculum_provisional_motion_fraction",
                    "curriculum_state_focus",
                    "curriculum_sampling_mass_shadow",
                    "curriculum_sampling_mass_eligible",
                    "curriculum_sampling_mass_terminal_replay",
                ):
                    self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
            if self._curriculum_quality_learning_enabled():
                for metric_name in (
                    "curriculum_quality_known_fraction",
                    "curriculum_quality_filler_fraction",
                    "curriculum_learning_pool_fraction",
                    "curriculum_sampling_mass_quality_filler",
                    "curriculum_sampling_mass_learning_pool",
                    "curriculum_quality_body_pos_score_mean",
                    "curriculum_quality_body_pos_score_cutoff",
                ):
                    self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
            if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
                for metric_name in (
                    "curriculum_unified_known_fraction",
                    "curriculum_unified_failure_rate",
                    "curriculum_unified_censored_fraction",
                    "curriculum_forward_known_fraction",
                    "curriculum_forward_risk_mean",
                    "curriculum_need_survival_mean",
                    "curriculum_need_forward_mean",
                    "curriculum_need_quality_mean",
                    "curriculum_need_uncertainty_mean",
                    "curriculum_target_mass_uniform",
                    "curriculum_target_mass_need",
                ):
                    self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
                if _cfg_curriculum_learnability_enabled(self.cfg):
                    for metric_name in (
                        "curriculum_learnability_primary_support_fraction",
                        "curriculum_learnability_primary_need_mean",
                        "curriculum_learnability_primary_scale",
                        "curriculum_learnability_base_fraction",
                        "curriculum_learnability_target_entropy",
                        "curriculum_learnability_quality_reweighted_fraction",
                        "curriculum_learnability_quality_base_mastered_mass",
                    ):
                        self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
            if getattr(self.cfg, "curriculum_detailed_metrics", True):
                for body_name in self.cfg.body_names:
                    self.metrics[f"curriculum_terminal_body_fraction_{body_name}"] = torch.zeros(
                        self.num_envs, device=self.device
                    )
        self._rebuild_active_motion_mapping()
        self._rebuild_sampling_distribution()

    def _curriculum_sampling_enabled(self) -> bool:
        """Return whether the opt-in start-bin curriculum is active."""

        return bool(getattr(self.cfg, "curriculum_sampling_enabled", False))

    def _curriculum_fixed_horizon_enabled(self) -> bool:
        """Return whether start difficulty uses a common forward horizon."""

        return getattr(self.cfg, "curriculum_start_horizon_frames", None) is not None

    def _curriculum_start_eligibility_enabled(self) -> bool:
        """Return whether comparable starts and censored tail starts are separated."""

        return bool(getattr(self.cfg, "curriculum_start_eligibility_enabled", False))

    def _curriculum_coverage_aware_enabled(self) -> bool:
        """Return whether state focus is ramped with eligible-start coverage."""

        return bool(getattr(self.cfg, "curriculum_coverage_aware_enabled", False))

    def _curriculum_quality_learning_enabled(self) -> bool:
        """Return whether successful-but-inaccurate starts may fill the learning pool."""

        return bool(getattr(self.cfg, "curriculum_quality_learning_enabled", False))

    def _curriculum_unified_window_enabled(self) -> bool:
        """Return whether every consecutive rollout window receives equal credit."""

        return bool(getattr(self.cfg, "curriculum_unified_window_enabled", False))

    def _curriculum_sampling_strategy(self) -> str:
        """Return the single unified-window sampling strategy."""

        return _cfg_curriculum_sampling_strategy(self.cfg)

    def _curriculum_learnability_sampling_enabled(self) -> bool:
        """Return whether unified needs use schema-9 learnability priority."""

        return _cfg_curriculum_learnability_enabled(self.cfg)

    def _curriculum_learnability_recipe(self) -> tuple[float, ...]:
        """Return existing evidence thresholds that define schema-9 priority."""

        return (
            float(self.cfg.curriculum_beta_prior_alpha),
            float(self.cfg.curriculum_beta_prior_beta),
            float(self.cfg.curriculum_min_known_trials),
            float(self.cfg.curriculum_min_window_trials),
            float(self.cfg.curriculum_no_progress_threshold),
            float(self.cfg.curriculum_improvement_threshold),
            float(self.cfg.curriculum_forward_risk_good_threshold),
            float(self.cfg.curriculum_forward_risk_bad_threshold),
            float(self.cfg.curriculum_quality_good_threshold),
            float(self.cfg.curriculum_quality_bad_threshold),
            float(self.cfg.curriculum_quality_min_trials),
            float(self.cfg.curriculum_quality_min_window_trials),
            float(self.cfg.curriculum_quality_min_windows),
            float(self.cfg.curriculum_quality_error_ema_alpha),
            float(self.cfg.curriculum_quality_body_pos_std),
        )

    def _curriculum_checkpoint_schema_version(self) -> int:
        if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            if _cfg_curriculum_sampling_strategy(self.cfg) == "learnability":
                return MotionCommand._CURRICULUM_LEARNABILITY_SCHEMA_VERSION
            return self._CURRICULUM_UNIFIED_WINDOW_SCHEMA_VERSION
        if self._curriculum_quality_learning_enabled():
            return self._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION
        if self._curriculum_start_eligibility_enabled() or self._curriculum_coverage_aware_enabled():
            return self._CURRICULUM_ELIGIBLE_COVERAGE_SCHEMA_VERSION
        if self._curriculum_fixed_horizon_enabled():
            return self._CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION
        return self._CURRICULUM_STATE_SCHEMA_VERSION

    def _validate_adaptive_sampling_cfg(self) -> None:
        if self.cfg.bin_size <= 0:
            raise ValueError("bin_size must be positive")
        if not 0.0 <= self.cfg.uniform_sampling_rate <= 1.0:
            raise ValueError("uniform_sampling_rate must be in [0, 1]")
        if (
            self.cfg.adp_samp_failure_rate_max_over_mean is not None
            and self.cfg.adp_samp_failure_rate_max_over_mean <= 0.0
        ):
            raise ValueError("adp_samp_failure_rate_max_over_mean must be positive or None")
        if self.cfg.pre_failure_sample_window < 0:
            raise ValueError("pre_failure_sample_window must be non-negative")
        if self.cfg.init_num_failures < 0:
            raise ValueError("init_num_failures must be non-negative")
        if self.cfg.adaptive_sampling_alpha is not None and not 0.0 < self.cfg.adaptive_sampling_alpha <= 1.0:
            raise ValueError("adaptive_sampling_alpha must be in (0, 1] or None")
        if self.cfg.adaptive_kernel_size < 1:
            raise ValueError("adaptive_kernel_size must be at least one")
        if not self._curriculum_sampling_enabled():
            return
        if self.cfg.curriculum_shadow_iterations < 0:
            raise ValueError("curriculum_shadow_iterations must be non-negative")
        if self.cfg.curriculum_blend_iterations < 0:
            raise ValueError("curriculum_blend_iterations must be non-negative")
        if self.cfg.curriculum_state_update_interval < 1:
            raise ValueError("curriculum_state_update_interval must be positive")
        for name in (
            "curriculum_min_window_trials",
            "curriculum_min_known_trials",
            "curriculum_min_quarantine_trials",
            "curriculum_min_exit_probe_trials",
        ):
            if getattr(self.cfg, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.cfg.curriculum_min_known_fraction <= 1.0:
            raise ValueError("curriculum_min_known_fraction must be in [0, 1]")
        if self.cfg.curriculum_beta_prior_alpha <= 0.0 or self.cfg.curriculum_beta_prior_beta <= 0.0:
            raise ValueError("curriculum beta prior parameters must be positive")
        for name in (
            "curriculum_mastered_enter_threshold",
            "curriculum_mastered_exit_threshold",
            "curriculum_stalled_enter_threshold",
            "curriculum_stalled_exit_threshold",
            "curriculum_quarantine_enter_threshold",
            "curriculum_quarantine_exit_threshold",
        ):
            if not 0.0 <= getattr(self.cfg, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.cfg.curriculum_mastered_enter_threshold >= self.cfg.curriculum_mastered_exit_threshold:
            raise ValueError("curriculum mastered enter threshold must be below its exit threshold")
        if self.cfg.curriculum_stalled_exit_threshold >= self.cfg.curriculum_stalled_enter_threshold:
            raise ValueError("curriculum stalled exit threshold must be below its enter threshold")
        if self.cfg.curriculum_quarantine_exit_threshold >= self.cfg.curriculum_quarantine_enter_threshold:
            raise ValueError("curriculum quarantine exit threshold must be below its enter threshold")
        if self.cfg.curriculum_no_progress_threshold < 0.0:
            raise ValueError("curriculum_no_progress_threshold must be non-negative")
        if self.cfg.curriculum_improvement_threshold <= self.cfg.curriculum_no_progress_threshold:
            raise ValueError("curriculum improvement threshold must exceed the no-progress threshold")
        self._validate_fixed_horizon_curriculum_cfg()
        state_weights = self.cfg.curriculum_state_sampling_weights
        if set(state_weights) != set(self._CURRICULUM_STATE_NAMES):
            raise ValueError(f"curriculum_state_sampling_weights must contain exactly {self._CURRICULUM_STATE_NAMES}")
        if any(float(weight) <= 0.0 for weight in state_weights.values()):
            raise ValueError("curriculum state sampling weights must all be positive")

    def _validate_fixed_horizon_curriculum_cfg(self) -> None:
        horizon_frames = getattr(self.cfg, "curriculum_start_horizon_frames", None)
        if horizon_frames is not None and horizon_frames < 1:
            raise ValueError("curriculum_start_horizon_frames must be positive or None")
        motion_length_exponent = float(getattr(self.cfg, "curriculum_motion_length_exponent", 1.0))
        if not 0.0 <= motion_length_exponent <= 1.0:
            raise ValueError("curriculum_motion_length_exponent must be in [0, 1]")
        min_terminal_visits = int(getattr(self.cfg, "curriculum_min_terminal_visits", 0))
        if min_terminal_visits < 0:
            raise ValueError("curriculum_min_terminal_visits must be non-negative")
        if horizon_frames is not None and min_terminal_visits < 1:
            raise ValueError("fixed-horizon curriculum requires positive curriculum_min_terminal_visits")
        min_terminal_hazard = float(getattr(self.cfg, "curriculum_quarantine_terminal_hazard_threshold", 0.0))
        if not 0.0 <= min_terminal_hazard <= 1.0:
            raise ValueError("curriculum_quarantine_terminal_hazard_threshold must be in [0, 1]")
        if int(getattr(self.cfg, "curriculum_terminal_hazard_window_bins", 1)) < 1:
            raise ValueError("curriculum_terminal_hazard_window_bins must be positive")
        probability_smoothing_alpha = float(getattr(self.cfg, "curriculum_probability_smoothing_alpha", 1.0))
        if not 0.0 < probability_smoothing_alpha <= 1.0:
            raise ValueError("curriculum_probability_smoothing_alpha must be in (0, 1]")
        eligibility_enabled = self._curriculum_start_eligibility_enabled()
        if eligibility_enabled and horizon_frames is None:
            raise ValueError("curriculum start eligibility requires a fixed start horizon")
        if eligibility_enabled and bool(getattr(self.cfg, "resample_at_motion_end", True)):
            raise ValueError("curriculum start eligibility requires resample_at_motion_end=False")
        terminal_replay_fraction = float(getattr(self.cfg, "curriculum_terminal_replay_fraction", 0.0))
        if not 0.0 <= terminal_replay_fraction < 1.0:
            raise ValueError("curriculum_terminal_replay_fraction must be in [0, 1)")
        if not eligibility_enabled and terminal_replay_fraction != 0.0:
            raise ValueError("terminal replay requires curriculum start eligibility")
        coverage_aware_enabled = self._curriculum_coverage_aware_enabled()
        if coverage_aware_enabled and not eligibility_enabled:
            raise ValueError("coverage-aware curriculum requires curriculum start eligibility")
        quality_learning_enabled = self._curriculum_quality_learning_enabled()
        unified_window_enabled = bool(getattr(self.cfg, "curriculum_unified_window_enabled", False))
        self._validate_unified_sampler_enablement(unified_window_enabled)
        if quality_learning_enabled and not (eligibility_enabled or unified_window_enabled):
            raise ValueError("quality learning requires curriculum start eligibility")
        if unified_window_enabled:
            self._validate_unified_window_curriculum_cfg(
                horizon_frames,
                eligibility_enabled=eligibility_enabled,
                coverage_aware_enabled=coverage_aware_enabled,
                terminal_replay_fraction=terminal_replay_fraction,
            )
        if quality_learning_enabled:
            learning_pool_fraction = float(getattr(self.cfg, "curriculum_learning_pool_min_fraction", 0.0))
            if not 0.0 < learning_pool_fraction <= 1.0:
                raise ValueError("curriculum_learning_pool_min_fraction must be in (0, 1]")
            quality_filler_weight = float(getattr(self.cfg, "curriculum_quality_filler_weight", 0.0))
            if not 0.0 < quality_filler_weight <= 1.0:
                raise ValueError("curriculum_quality_filler_weight must be in (0, 1]")
            for name in (
                "curriculum_quality_min_trials",
                "curriculum_quality_min_window_trials",
                "curriculum_quality_min_windows",
            ):
                if int(getattr(self.cfg, name, 0)) < 1:
                    raise ValueError(f"{name} must be positive")
            quality_ema_alpha = float(getattr(self.cfg, "curriculum_quality_error_ema_alpha", 0.0))
            if not 0.0 < quality_ema_alpha <= 1.0:
                raise ValueError("curriculum_quality_error_ema_alpha must be in (0, 1]")
            if float(getattr(self.cfg, "curriculum_quality_body_pos_std", 0.0)) <= 0.0:
                raise ValueError("curriculum_quality_body_pos_std must be positive")
        if eligibility_enabled:
            provisional_trials = int(getattr(self.cfg, "curriculum_provisional_known_trials", 1))
            if provisional_trials < 1 or provisional_trials > int(self.cfg.curriculum_min_known_trials):
                raise ValueError("curriculum provisional trials must be in [1, curriculum_min_known_trials]")
            min_provisional_fraction = float(getattr(self.cfg, "curriculum_min_provisional_known_fraction", 0.0))
            if not 0.0 <= min_provisional_fraction <= 1.0:
                raise ValueError("curriculum_min_provisional_known_fraction must be in [0, 1]")
            min_motion_fraction = float(getattr(self.cfg, "curriculum_min_provisional_motion_fraction", 0.0))
            if not 0.0 <= min_motion_fraction <= 1.0:
                raise ValueError("curriculum_min_provisional_motion_fraction must be in [0, 1]")
            motion_bin_fraction = float(getattr(self.cfg, "curriculum_provisional_motion_bin_fraction", 0.0))
            if not 0.0 <= motion_bin_fraction <= 1.0:
                raise ValueError("curriculum_provisional_motion_bin_fraction must be in [0, 1]")
            unconfirmed_floor = float(getattr(self.cfg, "curriculum_unconfirmed_min_coverage_ratio", 0.0))
            if not 0.0 <= unconfirmed_floor <= 1.0:
                raise ValueError("curriculum_unconfirmed_min_coverage_ratio must be in [0, 1]")
            focus_start = float(getattr(self.cfg, "curriculum_state_focus_start_fraction", 0.0))
            focus_end = float(getattr(self.cfg, "curriculum_state_focus_end_fraction", 1.0))
            if not 0.0 <= focus_start < focus_end <= 1.0:
                raise ValueError("curriculum state-focus coverage bounds must satisfy 0 <= start < end <= 1")

    def _validate_unified_sampler_enablement(self, unified_window_enabled: bool) -> None:
        strategy = _cfg_curriculum_sampling_strategy(self.cfg)
        if strategy not in ("need", "learnability"):
            raise ValueError("curriculum_sampling_strategy must be 'need' or 'learnability'")
        if strategy != "need" and not unified_window_enabled:
            raise ValueError("learnability sampling requires the unified-window curriculum")

    def _validate_unified_window_curriculum_cfg(
        self,
        horizon_frames: int | None,
        *,
        eligibility_enabled: bool,
        coverage_aware_enabled: bool,
        terminal_replay_fraction: float,
    ) -> None:
        if horizon_frames is None:
            raise ValueError("unified-window curriculum requires a fixed window horizon")
        if int(horizon_frames) != int(self.cfg.bin_size):
            raise ValueError("unified-window curriculum requires bin_size == curriculum_start_horizon_frames")
        if eligibility_enabled or coverage_aware_enabled or terminal_replay_fraction != 0.0:
            raise ValueError(
                "unified-window curriculum replaces start eligibility, coverage focus, and terminal replay"
            )
        if bool(getattr(self.cfg, "resample_at_motion_end", True)):
            raise ValueError("unified-window curriculum requires an explicit motion_time_out termination")
        if int(self.cfg.pre_failure_sample_window) != 0:
            raise ValueError("unified-window all-frame sampling requires pre_failure_sample_window=0")
        for name in (
            "curriculum_need_uniform_fraction",
            "curriculum_forward_risk_decay",
            "curriculum_forward_risk_good_threshold",
            "curriculum_forward_risk_bad_threshold",
            "curriculum_quality_good_threshold",
            "curriculum_quality_bad_threshold",
            "curriculum_quarantine_need_scale",
        ):
            value = float(getattr(self.cfg, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if int(getattr(self.cfg, "curriculum_forward_risk_horizon_bins")) < 1:
            raise ValueError("curriculum_forward_risk_horizon_bins must be positive")
        if not (
            float(self.cfg.curriculum_forward_risk_good_threshold)
            < float(self.cfg.curriculum_forward_risk_bad_threshold)
        ):
            raise ValueError("curriculum forward-risk good threshold must be below its bad threshold")
        if not float(self.cfg.curriculum_quality_good_threshold) < float(self.cfg.curriculum_quality_bad_threshold):
            raise ValueError("curriculum quality good threshold must be below its bad threshold")
        if _cfg_curriculum_learnability_enabled(self.cfg):
            if not self._curriculum_quality_learning_enabled():
                raise ValueError("learnability sampling requires curriculum quality learning")

    def _initialize_adaptive_sampling(self) -> None:
        motion_lengths = self._global_time_totals
        self.motion_bin_counts = torch.div(
            motion_lengths + self.cfg.bin_size - 1,
            self.cfg.bin_size,
            rounding_mode="floor",
        )
        self.bin_count = int(self.motion_bin_counts.sum().item())
        self.motion_bin_offsets = torch.zeros(self.global_num_motions, dtype=torch.long, device=self.device)
        if self.global_num_motions > 1:
            self.motion_bin_offsets[1:] = torch.cumsum(self.motion_bin_counts[:-1], dim=0)
        self.bin_motion_ids = torch.repeat_interleave(
            torch.arange(self.global_num_motions, device=self.device), self.motion_bin_counts
        )
        bin_indexes = torch.arange(self.bin_count, device=self.device)
        local_bin_indexes = bin_indexes - self.motion_bin_offsets[self.bin_motion_ids]
        self.bin_starts = local_bin_indexes * self.cfg.bin_size
        self.bin_ends = torch.minimum(
            self.bin_starts + self.cfg.bin_size,
            motion_lengths[self.bin_motion_ids],
        )
        self.bin_weights = (self.bin_ends - self.bin_starts).float()
        self.bin_weights /= self.bin_weights.mean()
        if self.cfg.sequence_length_agnostic:
            self.bin_weights /= self.motion_bin_counts[self.bin_motion_ids]
        if self._curriculum_start_eligibility_enabled():
            self._initialize_curriculum_start_geometry()
        elif bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            self._initialize_curriculum_unified_geometry()
        initial_count = 0.0 if self.cfg.adaptive_sampling_alpha is not None else float(self.cfg.init_num_failures)
        self.adp_samp_num_episodes = torch.full((self.bin_count,), initial_count, device=self.device)
        self.adp_samp_num_failures = torch.full((self.bin_count,), initial_count, device=self.device)
        self._current_adp_samp_num_episodes = torch.zeros(self.bin_count, device=self.device)
        self._current_adp_samp_num_failures = torch.zeros(self.bin_count, device=self.device)
        if self._curriculum_sampling_enabled():
            self._initialize_curriculum_sampling()

    def _initialize_curriculum_start_geometry(self) -> None:
        """Split every bin into comparable fixed-horizon and censored-tail starts."""

        horizon_frames = int(self.cfg.curriculum_start_horizon_frames)
        bin_motion_lengths = self._global_time_totals[self.bin_motion_ids]
        eligible_ends = torch.minimum(self.bin_ends, bin_motion_lengths - horizon_frames)
        self._curriculum_eligible_start_ends = torch.maximum(eligible_ends, self.bin_starts)
        self._curriculum_eligible_start_counts = self._curriculum_eligible_start_ends - self.bin_starts
        bin_lengths = self.bin_ends - self.bin_starts
        self._curriculum_terminal_start_counts = bin_lengths - self._curriculum_eligible_start_counts
        self._curriculum_eligible_bins = self._curriculum_eligible_start_counts > 0
        if not torch.any(self._curriculum_eligible_bins):
            raise ValueError("fixed-horizon curriculum has no eligible start frames in this motion dataset")
        self._curriculum_eligible_bin_count = int(self._curriculum_eligible_bins.sum().item())
        self._curriculum_eligible_motion_counts = torch.bincount(
            self.bin_motion_ids[self._curriculum_eligible_bins], minlength=self.global_num_motions
        )
        self._curriculum_eligible_motions = self._curriculum_eligible_motion_counts > 0
        required_fraction = float(getattr(self.cfg, "curriculum_provisional_motion_bin_fraction", 0.0))
        self._curriculum_provisional_motion_required_counts = (
            torch.ceil(self._curriculum_eligible_motion_counts.float() * required_fraction).long().clamp_min(1)
        )

    def _initialize_curriculum_unified_geometry(self) -> None:
        """Expose every valid frame through one ordinary, duration-weighted channel."""

        self._curriculum_eligible_start_ends = self.bin_ends.clone()
        self._curriculum_eligible_start_counts = self.bin_ends - self.bin_starts
        self._curriculum_terminal_start_counts = torch.zeros_like(self._curriculum_eligible_start_counts)
        self._curriculum_eligible_bins = torch.ones(self.bin_count, dtype=torch.bool, device=self.device)
        self._curriculum_eligible_bin_count = self.bin_count
        self._curriculum_eligible_motion_counts = self.motion_bin_counts.clone()
        self._curriculum_eligible_motions = torch.ones(self.global_num_motions, dtype=torch.bool, device=self.device)
        self._curriculum_provisional_motion_required_counts = torch.ones(
            self.global_num_motions, dtype=torch.long, device=self.device
        )

    def _initialize_curriculum_sampling(self) -> None:
        """Allocate opt-in start/terminal statistics without affecting the legacy sampler."""

        def zeros() -> torch.Tensor:
            return torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)

        self.curriculum_start_trials = zeros()
        self.curriculum_start_failures = zeros()
        self.curriculum_start_censored = zeros()
        self.curriculum_start_survival_steps = zeros()
        self.curriculum_start_completion_fraction = zeros()
        self.curriculum_terminal_visits = zeros()
        self.curriculum_terminal_failures = zeros()
        self.curriculum_terminal_body_failures = torch.zeros(
            len(self.cfg.body_names), dtype=torch.float32, device=self.device
        )

        self._current_curriculum_start_trials = zeros()
        self._current_curriculum_start_failures = zeros()
        self._current_curriculum_start_censored = zeros()
        self._current_curriculum_start_survival_steps = zeros()
        self._current_curriculum_start_completion_fraction = zeros()
        self._current_curriculum_terminal_visits = zeros()
        self._current_curriculum_terminal_failures = zeros()
        self._current_curriculum_terminal_body_failures = torch.zeros_like(self.curriculum_terminal_body_failures)
        self._curriculum_window_start_trials = zeros()
        self._curriculum_window_start_failures = zeros()

        if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            self.curriculum_forward_trials = zeros()
            self.curriculum_forward_risk_sum = zeros()
            self._current_curriculum_forward_trials = zeros()
            self._current_curriculum_forward_risk_sum = zeros()
            self._curriculum_window_forward_trials = zeros()
            self._curriculum_window_forward_risk_sum = zeros()
            self._curriculum_forward_risk_history = torch.full(
                (3, self.bin_count), float("nan"), dtype=torch.float32, device=self.device
            )
            self._curriculum_forward_risk_ema = torch.full(
                (self.bin_count,), float("nan"), dtype=torch.float32, device=self.device
            )
            self._curriculum_forward_observed_windows = torch.zeros(
                self.bin_count, dtype=torch.uint8, device=self.device
            )
            self._curriculum_last_survival_need = zeros()
            self._curriculum_last_forward_need = zeros()
            self._curriculum_last_quality_need = zeros()
            self._curriculum_last_uncertainty_need = zeros()
            self._curriculum_last_combined_need = zeros()
            if _cfg_curriculum_learnability_enabled(self.cfg):
                self._curriculum_last_learnability_base_fraction = 1.0
                self._curriculum_last_learnability_primary_support_fraction = 0.0
                self._curriculum_last_learnability_primary_need_mean = 0.0
                self._curriculum_last_learnability_primary_scale = 0.0
                self._curriculum_last_learnability_target_entropy = 1.0
                self._curriculum_last_learnability_quality_reweighted_fraction = 0.0
                self._curriculum_last_learnability_quality_mastered_mass = 0.0

        self._curriculum_failure_rate_history = torch.full(
            (3, self.bin_count), float("nan"), dtype=torch.float32, device=self.device
        )
        self._curriculum_states = torch.full(
            (self.bin_count,), self._CURRICULUM_UNKNOWN, dtype=torch.uint8, device=self.device
        )
        self._curriculum_state_entry_trials = zeros()

        self._episode_start_bins = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._episode_start_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_last_visited_bins = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._episode_curriculum_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_start_outcome_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._curriculum_start_eligibility_enabled():
            self._episode_start_label_enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._curriculum_body_id_mapping: torch.Tensor | None = None
        if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            self._episode_unified_window_bins = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            self._episode_unified_window_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._episode_unified_window_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._episode_unified_window_targets = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._episode_unified_window_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            risk_horizon = int(self.cfg.curriculum_forward_risk_horizon_bins)
            self._episode_forward_risk_bins = torch.full(
                (self.num_envs, risk_horizon), -1, dtype=torch.long, device=self.device
            )

        self._curriculum_iteration = 0
        self._curriculum_last_state_update_iteration = 0
        self._curriculum_blend_start_iteration = -1
        self._curriculum_known_fraction = 0.0
        self._curriculum_provisional_known_fraction = 0.0
        self._curriculum_provisional_motion_fraction = 0.0
        self._curriculum_effective_state_focus = 0.0
        self._curriculum_shadow_probabilities = zeros()
        self._curriculum_smoothed_probabilities = zeros()
        if self._curriculum_start_eligibility_enabled():
            self._curriculum_smoothed_eligible_probabilities = zeros()
            self._curriculum_smoothed_terminal_probabilities = zeros()
        self._curriculum_shadow_states = torch.full(
            (self.bin_count,), self._CURRICULUM_UNKNOWN, dtype=torch.uint8, device=self.device
        )
        self._curriculum_shadow_motion_length_exponent = 1.0
        self._curriculum_has_shadow_distribution = False
        self._curriculum_has_shadow_state_recipe = False
        self._curriculum_smoothed_probabilities_initialized = False
        self._curriculum_last_probability_smoothing_iteration = -1
        self._curriculum_metric_values: dict[str, torch.Tensor] = {}
        if self._curriculum_quality_learning_enabled():
            self._initialize_curriculum_quality_learning()

    def _initialize_curriculum_quality_learning(self) -> None:
        """Allocate the opt-in V17 successful-start quality statistics."""

        def zeros() -> torch.Tensor:
            return torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)

        self.curriculum_quality_trials = zeros()
        self._current_curriculum_quality_trials = zeros()
        self._current_curriculum_quality_error_sum = zeros()
        self._curriculum_quality_window_trials = zeros()
        self._curriculum_quality_window_error_sum = zeros()
        self._curriculum_quality_error_ema = torch.full(
            (self.bin_count,), float("nan"), dtype=torch.float32, device=self.device
        )
        self._curriculum_quality_observed_windows = torch.zeros(self.bin_count, dtype=torch.uint8, device=self.device)
        self._curriculum_quality_filler_mask = torch.zeros(self.bin_count, dtype=torch.bool, device=self.device)
        self._episode_curriculum_quality_error_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_curriculum_quality_observation_count = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_curriculum_quality_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_curriculum_quality_last_time_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        target_fraction = float(self.cfg.curriculum_learning_pool_min_fraction)
        learning_bin_count = (
            self.bin_count
            if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False))
            else self._curriculum_eligible_bin_count
        )
        self._curriculum_learning_pool_target_count = int(math.ceil(target_fraction * learning_bin_count))
        self._curriculum_quality_filler_count = 0
        self._curriculum_quality_error_cutoff = 0.0

    def _load_motion_collection(self, selected_global_ids: torch.Tensor) -> MotionCollection:
        motion = MotionCollection(
            self._motion_files,
            self.body_indexes,
            device=self.device,
            storage_device=self._motion_storage_device,
            joint_names=self.robot.joint_names,
            body_names=self.cfg.body_names,
            shard_by_rank=False,
            selected_global_ids=selected_global_ids.tolist(),
            global_time_totals=self._global_time_totals,
            max_chunk_frames=self.cfg.motion_chunk_frames,
            max_workers=self.cfg.motion_load_workers,
        )
        print(
            "[INFO] SONIC motion working set: "
            f"rank={motion.rank}/{motion.world_size}, "
            f"files={motion.rank_num_files}/{motion.global_num_motions}, "
            f"unique={motion.global_ids.unique().numel()}, "
            f"frames={motion.rank_num_frames:,}, chunks={motion.num_chunks}, "
            f"resident={motion.resident_bytes / 2**30:.2f} GiB, "
            f"storage={motion.storage_device}."
        )
        return motion

    def _rebuild_active_motion_mapping(self) -> None:
        active_bin_ids = []
        for global_motion_id in self.motion.global_ids.tolist():
            begin = int(self.motion_bin_offsets[global_motion_id])
            count = int(self.motion_bin_counts[global_motion_id])
            active_bin_ids.append(torch.arange(begin, begin + count, dtype=torch.long, device=self.device))
        self._active_bin_ids = torch.cat(active_bin_ids)
        self._active_local_motion_ids = torch.repeat_interleave(
            torch.arange(self.motion.num_motions, device=self.device),
            self.motion_bin_counts[self.motion.global_ids],
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def _refresh_motion_cache(self, env_ids: torch.Tensor | None = None) -> None:
        """Gather the current reference frame once for all downstream MDP terms."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        motion_ids = self.motion_ids[env_ids]
        time_steps = self.time_steps[env_ids]
        samples = self.motion.sample_many(self.motion._FIELDS, motion_ids, time_steps)
        for field, cache in self._motion_cache.items():
            cache[env_ids] = samples[field]

    def sample_body_window(self, time_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a motion window, reusing overlapping frames from the previous step."""
        motion_ids = self.motion_ids[:, None].expand_as(time_steps)
        if self._window_time_steps is None or self._window_time_steps.shape != time_steps.shape:
            samples = self.motion.sample_many(("body_pos_w", "body_quat_w"), motion_ids, time_steps)
            self._window_body_pos = samples["body_pos_w"]
            self._window_body_quat = samples["body_quat_w"]
        else:
            same_window = torch.all(time_steps == self._window_time_steps, dim=1) & (
                self.motion_ids == self._window_motion_ids
            )
            can_shift = (
                torch.all(time_steps[:, :-1] == self._window_time_steps[:, 1:], dim=1)
                & (self.motion_ids == self._window_motion_ids)
                & ~same_window
            )
            refresh = ~(same_window | can_shift)
            if torch.any(can_shift):
                self._window_body_pos[can_shift, :-1] = self._window_body_pos[can_shift, 1:]
                self._window_body_quat[can_shift, :-1] = self._window_body_quat[can_shift, 1:]
                next_motion_ids = self.motion_ids[can_shift]
                next_time_steps = time_steps[can_shift, -1]
                samples = self.motion.sample_many(("body_pos_w", "body_quat_w"), next_motion_ids, next_time_steps)
                self._window_body_pos[can_shift, -1] = samples["body_pos_w"]
                self._window_body_quat[can_shift, -1] = samples["body_quat_w"]
            if torch.any(refresh):
                refresh_motion_ids = motion_ids[refresh]
                refresh_time_steps = time_steps[refresh]
                samples = self.motion.sample_many(("body_pos_w", "body_quat_w"), refresh_motion_ids, refresh_time_steps)
                self._window_body_pos[refresh] = samples["body_pos_w"]
                self._window_body_quat[refresh] = samples["body_quat_w"]
        self._window_time_steps = time_steps.clone()
        self._window_motion_ids = self.motion_ids.clone()
        return self._window_body_pos, self._window_body_quat

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._motion_cache["joint_pos"]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._motion_cache["joint_vel"]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._motion_cache["body_pos_w"] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._motion_cache["body_quat_w"]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._motion_cache["body_lin_vel_w"]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._motion_cache["body_ang_vel_w"]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._motion_cache["body_pos_w"][:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._motion_cache["body_quat_w"][:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._motion_cache["body_lin_vel_w"][:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._motion_cache["body_ang_vel_w"][:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel.torch

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w.torch[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w.torch[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"].copy_(torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1))
        self.metrics["error_anchor_rot"].copy_(quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w))
        self.metrics["error_anchor_lin_vel"].copy_(
            torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        )
        self.metrics["error_anchor_ang_vel"].copy_(
            torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        )

        self.metrics["error_body_pos"].copy_(
            torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        )
        self.metrics["error_body_rot"].copy_(
            quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        )
        self.metrics["error_body_lin_vel"].copy_(
            torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        )
        self.metrics["error_body_ang_vel"].copy_(
            torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)
        )

        self.metrics["error_joint_pos"].copy_(torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1))
        self.metrics["error_joint_vel"].copy_(torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1))
        if self._curriculum_quality_learning_enabled():
            self._accumulate_curriculum_quality_error()

    def _accumulate_curriculum_quality_error(self) -> None:
        """Accumulate pre-reset, reward-aligned body-position error for the active window."""

        if not (
            hasattr(self, "last_global_body_pos_errors") and hasattr(self, "last_global_body_pos_error_time_steps")
        ):
            return

        if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            pending = (
                self._has_sampled
                & self._episode_unified_window_active
                & self._episode_curriculum_quality_valid
                & ~self._env.reset_buf
                & ~self._env.reset_terminated
                & ~self._env.reset_time_outs
                & (self.last_global_body_pos_error_time_steps == self.time_steps)
            )
        else:
            pending = (
                self._has_sampled
                & self._episode_start_label_enabled
                & ~self._episode_start_outcome_recorded
                & self._episode_curriculum_quality_valid
                & ~self._env.reset_buf
                & ~self._env.reset_terminated
                & ~self._env.reset_time_outs
                & (self.last_global_body_pos_error_time_steps == self.time_steps)
            )
        quality_mse = torch.square(self.last_global_body_pos_errors).mean(dim=-1)
        finite = torch.isfinite(quality_mse)
        self._episode_curriculum_quality_valid[pending & ~finite] = False
        observed = pending & finite
        self._episode_curriculum_quality_error_sum.add_(torch.where(observed, quality_mse, 0.0))
        self._episode_curriculum_quality_observation_count.add_(observed)
        self._episode_curriculum_quality_last_time_step[observed] = self.time_steps[observed]

    def _record_curriculum_start_quality(self, env_ids: torch.Tensor) -> None:
        """Record one complete-horizon quality observation per successful start."""

        if len(env_ids) == 0:
            return
        observation_count = self._episode_curriculum_quality_observation_count[env_ids]
        valid = self._episode_curriculum_quality_valid[env_ids] & (observation_count > 0.0)
        valid_env_ids = env_ids[valid]
        if len(valid_env_ids) == 0:
            return
        start_bins = self._episode_start_bins[valid_env_ids]
        mean_quality_mse = (
            self._episode_curriculum_quality_error_sum[valid_env_ids]
            / self._episode_curriculum_quality_observation_count[valid_env_ids]
        )
        std = float(self.cfg.curriculum_quality_body_pos_std)
        quality_score = 1.0 - torch.exp(-mean_quality_mse / (std * std))
        self._current_curriculum_quality_trials.index_add_(
            0, start_bins, torch.ones_like(start_bins, dtype=torch.float32)
        )
        self._current_curriculum_quality_error_sum.index_add_(0, start_bins, quality_score)

    def _reset_curriculum_window_quality(self, env_ids: torch.Tensor, *, valid: bool = True) -> None:
        """Reset quality accumulation independently for each unified window."""

        if not self._curriculum_quality_learning_enabled() or len(env_ids) == 0:
            return
        self._episode_curriculum_quality_error_sum[env_ids] = 0.0
        self._episode_curriculum_quality_observation_count[env_ids] = 0.0
        self._episode_curriculum_quality_valid[env_ids] = valid
        self._episode_curriculum_quality_last_time_step[env_ids] = -1

    def _record_curriculum_unified_window_quality(self, env_ids: torch.Tensor) -> None:
        """Write successful start and transit quality into the same per-bin arrays."""

        if not self._curriculum_quality_learning_enabled() or len(env_ids) == 0:
            return
        observation_count = self._episode_curriculum_quality_observation_count[env_ids]
        valid = self._episode_curriculum_quality_valid[env_ids] & (observation_count > 0.0)
        valid_env_ids = env_ids[valid]
        if len(valid_env_ids) == 0:
            return
        window_bins = self._episode_unified_window_bins[valid_env_ids]
        mean_quality_mse = (
            self._episode_curriculum_quality_error_sum[valid_env_ids]
            / self._episode_curriculum_quality_observation_count[valid_env_ids]
        )
        std = float(self.cfg.curriculum_quality_body_pos_std)
        quality_score = 1.0 - torch.exp(-mean_quality_mse / (std * std))
        self._current_curriculum_quality_trials.index_add_(
            0, window_bins, torch.ones_like(window_bins, dtype=torch.float32)
        )
        self._current_curriculum_quality_error_sum.index_add_(0, window_bins, quality_score)

    def _initialize_curriculum_unified_episode(self, env_ids: torch.Tensor) -> None:
        """Start one non-overlapping window chain at each environment's sampled frame."""

        if len(env_ids) == 0:
            return
        start_frames = self.time_steps[env_ids]
        start_bins = self._bucket_ids(self.motion_ids[env_ids], start_frames)
        horizon = int(self.cfg.curriculum_start_horizon_frames)
        remaining = (self.motion_lengths[env_ids] - start_frames).clamp_min(1)
        self._episode_unified_window_bins[env_ids] = start_bins
        self._episode_unified_window_frames[env_ids] = start_frames
        self._episode_unified_window_steps[env_ids] = 0
        self._episode_unified_window_targets[env_ids] = torch.minimum(remaining, torch.full_like(remaining, horizon))
        self._episode_unified_window_active[env_ids] = True
        self._episode_forward_risk_bins[env_ids] = -1
        self._reset_curriculum_window_quality(env_ids)

    def _record_curriculum_unified_window_outcomes(
        self, env_ids: torch.Tensor, *, failed: bool, include_terminal_frame: bool = False
    ) -> None:
        """Record one local trial for every active start or transit window."""

        if len(env_ids) == 0:
            return
        active = self._episode_unified_window_active[env_ids]
        env_ids = env_ids[active]
        if len(env_ids) == 0:
            return
        window_bins = self._episode_unified_window_bins[env_ids]
        ones = torch.ones_like(window_bins, dtype=torch.float32)
        self._current_curriculum_start_trials.index_add_(0, window_bins, ones)
        if failed:
            self._current_curriculum_start_failures.index_add_(0, window_bins, ones)
        else:
            self._record_curriculum_unified_window_quality(env_ids)
        steps = self._episode_unified_window_steps[env_ids].float()
        targets = self._episode_unified_window_targets[env_ids].clamp_min(1).float()
        if include_terminal_frame:
            # A terminating frame is observed by the termination manager but
            # does not reach _update_adaptive_exposure(). Include it here.
            steps = torch.minimum(steps + 1.0, targets)
        completion = torch.ones_like(steps) if not failed else (steps / targets).clamp_(0.0, 1.0)
        self._current_curriculum_start_survival_steps.index_add_(0, window_bins, steps)
        self._current_curriculum_start_completion_fraction.index_add_(0, window_bins, completion)
        self._episode_unified_window_active[env_ids] = False

    def _accumulate_curriculum_terminal_quality(self, env_ids: torch.Tensor) -> None:
        """Include the natural motion-ending frame in a successful tail window."""

        if not self._curriculum_quality_learning_enabled() or len(env_ids) == 0:
            return
        if not (
            hasattr(self, "last_global_body_pos_errors") and hasattr(self, "last_global_body_pos_error_time_steps")
        ):
            return
        active = self._episode_unified_window_active[env_ids] & self._episode_curriculum_quality_valid[env_ids]
        current = (self.last_global_body_pos_error_time_steps[env_ids] == self.time_steps[env_ids]) & (
            self._episode_curriculum_quality_last_time_step[env_ids] != self.time_steps[env_ids]
        )
        candidate_env_ids = env_ids[active & current]
        if len(candidate_env_ids) == 0:
            return
        quality_mse = torch.square(self.last_global_body_pos_errors[candidate_env_ids]).mean(dim=-1)
        finite = torch.isfinite(quality_mse)
        self._episode_curriculum_quality_valid[candidate_env_ids[~finite]] = False
        observed_env_ids = candidate_env_ids[finite]
        self._episode_curriculum_quality_error_sum[observed_env_ids] += quality_mse[finite]
        self._episode_curriculum_quality_observation_count[observed_env_ids] += 1.0
        self._episode_curriculum_quality_last_time_step[observed_env_ids] = self.time_steps[observed_env_ids]

    def _record_curriculum_unified_window_censored(self, env_ids: torch.Tensor) -> None:
        """Censor an interrupted partial window without turning it into a failure."""

        if len(env_ids) == 0:
            return
        active = self._episode_unified_window_active[env_ids]
        env_ids = env_ids[active]
        if len(env_ids) == 0:
            return
        window_bins = self._episode_unified_window_bins[env_ids]
        ones = torch.ones_like(window_bins, dtype=torch.float32)
        self._current_curriculum_start_censored.index_add_(0, window_bins, ones)
        steps = self._episode_unified_window_steps[env_ids].float()
        targets = self._episode_unified_window_targets[env_ids].clamp_min(1).float()
        self._current_curriculum_start_survival_steps.index_add_(0, window_bins, steps)
        self._current_curriculum_start_completion_fraction.index_add_(
            0, window_bins, (steps / targets).clamp_(0.0, 1.0)
        )
        self._episode_unified_window_active[env_ids] = False

    def _advance_curriculum_unified_windows(self, env_ids: torch.Tensor) -> None:
        """Close completed windows and open the next anchor without overlap."""

        if len(env_ids) == 0:
            return
        self._record_curriculum_unified_window_outcomes(env_ids, failed=False)
        next_frames = self._episode_unified_window_frames[env_ids] + self._episode_unified_window_targets[env_ids]
        has_next = next_frames < self.motion_lengths[env_ids]
        next_env_ids = env_ids[has_next]
        if len(next_env_ids) == 0:
            return
        next_frames = next_frames[has_next]
        horizon = int(self.cfg.curriculum_start_horizon_frames)
        remaining = self.motion_lengths[next_env_ids] - next_frames
        self._episode_unified_window_frames[next_env_ids] = next_frames
        self._episode_unified_window_bins[next_env_ids] = self._bucket_ids(self.motion_ids[next_env_ids], next_frames)
        self._episode_unified_window_steps[next_env_ids] = 0
        self._episode_unified_window_targets[next_env_ids] = torch.minimum(
            remaining, torch.full_like(remaining, horizon)
        )
        self._episode_unified_window_active[next_env_ids] = True
        self._reset_curriculum_window_quality(next_env_ids)

    def _record_curriculum_forward_outcomes(self, env_ids: torch.Tensor, *, failed: bool) -> None:
        """Resolve all pending K-bin lookahead cohorts for an episode."""

        if len(env_ids) == 0:
            return
        history = self._episode_forward_risk_bins[env_ids]
        valid = history >= 0
        if not torch.any(valid):
            self._episode_forward_risk_bins[env_ids] = -1
            return
        flat_bins = history[valid]
        ones = torch.ones_like(flat_bins, dtype=torch.float32)
        self._current_curriculum_forward_trials.index_add_(0, flat_bins, ones)
        if failed:
            distances = torch.arange(history.shape[1], dtype=torch.float32, device=self.device)
            weights = float(self.cfg.curriculum_forward_risk_decay) ** distances
            scores = weights.unsqueeze(0).expand_as(history)[valid]
            self._current_curriculum_forward_risk_sum.index_add_(0, flat_bins, scores)
        self._episode_forward_risk_bins[env_ids] = -1

    def _enter_curriculum_forward_bins(self, env_ids: torch.Tensor, bin_ids: torch.Tensor) -> None:
        """Open a lookahead cohort and safely resolve the one leaving K-bin history."""

        if len(env_ids) == 0:
            return
        history = self._episode_forward_risk_bins[env_ids]
        expired = history[:, -1]
        valid_expired = expired >= 0
        expired_bins = expired[valid_expired]
        self._current_curriculum_forward_trials.index_add_(
            0, expired_bins, torch.ones_like(expired_bins, dtype=torch.float32)
        )
        shifted = history.clone()
        if shifted.shape[1] > 1:
            shifted[:, 1:] = history[:, :-1]
        shifted[:, 0] = bin_ids
        self._episode_forward_risk_bins[env_ids] = shifted

    def _update_curriculum_unified_windows(self, env_ids: torch.Tensor) -> None:
        """Advance active V18 windows once for the current reference frame."""

        if len(env_ids) == 0:
            return
        active = self._episode_unified_window_active[env_ids]
        env_ids = env_ids[active]
        if len(env_ids) == 0:
            return
        self._episode_unified_window_steps[env_ids] += 1
        complete = self._episode_unified_window_steps[env_ids] >= self._episode_unified_window_targets[env_ids]
        self._advance_curriculum_unified_windows(env_ids[complete])

    def _bucket_ids(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        global_motion_ids = self.motion.global_ids[motion_ids]
        return self.motion_bin_offsets[global_motion_ids] + torch.div(
            time_steps, self.cfg.bin_size, rounding_mode="floor"
        )

    def _compute_failure_rate(self) -> torch.Tensor:
        failure_rate = torch.where(
            self.adp_samp_num_episodes > 0.0,
            self.adp_samp_num_failures / self.adp_samp_num_episodes.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(self.adp_samp_num_failures),
        )
        if self.cfg.adaptive_sampling_alpha is not None:
            failure_rate = self._smooth_failure_rate(failure_rate)
        if not self.cfg.use_failure_rate_decay:
            return failure_rate
        failure_rate_with_decay = torch.zeros_like(failure_rate)
        for step in reversed(range(self.bin_count)):
            if step == self.bin_count - 1:
                next_failure_rate = 0.0
            elif self.bin_motion_ids[step + 1] == self.bin_motion_ids[step]:
                next_failure_rate = self.cfg.decay_gamma * failure_rate_with_decay[step + 1]
            else:
                next_failure_rate = 0.0
            failure_rate_with_decay[step] = failure_rate[step] + next_failure_rate
        return failure_rate_with_decay

    def _smooth_failure_rate(self, failure_rate: torch.Tensor) -> torch.Tensor:
        """Apply MarmotLab's forward-looking kernel within each motion."""
        if self.cfg.adaptive_kernel_size == 1:
            return failure_rate
        kernel = failure_rate.new_tensor(
            [float(self.cfg.adaptive_kernel_lambda) ** index for index in range(self.cfg.adaptive_kernel_size)]
        )
        kernel /= kernel.sum()
        motion_end_bins = self.motion_bin_offsets[self.bin_motion_ids] + self.motion_bin_counts[self.bin_motion_ids] - 1
        bin_ids = torch.arange(self.bin_count, device=self.device)
        smoothed = torch.zeros_like(failure_rate)
        for offset, coefficient in enumerate(kernel):
            smoothed.add_(coefficient * failure_rate[torch.minimum(bin_ids + offset, motion_end_bins)])
        return smoothed

    @staticmethod
    def _cap_probabilities_at_uniform_ratio(probabilities: torch.Tensor, ratio: float) -> torch.Tensor:
        """Cap final probabilities and redistribute mass exactly as MarmotLab does."""
        probability_cap = ratio / float(probabilities.numel())
        if probability_cap >= 1.0:
            return probabilities
        capped = probabilities.clamp(max=probability_cap)
        deficit = 1.0 - capped.sum()
        if deficit <= 0.0:
            return capped
        available_capacity = (probability_cap - capped).clamp_min(0.0)
        capacity_sum = available_capacity.sum()
        if capacity_sum <= 0.0:
            return capped / capped.sum()
        return capped + deficit * available_capacity / capacity_sum

    def _marmot_probabilities(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.double()
        if scores.sum() <= 0.0:
            probabilities = torch.full_like(scores, 1.0 / len(scores))
        else:
            probabilities = scores / scores.sum()
        probabilities.mul_(1.0 - self.cfg.uniform_sampling_rate)
        probabilities.add_(self.cfg.uniform_sampling_rate / len(probabilities))
        if self.cfg.adp_samp_failure_rate_max_over_mean is not None:
            probabilities = self._cap_probabilities_at_uniform_ratio(
                probabilities, float(self.cfg.adp_samp_failure_rate_max_over_mean)
            )
        return probabilities

    def _clip_failure_rate(self, failure_rate: torch.Tensor) -> torch.Tensor:
        multiplier = self.cfg.adp_samp_failure_rate_max_over_mean
        if multiplier is None:
            return failure_rate
        return failure_rate.clamp(max=failure_rate.mean() * multiplier)

    def _configured_probability_cap(self, value: str | float | None, count: int) -> float | None:
        if value is None:
            return None
        if value == "auto":
            return float(self.cfg.adp_samp_failure_rate_max_over_mean) / count
        return float(value)

    def _apply_probability_caps(self, probabilities: torch.Tensor, active_bin_ids: torch.Tensor) -> torch.Tensor:
        max_prob_per_bin = self._configured_probability_cap(self.cfg.max_prob_per_bin, len(active_bin_ids))
        if max_prob_per_bin is not None and max_prob_per_bin > 0.0 and len(active_bin_ids) > 1.0 / max_prob_per_bin:
            probabilities = probabilities.clamp(max=max_prob_per_bin)
            probabilities /= probabilities.sum()

        active_motion_ids = self.bin_motion_ids[active_bin_ids]
        unique_motion_ids = active_motion_ids.unique()
        max_prob_per_motion = self._configured_probability_cap(self.cfg.max_prob_per_motion, len(unique_motion_ids))
        if (
            max_prob_per_motion is not None
            and max_prob_per_motion > 0.0
            and len(unique_motion_ids) > 1.0 / max_prob_per_motion
        ):
            for motion_id in unique_motion_ids:
                motion_mask = active_motion_ids == motion_id
                motion_probability = probabilities[motion_mask].sum()
                if motion_probability > max_prob_per_motion:
                    probabilities[motion_mask] *= max_prob_per_motion / motion_probability
            probabilities /= probabilities.sum()
        return probabilities

    def _curriculum_blend_factor(self) -> float:
        if not self._curriculum_sampling_enabled() or self._curriculum_blend_start_iteration < 0:
            return 0.0
        blend_iterations = int(self.cfg.curriculum_blend_iterations)
        if blend_iterations == 0:
            return 1.0
        elapsed = self._curriculum_iteration - self._curriculum_blend_start_iteration
        return min(max(elapsed / blend_iterations, 0.0), 1.0)

    def _curriculum_state_budget_probabilities(
        self,
        active_bin_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
        states: torch.Tensor | None = None,
        motion_length_exponent: float | None = None,
        preserve_absent_state_budgets: bool | None = None,
        state_weight_values: torch.Tensor | None = None,
        bin_base_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assign fixed state budgets, optionally tempering motion length within each state."""

        if states is None:
            states = self._curriculum_states
        active_states = states[active_bin_ids].long()
        state_counts = torch.bincount(active_states, minlength=len(self._CURRICULUM_STATE_NAMES)).to(dtype=dtype)
        if state_weight_values is None:
            state_weights = torch.tensor(
                [self.cfg.curriculum_state_sampling_weights[name] for name in self._CURRICULUM_STATE_NAMES],
                dtype=dtype,
                device=self.device,
            )
        else:
            state_weights = state_weight_values.to(device=self.device, dtype=dtype)
        state_weights /= state_weights.sum()
        state_is_present = state_counts > 0
        if preserve_absent_state_budgets is None:
            preserve_absent_state_budgets = bool(getattr(self.cfg, "curriculum_preserve_absent_state_budgets", False))
        if preserve_absent_state_budgets:
            absent_state_weight = state_weights[~state_is_present].sum()
        else:
            state_weights *= state_is_present
            state_weights /= state_weights.sum()
            absent_state_weight = state_weights.new_zeros(())
        if motion_length_exponent is None:
            motion_length_exponent = float(getattr(self.cfg, "curriculum_motion_length_exponent", 1.0))
        if bin_base_weights is not None:
            active_bin_weights = bin_base_weights.to(device=self.device, dtype=dtype)
            if active_bin_weights.shape != active_bin_ids.shape or torch.any(active_bin_weights <= 0.0):
                raise ValueError("curriculum bin base weights must be positive and match active_bin_ids")
        else:
            active_bin_weights = None
        if motion_length_exponent == 1.0 and active_bin_weights is not None:
            state_weight_sums = torch.bincount(
                active_states,
                weights=active_bin_weights,
                minlength=len(self._CURRICULUM_STATE_NAMES),
            )
            probabilities = (
                state_weights[active_states]
                * active_bin_weights
                / state_weight_sums[active_states].clamp_min(torch.finfo(dtype).tiny)
            )
        elif motion_length_exponent != 1.0:
            active_motion_ids = self.bin_motion_ids[active_bin_ids]
            state_motion_ids = active_states * self.global_num_motions + active_motion_ids
            if active_bin_weights is None:
                active_bin_weights = (self.bin_ends[active_bin_ids] - self.bin_starts[active_bin_ids]).to(dtype=dtype)
            state_motion_frame_counts = torch.bincount(
                state_motion_ids,
                weights=active_bin_weights,
                minlength=len(self._CURRICULUM_STATE_NAMES) * self.global_num_motions,
            )
            group_weights = torch.where(
                state_motion_frame_counts > 0.0,
                state_motion_frame_counts.pow(motion_length_exponent),
                torch.zeros_like(state_motion_frame_counts),
            )
            state_group_weight_sums = group_weights.reshape(len(self._CURRICULUM_STATE_NAMES), -1).sum(dim=1)
            active_group_frame_counts = state_motion_frame_counts[state_motion_ids]
            active_group_weights = group_weights[state_motion_ids]
            probability_per_group = (
                state_weights[active_states] * active_group_weights / state_group_weight_sums[active_states]
            )
            probabilities = probability_per_group * active_bin_weights / active_group_frame_counts
        else:
            probability_per_bin = state_weights / state_counts.clamp_min(1.0)
            probabilities = probability_per_bin[active_states]
        if preserve_absent_state_budgets:
            if active_bin_weights is None:
                replay_probabilities = probabilities.new_full(probabilities.shape, 1.0 / max(len(active_bin_ids), 1))
            else:
                replay_probabilities = active_bin_weights / active_bin_weights.sum()
            probabilities = probabilities + absent_state_weight * replay_probabilities
        return probabilities / probabilities.sum()

    def _curriculum_provisional_known_mask(self) -> torch.Tensor:
        trials = float(getattr(self.cfg, "curriculum_provisional_known_trials", 1))
        provisional = self.curriculum_start_trials >= trials
        if self._curriculum_start_eligibility_enabled():
            provisional &= self._curriculum_eligible_bins
        return provisional

    def _curriculum_mask_fraction(self, mask: torch.Tensor) -> float:
        if self._curriculum_start_eligibility_enabled():
            denominator = self._curriculum_eligible_bins.sum().clamp_min(1)
            return float((mask & self._curriculum_eligible_bins).sum().float().div(denominator).item())
        return float(mask.float().mean().item())

    def _curriculum_provisional_motion_fraction_value(self) -> float:
        if not self._curriculum_start_eligibility_enabled():
            return 1.0
        provisional = self._curriculum_provisional_known_mask()
        provisional_motion_counts = torch.bincount(self.bin_motion_ids[provisional], minlength=self.global_num_motions)
        covered_motions = self._curriculum_eligible_motions & (
            provisional_motion_counts >= self._curriculum_provisional_motion_required_counts
        )
        return float(covered_motions.sum().float().div(self._curriculum_eligible_motions.sum().clamp_min(1)).item())

    def _curriculum_state_focus_factor(self) -> float:
        """Smoothly introduce five-state focus after broad eligible coverage exists."""

        if not self._curriculum_coverage_aware_enabled():
            return 1.0
        start = float(self.cfg.curriculum_state_focus_start_fraction)
        end = float(self.cfg.curriculum_state_focus_end_fraction)
        progress = min(max((self._curriculum_known_fraction - start) / (end - start), 0.0), 1.0)
        return progress * progress * (3.0 - 2.0 * progress)

    def _curriculum_target_distribution_components(
        self, active_bin_ids: torch.Tensor, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return endpoint total, comparable-start, and terminal-replay mass."""

        if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            probabilities = self._curriculum_unified_need_probabilities(active_bin_ids, dtype=dtype)
            zeros = torch.zeros_like(probabilities)
            return probabilities, probabilities, zeros

        if not self._curriculum_start_eligibility_enabled():
            probabilities = self._curriculum_state_budget_probabilities(active_bin_ids, dtype=dtype)
            zeros = torch.zeros_like(probabilities)
            return probabilities, probabilities, zeros

        eligible_mask = self._curriculum_eligible_bins[active_bin_ids]
        terminal_counts = self._curriculum_terminal_start_counts[active_bin_ids].to(dtype=dtype)
        terminal_mask = terminal_counts > 0
        eligible_component = torch.zeros(len(active_bin_ids), dtype=dtype, device=self.device)
        terminal_component = torch.zeros_like(eligible_component)
        has_eligible = bool(torch.any(eligible_mask).item())
        has_terminal = bool(torch.any(terminal_mask).item())

        if has_eligible:
            eligible_bin_ids = active_bin_ids[eligible_mask]
            eligible_weights = self._curriculum_eligible_start_counts[eligible_bin_ids].to(dtype=dtype)
            coverage_probabilities = eligible_weights / eligible_weights.sum()
            sampling_states = None
            state_bin_weights = eligible_weights
            if self._curriculum_quality_learning_enabled() and self._curriculum_quality_filler_count > 0:
                sampling_states = self._curriculum_states.clone()
                sampling_states[self._curriculum_quality_filler_mask] = self._CURRICULUM_FRONTIER
                active_fillers = self._curriculum_quality_filler_mask[eligible_bin_ids]
                state_bin_weights = eligible_weights.clone()
                state_bin_weights[active_fillers] *= float(self.cfg.curriculum_quality_filler_weight)
            state_probabilities = self._curriculum_state_budget_probabilities(
                eligible_bin_ids,
                dtype=dtype,
                states=sampling_states,
                bin_base_weights=state_bin_weights,
            )
            focus = self._curriculum_state_focus_factor()
            eligible_probabilities = coverage_probabilities.lerp(state_probabilities, focus)
            effective_focus = focus
            floor_ratio = float(getattr(self.cfg, "curriculum_unconfirmed_min_coverage_ratio", 0.0))
            if floor_ratio > 0.0:
                confirmed = self._curriculum_known_mask()[eligible_bin_ids]
                unconfirmed = ~confirmed
                coverage_mass = coverage_probabilities[unconfirmed].sum()
                target_mass = eligible_probabilities[unconfirmed].sum()
                minimum_mass = coverage_mass * floor_ratio
                coverage_mix = ((minimum_mass - target_mass) / (coverage_mass - target_mass).clamp_min(1.0e-12)).clamp(
                    0.0, 1.0
                )
                eligible_probabilities.lerp_(coverage_probabilities, coverage_mix)
                effective_focus *= 1.0 - float(coverage_mix.item())
            self._curriculum_effective_state_focus = effective_focus
            eligible_component[eligible_mask] = eligible_probabilities

        if has_terminal:
            terminal_component[terminal_mask] = terminal_counts[terminal_mask] / terminal_counts[terminal_mask].sum()

        if has_eligible and has_terminal:
            terminal_fraction = float(self.cfg.curriculum_terminal_replay_fraction)
            eligible_component *= 1.0 - terminal_fraction
            terminal_component *= terminal_fraction
        elif has_eligible:
            terminal_component.zero_()
        elif has_terminal:
            eligible_component.zero_()
        else:
            raise RuntimeError("curriculum target has no eligible or terminal start frames")
        probabilities = eligible_component + terminal_component
        probabilities /= probabilities.sum()
        normalization = probabilities.new_tensor(1.0) / (eligible_component + terminal_component).sum()
        eligible_component *= normalization
        terminal_component *= normalization
        return probabilities, eligible_component, terminal_component

    @staticmethod
    def _normalize_curriculum_need(values: torch.Tensor, good: float, bad: float) -> torch.Tensor:
        """Map one calibrated diagnostic range into a bounded need score."""

        return ((values - good) / (bad - good)).clamp_(0.0, 1.0)

    def _curriculum_quality_evidence_mask(self) -> torch.Tensor:
        """Return bins with enough successful-window quality evidence."""

        return (
            (self.curriculum_quality_trials >= float(self.cfg.curriculum_quality_min_trials))
            & (self._curriculum_quality_observed_windows >= int(self.cfg.curriculum_quality_min_windows))
            & torch.isfinite(self._curriculum_quality_error_ema)
        )

    @staticmethod
    def _curriculum_normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        """Return Shannon entropy normalized by the distribution size."""

        if probabilities.numel() <= 1:
            return probabilities.new_ones(())
        tiny = torch.finfo(probabilities.dtype).tiny
        entropy = -(probabilities * probabilities.clamp_min(tiny).log()).sum()
        return entropy / math.log(probabilities.numel())

    def _curriculum_recent_progress_need(
        self,
        active_bin_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return positive local/forward progress calibrated by state thresholds."""

        no_progress = float(self.cfg.curriculum_no_progress_threshold)
        improvement = float(self.cfg.curriculum_improvement_threshold)

        def normalized(history: torch.Tensor) -> torch.Tensor:
            active_history = history[:, active_bin_ids].to(dtype=dtype)
            finite = torch.isfinite(active_history).all(dim=0)
            progress = active_history[0] - active_history[2]
            progress = ((progress - no_progress) / (improvement - no_progress)).clamp_(0.0, 1.0)
            return torch.where(finite, progress, torch.zeros_like(progress))

        return torch.maximum(
            normalized(self._curriculum_failure_rate_history),
            normalized(self._curriculum_forward_risk_history),
        )

    def _curriculum_quality_base_coverage(
        self,
        active_bin_ids: torch.Tensor,
        uniform_probabilities: torch.Tensor,
        quality_need: torch.Tensor,
    ) -> torch.Tensor:
        """Reweight quality within MASTERED while preserving its base mass exactly."""

        coverage = uniform_probabilities.clone()
        mastered = self._curriculum_states[active_bin_ids] == self._CURRICULUM_MASTERED
        quality_known = self._curriculum_quality_evidence_mask()[active_bin_ids]
        quality_mastered = mastered & quality_known
        mastered_mass = uniform_probabilities[mastered].sum()
        if mastered_mass > 0.0 and torch.any(quality_mastered):
            factors = torch.ones_like(quality_need)
            factors[quality_mastered] += quality_need[quality_mastered]
            reweighted = uniform_probabilities[mastered] * factors[mastered]
            coverage[mastered] = reweighted * (mastered_mass / reweighted.sum())
        return coverage / coverage.sum()

    def _curriculum_learnability_probabilities(
        self,
        active_bin_ids: torch.Tensor,
        frame_weights: torch.Tensor,
        survival_need: torch.Tensor,
        forward_need: torch.Tensor,
        quality_need: torch.Tensor,
        local_uncertainty_need: torch.Tensor,
        forward_uncertainty_need: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add self-normalized learnability priority to quality-shaped coverage."""

        uniform_probabilities = frame_weights / frame_weights.sum()
        quality_coverage = self._curriculum_quality_base_coverage(
            active_bin_ids,
            uniform_probabilities,
            quality_need,
        )

        uncertainty_need = torch.maximum(local_uncertainty_need, forward_uncertainty_need)
        progress_need = self._curriculum_recent_progress_need(active_bin_ids, dtype=dtype)
        difficulty_need = torch.maximum(survival_need, forward_need)
        learnability_need = torch.maximum(uncertainty_need, progress_need)
        primary_need = difficulty_need * learnability_need
        primary_support = primary_need > 0.0

        if torch.any(primary_support):
            # The scale is supplied by the current positive evidence itself:
            # an average learnable bin receives a 2x multiplier over its base
            # coverage, without a hand-set pool fraction or entropy target.
            positive_frame_weights = frame_weights[primary_support]
            primary_scale = (primary_need[primary_support] * positive_frame_weights).sum()
            primary_scale /= positive_frame_weights.sum()
            priority_multiplier = 1.0 + primary_need / primary_scale
            unnormalized = quality_coverage * priority_multiplier
            normalization = unnormalized.sum()
            probabilities = unnormalized / normalization
            base_fraction = float((1.0 / normalization).item())
        else:
            probabilities = quality_coverage
            primary_scale = primary_need.new_zeros(())
            base_fraction = 1.0

        self._curriculum_last_learnability_base_fraction = base_fraction
        self._curriculum_last_learnability_primary_support_fraction = float(primary_support.float().mean().item())
        self._curriculum_last_learnability_primary_need_mean = float(primary_need.mean().item())
        self._curriculum_last_learnability_primary_scale = float(primary_scale.item())
        self._curriculum_last_learnability_target_entropy = float(
            self._curriculum_normalized_entropy(probabilities).item()
        )
        mastered = self._curriculum_states[active_bin_ids] == self._CURRICULUM_MASTERED
        quality_reweighted = mastered & self._curriculum_quality_evidence_mask()[active_bin_ids] & (quality_need > 0.0)
        self._curriculum_last_learnability_quality_reweighted_fraction = float(
            quality_reweighted.float().mean().item()
        )
        self._curriculum_last_learnability_quality_mastered_mass = float(
            (quality_coverage[mastered].sum() * base_fraction).item()
        )
        return probabilities / probabilities.sum(), primary_need, uncertainty_need

    def _curriculum_unified_need_probabilities(
        self, active_bin_ids: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        """Mix an all-frame floor with continuous V18 learning need."""

        prior_alpha = float(self.cfg.curriculum_beta_prior_alpha)
        prior_beta = float(self.cfg.curriculum_beta_prior_beta)
        trials = self.curriculum_start_trials[active_bin_ids].to(dtype=dtype)
        failures = self.curriculum_start_failures[active_bin_ids].to(dtype=dtype)
        # Sampling need uses all unified start/transit outcomes.  Recent
        # decision windows remain deliberately separate for state hysteresis;
        # eight fresh outcomes must not replace a large cumulative estimate.
        local_rate = (failures + prior_alpha) / (trials + prior_alpha + prior_beta)
        survival_need = self._normalize_curriculum_need(local_rate, 0.10, 0.90)

        forward_known = self.curriculum_forward_trials[active_bin_ids] >= float(self.cfg.curriculum_min_known_trials)
        forward_rate = torch.nan_to_num(
            self._curriculum_forward_risk_ema[active_bin_ids].to(dtype=dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        recent_forward_trials = self._curriculum_window_forward_trials[active_bin_ids].to(dtype=dtype)
        recent_forward_rate = self._curriculum_window_forward_risk_sum[active_bin_ids].to(
            dtype=dtype
        ) / recent_forward_trials.clamp_min(1.0)
        forward_rate = torch.where(
            recent_forward_trials >= float(self.cfg.curriculum_min_window_trials),
            recent_forward_rate,
            forward_rate,
        )
        forward_need = self._normalize_curriculum_need(
            forward_rate,
            float(self.cfg.curriculum_forward_risk_good_threshold),
            float(self.cfg.curriculum_forward_risk_bad_threshold),
        )
        forward_need *= forward_known

        if self._curriculum_quality_learning_enabled():
            quality_known = self._curriculum_quality_evidence_mask()[active_bin_ids]
            quality_score = torch.nan_to_num(
                self._curriculum_quality_error_ema[active_bin_ids].to(dtype=dtype),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            quality_need = self._normalize_curriculum_need(
                quality_score,
                float(self.cfg.curriculum_quality_good_threshold),
                float(self.cfg.curriculum_quality_bad_threshold),
            )
            quality_need *= quality_known
        else:
            quality_need = torch.zeros_like(survival_need)
        uncertainty_need = (1.0 - trials / float(self.cfg.curriculum_min_known_trials)).clamp_(0.0, 1.0)
        forward_uncertainty_need = (
            1.0
            - self.curriculum_forward_trials[active_bin_ids].to(dtype=dtype)
            / float(self.cfg.curriculum_min_known_trials)
        ).clamp_(0.0, 1.0)

        need = torch.maximum(
            torch.maximum(survival_need, forward_need),
            torch.maximum(quality_need, uncertainty_need),
        )
        quarantine = self._curriculum_states[active_bin_ids] == self._CURRICULUM_QUARANTINE
        need[quarantine] *= float(self.cfg.curriculum_quarantine_need_scale)

        frame_weights = (self.bin_ends[active_bin_ids] - self.bin_starts[active_bin_ids]).to(dtype=dtype)
        if _cfg_curriculum_learnability_enabled(self.cfg):
            probabilities, need, uncertainty_need = self._curriculum_learnability_probabilities(
                active_bin_ids,
                frame_weights,
                survival_need,
                forward_need,
                quality_need,
                uncertainty_need,
                forward_uncertainty_need,
                dtype=dtype,
            )
        else:
            uniform_probabilities = frame_weights / frame_weights.sum()
            adaptive_weights = need * frame_weights
            if adaptive_weights.sum() > 0.0:
                need_probabilities = adaptive_weights / adaptive_weights.sum()
            else:
                need_probabilities = uniform_probabilities
            uniform_fraction = float(self.cfg.curriculum_need_uniform_fraction)
            probabilities = uniform_probabilities * uniform_fraction + need_probabilities * (1.0 - uniform_fraction)

        self._curriculum_last_survival_need[active_bin_ids] = survival_need.detach().float()
        self._curriculum_last_forward_need[active_bin_ids] = forward_need.detach().float()
        self._curriculum_last_quality_need[active_bin_ids] = quality_need.detach().float()
        self._curriculum_last_uncertainty_need[active_bin_ids] = uncertainty_need.detach().float()
        self._curriculum_last_combined_need[active_bin_ids] = need.detach().float()
        return probabilities / probabilities.sum()

    def _curriculum_shadow_distribution(
        self, legacy_probabilities: torch.Tensor, active_bin_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return the frozen source sampler conditioned on the active working set."""

        if not self._curriculum_has_shadow_distribution:
            return legacy_probabilities
        if self._curriculum_has_shadow_state_recipe:
            shadow = self._curriculum_state_budget_probabilities(
                active_bin_ids,
                dtype=legacy_probabilities.dtype,
                states=self._curriculum_shadow_states,
                motion_length_exponent=self._curriculum_shadow_motion_length_exponent,
                preserve_absent_state_budgets=False,
            )
            if self.cfg.adp_samp_failure_rate_max_over_mean is not None:
                shadow = self._cap_probabilities_at_uniform_ratio(
                    shadow, float(self.cfg.adp_samp_failure_rate_max_over_mean)
                )
            return shadow / shadow.sum()
        shadow = self._curriculum_shadow_probabilities[active_bin_ids].to(dtype=legacy_probabilities.dtype)
        shadow_mass = shadow.sum()
        return shadow / shadow_mass

    def _blend_curriculum_distribution_components(
        self, legacy_probabilities: torch.Tensor, active_bin_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Blend source and V16 components while preserving per-bin attribution."""

        blend = self._curriculum_blend_factor()
        shadow_probabilities = self._curriculum_shadow_distribution(legacy_probabilities, active_bin_ids)
        if blend <= 0.0:
            zeros = torch.zeros_like(shadow_probabilities)
            return shadow_probabilities, zeros, zeros
        curriculum_probabilities, target_eligible, target_terminal = self._curriculum_target_distribution_components(
            active_bin_ids, dtype=legacy_probabilities.dtype
        )
        probabilities = shadow_probabilities.lerp(curriculum_probabilities, blend)
        eligible_component = target_eligible * blend
        terminal_component = target_terminal * blend
        if self.cfg.adp_samp_failure_rate_max_over_mean is not None:
            capped_probabilities = self._cap_probabilities_at_uniform_ratio(
                probabilities, float(self.cfg.adp_samp_failure_rate_max_over_mean)
            )
            component_scale = capped_probabilities / probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
            eligible_component *= component_scale
            terminal_component *= component_scale
            probabilities = capped_probabilities
        normalization = probabilities.sum()
        probabilities /= normalization
        eligible_component /= normalization
        terminal_component /= normalization
        return probabilities, eligible_component, terminal_component

    def _blend_curriculum_distribution(
        self, legacy_probabilities: torch.Tensor, active_bin_ids: torch.Tensor
    ) -> torch.Tensor:
        """Blend from the exact restored sampler into the start-state curriculum."""

        probabilities, _, _ = self._blend_curriculum_distribution_components(legacy_probabilities, active_bin_ids)
        return probabilities

    def _smooth_curriculum_distribution(
        self,
        probabilities: torch.Tensor,
        *,
        advance: bool = True,
        eligible_component: torch.Tensor | None = None,
        terminal_component: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Smooth one full-dataset probability transition per PPO iteration."""

        alpha = float(getattr(self.cfg, "curriculum_probability_smoothing_alpha", 1.0))
        current = self._curriculum_smoothed_probabilities
        track_components = self._curriculum_start_eligibility_enabled()
        if track_components:
            if eligible_component is None:
                eligible_component = torch.zeros_like(probabilities)
            if terminal_component is None:
                terminal_component = torch.zeros_like(probabilities)
            current_eligible = self._curriculum_smoothed_eligible_probabilities
            current_terminal = self._curriculum_smoothed_terminal_probabilities
        should_initialize = not self._curriculum_smoothed_probabilities_initialized
        should_advance = advance and (
            self._curriculum_last_probability_smoothing_iteration != self._curriculum_iteration
        )
        if should_initialize or (should_advance and alpha >= 1.0):
            current.copy_(probabilities.to(dtype=current.dtype))
            if track_components:
                current_eligible.copy_(eligible_component.to(dtype=current_eligible.dtype))
                current_terminal.copy_(terminal_component.to(dtype=current_terminal.dtype))
        elif should_advance:
            current.lerp_(probabilities.to(dtype=current.dtype), alpha)
            if track_components:
                current_eligible.lerp_(eligible_component.to(dtype=current_eligible.dtype), alpha)
                current_terminal.lerp_(terminal_component.to(dtype=current_terminal.dtype), alpha)
        if should_initialize or should_advance:
            normalization = current.sum()
            current.div_(normalization)
            if track_components:
                current_eligible.div_(normalization)
                current_terminal.div_(normalization)
            self._curriculum_smoothed_probabilities_initialized = True
            self._curriculum_last_probability_smoothing_iteration = self._curriculum_iteration
        return current.to(dtype=probabilities.dtype)

    def _update_curriculum_metrics(self, active_bin_ids: torch.Tensor, probabilities: torch.Tensor) -> None:
        if not self._curriculum_sampling_enabled() or "curriculum_known_fraction" not in self.metrics:
            return
        if self._curriculum_start_eligibility_enabled():
            active_eligible_mask = self._curriculum_eligible_bins[active_bin_ids]
            metric_bin_ids = active_bin_ids[active_eligible_mask]
            metric_probabilities = probabilities[active_eligible_mask]
        else:
            active_eligible_mask = torch.ones(len(active_bin_ids), dtype=torch.bool, device=self.device)
            metric_bin_ids = active_bin_ids
            metric_probabilities = probabilities
        active_states = self._curriculum_states[metric_bin_ids].long()
        num_states = len(self._CURRICULUM_STATE_NAMES)
        state_counts = torch.bincount(active_states, minlength=num_states).float()
        state_fractions = state_counts / max(len(metric_bin_ids), 1)
        sampling_mass = torch.bincount(active_states, weights=metric_probabilities.float(), minlength=num_states)

        prior_alpha = float(self.cfg.curriculum_beta_prior_alpha)
        prior_beta = float(self.cfg.curriculum_beta_prior_beta)
        start_known = self._curriculum_known_mask()
        if torch.any(start_known):
            start_trials = self.curriculum_start_trials[start_known].sum()
            start_failures = self.curriculum_start_failures[start_known].sum()
            start_failure_rate = (start_failures + prior_alpha) / (start_trials + prior_alpha + prior_beta)
        else:
            start_failure_rate = probabilities.new_zeros(())
        start_outcomes = self.curriculum_start_trials.sum() + self.curriculum_start_censored.sum()
        if start_outcomes > 0.0:
            start_censored_fraction = self.curriculum_start_censored.sum() / start_outcomes
        else:
            start_censored_fraction = probabilities.new_zeros(())
        visited = self.curriculum_terminal_visits > 0.0
        if torch.any(visited):
            terminal_visits = self.curriculum_terminal_visits[visited].sum()
            terminal_failures = self.curriculum_terminal_failures[visited].sum()
            terminal_hazard = (terminal_failures + prior_alpha) / (terminal_visits + prior_alpha + prior_beta)
        else:
            terminal_hazard = probabilities.new_zeros(())

        metric_values = {
            "curriculum_known_fraction": probabilities.new_tensor(self._curriculum_known_fraction),
            "curriculum_start_known_fraction": probabilities.new_tensor(self._curriculum_mask_fraction(start_known)),
            "curriculum_blend": probabilities.new_tensor(self._curriculum_blend_factor()),
            "curriculum_start_failure_rate": start_failure_rate,
            "curriculum_start_censored_fraction": start_censored_fraction,
            "curriculum_terminal_hazard": terminal_hazard,
        }
        if self._curriculum_start_eligibility_enabled():
            eligible_mass = self._curriculum_active_eligible_component_probabilities.sum()
            terminal_mass = self._curriculum_active_terminal_component_probabilities.sum()
            shadow_mass = (probabilities.new_tensor(1.0) - eligible_mass - terminal_mass).clamp_min(0.0)
            metric_values.update(
                {
                    "curriculum_eligible_bin_fraction": self._curriculum_eligible_bins.float().mean(),
                    "curriculum_provisional_known_fraction": probabilities.new_tensor(
                        self._curriculum_provisional_known_fraction
                    ),
                    "curriculum_provisional_motion_fraction": probabilities.new_tensor(
                        self._curriculum_provisional_motion_fraction
                    ),
                    "curriculum_state_focus": probabilities.new_tensor(self._curriculum_effective_state_focus),
                    "curriculum_sampling_mass_shadow": shadow_mass,
                    "curriculum_sampling_mass_eligible": eligible_mass,
                    "curriculum_sampling_mass_terminal_replay": terminal_mass,
                }
            )
        if self._curriculum_quality_learning_enabled():
            quality_known = self._curriculum_quality_known_mask()[metric_bin_ids]
            quality_fillers = self._curriculum_quality_filler_mask[metric_bin_ids]
            survival_frontier = active_states == self._CURRICULUM_FRONTIER
            learning_pool = survival_frontier | quality_fillers
            if self._curriculum_start_eligibility_enabled():
                quality_sampling_probabilities = self._curriculum_active_eligible_component_probabilities[
                    active_eligible_mask
                ]
            else:
                quality_sampling_probabilities = metric_probabilities
            quality_known_float = quality_known.float()
            quality_errors = torch.nan_to_num(
                self._curriculum_quality_error_ema[metric_bin_ids], nan=0.0, posinf=0.0, neginf=0.0
            )
            quality_error_mean = (quality_errors * quality_known_float).sum() / quality_known_float.sum().clamp_min(1.0)
            denominator = max(len(metric_bin_ids), 1)
            metric_values.update(
                {
                    "curriculum_quality_known_fraction": quality_known_float.sum() / denominator,
                    "curriculum_quality_filler_fraction": quality_fillers.float().sum() / denominator,
                    "curriculum_learning_pool_fraction": learning_pool.float().sum() / denominator,
                    "curriculum_sampling_mass_quality_filler": quality_sampling_probabilities[quality_fillers].sum(),
                    "curriculum_sampling_mass_learning_pool": quality_sampling_probabilities[learning_pool].sum(),
                    "curriculum_quality_body_pos_score_mean": quality_error_mean,
                    "curriculum_quality_body_pos_score_cutoff": probabilities.new_tensor(
                        self._curriculum_quality_error_cutoff
                    ),
                }
            )
        if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            forward_known = self.curriculum_forward_trials[metric_bin_ids] >= float(
                self.cfg.curriculum_min_known_trials
            )
            forward_values = torch.nan_to_num(
                self._curriculum_forward_risk_ema[metric_bin_ids], nan=0.0, posinf=1.0, neginf=0.0
            )
            forward_mean = (forward_values * forward_known.float()).sum() / forward_known.float().sum().clamp_min(1.0)
            blend = self._curriculum_blend_factor()
            target_mass = probabilities.new_tensor(blend)
            if _cfg_curriculum_learnability_enabled(self.cfg):
                base_fraction = self._curriculum_last_learnability_base_fraction
            else:
                base_fraction = float(self.cfg.curriculum_need_uniform_fraction)
            metric_values.update(
                {
                    "curriculum_unified_known_fraction": probabilities.new_tensor(self._curriculum_known_fraction),
                    "curriculum_unified_failure_rate": start_failure_rate,
                    "curriculum_unified_censored_fraction": start_censored_fraction,
                    "curriculum_forward_known_fraction": forward_known.float().mean(),
                    "curriculum_forward_risk_mean": forward_mean,
                    "curriculum_need_survival_mean": self._curriculum_last_survival_need[metric_bin_ids].mean(),
                    "curriculum_need_forward_mean": self._curriculum_last_forward_need[metric_bin_ids].mean(),
                    "curriculum_need_quality_mean": self._curriculum_last_quality_need[metric_bin_ids].mean(),
                    "curriculum_need_uncertainty_mean": self._curriculum_last_uncertainty_need[metric_bin_ids].mean(),
                    "curriculum_target_mass_uniform": target_mass * base_fraction,
                    "curriculum_target_mass_need": target_mass * (1.0 - base_fraction),
                }
            )
            if _cfg_curriculum_learnability_enabled(self.cfg):
                metric_values.update(
                    {
                        "curriculum_learnability_primary_support_fraction": probabilities.new_tensor(
                            self._curriculum_last_learnability_primary_support_fraction
                        ),
                        "curriculum_learnability_primary_need_mean": probabilities.new_tensor(
                            self._curriculum_last_learnability_primary_need_mean
                        ),
                        "curriculum_learnability_primary_scale": probabilities.new_tensor(
                            self._curriculum_last_learnability_primary_scale
                        ),
                        "curriculum_learnability_base_fraction": probabilities.new_tensor(
                            self._curriculum_last_learnability_base_fraction
                        ),
                        "curriculum_learnability_target_entropy": probabilities.new_tensor(
                            self._curriculum_last_learnability_target_entropy
                        ),
                        "curriculum_learnability_quality_reweighted_fraction": probabilities.new_tensor(
                            self._curriculum_last_learnability_quality_reweighted_fraction
                        ),
                        "curriculum_learnability_quality_base_mastered_mass": probabilities.new_tensor(
                            self._curriculum_last_learnability_quality_mastered_mass
                        ),
                    }
                )
        for state_id, state_name in enumerate(self._CURRICULUM_STATE_NAMES):
            metric_values[f"curriculum_state_fraction_{state_name}"] = state_fractions[state_id]
            mass_metric_name = f"curriculum_sampling_mass_{state_name}"
            if mass_metric_name in self.metrics:
                metric_values[mass_metric_name] = sampling_mass[state_id]
        if getattr(self.cfg, "curriculum_detailed_metrics", True):
            body_failure_total = self.curriculum_terminal_body_failures.sum()
            if body_failure_total > 0.0:
                body_failure_fractions = self.curriculum_terminal_body_failures / body_failure_total
            else:
                body_failure_fractions = torch.zeros_like(self.curriculum_terminal_body_failures)
            for body_id, body_name in enumerate(self.cfg.body_names):
                metric_values[f"curriculum_terminal_body_fraction_{body_name}"] = body_failure_fractions[body_id]
        self._curriculum_metric_values = {name: value.detach() for name, value in metric_values.items()}
        for name, value in self._curriculum_metric_values.items():
            self.metrics[name][:] = value

    def _curriculum_known_mask(self) -> torch.Tensor:
        start_known = self.curriculum_start_trials >= float(self.cfg.curriculum_min_known_trials)
        if self._curriculum_start_eligibility_enabled():
            start_known &= self._curriculum_eligible_bins
        return start_known

    def _curriculum_forward_terminal_hard_mask(self) -> torch.Tensor:
        """Identify bins followed immediately by a locally high terminal hazard."""

        prior_alpha = float(self.cfg.curriculum_beta_prior_alpha)
        prior_beta = float(self.cfg.curriculum_beta_prior_beta)
        min_visits = float(getattr(self.cfg, "curriculum_min_terminal_visits", 0))
        hazard_threshold = float(getattr(self.cfg, "curriculum_quarantine_terminal_hazard_threshold", 0.0))
        window_bins = int(getattr(self.cfg, "curriculum_terminal_hazard_window_bins", 1))
        bin_ids = torch.arange(self.bin_count, device=self.device)
        motion_end_bins = self.motion_bin_offsets[self.bin_motion_ids] + self.motion_bin_counts[self.bin_motion_ids] - 1
        terminal_hard = torch.zeros(self.bin_count, dtype=torch.bool, device=self.device)
        for offset in range(window_bins):
            forward_bin_ids = torch.minimum(bin_ids + offset, motion_end_bins)
            visits = self.curriculum_terminal_visits[forward_bin_ids]
            failures = self.curriculum_terminal_failures[forward_bin_ids]
            hazard = (failures + prior_alpha) / (visits + prior_alpha + prior_beta)
            terminal_hard |= (visits >= min_visits) & (hazard >= hazard_threshold)
        return terminal_hard

    def _update_curriculum_states(self) -> None:
        """Advance the five-state hysteretic classifier from one disjoint window."""

        if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            self._update_curriculum_unified_states()
            return

        window_trials = self._curriculum_window_start_trials
        window_failures = self._curriculum_window_start_failures
        min_window_trials = float(self.cfg.curriculum_min_window_trials)
        complete_start_window = window_trials >= min_window_trials
        prior_alpha = float(self.cfg.curriculum_beta_prior_alpha)
        prior_beta = float(self.cfg.curriculum_beta_prior_beta)
        start_failure_rate = (window_failures + prior_alpha) / (window_trials + prior_alpha + prior_beta)
        start_known = self._curriculum_known_mask()
        valid_start = complete_start_window & start_known
        if self._curriculum_fixed_horizon_enabled():
            # Fixed-horizon state labels are comparable only when they come
            # from fixed-horizon start outcomes.  Cumulative terminal hazard is
            # retained solely as a local stalled->quarantine safety gate.
            failure_rate = start_failure_rate
            valid = valid_start
            known = valid_start
        else:
            failure_rate = start_failure_rate
            valid = complete_start_window
            known = valid & start_known

        history = self._curriculum_failure_rate_history
        if torch.any(valid):
            previous_middle = history[1, valid].clone()
            previous_latest = history[2, valid].clone()
            history[0, valid] = previous_middle
            history[1, valid] = previous_latest
            history[2, valid] = failure_rate[valid]

        finite = torch.isfinite(history)
        low_two = finite[1:].all(dim=0) & (history[1:] <= float(self.cfg.curriculum_mastered_enter_threshold)).all(
            dim=0
        )
        three_stalled = finite.all(dim=0) & (history >= float(self.cfg.curriculum_stalled_enter_threshold)).all(dim=0)
        three_quarantine = finite.all(dim=0) & (history >= float(self.cfg.curriculum_quarantine_enter_threshold)).all(
            dim=0
        )
        progress = history[0] - history[2]
        no_progress = finite.all(dim=0) & (progress < float(self.cfg.curriculum_no_progress_threshold))
        strong_improvement = finite.all(dim=0) & (progress >= float(self.cfg.curriculum_improvement_threshold))
        current_rate = history[2]
        previous_states = self._curriculum_states.clone()
        next_states = previous_states.clone()

        unknown = (previous_states == self._CURRICULUM_UNKNOWN) & known
        next_states[unknown & low_two] = self._CURRICULUM_MASTERED
        next_states[unknown & ~low_two] = self._CURRICULUM_FRONTIER

        mastered = (previous_states == self._CURRICULUM_MASTERED) & valid
        next_states[mastered & (current_rate > float(self.cfg.curriculum_mastered_exit_threshold))] = (
            self._CURRICULUM_FRONTIER
        )

        frontier = (previous_states == self._CURRICULUM_FRONTIER) & valid
        next_states[frontier & low_two] = self._CURRICULUM_MASTERED
        next_states[frontier & valid_start & start_known & ~low_two & three_stalled & no_progress] = (
            self._CURRICULUM_STALLED
        )

        stalled = (previous_states == self._CURRICULUM_STALLED) & valid
        stalled_to_mastered = stalled & low_two
        stalled_to_frontier = (
            stalled
            & ~stalled_to_mastered
            & (strong_improvement | (current_rate < float(self.cfg.curriculum_stalled_exit_threshold)))
        )
        stalled_to_quarantine = (
            stalled
            & ~stalled_to_mastered
            & ~stalled_to_frontier
            & (self.curriculum_start_trials >= float(self.cfg.curriculum_min_quarantine_trials))
            & three_quarantine
            & no_progress
        )
        if self._curriculum_fixed_horizon_enabled():
            stalled_to_quarantine &= self._curriculum_forward_terminal_hard_mask()
        next_states[stalled_to_mastered] = self._CURRICULUM_MASTERED
        next_states[stalled_to_frontier] = self._CURRICULUM_FRONTIER
        next_states[stalled_to_quarantine] = self._CURRICULUM_QUARANTINE

        entered_quarantine = (next_states == self._CURRICULUM_QUARANTINE) & (
            previous_states != self._CURRICULUM_QUARANTINE
        )
        self._curriculum_state_entry_trials[entered_quarantine] = self.curriculum_start_trials[entered_quarantine]
        quarantine = (previous_states == self._CURRICULUM_QUARANTINE) & valid
        probe_trials = self.curriculum_start_trials - self._curriculum_state_entry_trials
        quarantine_exit = (
            quarantine
            & (probe_trials >= float(self.cfg.curriculum_min_exit_probe_trials))
            & (strong_improvement | (current_rate < float(self.cfg.curriculum_quarantine_exit_threshold)))
        )
        next_states[quarantine_exit] = self._CURRICULUM_FRONTIER

        self._curriculum_states.copy_(next_states)
        # Consume only complete decision windows.  Sparse unknown/quarantine
        # probes retain their partial evidence across update boundaries.
        consumed_start_window = valid_start if self._curriculum_fixed_horizon_enabled() else complete_start_window
        self._curriculum_window_start_trials[consumed_start_window] = 0.0
        self._curriculum_window_start_failures[consumed_start_window] = 0.0
        if self._curriculum_quality_learning_enabled():
            self._update_curriculum_quality_statistics()
            self._refresh_curriculum_quality_learning_pool()
        self._curriculum_last_state_update_iteration = self._curriculum_iteration

    @staticmethod
    def _shift_curriculum_history(history: torch.Tensor, valid: torch.Tensor, values: torch.Tensor) -> None:
        """Append one value to the three-window history of each valid bin."""

        if not torch.any(valid):
            return
        previous_middle = history[1, valid].clone()
        previous_latest = history[2, valid].clone()
        history[0, valid] = previous_middle
        history[1, valid] = previous_latest
        history[2, valid] = values[valid]

    def _update_curriculum_unified_states(self) -> None:
        """Update V18 labels from unified local evidence plus non-cancelling future risk."""

        prior_alpha = float(self.cfg.curriculum_beta_prior_alpha)
        prior_beta = float(self.cfg.curriculum_beta_prior_beta)
        min_window = float(self.cfg.curriculum_min_window_trials)

        local_trials = self._curriculum_window_start_trials
        local_rate = (self._curriculum_window_start_failures + prior_alpha) / (local_trials + prior_alpha + prior_beta)
        local_known = self._curriculum_known_mask()
        local_valid = (local_trials >= min_window) & local_known
        self._shift_curriculum_history(self._curriculum_failure_rate_history, local_valid, local_rate)

        forward_trials = self._curriculum_window_forward_trials
        forward_rate = self._curriculum_window_forward_risk_sum / forward_trials.clamp_min(1.0)
        forward_known = self.curriculum_forward_trials >= float(self.cfg.curriculum_min_known_trials)
        forward_valid = (forward_trials >= min_window) & forward_known
        forward_history = self._curriculum_forward_risk_history
        self._shift_curriculum_history(forward_history, forward_valid, forward_rate)
        if torch.any(forward_valid):
            previous = self._curriculum_forward_risk_ema[forward_valid]
            alpha = 0.5
            updated = torch.where(
                torch.isfinite(previous), previous.lerp(forward_rate[forward_valid], alpha), forward_rate[forward_valid]
            )
            self._curriculum_forward_risk_ema[forward_valid] = updated
            observed = self._curriculum_forward_observed_windows[forward_valid].to(torch.int16)
            self._curriculum_forward_observed_windows[forward_valid] = (observed + 1).clamp_max(255).to(torch.uint8)

        local_history = self._curriculum_failure_rate_history
        local_finite = torch.isfinite(local_history)
        forward_finite = torch.isfinite(forward_history)
        local_low_two = local_finite[1:].all(dim=0) & (
            local_history[1:] <= float(self.cfg.curriculum_mastered_enter_threshold)
        ).all(dim=0)
        forward_low_two = forward_finite[1:].all(dim=0) & (forward_history[1:] <= 0.10).all(dim=0)
        forward_safe = ~forward_known | forward_low_two
        mastered_evidence = local_low_two & forward_safe

        local_progress = local_history[0] - local_history[2]
        forward_progress = forward_history[0] - forward_history[2]
        local_no_progress = local_finite.all(dim=0) & (
            local_progress < float(self.cfg.curriculum_no_progress_threshold)
        )
        forward_no_progress = forward_finite.all(dim=0) & (
            forward_progress < float(self.cfg.curriculum_no_progress_threshold)
        )
        local_strong_improvement = local_finite.all(dim=0) & (
            local_progress >= float(self.cfg.curriculum_improvement_threshold)
        )
        forward_strong_improvement = forward_finite.all(dim=0) & (
            forward_progress >= float(self.cfg.curriculum_improvement_threshold)
        )
        local_stalled = (
            local_finite.all(dim=0)
            & (local_history >= float(self.cfg.curriculum_stalled_enter_threshold)).all(dim=0)
            & local_no_progress
        )
        forward_stalled = (
            forward_finite.all(dim=0)
            & (forward_history >= float(self.cfg.curriculum_forward_risk_bad_threshold)).all(dim=0)
            & forward_no_progress
        )
        local_quarantine = (
            local_finite.all(dim=0)
            & (local_history >= float(self.cfg.curriculum_quarantine_enter_threshold)).all(dim=0)
            & local_no_progress
        )

        current_local = local_history[2]
        current_forward = forward_history[2]
        any_current_valid = local_valid | forward_valid
        mastered_exit = (local_valid & (current_local > float(self.cfg.curriculum_mastered_exit_threshold))) | (
            forward_valid & (current_forward > 0.20)
        )

        previous_states = self._curriculum_states.clone()
        next_states = previous_states.clone()
        unknown = (previous_states == self._CURRICULUM_UNKNOWN) & local_valid
        next_states[unknown & mastered_evidence] = self._CURRICULUM_MASTERED
        next_states[unknown & ~mastered_evidence] = self._CURRICULUM_FRONTIER

        mastered = (previous_states == self._CURRICULUM_MASTERED) & any_current_valid
        next_states[mastered & mastered_exit] = self._CURRICULUM_FRONTIER

        frontier = (previous_states == self._CURRICULUM_FRONTIER) & any_current_valid
        next_states[frontier & mastered_evidence] = self._CURRICULUM_MASTERED
        fresh_stall_evidence = (local_valid & local_stalled) | (forward_valid & forward_stalled)
        next_states[frontier & ~mastered_evidence & fresh_stall_evidence] = self._CURRICULUM_STALLED

        stalled = (previous_states == self._CURRICULUM_STALLED) & any_current_valid
        stalled_to_mastered = stalled & mastered_evidence
        all_active_reasons_improved = (
            ~local_stalled
            | local_strong_improvement
            | (current_local < float(self.cfg.curriculum_stalled_exit_threshold))
        ) & (~forward_stalled | forward_strong_improvement | (current_forward < 0.50))
        stalled_to_frontier = stalled & ~stalled_to_mastered & all_active_reasons_improved
        stalled_to_quarantine = (
            stalled
            & local_valid
            & ~stalled_to_mastered
            & ~stalled_to_frontier
            & (self.curriculum_start_trials >= float(self.cfg.curriculum_min_quarantine_trials))
            & local_quarantine
            & self._curriculum_forward_terminal_hard_mask()
        )
        next_states[stalled_to_mastered] = self._CURRICULUM_MASTERED
        next_states[stalled_to_frontier] = self._CURRICULUM_FRONTIER
        next_states[stalled_to_quarantine] = self._CURRICULUM_QUARANTINE

        entered_quarantine = (next_states == self._CURRICULUM_QUARANTINE) & (
            previous_states != self._CURRICULUM_QUARANTINE
        )
        self._curriculum_state_entry_trials[entered_quarantine] = self.curriculum_start_trials[entered_quarantine]
        quarantine = (previous_states == self._CURRICULUM_QUARANTINE) & local_valid
        probe_trials = self.curriculum_start_trials - self._curriculum_state_entry_trials
        quarantine_exit = (
            quarantine
            & (probe_trials >= float(self.cfg.curriculum_min_exit_probe_trials))
            & (local_strong_improvement | (current_local < float(self.cfg.curriculum_quarantine_exit_threshold)))
        )
        next_states[quarantine_exit] = self._CURRICULUM_FRONTIER
        self._curriculum_states.copy_(next_states)

        self._curriculum_window_start_trials[local_valid] = 0.0
        self._curriculum_window_start_failures[local_valid] = 0.0
        self._curriculum_window_forward_trials[forward_valid] = 0.0
        self._curriculum_window_forward_risk_sum[forward_valid] = 0.0
        if self._curriculum_quality_learning_enabled():
            self._update_curriculum_quality_statistics()
            self._refresh_curriculum_quality_learning_pool()
        self._curriculum_last_state_update_iteration = self._curriculum_iteration

    def _curriculum_quality_known_mask(self) -> torch.Tensor:
        """Return MASTERED bins with enough successful, finite quality evidence."""

        return (
            self._curriculum_eligible_bins
            & (self._curriculum_states == self._CURRICULUM_MASTERED)
            & (self.curriculum_quality_trials >= float(self.cfg.curriculum_quality_min_trials))
            & (self._curriculum_quality_observed_windows >= int(self.cfg.curriculum_quality_min_windows))
            & torch.isfinite(self._curriculum_quality_error_ema)
        )

    def _update_curriculum_quality_statistics(self) -> None:
        """Consume complete quality windows and update a per-bin error EMA."""

        complete = self._curriculum_quality_window_trials >= float(self.cfg.curriculum_quality_min_window_trials)
        if not torch.any(complete):
            return
        window_mean = (
            self._curriculum_quality_window_error_sum[complete] / self._curriculum_quality_window_trials[complete]
        )
        previous = self._curriculum_quality_error_ema[complete]
        alpha = float(self.cfg.curriculum_quality_error_ema_alpha)
        updated = torch.where(torch.isfinite(previous), previous.lerp(window_mean, alpha), window_mean)
        self._curriculum_quality_error_ema[complete] = updated
        observed = self._curriculum_quality_observed_windows[complete].to(dtype=torch.int16)
        self._curriculum_quality_observed_windows[complete] = (observed + 1).clamp_max(255).to(torch.uint8)
        self._curriculum_quality_window_trials[complete] = 0.0
        self._curriculum_quality_window_error_sum[complete] = 0.0

    @staticmethod
    def _select_deterministic_topk_ids(
        candidate_ids: torch.Tensor, candidate_scores: torch.Tensor, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select exactly ``count`` IDs, resolving equal-score ties by global bin ID."""

        if count >= len(candidate_ids):
            selected = candidate_ids
            cutoff = candidate_scores.min() if len(candidate_scores) > 0 else candidate_scores.new_zeros(())
            return selected, cutoff
        top_values = torch.topk(candidate_scores, count, largest=True, sorted=False).values
        cutoff = top_values.min()
        above = candidate_ids[candidate_scores > cutoff]
        tied = candidate_ids[candidate_scores == cutoff]
        remaining = count - len(above)
        if remaining > 0:
            tie_selection = torch.topk(tied, remaining, largest=False, sorted=False).values
            selected = torch.cat((above, tie_selection))
        else:
            selected = above
        return selected, cutoff

    def _refresh_curriculum_quality_learning_pool(self) -> None:
        """Keep every survival frontier bin and fill the pool with high-error MASTERED bins."""

        self._curriculum_quality_filler_mask.zero_()
        if _cfg_curriculum_learnability_enabled(self.cfg):
            # Schema 9 uses quality only to reshape MASTERED base coverage;
            # the legacy fixed-size filler pool must not leak into either the
            # probability recipe or its diagnostics.
            self._curriculum_quality_filler_count = 0
            self._curriculum_quality_error_cutoff = 0.0
            return
        frontier_count = int(
            (self._curriculum_eligible_bins & (self._curriculum_states == self._CURRICULUM_FRONTIER)).sum().item()
        )
        filler_count = max(self._curriculum_learning_pool_target_count - frontier_count, 0)
        if filler_count == 0:
            self._curriculum_quality_filler_count = 0
            self._curriculum_quality_error_cutoff = 0.0
            return
        candidate_mask = self._curriculum_quality_known_mask()
        candidate_ids = torch.where(candidate_mask)[0]
        filler_count = min(filler_count, len(candidate_ids))
        if filler_count == 0:
            self._curriculum_quality_filler_count = 0
            self._curriculum_quality_error_cutoff = 0.0
            return
        candidate_scores = self._curriculum_quality_error_ema[candidate_ids]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # All ranks have identical all-reduced statistics. Let rank zero
            # choose tie membership once so the cached masks cannot diverge.
            if torch.distributed.get_rank() == 0:
                selected_ids, cutoff = self._select_deterministic_topk_ids(
                    candidate_ids, candidate_scores, filler_count
                )
            else:
                selected_ids = torch.empty(filler_count, dtype=torch.long, device=self.device)
                cutoff = torch.zeros((), dtype=torch.float32, device=self.device)
            torch.distributed.broadcast(selected_ids, src=0)
            torch.distributed.broadcast(cutoff, src=0)
        else:
            selected_ids, cutoff = self._select_deterministic_topk_ids(candidate_ids, candidate_scores, filler_count)
        self._curriculum_quality_filler_mask[selected_ids] = True
        self._curriculum_quality_filler_count = filler_count
        self._curriculum_quality_error_cutoff = float(cutoff.item())

    def _flush_curriculum_statistics(self) -> None:
        """SUM raw per-rank deltas, then update identical global sufficient statistics."""

        base_pending_statistics = (
            self._current_curriculum_start_trials,
            self._current_curriculum_start_failures,
            self._current_curriculum_start_censored,
            self._current_curriculum_start_survival_steps,
            self._current_curriculum_start_completion_fraction,
            self._current_curriculum_terminal_visits,
            self._current_curriculum_terminal_failures,
        )
        quality_pending_statistics = (
            (
                self._current_curriculum_quality_error_sum,
                self._current_curriculum_quality_trials,
            )
            if self._curriculum_quality_learning_enabled()
            else ()
        )
        forward_pending_statistics = (
            (
                self._current_curriculum_forward_trials,
                self._current_curriculum_forward_risk_sum,
            )
            if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False))
            else ()
        )
        pending_statistics = (*base_pending_statistics, *forward_pending_statistics, *quality_pending_statistics)
        packed = torch.cat(pending_statistics)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        deltas = packed.chunk(len(pending_statistics))
        base_deltas = deltas[: len(base_pending_statistics)]
        totals = (
            self.curriculum_start_trials,
            self.curriculum_start_failures,
            self.curriculum_start_censored,
            self.curriculum_start_survival_steps,
            self.curriculum_start_completion_fraction,
            self.curriculum_terminal_visits,
            self.curriculum_terminal_failures,
        )
        for total, delta in zip(totals, base_deltas):
            total.add_(delta)
        self._curriculum_window_start_trials.add_(base_deltas[0])
        self._curriculum_window_start_failures.add_(base_deltas[1])
        next_delta = len(base_pending_statistics)
        if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            forward_trials_delta, forward_risk_delta = deltas[next_delta : next_delta + 2]
            self.curriculum_forward_trials.add_(forward_trials_delta)
            self.curriculum_forward_risk_sum.add_(forward_risk_delta)
            self._curriculum_window_forward_trials.add_(forward_trials_delta)
            self._curriculum_window_forward_risk_sum.add_(forward_risk_delta)
            next_delta += 2
        if self._curriculum_quality_learning_enabled():
            quality_error_delta, quality_trials_delta = deltas[next_delta:]
            self.curriculum_quality_trials.add_(quality_trials_delta)
            self._curriculum_quality_window_trials.add_(quality_trials_delta)
            self._curriculum_quality_window_error_sum.add_(quality_error_delta)
        for pending in pending_statistics:
            pending.zero_()
        body_failure_delta = self._current_curriculum_terminal_body_failures.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(body_failure_delta, op=torch.distributed.ReduceOp.SUM)
        self.curriculum_terminal_body_failures.add_(body_failure_delta)
        self._current_curriculum_terminal_body_failures.zero_()

    def _advance_curriculum_sampling(self, *, sync_across_ranks: bool) -> None:
        self._curriculum_iteration += 1
        multi_rank = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if multi_rank and not sync_across_ranks:
            return
        self._flush_curriculum_statistics()
        if self._curriculum_iteration - self._curriculum_last_state_update_iteration >= int(
            self.cfg.curriculum_state_update_interval
        ):
            self._update_curriculum_states()
        known = self._curriculum_known_mask()
        self._curriculum_known_fraction = self._curriculum_mask_fraction(known)
        if self._curriculum_start_eligibility_enabled():
            provisional = self._curriculum_provisional_known_mask()
            self._curriculum_provisional_known_fraction = self._curriculum_mask_fraction(provisional)
            self._curriculum_provisional_motion_fraction = self._curriculum_provisional_motion_fraction_value()
        else:
            # V14/V15 do not use or log provisional coverage. Avoid adding a
            # full-dataset scan and host synchronization to their hot path.
            self._curriculum_provisional_known_fraction = 1.0
            self._curriculum_provisional_motion_fraction = 1.0
        if (
            self._curriculum_blend_start_iteration < 0
            and self._curriculum_iteration >= int(self.cfg.curriculum_shadow_iterations)
            and self._curriculum_known_fraction >= float(self.cfg.curriculum_min_known_fraction)
            and self._curriculum_provisional_known_fraction
            >= float(getattr(self.cfg, "curriculum_min_provisional_known_fraction", 0.0))
            and self._curriculum_provisional_motion_fraction
            >= float(getattr(self.cfg, "curriculum_min_provisional_motion_fraction", 0.0))
        ):
            self._curriculum_blend_start_iteration = self._curriculum_iteration

    def _legacy_sampling_probabilities(self, bin_ids: torch.Tensor) -> torch.Tensor:
        """Build the pre-curriculum adaptive distribution on a bin subset."""

        failure_rate = self._compute_failure_rate()[bin_ids]
        if self.cfg.adaptive_sampling_alpha is not None:
            return self._marmot_probabilities(failure_rate)
        failure_rate = self._clip_failure_rate(failure_rate).double()
        if failure_rate.sum() <= 0.0:
            failure_probabilities = torch.full_like(failure_rate, 1.0 / max(len(failure_rate), 1))
        else:
            failure_probabilities = failure_rate / failure_rate.sum()
        uniform_probabilities = torch.ones_like(failure_probabilities) / max(len(failure_probabilities), 1)
        probabilities = (
            failure_probabilities * (1.0 - self.cfg.uniform_sampling_rate)
            + uniform_probabilities * self.cfg.uniform_sampling_rate
        )
        probabilities *= self.bin_weights[bin_ids]
        probabilities /= probabilities.sum()
        return self._apply_probability_caps(probabilities, bin_ids)

    def _rebuild_global_sampling_distribution(self, *, advance_curriculum_smoothing: bool = True) -> None:
        """Match SONIC's full-dataset distribution used to draw a working set."""

        all_bin_ids = torch.arange(self.bin_count, device=self.device)
        probabilities = self._legacy_sampling_probabilities(all_bin_ids)
        if getattr(self.cfg, "curriculum_sampling_enabled", False):
            probabilities, eligible_component, terminal_component = self._blend_curriculum_distribution_components(
                probabilities, all_bin_ids
            )
            probabilities = self._smooth_curriculum_distribution(
                probabilities,
                advance=advance_curriculum_smoothing,
                eligible_component=eligible_component,
                terminal_component=terminal_component,
            )
        self.adp_sampling_prob = probabilities.float()
        motion_sampling_probabilities = torch.zeros(self.global_num_motions, dtype=torch.float32, device=self.device)
        motion_sampling_probabilities.index_add_(0, self.bin_motion_ids, self.adp_sampling_prob)
        self._motion_sampling_probabilities = motion_sampling_probabilities / motion_sampling_probabilities.sum()

    def _rebuild_sampling_distribution(self) -> None:
        """Match SONIC's adaptive distribution over the active working set."""

        eligibility_enabled = bool(getattr(self.cfg, "curriculum_start_eligibility_enabled", False))
        if (
            getattr(self.cfg, "curriculum_sampling_enabled", False)
            and self._curriculum_fixed_horizon_enabled()
            and self._curriculum_smoothed_probabilities_initialized
            and (
                self._curriculum_blend_factor() > 0.0
                or (self._curriculum_has_shadow_distribution and not self._curriculum_has_shadow_state_recipe)
            )
        ):
            active_global_mass = self.adp_sampling_prob[self._active_bin_ids].double()
            active_normalization = active_global_mass.sum()
            probabilities = active_global_mass / active_normalization
            if eligibility_enabled:
                active_eligible_component = (
                    self._curriculum_smoothed_eligible_probabilities[self._active_bin_ids].double()
                    / active_normalization
                )
                active_terminal_component = (
                    self._curriculum_smoothed_terminal_probabilities[self._active_bin_ids].double()
                    / active_normalization
                )
            if self.cfg.adp_samp_failure_rate_max_over_mean is not None:
                capped_probabilities = self._cap_probabilities_at_uniform_ratio(
                    probabilities, float(self.cfg.adp_samp_failure_rate_max_over_mean)
                )
                if eligibility_enabled:
                    component_scale = capped_probabilities / probabilities.clamp_min(
                        torch.finfo(probabilities.dtype).tiny
                    )
                    active_eligible_component *= component_scale
                    active_terminal_component *= component_scale
                probabilities = capped_probabilities
        else:
            probabilities = self._legacy_sampling_probabilities(self._active_bin_ids)
            if getattr(self.cfg, "curriculum_sampling_enabled", False):
                probabilities = self._blend_curriculum_distribution(probabilities, self._active_bin_ids)
            if eligibility_enabled:
                active_eligible_component = torch.zeros_like(probabilities)
                active_terminal_component = torch.zeros_like(probabilities)
        if not torch.all(torch.isfinite(probabilities)) or torch.any(probabilities < 0.0):
            raise RuntimeError("Adaptive motion sampling produced invalid probabilities")
        if eligibility_enabled:
            components = active_eligible_component + active_terminal_component
            if (
                not torch.all(torch.isfinite(components))
                or torch.any(active_eligible_component < -1.0e-8)
                or torch.any(active_terminal_component < -1.0e-8)
                or torch.any(components > probabilities + 1.0e-7)
            ):
                raise RuntimeError("Curriculum sampling components are inconsistent with total probabilities")
            active_eligible_component.clamp_min_(0.0)
            active_terminal_component.clamp_min_(0.0)
        self.adp_sampling_active_prob = probabilities.float()
        if eligibility_enabled:
            self._curriculum_active_eligible_component_probabilities = active_eligible_component.float()
            self._curriculum_active_terminal_component_probabilities = active_terminal_component.float()

        entropy = -(self.adp_sampling_active_prob * self.adp_sampling_active_prob.clamp_min(1.0e-12).log()).sum()
        entropy_normalized = (
            entropy / math.log(len(self.adp_sampling_active_prob))
            if len(self.adp_sampling_active_prob) > 1
            else torch.ones_like(entropy)
        )
        probability_max = self.adp_sampling_active_prob.max()
        motion_probabilities = torch.bincount(
            self.bin_motion_ids[self._active_bin_ids],
            weights=self.adp_sampling_active_prob,
            minlength=self.global_num_motions,
        )
        motion_probability_max = motion_probabilities.max()
        self._sampling_entropy_normalized = entropy_normalized
        self._sampling_probability_max = probability_max
        self._sampling_motion_probability_max = motion_probability_max
        if "sampling_entropy" in self.metrics:
            self.metrics["sampling_entropy"][:] = entropy_normalized
            self.metrics["sampling_top1_prob"][:] = probability_max
            if "sampling_top1_motion_prob" in self.metrics:
                self.metrics["sampling_top1_motion_prob"][:] = motion_probability_max
        if getattr(self.cfg, "curriculum_sampling_enabled", False):
            self._update_curriculum_metrics(self._active_bin_ids, self.adp_sampling_active_prob)

    def _record_curriculum_start_outcomes(self, env_ids: torch.Tensor, episode_failed: torch.Tensor) -> None:
        """Record one comparable start outcome for each environment."""

        start_bins = self._episode_start_bins[env_ids]
        ones = torch.ones_like(start_bins, dtype=torch.float32)
        self._current_curriculum_start_trials.index_add_(0, start_bins, ones)
        failed_start_bins = start_bins[episode_failed]
        self._current_curriculum_start_failures.index_add_(
            0, failed_start_bins, torch.ones_like(failed_start_bins, dtype=torch.float32)
        )
        survival_steps = self._episode_curriculum_steps[env_ids].float()
        remaining_frames = (self.motion_lengths[env_ids] - self._episode_start_frames[env_ids]).clamp_min(1)
        completion_fraction = (survival_steps / remaining_frames.float()).clamp_(0.0, 1.0)
        self._current_curriculum_start_survival_steps.index_add_(0, start_bins, survival_steps)
        self._current_curriculum_start_completion_fraction.index_add_(0, start_bins, completion_fraction)
        self._episode_start_outcome_recorded[env_ids] = True

    def _record_curriculum_start_censored(self, env_ids: torch.Tensor) -> None:
        """Record starts whose motion/reset ended before the common horizon."""

        start_bins = self._episode_start_bins[env_ids]
        self._current_curriculum_start_censored.index_add_(
            0, start_bins, torch.ones_like(start_bins, dtype=torch.float32)
        )
        survival_steps = self._episode_curriculum_steps[env_ids].float()
        remaining_frames = (self.motion_lengths[env_ids] - self._episode_start_frames[env_ids]).clamp_min(1)
        completion_fraction = (survival_steps / remaining_frames.float()).clamp_(0.0, 1.0)
        self._current_curriculum_start_survival_steps.index_add_(0, start_bins, survival_steps)
        self._current_curriculum_start_completion_fraction.index_add_(0, start_bins, completion_fraction)
        self._episode_start_outcome_recorded[env_ids] = True

    def _update_adaptive_exposure(self) -> None:
        """Accumulate length-normalized exposure for every active motion bin."""

        active_env_ids = torch.where(self._has_sampled)[0]
        if len(active_env_ids) == 0:
            return
        time_steps = torch.minimum(self.time_steps[active_env_ids], self.motion_lengths[active_env_ids] - 1)
        bin_ids = self._bucket_ids(self.motion_ids[active_env_ids], time_steps)
        if getattr(self.cfg, "curriculum_sampling_enabled", False):
            last_visited_bins = self._episode_last_visited_bins[active_env_ids]
            entered_new_bin = bin_ids != last_visited_bins
            entered_bins = bin_ids[entered_new_bin]
            self._current_curriculum_terminal_visits.index_add_(
                0, entered_bins, torch.ones_like(entered_bins, dtype=torch.float32)
            )
            entered_env_ids = active_env_ids[entered_new_bin]
            self._episode_last_visited_bins[entered_env_ids] = entered_bins
            self._episode_curriculum_steps[active_env_ids] += 1
            if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
                self._enter_curriculum_forward_bins(entered_env_ids, entered_bins)
                self._update_curriculum_unified_windows(active_env_ids)
            elif self._curriculum_fixed_horizon_enabled():
                horizon_frames = int(self.cfg.curriculum_start_horizon_frames)
                reached_horizon = (
                    ~self._episode_start_outcome_recorded[active_env_ids]
                    & (self._episode_start_bins[active_env_ids] >= 0)
                    & (self._episode_curriculum_steps[active_env_ids] >= horizon_frames)
                )
                horizon_env_ids = active_env_ids[reached_horizon]
                if self._curriculum_quality_learning_enabled():
                    self._record_curriculum_start_quality(horizon_env_ids)
                self._record_curriculum_start_outcomes(
                    horizon_env_ids,
                    torch.zeros_like(horizon_env_ids, dtype=torch.bool),
                )
        if self.cfg.adaptive_sampling_alpha is not None:
            self._current_adp_samp_num_episodes.index_add_(0, bin_ids, torch.ones_like(bin_ids, dtype=torch.float))
        else:
            bin_lengths = (self.bin_ends[bin_ids] - self.bin_starts[bin_ids]).float()
            self.adp_samp_num_episodes.index_add_(0, bin_ids, bin_lengths.reciprocal())

    def set_fixed_motion_ids(self, motion_ids: Sequence[int] | torch.Tensor) -> None:
        """Pin every environment to one process-local motion for deterministic evaluation.

        The assignment is retained across environment resets.  Training and
        normal playback remain adaptive until this method is called.
        """

        fixed_motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        if fixed_motion_ids.shape != (self.num_envs,):
            raise ValueError(
                f"fixed motion assignment must have shape ({self.num_envs},), got {tuple(fixed_motion_ids.shape)}"
            )
        if torch.any(fixed_motion_ids < 0) or torch.any(fixed_motion_ids >= self.motion.num_motions):
            raise ValueError(f"fixed motion ids must be in [0, {self.motion.num_motions})")
        self._fixed_motion_ids = fixed_motion_ids.clone()

    def clear_fixed_motion_ids(self) -> None:
        """Restore adaptive motion selection after deterministic evaluation."""

        self._fixed_motion_ids = None

    def _finalize_curriculum_unified_episodes(
        self,
        env_ids: torch.Tensor,
        terminal_bins: torch.Tensor,
        episode_failed: torch.Tensor,
        invalid_state_failed: torch.Tensor,
        motion_completed: torch.Tensor,
    ) -> None:
        """Close V18 local and lookahead windows, with failure taking precedence."""

        if len(env_ids) == 0:
            return
        terminal_not_seen = self._episode_last_visited_bins[env_ids] != terminal_bins
        terminal_env_ids = env_ids[terminal_not_seen]
        terminal_new_bins = terminal_bins[terminal_not_seen]
        self._enter_curriculum_forward_bins(terminal_env_ids, terminal_new_bins)
        self._episode_last_visited_bins[terminal_env_ids] = terminal_new_bins

        valid_failure = episode_failed & ~invalid_state_failed
        successful_motion_end = motion_completed & ~valid_failure & ~invalid_state_failed
        interrupted = ~(valid_failure | successful_motion_end)

        self._accumulate_curriculum_terminal_quality(env_ids[successful_motion_end])
        self._record_curriculum_unified_window_outcomes(
            env_ids[valid_failure], failed=True, include_terminal_frame=True
        )
        self._record_curriculum_unified_window_outcomes(
            env_ids[successful_motion_end], failed=False, include_terminal_frame=True
        )
        self._record_curriculum_unified_window_censored(env_ids[interrupted])
        self._record_curriculum_forward_outcomes(env_ids[valid_failure], failed=True)
        self._record_curriculum_forward_outcomes(env_ids[successful_motion_end], failed=False)
        # Manual reset, invalid state, or an unrelated timeout is censored in
        # both channels and must not leak pending history into the next motion.
        self._episode_forward_risk_bins[env_ids[interrupted]] = -1

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        eligibility_enabled = bool(getattr(self.cfg, "curriculum_start_eligibility_enabled", False))
        previously_sampled = self._has_sampled[env_ids]
        # Boolean indexing and empty index_add operations are valid, so avoid a
        # per-reset CUDA scalar synchronization just to special-case first use.
        if len(env_ids) > 0:
            previous_env_ids = env_ids[previously_sampled]
            previous_time_steps = torch.minimum(
                self.time_steps[previous_env_ids], self.motion_lengths[previous_env_ids] - 1
            )
            previous_bins = self._bucket_ids(self.motion_ids[previous_env_ids], previous_time_steps)
            episode_failed = self._env.termination_manager.terminated[previous_env_ids]
            if getattr(self.cfg, "curriculum_sampling_enabled", False):
                # Termination is evaluated before CommandTerm.compute().  If the
                # final frame crossed a bin boundary, count that reached bin here.
                terminal_bin_not_visited = self._episode_last_visited_bins[previous_env_ids] != previous_bins
                newly_visited_terminal_bins = previous_bins[terminal_bin_not_visited]
                self._current_curriculum_terminal_visits.index_add_(
                    0,
                    newly_visited_terminal_bins,
                    torch.ones_like(newly_visited_terminal_bins, dtype=torch.float32),
                )
                start_bins = self._episode_start_bins[previous_env_ids]
                valid_start = start_bins >= 0
                valid_env_ids = previous_env_ids[valid_start]
                valid_episode_failed = episode_failed[valid_start]
                try:
                    invalid_state_failed = self._env.termination_manager.get_term("invalid_robot_state")[
                        previous_env_ids
                    ]
                except (AttributeError, KeyError, ValueError):
                    invalid_state_failed = torch.zeros_like(episode_failed)
                if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
                    try:
                        motion_completed = self._env.termination_manager.get_term("motion_time_out")[previous_env_ids]
                    except (AttributeError, KeyError, ValueError):
                        motion_completed = torch.zeros_like(episode_failed)
                    self._finalize_curriculum_unified_episodes(
                        previous_env_ids,
                        previous_bins,
                        episode_failed,
                        invalid_state_failed,
                        motion_completed,
                    )
                elif self._curriculum_fixed_horizon_enabled():
                    pending_outcome = ~self._episode_start_outcome_recorded[valid_env_ids]
                    within_horizon = self._episode_curriculum_steps[valid_env_ids] <= int(
                        self.cfg.curriculum_start_horizon_frames
                    )
                    label_failure = valid_episode_failed & within_horizon
                    if getattr(self.cfg, "curriculum_exclude_invalid_failures", False):
                        label_failure &= ~invalid_state_failed[valid_start]
                    labeled_env_ids = valid_env_ids[pending_outcome & label_failure]
                    self._record_curriculum_start_outcomes(
                        labeled_env_ids,
                        torch.ones_like(labeled_env_ids, dtype=torch.bool),
                    )
                    censored_env_ids = valid_env_ids[pending_outcome & ~label_failure]
                    self._record_curriculum_start_censored(censored_env_ids)
                else:
                    self._record_curriculum_start_outcomes(valid_env_ids, valid_episode_failed)

                terminal_failure_mask = episode_failed
                if getattr(self.cfg, "curriculum_exclude_invalid_failures", False):
                    terminal_failure_mask = terminal_failure_mask & ~invalid_state_failed
                failed_terminal_bins = previous_bins[terminal_failure_mask]
                self._current_curriculum_terminal_failures.index_add_(
                    0,
                    failed_terminal_bins,
                    torch.ones_like(failed_terminal_bins, dtype=torch.float32),
                )
                try:
                    body_pos_failed = self._env.termination_manager.get_term("body_pos")[previous_env_ids]
                except (AttributeError, KeyError, ValueError):
                    body_pos_failed = torch.zeros_like(episode_failed)
                body_pos_failed &= episode_failed & ~invalid_state_failed
                if hasattr(self, "last_global_body_pos_errors"):
                    failure_errors = torch.nan_to_num(
                        self.last_global_body_pos_errors[previous_env_ids[body_pos_failed]],
                        nan=float("-inf"),
                    )
                    error_body_ids = failure_errors.argmax(dim=1)
                    if self._curriculum_body_id_mapping is None:
                        error_body_names = getattr(self, "last_global_body_pos_error_names", self.cfg.body_names)
                        self._curriculum_body_id_mapping = torch.tensor(
                            [self.cfg.body_names.index(name) for name in error_body_names],
                            dtype=torch.long,
                            device=self.device,
                        )
                    body_ids = self._curriculum_body_id_mapping[error_body_ids]
                    self._current_curriculum_terminal_body_failures.index_add_(
                        0, body_ids, torch.ones_like(body_ids, dtype=torch.float32)
                    )
            failed_bins = previous_bins[episode_failed]
            failure_buffer = (
                self._current_adp_samp_num_failures
                if self.cfg.adaptive_sampling_alpha is not None
                else self.adp_samp_num_failures
            )
            failure_buffer.index_add_(
                0,
                failed_bins,
                torch.full_like(
                    failed_bins,
                    self.cfg.failure_counts_multiplier,
                    dtype=self.adp_samp_num_failures.dtype,
                ),
            )

        if self._fixed_motion_ids is None:
            sampled_active_bin_ids = torch.multinomial(
                self.adp_sampling_active_prob,
                num_samples=len(env_ids),
                replacement=True,
            )
            sampled_bins = self._active_bin_ids[sampled_active_bin_ids]
            sampled_motion_ids = self._active_local_motion_ids[sampled_active_bin_ids]
            if eligibility_enabled:
                sampled_bin_probabilities = self.adp_sampling_active_prob[sampled_active_bin_ids].clamp_min(1.0e-12)
                eligible_probability = (
                    self._curriculum_active_eligible_component_probabilities[sampled_active_bin_ids]
                    / sampled_bin_probabilities
                ).clamp_(0.0, 1.0)
                terminal_probability = (
                    self._curriculum_active_terminal_component_probabilities[sampled_active_bin_ids]
                    / sampled_bin_probabilities
                ).clamp_(0.0, 1.0)
                component_samples = torch.rand(len(env_ids), device=self.device)
                sampled_terminal_component = component_samples < terminal_probability
                sampled_eligible_component = ~sampled_terminal_component & (
                    component_samples < (terminal_probability + eligible_probability).clamp_max(1.0)
                )
        else:
            sampled_motion_ids = self._fixed_motion_ids[env_ids]
            sampled_global_motion_ids = self.motion.global_ids[sampled_motion_ids]
            sampled_motion_bin_counts = self.motion_bin_counts[sampled_global_motion_ids]
            sampled_local_bin_ids = (torch.rand(len(env_ids), device=self.device) * sampled_motion_bin_counts).long()
            sampled_bins = self.motion_bin_offsets[sampled_global_motion_ids] + sampled_local_bin_ids
            if eligibility_enabled:
                sampled_eligible_component = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
                sampled_terminal_component = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        self.motion_ids[env_ids] = sampled_motion_ids
        self.motion_lengths[env_ids] = self.motion.lengths(sampled_motion_ids)
        bin_lengths = self.bin_ends[sampled_bins] - self.bin_starts[sampled_bins]
        playback_start_frame = getattr(self.cfg, "playback_start_frame", None)
        if playback_start_frame is not None:
            start_frame = int(playback_start_frame)
            invalid = start_frame >= self.motion_lengths[env_ids]
            if torch.any(invalid):
                shortest_length = int(self.motion_lengths[env_ids][invalid].min().item())
                raise ValueError(
                    f"Playback start frame {start_frame} is outside the selected motion "
                    f"(length: {shortest_length} frames; maximum start frame: {shortest_length - 1})."
                )
            sampled_time_steps = torch.full((len(env_ids),), start_frame, dtype=torch.long, device=self.device)
        elif self.cfg.start_at_motion_beginning:
            sampled_time_steps = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        else:
            if eligibility_enabled:
                # Source/shadow keeps the historical full-bin sampler. V16
                # endpoint components sample their disjoint intervals exactly.
                sampled_time_steps = (
                    self.bin_starts[sampled_bins] + (torch.rand(len(env_ids), device=self.device) * bin_lengths).long()
                )
                eligible_counts = self._curriculum_eligible_start_counts[sampled_bins]
                terminal_counts = self._curriculum_terminal_start_counts[sampled_bins]
                eligible_envs = sampled_eligible_component
                terminal_envs = sampled_terminal_component
                interval_samples = torch.rand(len(env_ids), device=self.device)
                sampled_time_steps[eligible_envs] = (
                    self.bin_starts[sampled_bins[eligible_envs]]
                    + (interval_samples[eligible_envs] * eligible_counts[eligible_envs]).long()
                )
                sampled_time_steps[terminal_envs] = (
                    self._curriculum_eligible_start_ends[sampled_bins[terminal_envs]]
                    + (interval_samples[terminal_envs] * terminal_counts[terminal_envs]).long()
                )
                if self.cfg.pre_failure_sample_window > 0:
                    shadow_envs = ~(eligible_envs | terminal_envs)
                    pre_failure_offsets = torch.randint(
                        self.cfg.pre_failure_sample_window,
                        (len(env_ids),),
                        device=self.device,
                    )
                    sampled_time_steps[shadow_envs] = (
                        sampled_time_steps[shadow_envs] - pre_failure_offsets[shadow_envs]
                    ).clamp_min(0)
            else:
                sampled_time_steps = (
                    self.bin_starts[sampled_bins] + (torch.rand(len(env_ids), device=self.device) * bin_lengths).long()
                )
                if self.cfg.pre_failure_sample_window > 0:
                    pre_failure_offsets = torch.randint(
                        self.cfg.pre_failure_sample_window,
                        (len(env_ids),),
                        device=self.device,
                    )
                    sampled_time_steps = (sampled_time_steps - pre_failure_offsets).clamp_min(0)
        self.time_steps[env_ids] = sampled_time_steps
        self._has_sampled[env_ids] = True
        if getattr(self.cfg, "curriculum_sampling_enabled", False):
            actual_start_bins = self._bucket_ids(sampled_motion_ids, sampled_time_steps)
            if eligibility_enabled:
                label_enabled = sampled_time_steps < self._curriculum_eligible_start_ends[actual_start_bins]
                self._episode_start_label_enabled[env_ids] = label_enabled
                self._episode_start_bins[env_ids] = torch.where(
                    label_enabled, actual_start_bins, torch.full_like(actual_start_bins, -1)
                )
            else:
                self._episode_start_bins[env_ids] = actual_start_bins
            self._episode_start_frames[env_ids] = sampled_time_steps
            self._episode_last_visited_bins[env_ids] = -1
            self._episode_curriculum_steps[env_ids] = 0
            self._episode_start_outcome_recorded[env_ids] = False
            if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
                self._initialize_curriculum_unified_episode(env_ids)
            elif self._curriculum_quality_learning_enabled():
                self._episode_curriculum_quality_error_sum[env_ids] = 0.0
                self._episode_curriculum_quality_observation_count[env_ids] = 0.0
                self._episode_curriculum_quality_valid[env_ids] = self._episode_start_label_enabled[env_ids]
        # CommandTerm.reset clears metrics for reset environments before
        # resampling. Restore these global distribution diagnostics so high
        # reset rates do not make them appear to collapse toward zero.
        self.metrics["sampling_entropy"][env_ids] = self._sampling_entropy_normalized
        self.metrics["sampling_top1_prob"][env_ids] = self._sampling_probability_max
        if "sampling_top1_motion_prob" in self.metrics:
            self.metrics["sampling_top1_motion_prob"][env_ids] = self._sampling_motion_probability_max
        if getattr(self.cfg, "curriculum_sampling_enabled", False):
            for name, value in self._curriculum_metric_values.items():
                self.metrics[name][env_ids] = value

    def _check_adaptive_distributed_layout(self) -> None:
        """Fail coherently before any variable-size adaptive collective."""

        if self._adaptive_layout_checked:
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            # The environment is constructed before the runner initializes
            # the process group. Preserve the future cross-rank check.
            self._adaptive_layout_checked = self.motion.world_size == 1
            return
        learnability_enabled = _cfg_curriculum_learnability_enabled(self.cfg)
        learnability_recipe = self._curriculum_learnability_recipe() if learnability_enabled else (0.0,) * 15
        layout = torch.tensor(
            [
                self.global_num_motions,
                self.bin_count,
                self._curriculum_checkpoint_schema_version()
                if bool(getattr(self.cfg, "curriculum_sampling_enabled", False))
                else 0,
                int(getattr(self.cfg, "curriculum_forward_risk_horizon_bins", 0))
                if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False))
                else 0,
                int(learnability_enabled),
                *(int(round(value * 1_000_000_000)) for value in learnability_recipe),
                *self.motion.manifest_fingerprint_words,
            ],
            dtype=torch.long,
            device=self.device,
        )
        layout_min = layout.clone()
        layout_max = layout.clone()
        torch.distributed.all_reduce(layout_min, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(layout_max, op=torch.distributed.ReduceOp.MAX)
        if not torch.equal(layout_min, layout_max):
            raise RuntimeError(
                "Distributed workers disagree on the global motion/bin layout, curriculum sampler, "
                "or manifest fingerprint: "
                f"local={layout.tolist()}, min={layout_min.tolist()}, max={layout_max.tolist()}"
            )
        self._adaptive_layout_checked = True

    def sync_and_compute_adaptive_sampling(self, *, sync_across_ranks: bool) -> None:
        """Advance EMA every PPO iteration and periodically average it across ranks."""

        if sync_across_ranks:
            self._check_adaptive_distributed_layout()
        if self.cfg.adaptive_sampling_alpha is not None:
            alpha = float(self.cfg.adaptive_sampling_alpha)
            self.adp_samp_num_episodes.mul_(1.0 - alpha).add_(self._current_adp_samp_num_episodes, alpha=alpha)
            self.adp_samp_num_failures.mul_(1.0 - alpha).add_(self._current_adp_samp_num_failures, alpha=alpha)
            self._current_adp_samp_num_episodes.zero_()
            self._current_adp_samp_num_failures.zero_()
        if sync_across_ranks and torch.distributed.is_available() and torch.distributed.is_initialized():
            statistics = (self.adp_samp_num_episodes, self.adp_samp_num_failures)
            packed = torch.cat(statistics)
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            packed /= torch.distributed.get_world_size()
            episodes, failures = packed.chunk(2)
            statistics[0].copy_(episodes)
            statistics[1].copy_(failures)
        if getattr(self.cfg, "curriculum_sampling_enabled", False):
            self._advance_curriculum_sampling(sync_across_ranks=sync_across_ranks)
        self._rebuild_global_sampling_distribution()
        self._rebuild_sampling_distribution()

    def _checkpoint_curriculum_distribution(
        self, state_dict: dict[str, torch.Tensor], schema_version: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, float, bool]:
        """Reconstruct the sampler active at a compatible older checkpoint."""

        all_bin_ids = torch.arange(self.bin_count, device=self.device)
        legacy_probabilities = self._legacy_sampling_probabilities(all_bin_ids)
        saved_effective_probabilities = state_dict.get("curriculum_smoothed_probabilities")
        if (
            schema_version >= self._CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION
            and isinstance(saved_effective_probabilities, torch.Tensor)
            and saved_effective_probabilities.shape == self._curriculum_smoothed_probabilities.shape
        ):
            probabilities = saved_effective_probabilities.to(device=self.device, dtype=torch.float32)
            if (
                torch.all(torch.isfinite(probabilities))
                and torch.all(probabilities >= 0.0)
                and probabilities.sum() > 0.0
            ):
                return probabilities / probabilities.sum(), None, 1.0, False
        saved_states = state_dict.get("curriculum_states")
        if not isinstance(saved_states, torch.Tensor) or saved_states.shape != self._curriculum_states.shape:
            return legacy_probabilities.float(), None, 1.0, False
        saved_states = saved_states.to(device=self.device, dtype=torch.uint8)
        if torch.any(saved_states >= len(self._CURRICULUM_STATE_NAMES)):
            return legacy_probabilities.float(), None, 1.0, False
        if schema_version == self._CURRICULUM_STATE_SCHEMA_VERSION:
            # V14/schema2 assigned equal probability to every bin.
            source_exponent = 1.0
        elif schema_version == self._CURRICULUM_BIASED_FIXED_HORIZON_SCHEMA_VERSION:
            # The original V15/schema3 sampler tempered aggregate motion mass
            # by square-root length. Reconstruct that historical endpoint even
            # when the current V15 target has returned to equal-bin sampling.
            source_exponent = 0.5
        else:
            source_exponent = float(getattr(self.cfg, "curriculum_motion_length_exponent", 1.0))
        source_curriculum = self._curriculum_state_budget_probabilities(
            all_bin_ids,
            dtype=legacy_probabilities.dtype,
            states=saved_states,
            motion_length_exponent=source_exponent,
            preserve_absent_state_budgets=False,
        )
        saved_iteration = state_dict.get("curriculum_iteration")
        saved_blend_start = state_dict.get("curriculum_blend_start_iteration")
        if (
            not isinstance(saved_iteration, torch.Tensor)
            or not isinstance(saved_blend_start, torch.Tensor)
            or int(saved_blend_start.item()) < 0
        ):
            source_blend = 0.0
        elif int(self.cfg.curriculum_blend_iterations) == 0:
            source_blend = 1.0
        else:
            source_blend = min(
                max(
                    (int(saved_iteration.item()) - int(saved_blend_start.item()))
                    / float(self.cfg.curriculum_blend_iterations),
                    0.0,
                ),
                1.0,
            )
        probabilities = legacy_probabilities.lerp(source_curriculum, source_blend)
        if self.cfg.adp_samp_failure_rate_max_over_mean is not None:
            probabilities = self._cap_probabilities_at_uniform_ratio(
                probabilities, float(self.cfg.adp_samp_failure_rate_max_over_mean)
            )
        probabilities /= probabilities.sum()
        return probabilities.float(), saved_states, source_exponent, source_blend >= 1.0

    def get_adaptive_sampling_state(self) -> dict[str, torch.Tensor]:
        state = {
            "adp_samp_num_episodes": self.adp_samp_num_episodes.detach().cpu(),
            "adp_samp_num_failures": self.adp_samp_num_failures.detach().cpu(),
        }
        if not getattr(self.cfg, "curriculum_sampling_enabled", False):
            return state
        schema_version = self._curriculum_checkpoint_schema_version()
        state.update(
            {
                "curriculum_state_schema_version": torch.tensor(schema_version),
                "curriculum_manifest_fingerprint_words": torch.tensor(
                    self.motion.manifest_fingerprint_words, dtype=torch.long
                ),
                "curriculum_bin_size": torch.tensor(self.cfg.bin_size, dtype=torch.long),
                "curriculum_start_trials": self.curriculum_start_trials.detach().cpu(),
                "curriculum_start_failures": self.curriculum_start_failures.detach().cpu(),
                "curriculum_start_survival_steps": self.curriculum_start_survival_steps.detach().cpu(),
                "curriculum_start_completion_fraction": (self.curriculum_start_completion_fraction.detach().cpu()),
                "curriculum_terminal_visits": self.curriculum_terminal_visits.detach().cpu(),
                "curriculum_terminal_failures": self.curriculum_terminal_failures.detach().cpu(),
                "curriculum_terminal_body_failures": (self.curriculum_terminal_body_failures.detach().cpu()),
                "curriculum_window_start_trials": self._curriculum_window_start_trials.detach().cpu(),
                "curriculum_window_start_failures": self._curriculum_window_start_failures.detach().cpu(),
                "curriculum_failure_rate_history": self._curriculum_failure_rate_history.detach().cpu(),
                "curriculum_states": self._curriculum_states.detach().cpu(),
                "curriculum_state_entry_trials": self._curriculum_state_entry_trials.detach().cpu(),
                "curriculum_iteration": torch.tensor(self._curriculum_iteration, dtype=torch.long),
                "curriculum_last_state_update_iteration": torch.tensor(
                    self._curriculum_last_state_update_iteration, dtype=torch.long
                ),
                "curriculum_blend_start_iteration": torch.tensor(
                    self._curriculum_blend_start_iteration, dtype=torch.long
                ),
            }
        )
        if schema_version >= self._CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION:
            state.update(
                {
                    "curriculum_start_horizon_frames": torch.tensor(
                        int(self.cfg.curriculum_start_horizon_frames), dtype=torch.long
                    ),
                    "curriculum_start_censored": self.curriculum_start_censored.detach().cpu(),
                    "curriculum_shadow_probabilities": self._curriculum_shadow_probabilities.detach().cpu(),
                    "curriculum_smoothed_probabilities": self._curriculum_smoothed_probabilities.detach().cpu(),
                    "curriculum_shadow_states": self._curriculum_shadow_states.detach().cpu(),
                    "curriculum_shadow_motion_length_exponent": torch.tensor(
                        self._curriculum_shadow_motion_length_exponent, dtype=torch.float32
                    ),
                    "curriculum_has_shadow_distribution": torch.tensor(self._curriculum_has_shadow_distribution),
                    "curriculum_has_shadow_state_recipe": torch.tensor(self._curriculum_has_shadow_state_recipe),
                    "curriculum_smoothed_probabilities_initialized": torch.tensor(
                        self._curriculum_smoothed_probabilities_initialized
                    ),
                    "curriculum_last_probability_smoothing_iteration": torch.tensor(
                        self._curriculum_last_probability_smoothing_iteration, dtype=torch.long
                    ),
                }
            )
        if self._curriculum_start_eligibility_enabled():
            state.update(
                {
                    "curriculum_terminal_replay_fraction": torch.tensor(
                        float(self.cfg.curriculum_terminal_replay_fraction), dtype=torch.float32
                    ),
                    "curriculum_smoothed_eligible_probabilities": (
                        self._curriculum_smoothed_eligible_probabilities.detach().cpu()
                    ),
                    "curriculum_smoothed_terminal_probabilities": (
                        self._curriculum_smoothed_terminal_probabilities.detach().cpu()
                    ),
                }
            )
        if self._curriculum_quality_learning_enabled():
            state.update(
                {
                    "curriculum_quality_config": torch.tensor(
                        [
                            float(self.cfg.curriculum_learning_pool_min_fraction),
                            float(self.cfg.curriculum_quality_filler_weight),
                            float(self.cfg.curriculum_quality_min_trials),
                            float(self.cfg.curriculum_quality_min_window_trials),
                            float(self.cfg.curriculum_quality_min_windows),
                            float(self.cfg.curriculum_quality_error_ema_alpha),
                            float(self.cfg.curriculum_quality_body_pos_std),
                        ],
                        dtype=torch.float32,
                    ),
                    "curriculum_quality_trials": self.curriculum_quality_trials.detach().cpu(),
                    "curriculum_quality_window_trials": self._curriculum_quality_window_trials.detach().cpu(),
                    "curriculum_quality_window_error_sum": (self._curriculum_quality_window_error_sum.detach().cpu()),
                    "curriculum_quality_error_ema": self._curriculum_quality_error_ema.detach().cpu(),
                    "curriculum_quality_observed_windows": (self._curriculum_quality_observed_windows.detach().cpu()),
                }
            )
        if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            state.update(
                {
                    "curriculum_unified_config": torch.tensor(
                        [
                            float(self.cfg.curriculum_need_uniform_fraction),
                            float(self.cfg.curriculum_forward_risk_horizon_bins),
                            float(self.cfg.curriculum_forward_risk_decay),
                            float(self.cfg.curriculum_forward_risk_good_threshold),
                            float(self.cfg.curriculum_forward_risk_bad_threshold),
                            float(self.cfg.curriculum_quality_good_threshold),
                            float(self.cfg.curriculum_quality_bad_threshold),
                            float(self.cfg.curriculum_quarantine_need_scale),
                        ],
                        dtype=torch.float32,
                    ),
                    "curriculum_forward_trials": self.curriculum_forward_trials.detach().cpu(),
                    "curriculum_forward_risk_sum": self.curriculum_forward_risk_sum.detach().cpu(),
                    "curriculum_window_forward_trials": (self._curriculum_window_forward_trials.detach().cpu()),
                    "curriculum_window_forward_risk_sum": (self._curriculum_window_forward_risk_sum.detach().cpu()),
                    "curriculum_forward_risk_history": self._curriculum_forward_risk_history.detach().cpu(),
                    "curriculum_forward_risk_ema": self._curriculum_forward_risk_ema.detach().cpu(),
                    "curriculum_forward_observed_windows": (self._curriculum_forward_observed_windows.detach().cpu()),
                }
            )
            if _cfg_curriculum_learnability_enabled(self.cfg):
                state["curriculum_learnability_recipe"] = torch.tensor(
                    self._curriculum_learnability_recipe(),
                    dtype=torch.float32,
                )
        return state

    def _reset_curriculum_sampling_state(self) -> None:
        for tensor in (
            self.curriculum_start_trials,
            self.curriculum_start_failures,
            self.curriculum_start_censored,
            self.curriculum_start_survival_steps,
            self.curriculum_start_completion_fraction,
            self.curriculum_terminal_visits,
            self.curriculum_terminal_failures,
            self.curriculum_terminal_body_failures,
            self._current_curriculum_start_trials,
            self._current_curriculum_start_failures,
            self._current_curriculum_start_censored,
            self._current_curriculum_start_survival_steps,
            self._current_curriculum_start_completion_fraction,
            self._current_curriculum_terminal_visits,
            self._current_curriculum_terminal_failures,
            self._current_curriculum_terminal_body_failures,
            self._curriculum_window_start_trials,
            self._curriculum_window_start_failures,
            self._curriculum_state_entry_trials,
        ):
            tensor.zero_()
        self._curriculum_failure_rate_history.fill_(float("nan"))
        self._curriculum_states.fill_(self._CURRICULUM_UNKNOWN)
        if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
            for tensor in (
                self.curriculum_forward_trials,
                self.curriculum_forward_risk_sum,
                self._current_curriculum_forward_trials,
                self._current_curriculum_forward_risk_sum,
                self._curriculum_window_forward_trials,
                self._curriculum_window_forward_risk_sum,
                self._curriculum_forward_observed_windows,
                self._curriculum_last_survival_need,
                self._curriculum_last_forward_need,
                self._curriculum_last_quality_need,
                self._curriculum_last_uncertainty_need,
                self._curriculum_last_combined_need,
            ):
                tensor.zero_()
            self._curriculum_forward_risk_history.fill_(float("nan"))
            self._curriculum_forward_risk_ema.fill_(float("nan"))
            self._episode_unified_window_bins.fill_(-1)
            self._episode_unified_window_frames.zero_()
            self._episode_unified_window_steps.zero_()
            self._episode_unified_window_targets.zero_()
            self._episode_unified_window_active.zero_()
            self._episode_forward_risk_bins.fill_(-1)
        if self._curriculum_quality_learning_enabled():
            for tensor in (
                self.curriculum_quality_trials,
                self._current_curriculum_quality_trials,
                self._current_curriculum_quality_error_sum,
                self._curriculum_quality_window_trials,
                self._curriculum_quality_window_error_sum,
            ):
                tensor.zero_()
            self._curriculum_quality_error_ema.fill_(float("nan"))
            self._curriculum_quality_observed_windows.zero_()
            self._curriculum_quality_filler_mask.zero_()
            self._episode_curriculum_quality_error_sum.zero_()
            self._episode_curriculum_quality_observation_count.zero_()
            self._episode_curriculum_quality_valid.zero_()
            self._episode_curriculum_quality_last_time_step.fill_(-1)
            self._curriculum_quality_filler_count = 0
            self._curriculum_quality_error_cutoff = 0.0
        self._curriculum_shadow_probabilities.zero_()
        self._curriculum_smoothed_probabilities.zero_()
        if self._curriculum_start_eligibility_enabled():
            self._curriculum_smoothed_eligible_probabilities.zero_()
            self._curriculum_smoothed_terminal_probabilities.zero_()
        self._curriculum_shadow_states.fill_(self._CURRICULUM_UNKNOWN)
        self._episode_start_bins.fill_(-1)
        self._episode_start_frames.zero_()
        self._episode_last_visited_bins.fill_(-1)
        self._episode_curriculum_steps.zero_()
        self._episode_start_outcome_recorded.zero_()
        if self._curriculum_start_eligibility_enabled():
            self._episode_start_label_enabled.zero_()
        self._curriculum_iteration = 0
        self._curriculum_last_state_update_iteration = 0
        self._curriculum_blend_start_iteration = -1
        self._curriculum_known_fraction = 0.0
        self._curriculum_provisional_known_fraction = 0.0
        self._curriculum_provisional_motion_fraction = 0.0
        self._curriculum_effective_state_focus = 0.0
        self._curriculum_shadow_motion_length_exponent = 1.0
        self._curriculum_has_shadow_distribution = False
        self._curriculum_has_shadow_state_recipe = False
        self._curriculum_smoothed_probabilities_initialized = False
        self._curriculum_last_probability_smoothing_iteration = -1
        if _cfg_curriculum_learnability_enabled(self.cfg):
            self._curriculum_last_learnability_base_fraction = 1.0
            self._curriculum_last_learnability_primary_support_fraction = 0.0
            self._curriculum_last_learnability_primary_need_mean = 0.0
            self._curriculum_last_learnability_primary_scale = 0.0
            self._curriculum_last_learnability_target_entropy = 1.0
            self._curriculum_last_learnability_quality_reweighted_fraction = 0.0
            self._curriculum_last_learnability_quality_mastered_mass = 0.0
        self._curriculum_metric_values = {}

    def _curriculum_quality_checkpoint_targets(self) -> dict[str, torch.Tensor]:
        """Return the schema-6 quality tensors that survive a checkpoint."""

        return {
            "curriculum_quality_trials": self.curriculum_quality_trials,
            "curriculum_quality_window_trials": self._curriculum_quality_window_trials,
            "curriculum_quality_window_error_sum": self._curriculum_quality_window_error_sum,
            "curriculum_quality_error_ema": self._curriculum_quality_error_ema,
            "curriculum_quality_observed_windows": self._curriculum_quality_observed_windows,
        }

    def _curriculum_quality_checkpoint_state_valid(
        self,
        state_dict: dict[str, torch.Tensor],
        schema_version: int,
        runtime_schema_version: int,
    ) -> bool:
        """Validate schema-6 quality evidence independently of the base sampler."""

        if runtime_schema_version < self._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION:
            return True
        if schema_version != runtime_schema_version:
            return False
        saved_quality_config = state_dict.get("curriculum_quality_config")
        runtime_quality_config = torch.tensor(
            [
                float(self.cfg.curriculum_learning_pool_min_fraction),
                float(self.cfg.curriculum_quality_filler_weight),
                float(self.cfg.curriculum_quality_min_trials),
                float(self.cfg.curriculum_quality_min_window_trials),
                float(self.cfg.curriculum_quality_min_windows),
                float(self.cfg.curriculum_quality_error_ema_alpha),
                float(self.cfg.curriculum_quality_body_pos_std),
            ],
            dtype=torch.float32,
        )
        targets = self._curriculum_quality_checkpoint_targets()
        if not (
            isinstance(saved_quality_config, torch.Tensor)
            and saved_quality_config.shape == runtime_quality_config.shape
            # The learning-pool size and filler weight are sampler recipe.
            # The remaining fields define how the persisted quality evidence
            # was measured and accumulated, so only those are evidence-critical.
            and torch.allclose(
                saved_quality_config.cpu().float()[2:], runtime_quality_config[2:], rtol=0.0, atol=1.0e-7
            )
            and all(
                isinstance(state_dict.get(name), torch.Tensor) and state_dict[name].shape == target.shape
                for name, target in targets.items()
            )
        ):
            return False
        quality_trials = state_dict["curriculum_quality_trials"].float()
        quality_window_trials = state_dict["curriculum_quality_window_trials"].float()
        quality_window_error_sum = state_dict["curriculum_quality_window_error_sum"].float()
        quality_error_ema = state_dict["curriculum_quality_error_ema"].float()
        quality_observed_windows = state_dict["curriculum_quality_observed_windows"].float()
        finite_quality_ema = quality_error_ema[torch.isfinite(quality_error_ema)]
        return bool(
            (
                torch.all(torch.isfinite(quality_trials))
                & torch.all(torch.isfinite(quality_window_trials))
                & torch.all(torch.isfinite(quality_window_error_sum))
                & torch.all(~torch.isinf(quality_error_ema))
                & torch.all(quality_trials >= 0.0)
                & torch.all(quality_window_trials >= 0.0)
                & torch.all(quality_window_error_sum >= 0.0)
                & torch.all(quality_window_trials <= quality_trials + 1.0e-6)
                & torch.all(quality_window_error_sum <= quality_window_trials + 1.0e-5)
                & torch.all(torch.isfinite(quality_observed_windows))
                & torch.all(quality_observed_windows >= 0.0)
                & torch.all(quality_observed_windows <= 255.0)
                & torch.all(finite_quality_ema >= 0.0)
                & torch.all(finite_quality_ema <= 1.0)
            ).item()
        )

    def _curriculum_quality_checkpoint_recipe_valid(self, state_dict: dict[str, torch.Tensor]) -> bool:
        """Validate the full quality-learning recipe for an exact resume."""

        saved_config = state_dict.get("curriculum_quality_config")
        runtime_config = torch.tensor(
            [
                float(self.cfg.curriculum_learning_pool_min_fraction),
                float(self.cfg.curriculum_quality_filler_weight),
                float(self.cfg.curriculum_quality_min_trials),
                float(self.cfg.curriculum_quality_min_window_trials),
                float(self.cfg.curriculum_quality_min_windows),
                float(self.cfg.curriculum_quality_error_ema_alpha),
                float(self.cfg.curriculum_quality_body_pos_std),
            ],
            dtype=torch.float32,
        )
        return bool(
            isinstance(saved_config, torch.Tensor)
            and saved_config.shape == runtime_config.shape
            and torch.allclose(saved_config.cpu().float(), runtime_config, rtol=0.0, atol=1.0e-7)
        )

    def _restore_curriculum_quality_checkpoint_state(
        self,
        state_dict: dict[str, torch.Tensor],
        schema_version: int,
        runtime_schema_version: int,
        quality_state_valid: bool,
    ) -> bool:
        """Restore exact V17 evidence, or leave reset/cold evidence untouched."""

        if (
            runtime_schema_version < self._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION
            or schema_version != runtime_schema_version
            or not quality_state_valid
        ):
            return False
        for name, target in self._curriculum_quality_checkpoint_targets().items():
            target.copy_(state_dict[name].to(device=self.device, dtype=target.dtype))
        return True

    def _finish_curriculum_quality_checkpoint_restore(self, schema_version: int, restore_quality_state: bool) -> None:
        """Rebuild the derived pool and report exact-base/cold-quality migration."""

        if not self._curriculum_quality_learning_enabled():
            return
        self._refresh_curriculum_quality_learning_pool()
        if schema_version == self._CURRICULUM_ELIGIBLE_COVERAGE_SCHEMA_VERSION:
            print("[INFO] Restored V16 curriculum exactly; V17 quality learning starts from cold statistics.")
        elif not restore_quality_state:
            print(
                "[WARN] V17 quality checkpoint state is incompatible; "
                "the base curriculum was restored exactly and quality learning restarts cold."
            )

    def _curriculum_unified_checkpoint_targets(self) -> dict[str, torch.Tensor]:
        """Return schema-7 forward-risk tensors that survive a checkpoint."""

        return {
            "curriculum_forward_trials": self.curriculum_forward_trials,
            "curriculum_forward_risk_sum": self.curriculum_forward_risk_sum,
            "curriculum_window_forward_trials": self._curriculum_window_forward_trials,
            "curriculum_window_forward_risk_sum": self._curriculum_window_forward_risk_sum,
            "curriculum_forward_risk_history": self._curriculum_forward_risk_history,
            "curriculum_forward_risk_ema": self._curriculum_forward_risk_ema,
            "curriculum_forward_observed_windows": self._curriculum_forward_observed_windows,
        }

    def _curriculum_learnability_recipe_valid(self, state_dict: dict[str, torch.Tensor]) -> bool:
        """Validate the schema-9 recipe for an exact resume."""

        saved_config = state_dict.get("curriculum_learnability_recipe")
        runtime_config = torch.tensor(self._curriculum_learnability_recipe(), dtype=torch.float32)
        return bool(
            isinstance(saved_config, torch.Tensor)
            and saved_config.shape == runtime_config.shape
            and torch.allclose(saved_config.cpu().float(), runtime_config, rtol=0.0, atol=1.0e-7)
        )

    def _curriculum_unified_checkpoint_state_valid(self, state_dict: dict[str, torch.Tensor]) -> bool:
        """Validate evidence-critical V18 config and forward-risk evidence."""

        saved_config = state_dict.get("curriculum_unified_config")
        runtime_config = torch.tensor(
            [
                float(self.cfg.curriculum_need_uniform_fraction),
                float(self.cfg.curriculum_forward_risk_horizon_bins),
                float(self.cfg.curriculum_forward_risk_decay),
                float(self.cfg.curriculum_forward_risk_good_threshold),
                float(self.cfg.curriculum_forward_risk_bad_threshold),
                float(self.cfg.curriculum_quality_good_threshold),
                float(self.cfg.curriculum_quality_bad_threshold),
                float(self.cfg.curriculum_quarantine_need_scale),
            ],
            dtype=torch.float32,
        )
        targets = self._curriculum_unified_checkpoint_targets()
        if not (
            isinstance(saved_config, torch.Tensor)
            and saved_config.shape == runtime_config.shape
            # Forward horizon and decay define the risk observations. Uniform
            # mass, decision thresholds, and quarantine scaling are recipe and
            # may change without invalidating the accumulated raw evidence.
            and torch.allclose(saved_config.cpu().float()[1:3], runtime_config[1:3], rtol=0.0, atol=1.0e-7)
            and all(
                isinstance(state_dict.get(name), torch.Tensor) and state_dict[name].shape == target.shape
                for name, target in targets.items()
            )
        ):
            return False
        trials = state_dict["curriculum_forward_trials"].float()
        risk_sum = state_dict["curriculum_forward_risk_sum"].float()
        window_trials = state_dict["curriculum_window_forward_trials"].float()
        window_risk_sum = state_dict["curriculum_window_forward_risk_sum"].float()
        history = state_dict["curriculum_forward_risk_history"].float()
        risk_ema = state_dict["curriculum_forward_risk_ema"].float()
        observed = state_dict["curriculum_forward_observed_windows"].float()
        finite_history = history[torch.isfinite(history)]
        finite_ema = risk_ema[torch.isfinite(risk_ema)]
        return bool(
            (
                torch.all(torch.isfinite(trials))
                & torch.all(torch.isfinite(risk_sum))
                & torch.all(torch.isfinite(window_trials))
                & torch.all(torch.isfinite(window_risk_sum))
                & torch.all(~torch.isinf(history))
                & torch.all(~torch.isinf(risk_ema))
                & torch.all(trials >= 0.0)
                & torch.all(risk_sum >= 0.0)
                & torch.all(risk_sum <= trials + 1.0e-5)
                & torch.all(window_trials >= 0.0)
                & torch.all(window_trials <= trials + 1.0e-5)
                & torch.all(window_risk_sum >= 0.0)
                & torch.all(window_risk_sum <= window_trials + 1.0e-5)
                & torch.all(finite_history >= 0.0)
                & torch.all(finite_history <= 1.0)
                & torch.all(finite_ema >= 0.0)
                & torch.all(finite_ema <= 1.0)
                & torch.all(torch.isfinite(observed))
                & torch.all(observed >= 0.0)
                & torch.all(observed <= 255.0)
            ).item()
        )

    def _curriculum_unified_checkpoint_recipe_valid(self, state_dict: dict[str, torch.Tensor]) -> bool:
        """Validate the full unified state/sampler recipe for an exact resume."""

        saved_config = state_dict.get("curriculum_unified_config")
        runtime_config = torch.tensor(
            [
                float(self.cfg.curriculum_need_uniform_fraction),
                float(self.cfg.curriculum_forward_risk_horizon_bins),
                float(self.cfg.curriculum_forward_risk_decay),
                float(self.cfg.curriculum_forward_risk_good_threshold),
                float(self.cfg.curriculum_forward_risk_bad_threshold),
                float(self.cfg.curriculum_quality_good_threshold),
                float(self.cfg.curriculum_quality_bad_threshold),
                float(self.cfg.curriculum_quarantine_need_scale),
            ],
            dtype=torch.float32,
        )
        return bool(
            isinstance(saved_config, torch.Tensor)
            and saved_config.shape == runtime_config.shape
            and torch.allclose(saved_config.cpu().float(), runtime_config, rtol=0.0, atol=1.0e-7)
        )

    def _load_curriculum_unified_checkpoint(
        self,
        state_dict: dict[str, torch.Tensor],
        schema_version: int,
        *,
        layout_matches: bool,
    ) -> bool:
        """Restore unified schemas or migrate an older recipe behind its frozen sampler."""

        base_targets = {
            "curriculum_start_trials": self.curriculum_start_trials,
            "curriculum_start_failures": self.curriculum_start_failures,
            "curriculum_start_censored": self.curriculum_start_censored,
            "curriculum_start_survival_steps": self.curriculum_start_survival_steps,
            "curriculum_start_completion_fraction": self.curriculum_start_completion_fraction,
            "curriculum_terminal_visits": self.curriculum_terminal_visits,
            "curriculum_terminal_failures": self.curriculum_terminal_failures,
            "curriculum_terminal_body_failures": self.curriculum_terminal_body_failures,
            "curriculum_window_start_trials": self._curriculum_window_start_trials,
            "curriculum_window_start_failures": self._curriculum_window_start_failures,
            "curriculum_failure_rate_history": self._curriculum_failure_rate_history,
            "curriculum_states": self._curriculum_states,
            "curriculum_state_entry_trials": self._curriculum_state_entry_trials,
        }
        fixed_targets = {
            "curriculum_shadow_probabilities": self._curriculum_shadow_probabilities,
            "curriculum_smoothed_probabilities": self._curriculum_smoothed_probabilities,
            "curriculum_shadow_states": self._curriculum_shadow_states,
        }
        saved_horizon = state_dict.get("curriculum_start_horizon_frames")
        horizon_matches = isinstance(saved_horizon, torch.Tensor) and int(saved_horizon.item()) == int(
            self.cfg.curriculum_start_horizon_frames
        )
        base_matches = (
            layout_matches
            and horizon_matches
            and all(
                isinstance(state_dict.get(name), torch.Tensor) and state_dict[name].shape == target.shape
                for name, target in base_targets.items()
            )
        )
        saved_states = state_dict.get("curriculum_states")
        states_valid = isinstance(saved_states, torch.Tensor) and not torch.any(
            saved_states >= len(self._CURRICULUM_STATE_NAMES)
        )

        runtime_schema_version = self._curriculum_checkpoint_schema_version()
        learnability_upgrade = (
            schema_version
            in (
                self._CURRICULUM_UNIFIED_WINDOW_SCHEMA_VERSION,
                MotionCommand._CURRICULUM_STAGED_SAMPLING_SCHEMA_VERSION,
            )
            and runtime_schema_version == MotionCommand._CURRICULUM_LEARNABILITY_SCHEMA_VERSION
        )
        same_unified_schema = schema_version == runtime_schema_version and schema_version in (
            self._CURRICULUM_UNIFIED_WINDOW_SCHEMA_VERSION,
            MotionCommand._CURRICULUM_LEARNABILITY_SCHEMA_VERSION,
        )
        if learnability_upgrade or same_unified_schema:
            scalar_names = (
                "curriculum_iteration",
                "curriculum_last_state_update_iteration",
                "curriculum_blend_start_iteration",
                "curriculum_shadow_motion_length_exponent",
                "curriculum_has_shadow_distribution",
                "curriculum_has_shadow_state_recipe",
                "curriculum_smoothed_probabilities_initialized",
                "curriculum_last_probability_smoothing_iteration",
            )
            fixed_matches = all(
                isinstance(state_dict.get(name), torch.Tensor) and state_dict[name].shape == target.shape
                for name, target in fixed_targets.items()
            ) and all(isinstance(state_dict.get(name), torch.Tensor) for name in scalar_names)
            quality_valid = self._curriculum_quality_checkpoint_state_valid(state_dict, schema_version, schema_version)
            quality_recipe_valid = MotionCommand._curriculum_quality_checkpoint_recipe_valid(self, state_dict)
            unified_evidence_valid = self._curriculum_unified_checkpoint_state_valid(state_dict)
            unified_recipe_valid = MotionCommand._curriculum_unified_checkpoint_recipe_valid(self, state_dict)
            if schema_version == MotionCommand._CURRICULUM_LEARNABILITY_SCHEMA_VERSION:
                recipe_valid = MotionCommand._curriculum_learnability_recipe_valid(self, state_dict)
            else:
                recipe_valid = quality_recipe_valid and unified_recipe_valid
            restart_transition = learnability_upgrade or (same_unified_schema and not recipe_valid)
            saved_sampler = state_dict.get("curriculum_smoothed_probabilities")
            transition_shadow_valid = not restart_transition or bool(
                isinstance(saved_sampler, torch.Tensor)
                and torch.all(torch.isfinite(saved_sampler))
                and torch.all(saved_sampler >= 0.0)
                and saved_sampler.sum() > 0.0
            )
            evidence_compatible = (
                base_matches and states_valid and fixed_matches and quality_valid and unified_evidence_valid
            )
            if evidence_compatible and transition_shadow_valid:
                self._reset_curriculum_sampling_state()
                for name, target in (*base_targets.items(), *fixed_targets.items()):
                    target.copy_(state_dict[name].to(device=self.device, dtype=target.dtype))
                for name, target in self._curriculum_unified_checkpoint_targets().items():
                    target.copy_(state_dict[name].to(device=self.device, dtype=target.dtype))
                self._restore_curriculum_quality_checkpoint_state(
                    state_dict,
                    schema_version,
                    schema_version,
                    quality_valid,
                )
                if restart_transition:
                    saved_sampler = state_dict["curriculum_smoothed_probabilities"].to(
                        device=self.device, dtype=torch.float32
                    )
                    self._curriculum_shadow_probabilities.copy_(saved_sampler)
                    self._curriculum_smoothed_probabilities.copy_(saved_sampler)
                    self._curriculum_shadow_motion_length_exponent = 1.0
                    self._curriculum_has_shadow_distribution = True
                    self._curriculum_has_shadow_state_recipe = False
                    self._curriculum_smoothed_probabilities_initialized = True
                    self._curriculum_iteration = 0
                    self._curriculum_last_state_update_iteration = 0
                    self._curriculum_blend_start_iteration = -1
                    self._curriculum_last_probability_smoothing_iteration = 0
                else:
                    self._curriculum_iteration = int(state_dict["curriculum_iteration"].item())
                    self._curriculum_last_state_update_iteration = int(
                        state_dict["curriculum_last_state_update_iteration"].item()
                    )
                    self._curriculum_blend_start_iteration = int(state_dict["curriculum_blend_start_iteration"].item())
                    self._curriculum_shadow_motion_length_exponent = float(
                        state_dict["curriculum_shadow_motion_length_exponent"].item()
                    )
                    self._curriculum_has_shadow_distribution = bool(
                        state_dict["curriculum_has_shadow_distribution"].item()
                    )
                    self._curriculum_has_shadow_state_recipe = bool(
                        state_dict["curriculum_has_shadow_state_recipe"].item()
                    )
                    self._curriculum_smoothed_probabilities_initialized = bool(
                        state_dict["curriculum_smoothed_probabilities_initialized"].item()
                    )
                    self._curriculum_last_probability_smoothing_iteration = int(
                        state_dict["curriculum_last_probability_smoothing_iteration"].item()
                    )
                self._curriculum_known_fraction = self._curriculum_mask_fraction(self._curriculum_known_mask())
                self._curriculum_provisional_known_fraction = 1.0
                self._curriculum_provisional_motion_fraction = 1.0
                self._refresh_curriculum_quality_learning_pool()
                if learnability_upgrade:
                    print(
                        "[INFO] Migrated unified curriculum evidence into learnability schema 9; "
                        "the saved sampler remains frozen through the new shadow/blend transition."
                    )
                elif not recipe_valid:
                    print(
                        "[INFO] Curriculum sampling recipe changed; accumulated evidence was preserved "
                        "behind the saved sampler for a fresh shadow/blend transition."
                    )
                return True
            print("[WARN] Unified curriculum checkpoint is incompatible; restarting statistics in shadow mode.")
            self._reset_curriculum_sampling_state()
            return False

        if schema_version == self._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION and base_matches and states_valid:
            quality_valid = self._curriculum_quality_checkpoint_state_valid(
                state_dict,
                self._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION,
                self._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION,
            )
            shadow_probabilities, _, _, _ = self._checkpoint_curriculum_distribution(state_dict, schema_version)
            self._reset_curriculum_sampling_state()
            for name, target in base_targets.items():
                target.copy_(state_dict[name].to(device=self.device, dtype=target.dtype))
            if quality_valid:
                self._restore_curriculum_quality_checkpoint_state(
                    state_dict,
                    self._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION,
                    self._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION,
                    True,
                )
            self._curriculum_shadow_probabilities.copy_(shadow_probabilities)
            self._curriculum_smoothed_probabilities.copy_(shadow_probabilities)
            self._curriculum_has_shadow_distribution = True
            self._curriculum_has_shadow_state_recipe = False
            self._curriculum_smoothed_probabilities_initialized = True
            self._curriculum_last_probability_smoothing_iteration = 0
            self._curriculum_iteration = 0
            self._curriculum_last_state_update_iteration = 0
            self._curriculum_blend_start_iteration = -1
            self._curriculum_known_fraction = self._curriculum_mask_fraction(self._curriculum_known_mask())
            self._curriculum_provisional_known_fraction = 1.0
            self._curriculum_provisional_motion_fraction = 1.0
            self._refresh_curriculum_quality_learning_pool()
            print(
                "[INFO] Migrated V17 start and quality evidence into unified V18 windows; "
                "forward risk starts cold behind the frozen V17 sampler."
            )
            return True

        print("[WARN] Curriculum checkpoint cannot migrate to the unified runtime; starting statistics cold.")
        self._reset_curriculum_sampling_state()
        return False

    def load_adaptive_sampling_state(self, state_dict: dict[str, torch.Tensor]) -> bool:
        episodes = state_dict.get("adp_samp_num_episodes")
        failures = state_dict.get("adp_samp_num_failures")
        if episodes is None or failures is None:
            return False
        if episodes.shape != self.adp_samp_num_episodes.shape or failures.shape != self.adp_samp_num_failures.shape:
            print("[WARN] Adaptive sampling state does not match the current dataset; skipping restore.")
            return False
        self.adp_samp_num_episodes.copy_(episodes.to(self.device))
        self.adp_samp_num_failures.copy_(failures.to(self.device))
        self._current_adp_samp_num_episodes.zero_()
        self._current_adp_samp_num_failures.zero_()
        # The environment is freshly reset after a restore.  Do not let the
        # construction-time episodes contaminate either legacy or v14 stats.
        self._has_sampled.zero_()
        exact_probability_state_restored = False
        if getattr(self.cfg, "curriculum_sampling_enabled", False):
            schema_value = state_dict.get("curriculum_state_schema_version")
            schema_version = int(schema_value.item()) if isinstance(schema_value, torch.Tensor) else 1
            runtime_schema_version = self._curriculum_checkpoint_schema_version()
            fingerprint = state_dict.get("curriculum_manifest_fingerprint_words")
            saved_bin_size = state_dict.get("curriculum_bin_size")
            expected_fingerprint = torch.tensor(self.motion.manifest_fingerprint_words, dtype=torch.long)
            layout_matches = (
                isinstance(fingerprint, torch.Tensor)
                and torch.equal(fingerprint.cpu().long(), expected_fingerprint)
                and isinstance(saved_bin_size, torch.Tensor)
                and int(saved_bin_size.item()) == int(self.cfg.bin_size)
            )
            if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
                exact_probability_state_restored = self._load_curriculum_unified_checkpoint(
                    state_dict,
                    schema_version,
                    layout_matches=layout_matches,
                )
                self._rebuild_global_sampling_distribution(
                    advance_curriculum_smoothing=not exact_probability_state_restored
                )
                self._rebuild_sampling_distribution()
                return True
            expected_shapes = {
                "curriculum_start_trials": self.curriculum_start_trials.shape,
                "curriculum_start_failures": self.curriculum_start_failures.shape,
                "curriculum_start_survival_steps": self.curriculum_start_survival_steps.shape,
                "curriculum_start_completion_fraction": self.curriculum_start_completion_fraction.shape,
                "curriculum_terminal_visits": self.curriculum_terminal_visits.shape,
                "curriculum_terminal_failures": self.curriculum_terminal_failures.shape,
                "curriculum_terminal_body_failures": self.curriculum_terminal_body_failures.shape,
                "curriculum_window_start_trials": self._curriculum_window_start_trials.shape,
                "curriculum_window_start_failures": self._curriculum_window_start_failures.shape,
                "curriculum_failure_rate_history": self._curriculum_failure_rate_history.shape,
                "curriculum_states": self._curriculum_states.shape,
                "curriculum_state_entry_trials": self._curriculum_state_entry_trials.shape,
            }
            restored_tensors = {
                "curriculum_start_trials": self.curriculum_start_trials,
                "curriculum_start_failures": self.curriculum_start_failures,
                "curriculum_start_survival_steps": self.curriculum_start_survival_steps,
                "curriculum_start_completion_fraction": self.curriculum_start_completion_fraction,
                "curriculum_terminal_visits": self.curriculum_terminal_visits,
                "curriculum_terminal_failures": self.curriculum_terminal_failures,
                "curriculum_terminal_body_failures": self.curriculum_terminal_body_failures,
                "curriculum_window_start_trials": self._curriculum_window_start_trials,
                "curriculum_window_start_failures": self._curriculum_window_start_failures,
                "curriculum_failure_rate_history": self._curriculum_failure_rate_history,
                "curriculum_states": self._curriculum_states,
                "curriculum_state_entry_trials": self._curriculum_state_entry_trials,
            }
            horizon_matches = True
            if runtime_schema_version >= self._CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION:
                saved_horizon = state_dict.get("curriculum_start_horizon_frames")
                horizon_matches = isinstance(saved_horizon, torch.Tensor) and int(saved_horizon.item()) == int(
                    self.cfg.curriculum_start_horizon_frames
                )
                expected_shapes.update(
                    {
                        "curriculum_start_censored": self.curriculum_start_censored.shape,
                        "curriculum_shadow_probabilities": self._curriculum_shadow_probabilities.shape,
                        "curriculum_smoothed_probabilities": self._curriculum_smoothed_probabilities.shape,
                        "curriculum_shadow_states": self._curriculum_shadow_states.shape,
                    }
                )
                restored_tensors.update(
                    {
                        "curriculum_start_censored": self.curriculum_start_censored,
                        "curriculum_shadow_probabilities": self._curriculum_shadow_probabilities,
                        "curriculum_smoothed_probabilities": self._curriculum_smoothed_probabilities,
                        "curriculum_shadow_states": self._curriculum_shadow_states,
                    }
                )
            if runtime_schema_version >= self._CURRICULUM_ELIGIBLE_COVERAGE_SCHEMA_VERSION:
                expected_shapes.update(
                    {
                        "curriculum_smoothed_eligible_probabilities": (
                            self._curriculum_smoothed_eligible_probabilities.shape
                        ),
                        "curriculum_smoothed_terminal_probabilities": (
                            self._curriculum_smoothed_terminal_probabilities.shape
                        ),
                    }
                )
                restored_tensors.update(
                    {
                        "curriculum_smoothed_eligible_probabilities": (
                            self._curriculum_smoothed_eligible_probabilities
                        ),
                        "curriculum_smoothed_terminal_probabilities": (
                            self._curriculum_smoothed_terminal_probabilities
                        ),
                    }
                )
            counter_names = [
                "curriculum_iteration",
                "curriculum_last_state_update_iteration",
                "curriculum_blend_start_iteration",
            ]
            if runtime_schema_version >= self._CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION:
                counter_names.extend(
                    [
                        "curriculum_shadow_motion_length_exponent",
                        "curriculum_has_shadow_distribution",
                        "curriculum_has_shadow_state_recipe",
                        "curriculum_smoothed_probabilities_initialized",
                        "curriculum_last_probability_smoothing_iteration",
                    ]
                )
            counters_present = all(isinstance(state_dict.get(name), torch.Tensor) for name in counter_names)
            replay_fraction_matches = True
            if runtime_schema_version >= self._CURRICULUM_ELIGIBLE_COVERAGE_SCHEMA_VERSION:
                saved_replay_fraction = state_dict.get("curriculum_terminal_replay_fraction")
                replay_fraction_matches = isinstance(saved_replay_fraction, torch.Tensor) and math.isclose(
                    float(saved_replay_fraction.item()),
                    float(self.cfg.curriculum_terminal_replay_fraction),
                    rel_tol=0.0,
                    abs_tol=1.0e-7,
                )
            component_state_valid = True
            if runtime_schema_version >= self._CURRICULUM_ELIGIBLE_COVERAGE_SCHEMA_VERSION:
                probability_names = (
                    "curriculum_smoothed_probabilities",
                    "curriculum_smoothed_eligible_probabilities",
                    "curriculum_smoothed_terminal_probabilities",
                )
                component_state_valid = all(
                    isinstance(state_dict.get(name), torch.Tensor)
                    and state_dict[name].shape == restored_tensors[name].shape
                    for name in probability_names
                )
                initialized_value = state_dict.get("curriculum_smoothed_probabilities_initialized")
                component_state_valid &= isinstance(initialized_value, torch.Tensor)
                if component_state_valid:
                    total_probabilities, eligible_probabilities, terminal_probabilities = (
                        state_dict[name].to(device=self.device, dtype=torch.float32) for name in probability_names
                    )
                    tolerance = 1.0e-7
                    finite_and_nonnegative = bool(
                        (
                            torch.all(torch.isfinite(total_probabilities))
                            & torch.all(torch.isfinite(eligible_probabilities))
                            & torch.all(torch.isfinite(terminal_probabilities))
                            & torch.all(total_probabilities >= 0.0)
                            & torch.all(eligible_probabilities >= 0.0)
                            & torch.all(terminal_probabilities >= 0.0)
                        ).item()
                    )
                    initialized = bool(initialized_value.item())
                    if initialized:
                        component_state_valid = (
                            finite_and_nonnegative
                            and math.isclose(
                                float(total_probabilities.sum().item()), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-6
                            )
                            and bool(
                                torch.all(
                                    eligible_probabilities + terminal_probabilities <= total_probabilities + tolerance
                                ).item()
                            )
                            and bool(
                                torch.all(
                                    eligible_probabilities[~self._curriculum_eligible_bins].abs() <= tolerance
                                ).item()
                            )
                            and bool(
                                torch.all(
                                    terminal_probabilities[self._curriculum_terminal_start_counts == 0].abs()
                                    <= tolerance
                                ).item()
                            )
                        )
                    else:
                        component_state_valid = finite_and_nonnegative and bool(
                            (
                                torch.all(total_probabilities.abs() <= tolerance)
                                & torch.all(eligible_probabilities.abs() <= tolerance)
                                & torch.all(terminal_probabilities.abs() <= tolerance)
                            ).item()
                        )
            quality_state_valid = self._curriculum_quality_checkpoint_state_valid(
                state_dict, schema_version, runtime_schema_version
            )
            schema_is_exact_or_v16_upgrade = schema_version == runtime_schema_version or (
                runtime_schema_version == self._CURRICULUM_QUALITY_LEARNING_SCHEMA_VERSION
                and schema_version == self._CURRICULUM_ELIGIBLE_COVERAGE_SCHEMA_VERSION
            )
            exact_state_matches = (
                schema_is_exact_or_v16_upgrade
                and layout_matches
                and horizon_matches
                and replay_fraction_matches
                and component_state_valid
                and counters_present
                and all(
                    isinstance(state_dict.get(name), torch.Tensor) and state_dict[name].shape == shape
                    for name, shape in expected_shapes.items()
                )
            )
            if exact_state_matches:
                self._reset_curriculum_sampling_state()
                for name, target in restored_tensors.items():
                    target.copy_(state_dict[name].to(device=self.device, dtype=target.dtype))
                restore_quality_state = self._restore_curriculum_quality_checkpoint_state(
                    state_dict,
                    schema_version,
                    runtime_schema_version,
                    quality_state_valid,
                )
                if torch.any(self._curriculum_states >= len(self._CURRICULUM_STATE_NAMES)):
                    print("[WARN] Curriculum checkpoint contains invalid states; starting it in shadow mode.")
                    self._reset_curriculum_sampling_state()
                else:
                    self._curriculum_iteration = int(state_dict["curriculum_iteration"].item())
                    self._curriculum_last_state_update_iteration = int(
                        state_dict["curriculum_last_state_update_iteration"].item()
                    )
                    self._curriculum_blend_start_iteration = int(state_dict["curriculum_blend_start_iteration"].item())
                    if runtime_schema_version >= self._CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION:
                        self._curriculum_shadow_motion_length_exponent = float(
                            state_dict["curriculum_shadow_motion_length_exponent"].item()
                        )
                        self._curriculum_has_shadow_distribution = bool(
                            state_dict["curriculum_has_shadow_distribution"].item()
                        )
                        self._curriculum_has_shadow_state_recipe = bool(
                            state_dict["curriculum_has_shadow_state_recipe"].item()
                        )
                        self._curriculum_smoothed_probabilities_initialized = bool(
                            state_dict["curriculum_smoothed_probabilities_initialized"].item()
                        )
                        self._curriculum_last_probability_smoothing_iteration = int(
                            state_dict["curriculum_last_probability_smoothing_iteration"].item()
                        )
                        exact_probability_state_restored = True
                    known = self._curriculum_known_mask()
                    self._curriculum_known_fraction = self._curriculum_mask_fraction(known)
                    provisional = self._curriculum_provisional_known_mask()
                    self._curriculum_provisional_known_fraction = self._curriculum_mask_fraction(provisional)
                    self._curriculum_provisional_motion_fraction = self._curriculum_provisional_motion_fraction_value()
                    self._finish_curriculum_quality_checkpoint_restore(schema_version, restore_quality_state)
            elif (
                runtime_schema_version >= self._CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION
                and schema_version
                in (
                    self._CURRICULUM_STATE_SCHEMA_VERSION,
                    self._CURRICULUM_BIASED_FIXED_HORIZON_SCHEMA_VERSION,
                    self._CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION,
                    self._CURRICULUM_ELIGIBLE_COVERAGE_SCHEMA_VERSION,
                    runtime_schema_version,
                )
                and layout_matches
            ):
                terminal_names = (
                    "curriculum_terminal_visits",
                    "curriculum_terminal_failures",
                    "curriculum_terminal_body_failures",
                )
                terminal_state_matches = all(
                    isinstance(state_dict.get(name), torch.Tensor)
                    and state_dict[name].shape == restored_tensors[name].shape
                    for name in terminal_names
                )
                shadow_probabilities, shadow_states, shadow_exponent, shadow_uses_state_recipe = (
                    self._checkpoint_curriculum_distribution(state_dict, schema_version)
                )
                saved_horizon = state_dict.get("curriculum_start_horizon_frames")
                can_preserve_fixed_horizon_start = (
                    runtime_schema_version == self._CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION
                    and schema_version == self._CURRICULUM_BIASED_FIXED_HORIZON_SCHEMA_VERSION
                    and isinstance(saved_horizon, torch.Tensor)
                    and int(saved_horizon.item()) == int(self.cfg.curriculum_start_horizon_frames)
                )
                fixed_horizon_sufficient_names = (
                    "curriculum_start_trials",
                    "curriculum_start_failures",
                    "curriculum_start_censored",
                    "curriculum_start_survival_steps",
                    "curriculum_start_completion_fraction",
                )
                fixed_horizon_state_matches = can_preserve_fixed_horizon_start and all(
                    isinstance(state_dict.get(name), torch.Tensor)
                    and state_dict[name].shape == restored_tensors[name].shape
                    for name in fixed_horizon_sufficient_names
                )
                can_preserve_fully_eligible_start = (
                    runtime_schema_version == self._CURRICULUM_ELIGIBLE_COVERAGE_SCHEMA_VERSION
                    and schema_version == self._CURRICULUM_FIXED_HORIZON_SCHEMA_VERSION
                    and horizon_matches
                )
                fully_eligible_state_names = (
                    *fixed_horizon_sufficient_names,
                    "curriculum_window_start_trials",
                    "curriculum_window_start_failures",
                    "curriculum_failure_rate_history",
                    "curriculum_states",
                    "curriculum_state_entry_trials",
                )
                fully_eligible_state_matches = can_preserve_fully_eligible_start and all(
                    isinstance(state_dict.get(name), torch.Tensor)
                    and state_dict[name].shape == restored_tensors[name].shape
                    for name in fully_eligible_state_names
                )
                if fully_eligible_state_matches:
                    saved_states = state_dict["curriculum_states"]
                    fully_eligible_state_matches = not torch.any(saved_states >= len(self._CURRICULUM_STATE_NAMES))
                self._reset_curriculum_sampling_state()
                if terminal_state_matches:
                    for name in terminal_names:
                        target = restored_tensors[name]
                        target.copy_(state_dict[name].to(device=self.device, dtype=target.dtype))
                    if fixed_horizon_state_matches:
                        for name in fixed_horizon_sufficient_names:
                            target = restored_tensors[name]
                            target.copy_(state_dict[name].to(device=self.device, dtype=target.dtype))
                    elif fully_eligible_state_matches:
                        fully_eligible = self._curriculum_eligible_bins & (self._curriculum_terminal_start_counts == 0)
                        for name in fully_eligible_state_names:
                            target = restored_tensors[name]
                            source = state_dict[name].to(device=self.device, dtype=target.dtype)
                            if target.ndim == 1:
                                target[fully_eligible] = source[fully_eligible]
                            else:
                                target[:, fully_eligible] = source[:, fully_eligible]
                    self._curriculum_shadow_probabilities.copy_(shadow_probabilities)
                    self._curriculum_smoothed_probabilities.copy_(shadow_probabilities)
                    self._curriculum_has_shadow_distribution = True
                    self._curriculum_smoothed_probabilities_initialized = True
                    self._curriculum_last_probability_smoothing_iteration = 0
                    if shadow_states is not None and shadow_uses_state_recipe:
                        self._curriculum_shadow_states.copy_(shadow_states)
                        self._curriculum_shadow_motion_length_exponent = shadow_exponent
                        self._curriculum_has_shadow_state_recipe = True
                    known = self._curriculum_known_mask()
                    self._curriculum_known_fraction = self._curriculum_mask_fraction(known)
                    provisional = self._curriculum_provisional_known_mask()
                    self._curriculum_provisional_known_fraction = self._curriculum_mask_fraction(provisional)
                    self._curriculum_provisional_motion_fraction = self._curriculum_provisional_motion_fraction_value()
                    if fixed_horizon_state_matches:
                        print(
                            "[INFO] Migrated fixed-horizon sufficient statistics; "
                            "the corrected classifier restarts from the checkpoint sampler in shadow."
                        )
                    elif fully_eligible_state_matches:
                        print(
                            "[INFO] Migrated fully eligible fixed-horizon states; "
                            "mixed/tail start state restarts from the checkpoint sampler in shadow."
                        )
                    else:
                        print(
                            "[INFO] Migrated terminal curriculum statistics; "
                            "fixed-horizon start state begins from the checkpoint sampler in shadow."
                        )
                else:
                    print("[WARN] Curriculum terminal state is incompatible; starting it in shadow mode.")
            else:
                if schema_version > runtime_schema_version:
                    print("[WARN] Curriculum checkpoint schema is newer than this runtime; starting it in shadow mode.")
                elif schema_version == runtime_schema_version:
                    print("[WARN] Curriculum state does not match the current dataset; starting it in shadow mode.")
                self._reset_curriculum_sampling_state()
        self._rebuild_global_sampling_distribution(advance_curriculum_smoothing=not exact_probability_state_restored)
        self._rebuild_sampling_distribution()
        return True

    def resample_motion_working_set(self) -> bool:
        if self.all_motions_loaded:
            return False
        self._rebuild_global_sampling_distribution()
        selected_global_ids = torch.multinomial(
            self._motion_sampling_probabilities,
            num_samples=self.max_num_load_motions,
            replacement=self.cfg.working_set_replacement,
        )
        self._has_sampled.zero_()
        self._fixed_motion_ids = None
        self.motion_ids.zero_()
        self.motion_lengths.zero_()
        if getattr(self.cfg, "curriculum_sampling_enabled", False):
            self._episode_start_bins.fill_(-1)
            self._episode_start_frames.zero_()
            self._episode_last_visited_bins.fill_(-1)
            self._episode_curriculum_steps.zero_()
            self._episode_start_outcome_recorded.zero_()
            if self._curriculum_start_eligibility_enabled():
                self._episode_start_label_enabled.zero_()
            if bool(getattr(self.cfg, "curriculum_unified_window_enabled", False)):
                self._episode_unified_window_bins.fill_(-1)
                self._episode_unified_window_frames.zero_()
                self._episode_unified_window_steps.zero_()
                self._episode_unified_window_targets.zero_()
                self._episode_unified_window_active.zero_()
                self._episode_forward_risk_bins.fill_(-1)
            if self._curriculum_quality_learning_enabled():
                self._episode_curriculum_quality_error_sum.zero_()
                self._episode_curriculum_quality_observation_count.zero_()
                self._episode_curriculum_quality_valid.zero_()
                self._episode_curriculum_quality_last_time_step.fill_(-1)
        del self.motion
        gc.collect()
        self.motion = self._load_motion_collection(selected_global_ids)
        self._rebuild_active_motion_mapping()
        self._rebuild_sampling_distribution()
        self._window_time_steps = None
        self._window_motion_ids = None
        self._window_body_pos = None
        self._window_body_quat = None
        return True

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._adaptive_sampling(env_ids)
        self._refresh_motion_cache(env_ids)

        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat([root_pos, root_ori], dim=-1),
            env_ids=env_ids,
        )
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=torch.cat([root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self._update_adaptive_exposure()
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion_lengths)[0]
        if self.cfg.resample_at_motion_end:
            self._resample_command(env_ids)
        if len(env_ids) > 0:
            active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            active_mask[env_ids] = False
            active_env_ids = torch.where(active_mask)[0]
        else:
            active_env_ids = torch.arange(self.num_envs, device=self.device)
        self._refresh_motion_cache(active_env_ids)

        anchor_pos_w = self.anchor_pos_w[:, None, :]
        anchor_quat_w = self.anchor_quat_w[:, None, :]
        robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :]
        robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :]

        delta_pos_w = robot_anchor_pos_w.expand(-1, len(self.cfg.body_names), -1).clone()
        delta_pos_w[..., 2] = anchor_pos_w[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w))).expand(
            -1, len(self.cfg.body_names), -1
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    motion_data_device: str = "auto"
    # SONIC loads all motions when they fit; otherwise each rank independently
    # draws a replacement-sampled working set of min(num_envs, 1024) motions.
    max_num_load_motions: int | None = None
    # Sampling with replacement matches SONIC. Disable it to keep every
    # process-local working set unique while allowing overlap across ranks.
    working_set_replacement: bool = True
    # NPZ files are decoded concurrently but consumed in manifest order.
    motion_load_workers: int = 4
    # Upper bound for one process-local construction/runtime chunk.
    motion_chunk_frames: int = 262_144
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # When disabled, the environment is expected to terminate at the final
    # motion frame instead of silently switching to another motion.
    resample_at_motion_end: bool = True
    # Playback/debug option. Training keeps random motion-time initialization
    # unless this is explicitly enabled by a caller such as play.py.
    start_at_motion_beginning: bool = False
    # Explicit playback-only start frame. None preserves the configured sampler;
    # an integer overrides start_at_motion_beginning and random/bin sampling.
    playback_start_frame: int | None = None

    bin_size: int = 50
    sequence_length_agnostic: bool = True
    init_num_failures: float = 1.0
    uniform_sampling_rate: float = 0.1
    pre_failure_sample_window: int = 200
    use_failure_rate_decay: bool = False
    decay_gamma: float = 0.8
    # None retains the legacy cumulative SONIC sampler. A value enables
    # MarmotLab-style EMA statistics and forward-kernel smoothing.
    adaptive_sampling_alpha: float | None = None
    adaptive_kernel_size: int = 1
    adaptive_kernel_lambda: float = 0.8
    adp_samp_failure_rate_max_over_mean: float | None = 200.0
    failure_counts_multiplier: float = 1.0
    max_prob_per_bin: str | float | None = None
    max_prob_per_motion: str | float | None = None

    # Opt-in v14 curriculum.  Keeping this disabled preserves the v13 sampler,
    # memory footprint, probability distribution, and checkpoint schema.
    curriculum_sampling_enabled: bool = False
    curriculum_shadow_iterations: int = 100
    curriculum_blend_iterations: int = 50
    curriculum_state_update_interval: int = 50
    curriculum_min_window_trials: int = 8
    curriculum_min_known_trials: int = 32
    curriculum_min_quarantine_trials: int = 128
    curriculum_min_exit_probe_trials: int = 32
    curriculum_min_known_fraction: float = 0.10
    curriculum_beta_prior_alpha: float = 1.0
    curriculum_beta_prior_beta: float = 1.0
    curriculum_mastered_enter_threshold: float = 0.10
    curriculum_mastered_exit_threshold: float = 0.15
    curriculum_stalled_enter_threshold: float = 0.80
    curriculum_stalled_exit_threshold: float = 0.75
    curriculum_quarantine_enter_threshold: float = 0.90
    curriculum_quarantine_exit_threshold: float = 0.80
    curriculum_no_progress_threshold: float = 0.02
    curriculum_improvement_threshold: float = 0.05
    # None preserves v14's full-suffix start label.  A positive value records
    # start failure only within a common number of motion frames.
    curriculum_start_horizon_frames: int | None = None
    curriculum_exclude_invalid_failures: bool = False
    # V16 splits each bin into starts that can expose the full horizon and a
    # censored suffix. Only comparable starts feed the five-state classifier.
    curriculum_start_eligibility_enabled: bool = False
    # Reset-start probability reserved for the censored suffix. This retains
    # motion-ending training without letting one-sided labels occupy unknown.
    curriculum_terminal_replay_fraction: float = 0.0
    # Coverage-aware focus keeps comparable starts broadly sampled first, then
    # transitions to the configured five-state budgets as coverage matures.
    curriculum_coverage_aware_enabled: bool = False
    curriculum_provisional_known_trials: int = 8
    curriculum_min_provisional_known_fraction: float = 0.0
    curriculum_min_provisional_motion_fraction: float = 0.0
    curriculum_provisional_motion_bin_fraction: float = 0.0
    # Lower bound relative to broad eligible-frame sampling for all bins that
    # have not yet accumulated curriculum_min_known_trials.
    curriculum_unconfirmed_min_coverage_ratio: float = 0.0
    curriculum_state_focus_start_fraction: float = 0.0
    curriculum_state_focus_end_fraction: float = 1.0
    # V17 keeps every survival frontier bin and fills a minimum-sized learning
    # pool with successful MASTERED bins that still have high global body-pos
    # tracking error. Disabled by default to preserve the V16 schema/path.
    curriculum_quality_learning_enabled: bool = False
    curriculum_learning_pool_min_fraction: float = 0.15
    curriculum_quality_filler_weight: float = 0.5
    curriculum_quality_min_trials: int = 32
    curriculum_quality_min_window_trials: int = 8
    curriculum_quality_min_windows: int = 2
    curriculum_quality_error_ema_alpha: float = 0.5
    curriculum_quality_body_pos_std: float = 0.3
    # V18 assigns the same local survival and quality meaning to the initial
    # rollout window and every naturally reached non-overlapping window.
    curriculum_unified_window_enabled: bool = False
    # V18 samples 20% from all valid frames and 80% from the maximum normalized
    # survival, forward-risk, quality, or uncertainty need of each bin.
    curriculum_need_uniform_fraction: float = 0.20
    # V18 uses "need". V19 uses "learnability": difficult bins receive extra
    # priority only while uncertainty or recent improvement says they can still
    # learn; tracking quality only reshapes MASTERED's existing base coverage.
    curriculum_sampling_strategy: str = "need"
    curriculum_forward_risk_horizon_bins: int = 5
    curriculum_forward_risk_decay: float = 0.80
    curriculum_forward_risk_good_threshold: float = 0.05
    curriculum_forward_risk_bad_threshold: float = 0.60
    curriculum_quality_good_threshold: float = 0.08
    curriculum_quality_bad_threshold: float = 0.30
    curriculum_quarantine_need_scale: float = 0.25
    # Alpha=1 is v14's equal-bin distribution.  Alpha=0 chooses motions
    # equally within each state; intermediate values temper length exposure.
    curriculum_motion_length_exponent: float = 1.0
    curriculum_min_terminal_visits: int = 0
    curriculum_quarantine_terminal_hazard_threshold: float = 0.0
    curriculum_terminal_hazard_window_bins: int = 1
    # Preserve each configured state budget even when that state is empty;
    # missing mass becomes uniform replay instead of amplifying other states.
    curriculum_preserve_absent_state_budgets: bool = False
    # EMA applied once per PPO iteration to the final full-dataset sampler.
    # One preserves v14's immediate probability updates.
    curriculum_probability_smoothing_alpha: float = 1.0
    curriculum_detailed_metrics: bool = True
    curriculum_state_sampling_weights: dict[str, float] = {
        "unknown": 0.20,
        "mastered": 0.10,
        "frontier": 0.55,
        "stalled": 0.10,
        "quarantine": 0.05,
    }

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
