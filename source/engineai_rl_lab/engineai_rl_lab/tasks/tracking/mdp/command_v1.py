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
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from ..robots.actuator import DelayedImplicitActuator
from .motion_data import MotionCollection, resolve_motion_catalog, select_motion_shard

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["AdaptiveSamplerV1", "AdaptiveSamplerV1Cfg", "MotionCommandV1", "MotionCommandV1Cfg"]

_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


@configclass
class AdaptiveSamplerV1Cfg:
    """Small curriculum with explicit coverage, difficulty, and learnability.

    The adaptive part moves continuously between two distributions:

    - ``hard`` follows current failure/tracking difficulty.
    - ``learnable`` follows difficulty times uncertainty or recent improvement.

    A single global gate gives most adaptive mass to ``learnable`` while useful
    learning opportunities remain.  As they disappear, the same mass returns to
    ``hard``.  ``coverage_fraction`` is always reserved for dataset coverage.
    """

    bin_size: int = 50
    equal_motion_weighting: bool = False
    coverage_fraction: float = 0.20
    pre_failure_window: int = 0
    tracking_error_scale: float = 0.30
    global_tracking_error_weight: float = 1.0
    relative_position_error_weight: float = 0.0
    relative_position_error_scale: float = 0.30
    relative_orientation_error_weight: float = 0.0
    relative_orientation_error_scale: float = 0.40
    local_position_error_weight: float = 0.0
    local_position_error_scale: float = 0.10
    local_position_body_names: list[str] = []
    local_position_body_offsets: list[list[float]] = []
    deduplicate_failure_events: bool = False
    fast_half_life: float = 32.0
    slow_half_life: float = 64.0
    uncertainty_exposure: float = 32.0
    learnability_full_scale: float = 0.05
    max_learnable_fraction: float = 0.75
    # Strict upper bound on final per-bin probability relative to frame coverage.
    probability_cap_ratio: float = 200.0


class AdaptiveSamplerV1:
    """Per-bin adaptive sampler with three persistent statistics.

    Persistent state is only total exposure plus a fast and slow EMA of one
    bounded difficulty signal. The signal is the maximum of the local failure
    rate and reward-aligned body-position tracking error.
    """

    STATE_VERSION = 2

    def __init__(
        self,
        motion_lengths: torch.Tensor,
        cfg: AdaptiveSamplerV1Cfg,
        *,
        device: str | torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self._validate_cfg()

        motion_lengths = torch.as_tensor(motion_lengths, dtype=torch.long, device=self.device)
        if motion_lengths.ndim != 1 or len(motion_lengths) == 0 or torch.any(motion_lengths <= 0):
            raise ValueError("motion_lengths must be a non-empty vector of positive frame counts")
        self.num_motions = len(motion_lengths)
        self.motion_bin_counts = torch.div(
            motion_lengths + cfg.bin_size - 1,
            cfg.bin_size,
            rounding_mode="floor",
        )
        self.bin_count = int(self.motion_bin_counts.sum().item())
        self.motion_bin_offsets = torch.zeros(self.num_motions, dtype=torch.long, device=self.device)
        if self.num_motions > 1:
            self.motion_bin_offsets[1:] = torch.cumsum(self.motion_bin_counts[:-1], dim=0)
        self.bin_motion_ids = torch.repeat_interleave(
            torch.arange(self.num_motions, device=self.device), self.motion_bin_counts
        )
        bin_ids = torch.arange(self.bin_count, device=self.device)
        local_bin_ids = bin_ids - self.motion_bin_offsets[self.bin_motion_ids]
        self.bin_starts = local_bin_ids * cfg.bin_size
        self.bin_ends = torch.minimum(
            self.bin_starts + cfg.bin_size,
            motion_lengths[self.bin_motion_ids],
        )
        self.bin_lengths = self.bin_ends - self.bin_starts
        self.base_weights = self.bin_lengths.double()
        if cfg.equal_motion_weighting:
            self.base_weights /= motion_lengths[self.bin_motion_ids]

        # A negative sentinel distinguishes unknown bins without another tensor.
        # Distribution construction treats them as difficult and learnable; the
        # first evidence initializes both EMAs to the same measured difficulty.
        self.total_exposure = torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)
        self.difficulty_fast = torch.full((self.bin_count,), -1.0, dtype=torch.float32, device=self.device)
        self.difficulty_slow = torch.full_like(self.difficulty_fast, -1.0)

        # Raw sufficient statistics are summed across ranks before an EMA update.
        self._exposure_delta = torch.zeros_like(self.total_exposure)
        self._failure_delta = torch.zeros_like(self.total_exposure)
        self._tracking_error_sum_delta = torch.zeros_like(self.total_exposure)
        self._tracking_error_count_delta = torch.zeros_like(self.total_exposure)

        self.last_gate = 0.0
        self.last_remaining_learnability = 0.0
        self.last_difficulty_mean = 1.0
        self.last_coverage_mass = cfg.coverage_fraction
        self.last_hard_mass = 1.0 - cfg.coverage_fraction
        self.last_learnable_mass = 0.0

    def _validate_cfg(self) -> None:
        cfg = self.cfg
        if cfg.bin_size <= 0:
            raise ValueError("adaptive_sampling.bin_size must be positive")
        if not 0.0 < cfg.coverage_fraction < 1.0:
            raise ValueError("adaptive_sampling.coverage_fraction must be in (0, 1)")
        if cfg.pre_failure_window < 0:
            raise ValueError("adaptive_sampling.pre_failure_window must be non-negative")
        if cfg.tracking_error_scale <= 0.0:
            raise ValueError("adaptive_sampling.tracking_error_scale must be positive")
        error_weights = (
            cfg.global_tracking_error_weight,
            cfg.relative_position_error_weight,
            cfg.relative_orientation_error_weight,
            cfg.local_position_error_weight,
        )
        if any(weight < 0.0 for weight in error_weights) or sum(error_weights) <= 0.0:
            raise ValueError("adaptive sampling tracking-error weights must be non-negative with a positive sum")
        if cfg.relative_position_error_scale <= 0.0:
            raise ValueError("adaptive_sampling.relative_position_error_scale must be positive")
        if cfg.relative_orientation_error_scale <= 0.0:
            raise ValueError("adaptive_sampling.relative_orientation_error_scale must be positive")
        if cfg.local_position_error_scale <= 0.0:
            raise ValueError("adaptive_sampling.local_position_error_scale must be positive")
        if cfg.fast_half_life <= 0.0 or cfg.slow_half_life <= cfg.fast_half_life:
            raise ValueError("adaptive sampling requires 0 < fast_half_life < slow_half_life")
        if cfg.uncertainty_exposure <= 0.0:
            raise ValueError("adaptive_sampling.uncertainty_exposure must be positive")
        if not 0.0 < cfg.learnability_full_scale <= 1.0:
            raise ValueError("adaptive_sampling.learnability_full_scale must be in (0, 1]")
        if not 0.0 <= cfg.max_learnable_fraction <= 1.0:
            raise ValueError("adaptive_sampling.max_learnable_fraction must be in [0, 1]")
        if cfg.probability_cap_ratio < 1.0:
            raise ValueError("adaptive_sampling.probability_cap_ratio must be at least one")

    def recipe_tensor(self) -> torch.Tensor:
        """Return metadata that changes the meaning of persisted statistics."""
        recipe = [
            self.cfg.bin_size,
            int(self.cfg.equal_motion_weighting),
            self.cfg.coverage_fraction,
            self.cfg.pre_failure_window,
            self.cfg.tracking_error_scale,
            self.cfg.fast_half_life,
            self.cfg.slow_half_life,
            self.cfg.uncertainty_exposure,
            self.cfg.learnability_full_scale,
            self.cfg.max_learnable_fraction,
            self.cfg.probability_cap_ratio,
        ]
        extended_recipe = (
            self.cfg.global_tracking_error_weight != 1.0
            or self.cfg.relative_position_error_weight != 0.0
            or self.cfg.relative_orientation_error_weight != 0.0
            or self.cfg.local_position_error_weight != 0.0
            or self.cfg.deduplicate_failure_events
        )
        if extended_recipe:
            recipe.extend(
                [
                    self.cfg.global_tracking_error_weight,
                    self.cfg.relative_position_error_weight,
                    self.cfg.relative_position_error_scale,
                    self.cfg.relative_orientation_error_weight,
                    self.cfg.relative_orientation_error_scale,
                    self.cfg.local_position_error_weight,
                    self.cfg.local_position_error_scale,
                    int(self.cfg.deduplicate_failure_events),
                ]
            )
        return torch.tensor(recipe, dtype=torch.float64)

    def bucket_ids(self, global_motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        return self.motion_bin_offsets[global_motion_ids] + torch.div(
            time_steps, self.cfg.bin_size, rounding_mode="floor"
        )

    def active_mapping(self, global_motion_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a process-local motion list to flat global bins without per-motion loops."""

        global_motion_ids = torch.as_tensor(global_motion_ids, dtype=torch.long, device=self.device)
        if global_motion_ids.ndim != 1 or len(global_motion_ids) == 0:
            raise ValueError("global_motion_ids must be a non-empty vector")
        counts = self.motion_bin_counts[global_motion_ids]
        total_bins = int(counts.sum().item())
        local_motion_ids = torch.repeat_interleave(
            torch.arange(len(global_motion_ids), device=self.device),
            counts,
            output_size=total_bins,
        )
        flat_motion_starts = torch.cumsum(counts, dim=0) - counts
        global_bin_starts = self.motion_bin_offsets[global_motion_ids]
        flat_bin_offsets = torch.repeat_interleave(
            global_bin_starts - flat_motion_starts,
            counts,
            output_size=total_bins,
        )
        active_bin_ids = torch.arange(total_bins, device=self.device) + flat_bin_offsets
        return active_bin_ids, local_motion_ids

    def record_exposure(self, bin_ids: torch.Tensor) -> None:
        if len(bin_ids) == 0:
            return
        exposure = self.bin_lengths[bin_ids].float().reciprocal()
        self._exposure_delta.index_add_(0, bin_ids, exposure)

    def record_failures(self, bin_ids: torch.Tensor, event_ids: torch.Tensor | None = None) -> None:
        if len(bin_ids) == 0:
            return
        if event_ids is not None:
            if event_ids.shape != bin_ids.shape:
                raise ValueError("event_ids must have the same shape as bin_ids")
            if self.cfg.deduplicate_failure_events:
                event_bin_pairs = event_ids.to(torch.long) * self.bin_count + bin_ids.to(torch.long)
                bin_ids = torch.remainder(torch.unique(event_bin_pairs), self.bin_count)
        self._failure_delta.index_add_(0, bin_ids, torch.ones_like(bin_ids, dtype=torch.float32))

    @staticmethod
    def _tracking_error_score(errors: torch.Tensor, scale: float) -> torch.Tensor:
        return 1.0 - torch.exp(-errors.float().square().mean(dim=-1) / float(scale) ** 2)

    def record_tracking_error(
        self,
        bin_ids: torch.Tensor,
        body_position_errors: torch.Tensor,
        *,
        relative_position_errors: torch.Tensor | None = None,
        relative_orientation_errors: torch.Tensor | None = None,
        local_position_errors: torch.Tensor | None = None,
    ) -> None:
        """Record a configurable, reward-aligned tracking difficulty in ``[0, 1]``."""

        if len(bin_ids) == 0:
            return
        if body_position_errors.ndim < 2 or body_position_errors.shape[0] != len(bin_ids):
            raise ValueError("body_position_errors must have shape (num_samples, num_bodies)")
        weighted_scores = self._tracking_error_score(body_position_errors, self.cfg.tracking_error_scale) * float(
            self.cfg.global_tracking_error_weight
        )
        total_weight = float(self.cfg.global_tracking_error_weight)
        optional_errors = (
            (
                relative_position_errors,
                self.cfg.relative_position_error_weight,
                self.cfg.relative_position_error_scale,
                "relative_position_errors",
            ),
            (
                relative_orientation_errors,
                self.cfg.relative_orientation_error_weight,
                self.cfg.relative_orientation_error_scale,
                "relative_orientation_errors",
            ),
            (
                local_position_errors,
                self.cfg.local_position_error_weight,
                self.cfg.local_position_error_scale,
                "local_position_errors",
            ),
        )
        for errors, weight, scale, name in optional_errors:
            if weight <= 0.0:
                continue
            if errors is None or errors.ndim < 2 or errors.shape[0] != len(bin_ids):
                raise ValueError(f"{name} must have shape (num_samples, num_terms) when its weight is positive")
            weighted_scores += self._tracking_error_score(errors, scale) * float(weight)
            total_weight += float(weight)
        score = weighted_scores / total_weight
        finite = torch.isfinite(score)
        valid_bins = bin_ids[finite]
        self._tracking_error_sum_delta.index_add_(0, valid_bins, score[finite])
        self._tracking_error_count_delta.index_add_(0, valid_bins, torch.ones_like(valid_bins, dtype=torch.float32))

    def _sync_deltas(self) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        packed = torch.cat(
            (
                self._exposure_delta,
                self._failure_delta,
                self._tracking_error_sum_delta,
                self._tracking_error_count_delta,
            )
        )
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        exposure, failure, error_sum, error_count = packed.chunk(4)
        self._exposure_delta.copy_(exposure)
        self._failure_delta.copy_(failure)
        self._tracking_error_sum_delta.copy_(error_sum)
        self._tracking_error_count_delta.copy_(error_count)

    @staticmethod
    def _ema_alpha(exposure: torch.Tensor, half_life: float) -> torch.Tensor:
        return 1.0 - torch.exp(-math.log(2.0) * exposure / half_life)

    def update(self, *, sync_across_ranks: bool) -> bool:
        """Consume sufficient statistics and update both difficulty time scales.

        Distributed workers intentionally wait for a requested synchronization,
        then all consume the same globally summed deltas. Single-process runs
        update on every call.
        """

        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed and torch.distributed.get_world_size() > 1 and not sync_across_ranks:
            return False
        if distributed and sync_across_ranks:
            self._sync_deltas()

        observed = (self._exposure_delta > 0.0) | (self._failure_delta > 0.0) | (self._tracking_error_count_delta > 0.0)
        if torch.any(observed):
            exposure = self._exposure_delta[observed]
            minimum_exposure = self.bin_lengths[observed].float().reciprocal()
            failure = self._failure_delta[observed]
            failure_rate = (failure / exposure.clamp_min(minimum_exposure)).clamp_(0.0, 1.0)
            error_count = self._tracking_error_count_delta[observed]
            tracking_error = torch.where(
                error_count > 0.0,
                self._tracking_error_sum_delta[observed] / error_count.clamp_min(1.0),
                torch.zeros_like(error_count),
            ).clamp_(0.0, 1.0)
            batch_difficulty = torch.maximum(failure_rate, tracking_error)
            error_exposure = error_count / self.bin_lengths[observed].float()
            evidence = torch.maximum(exposure, torch.maximum(failure, error_exposure))
            fast_alpha = self._ema_alpha(evidence, float(self.cfg.fast_half_life))
            slow_alpha = self._ema_alpha(evidence, float(self.cfg.slow_half_life))
            fast = self.difficulty_fast[observed]
            slow = self.difficulty_slow[observed]
            first_evidence = fast < 0.0
            self.difficulty_fast[observed] = torch.where(
                first_evidence,
                batch_difficulty,
                fast + fast_alpha * (batch_difficulty - fast),
            )
            self.difficulty_slow[observed] = torch.where(
                first_evidence,
                batch_difficulty,
                slow + slow_alpha * (batch_difficulty - slow),
            )
            updated_exposure = self.total_exposure[observed] + exposure
            self.total_exposure[observed] = updated_exposure.clamp_max(float(self.cfg.uncertainty_exposure))

        self._exposure_delta.zero_()
        self._failure_delta.zero_()
        self._tracking_error_sum_delta.zero_()
        self._tracking_error_count_delta.zero_()
        return True

    @staticmethod
    def _normalize_or(values: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        total = values.sum()
        normalized = values / total.clamp_min(torch.finfo(values.dtype).tiny)
        return torch.where(total > torch.finfo(values.dtype).tiny, normalized, fallback)

    def _cap_relative_to_coverage(
        self,
        probabilities: torch.Tensor,
        coverage: torch.Tensor,
        *,
        cap_ratio: float,
    ) -> torch.Tensor:
        cap = coverage * cap_ratio
        capped = torch.minimum(probabilities, cap)
        deficit = 1.0 - capped.sum()
        if deficit <= torch.finfo(capped.dtype).eps:
            return capped / capped.sum()
        capacity = (cap - capped).clamp_min(0.0)
        capacity_sum = capacity.sum()
        if capacity_sum <= torch.finfo(capped.dtype).tiny:
            return capped / capped.sum()
        capped += capacity * (deficit / capacity_sum)
        return capped / capped.sum()

    def distribution(self, bin_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Build the global mixture, then condition it on an optional bin subset."""

        if bin_ids is not None and len(bin_ids) == 0:
            raise ValueError("cannot build a distribution over an empty bin set")

        all_bin_ids = torch.arange(self.bin_count, dtype=torch.long, device=self.device)
        coverage = self.base_weights[all_bin_ids]
        coverage = coverage / coverage.sum()
        initialized = (self.difficulty_fast >= 0.0) & (self.difficulty_slow >= 0.0)
        difficulty = torch.where(
            initialized,
            self.difficulty_fast.double().clamp(0.0, 1.0),
            torch.ones(self.bin_count, dtype=torch.float64, device=self.device),
        )
        slow = torch.where(
            initialized,
            self.difficulty_slow.double().clamp(0.0, 1.0),
            torch.ones_like(difficulty),
        )
        exposure = self.total_exposure.double()
        uncertainty = (1.0 - exposure / float(self.cfg.uncertainty_exposure)).clamp_(0.0, 1.0)
        measured_progress = ((slow - difficulty).clamp_min_(0.0) / slow.clamp_min(1.0e-6)).clamp_max_(1.0)
        progress = torch.where(initialized, measured_progress, torch.zeros_like(measured_progress))
        learnability = torch.maximum(uncertainty, progress)

        hard_raw = coverage * difficulty
        hard = self._normalize_or(hard_raw, coverage)
        learnable_raw = hard_raw * learnability
        learnable = self._normalize_or(learnable_raw, hard)

        has_support = (hard_raw.sum() > torch.finfo(hard_raw.dtype).tiny) & (
            learnable_raw.sum() > torch.finfo(learnable_raw.dtype).tiny
        )
        remaining = torch.where(has_support, (hard * learnability).sum(), coverage.new_zeros(()))
        gate_position = (remaining / float(self.cfg.learnability_full_scale)).clamp_(0.0, 1.0)
        gate_position = gate_position.square() * (3.0 - 2.0 * gate_position)
        gate = gate_position * float(self.cfg.max_learnable_fraction)

        coverage_fraction = float(self.cfg.coverage_fraction)
        adaptive_fraction = 1.0 - coverage_fraction
        # The final distribution contains an uncapped coverage floor. Convert
        # the requested final cap into the corresponding adaptive-component cap:
        #   coverage_fraction + adaptive_fraction * adaptive_cap_ratio = final_cap_ratio.
        adaptive_cap_ratio = (float(self.cfg.probability_cap_ratio) - coverage_fraction) / adaptive_fraction
        hard = self._cap_relative_to_coverage(hard, coverage, cap_ratio=adaptive_cap_ratio)
        learnable = self._cap_relative_to_coverage(learnable, coverage, cap_ratio=adaptive_cap_ratio)
        adaptive = hard.lerp(learnable, gate)
        probabilities = coverage * coverage_fraction
        probabilities += adaptive * adaptive_fraction
        probabilities /= probabilities.sum()
        if not torch.all(torch.isfinite(probabilities)) or torch.any(probabilities < 0.0):
            raise RuntimeError("adaptive sampling produced invalid probabilities")

        self.last_gate = float(gate.item())
        self.last_remaining_learnability = float(remaining.item())
        self.last_difficulty_mean = float((coverage * difficulty).sum().item())
        self.last_coverage_mass = float(self.cfg.coverage_fraction)
        self.last_learnable_mass = (1.0 - float(self.cfg.coverage_fraction)) * self.last_gate
        self.last_hard_mass = 1.0 - self.last_coverage_mass - self.last_learnable_mass
        if bin_ids is None:
            return probabilities.float()
        conditioned = probabilities[bin_ids]
        subset_coverage = self.base_weights[bin_ids]
        subset_coverage /= subset_coverage.sum()
        return self._normalize_or(conditioned, subset_coverage).float()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "version": torch.tensor(self.STATE_VERSION, dtype=torch.long),
            "recipe": self.recipe_tensor(),
            "total_exposure": self.total_exposure.detach().cpu().clone(),
            "difficulty_fast": self.difficulty_fast.detach().cpu().clone(),
            "difficulty_slow": self.difficulty_slow.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> bool:
        version = state.get("version")
        if (
            not isinstance(version, torch.Tensor)
            or version.numel() != 1
            or version.dtype not in _INTEGER_DTYPES
            or int(version.item()) != self.STATE_VERSION
        ):
            return False
        recipe = state.get("recipe")
        if not isinstance(recipe, torch.Tensor) or not torch.equal(recipe.cpu().double(), self.recipe_tensor()):
            return False
        names = ("total_exposure", "difficulty_fast", "difficulty_slow")
        if any(not isinstance(state.get(name), torch.Tensor) for name in names):
            return False
        if any(state[name].shape != (self.bin_count,) for name in names):
            return False
        exposure = state["total_exposure"].to(device=self.device, dtype=torch.float32)
        fast = state["difficulty_fast"].to(device=self.device, dtype=torch.float32)
        slow = state["difficulty_slow"].to(device=self.device, dtype=torch.float32)
        uninitialized = (fast == -1.0) & (slow == -1.0)
        initialized = (fast >= 0.0) & (fast <= 1.0) & (slow >= 0.0) & (slow <= 1.0)
        if (
            not torch.all(torch.isfinite(exposure))
            or not torch.all(torch.isfinite(fast))
            or not torch.all(torch.isfinite(slow))
            or torch.any(exposure < 0.0)
            or torch.any(~(uninitialized | initialized))
            or torch.any(uninitialized & (exposure != 0.0))
        ):
            return False
        self.total_exposure.copy_(exposure.clamp_max(float(self.cfg.uncertainty_exposure)))
        self.difficulty_fast.copy_(fast)
        self.difficulty_slow.copy_(slow)
        self._exposure_delta.zero_()
        self._failure_delta.zero_()
        self._tracking_error_sum_delta.zero_()
        self._tracking_error_count_delta.zero_()
        return True


class MotionCommandV1(CommandTerm):
    cfg: MotionCommandV1Cfg

    @staticmethod
    def _select_rank_shard(
        motion_files: Sequence[str],
        *,
        world_size: int | None = None,
        rank: int | None = None,
    ):
        """Return the deterministic global-ID slice owned by one rank."""

        return select_motion_shard(
            motion_files,
            shard_by_rank=True,
            world_size=world_size,
            rank=rank,
        )

    def __init__(self, cfg: MotionCommandV1Cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        local_body_names = self.cfg.adaptive_sampling.local_position_body_names
        if self.cfg.adaptive_sampling.local_position_error_weight > 0.0 and not local_body_names:
            raise ValueError("local_position_body_names must be provided when local-position difficulty is enabled")
        self._local_position_body_indexes = torch.tensor(
            [self.cfg.body_names.index(name) for name in local_body_names],
            dtype=torch.long,
            device=self.device,
        )
        local_offsets = self.cfg.adaptive_sampling.local_position_body_offsets
        if local_offsets:
            self._local_position_body_offsets = torch.tensor(local_offsets, dtype=torch.float32, device=self.device)
            if self._local_position_body_offsets.shape != (len(local_body_names), 3):
                raise ValueError(
                    "local_position_body_offsets must have shape "
                    f"({len(local_body_names)}, 3), got {tuple(self._local_position_body_offsets.shape)}"
                )
        else:
            self._local_position_body_offsets = torch.zeros(
                (len(local_body_names), 3), dtype=torch.float32, device=self.device
            )
        if self.cfg.fixed_joint_positions:
            fixed_joint_ids, fixed_joint_names = self.robot.find_joints(
                list(self.cfg.fixed_joint_positions), preserve_order=True, as_proxy=True
            )
            self._fixed_joint_ids = fixed_joint_ids.torch
            self._fixed_joint_positions = torch.tensor(
                [self.cfg.fixed_joint_positions[name] for name in fixed_joint_names],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self._fixed_joint_ids = torch.empty(0, dtype=torch.long, device=self.device)
            self._fixed_joint_positions = torch.empty(0, dtype=torch.float32, device=self.device)
        tracked_joint_mask = torch.ones(self.robot.num_joints, dtype=torch.bool, device=self.device)
        tracked_joint_mask[self._fixed_joint_ids] = False
        self._tracked_joint_ids = torch.where(tracked_joint_mask)[0]

        if self.cfg.playback_start_frame is not None and self.cfg.playback_start_frame < 0:
            raise ValueError("playback_start_frame must be non-negative or None")
        self._motion_files, global_time_totals = resolve_motion_catalog(
            self.cfg.motion_file,
            cache_path=self.cfg.motion_catalog_cache,
            max_workers=self.cfg.motion_load_workers,
        )
        self._motion_storage_device = self.cfg.motion_data_device
        if self._motion_storage_device == "auto":
            self._motion_storage_device = "cpu" if len(self._motion_files) > 1 else self.device
        self._global_time_totals = torch.tensor(global_time_totals, dtype=torch.long, device=self.device)
        self.global_num_motions = len(self._motion_files)
        self._initialize_adaptive_sampling()
        self._adaptive_layout_checked = False
        self._rebuild_global_sampling_distribution()

        if self.cfg.motion_shard_across_ranks:
            selection = self._select_rank_shard(self._motion_files)
            selected_global_ids = torch.tensor(selection.global_ids, dtype=torch.long, device=self.device)
            # A deterministic rank shard is a static working set: every motion
            # assigned to this rank is resident, and the union of rank shards is
            # the complete global manifest.  Working-set resampling must remain
            # disabled even though one process does not own every global motion.
            self.max_num_load_motions = len(selection.global_ids)
            self.all_motions_loaded = True
        else:
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
        self._quality_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        self.metrics["sampling_coverage_mass"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_hard_mass"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_learnable_mass"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_learnability_gate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_remaining_learnability"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_difficulty_mean"] = torch.zeros(self.num_envs, device=self.device)
        self._rebuild_active_motion_mapping()
        self._rebuild_sampling_distribution()

    def _initialize_adaptive_sampling(self) -> None:
        self.adaptive_sampler = AdaptiveSamplerV1(
            self._global_time_totals,
            self.cfg.adaptive_sampling,
            device=self.device,
        )
        self.motion_bin_counts = self.adaptive_sampler.motion_bin_counts
        self.bin_count = self.adaptive_sampler.bin_count
        self.motion_bin_offsets = self.adaptive_sampler.motion_bin_offsets
        self.bin_motion_ids = self.adaptive_sampler.bin_motion_ids
        self.bin_starts = self.adaptive_sampler.bin_starts
        self.bin_ends = self.adaptive_sampler.bin_ends

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
            "[INFO] Motion working set: "
            f"rank={motion.rank}/{motion.world_size}, "
            f"files={motion.rank_num_files}/{motion.global_num_motions}, "
            f"unique={motion.global_ids.unique().numel()}, "
            f"frames={motion.rank_num_frames:,}, chunks={motion.num_chunks}, "
            f"resident={motion.resident_bytes / 2**30:.2f} GiB, "
            f"storage={motion.storage_device}."
        )
        return motion

    def _rebuild_active_motion_mapping(self) -> None:
        self._active_bin_ids, self._active_local_motion_ids = self.adaptive_sampler.active_mapping(
            self.motion.global_ids
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [
                self.joint_pos[:, self._tracked_joint_ids],
                self.joint_vel[:, self._tracked_joint_ids],
            ],
            dim=1,
        )

    def _refresh_motion_cache(self, env_ids: torch.Tensor | None = None) -> None:
        """Gather the current reference frame once for all downstream MDP terms."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        motion_ids = self.motion_ids[env_ids]
        time_steps = self.time_steps[env_ids]
        samples = self.motion.sample_many(self.motion._FIELDS, motion_ids, time_steps)
        self._apply_fixed_joint_state(samples["joint_pos"], samples["joint_vel"])
        for field, cache in self._motion_cache.items():
            cache[env_ids] = samples[field]

    def _apply_fixed_joint_state(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
        """Overwrite configured fixed joints with their held position and zero velocity."""
        if self._fixed_joint_ids.numel() == 0:
            return
        joint_pos[:, self._fixed_joint_ids] = self._fixed_joint_positions
        joint_vel[:, self._fixed_joint_ids] = 0.0

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

    def _local_position_errors(self) -> torch.Tensor | None:
        """Return anchor-frame key-point errors used by the optional sampler term."""
        if self._local_position_body_indexes.numel() == 0:
            return None
        body_ids = self._local_position_body_indexes
        offsets = self._local_position_body_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        ref_body_pos_w = self.body_pos_w[:, body_ids] + quat_apply(self.body_quat_w[:, body_ids], offsets)
        robot_body_pos_w = self.robot_body_pos_w[:, body_ids] + quat_apply(self.robot_body_quat_w[:, body_ids], offsets)
        ref_anchor_quat = self.anchor_quat_w[:, None, :].expand(-1, len(body_ids), -1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].expand(-1, len(body_ids), -1)
        ref_pos_b = quat_apply_inverse(ref_anchor_quat, ref_body_pos_w - self.anchor_pos_w[:, None, :])
        robot_pos_b = quat_apply_inverse(robot_anchor_quat, robot_body_pos_w - self.robot_anchor_pos_w[:, None, :])
        return torch.norm(ref_pos_b - robot_pos_b, dim=-1)

    def _update_metrics(self):
        self.metrics["error_anchor_pos"].copy_(torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1))
        self.metrics["error_anchor_rot"].copy_(quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w))
        self.metrics["error_anchor_lin_vel"].copy_(
            torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        )
        self.metrics["error_anchor_ang_vel"].copy_(
            torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        )

        relative_body_position_errors = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1)
        self.metrics["error_body_pos"].copy_(relative_body_position_errors.mean(dim=-1))
        relative_body_orientation_errors = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w)
        self.metrics["error_body_rot"].copy_(relative_body_orientation_errors.mean(dim=-1))
        self.metrics["error_body_lin_vel"].copy_(
            torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        )
        self.metrics["error_body_ang_vel"].copy_(
            torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)
        )

        tracked_joint_ids = self._tracked_joint_ids
        self.metrics["error_joint_pos"].copy_(
            torch.norm(
                self.joint_pos[:, tracked_joint_ids] - self.robot_joint_pos[:, tracked_joint_ids],
                dim=-1,
            )
        )
        self.metrics["error_joint_vel"].copy_(
            torch.norm(
                self.joint_vel[:, tracked_joint_ids] - self.robot_joint_vel[:, tracked_joint_ids],
                dim=-1,
            )
        )

        quality_env_ids = torch.where(self._has_sampled & self._quality_valid)[0]
        quality_time_steps = torch.minimum(self.time_steps[quality_env_ids], self.motion_lengths[quality_env_ids] - 1)
        quality_bin_ids = self._bucket_ids(self.motion_ids[quality_env_ids], quality_time_steps)
        global_body_position_errors = torch.norm(self.body_pos_w - self.robot_body_pos_w, dim=-1)
        local_position_errors = self._local_position_errors()
        self.adaptive_sampler.record_tracking_error(
            quality_bin_ids,
            global_body_position_errors[quality_env_ids],
            relative_position_errors=relative_body_position_errors[quality_env_ids],
            relative_orientation_errors=relative_body_orientation_errors[quality_env_ids],
            local_position_errors=(None if local_position_errors is None else local_position_errors[quality_env_ids]),
        )
        self._quality_valid[self._has_sampled] = True

    def _bucket_ids(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        global_motion_ids = self.motion.global_ids[motion_ids]
        return self.adaptive_sampler.bucket_ids(global_motion_ids, time_steps)

    def snapshot_transition_sampling_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Snapshot each environment's global bin and adaptive/coverage ratio.

        The context is read immediately before an action is sampled so rollout
        consumers can associate the transition with the reference bin that
        produced its observation.  It is deliberately read-only and therefore
        does not change the V20--V25 sampling recipe or update schedule.  Both
        probabilities are conditioned on this rank's active motion set, exactly
        like the reset sampler itself.
        """

        bin_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        probability_ratios = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        env_ids = torch.where(self._has_sampled)[0]
        if len(env_ids) == 0:
            return bin_ids, probability_ratios

        time_steps = self.time_steps[env_ids]
        motion_lengths = self.motion_lengths[env_ids]
        if torch.any(time_steps < 0) or torch.any(time_steps > motion_lengths):
            raise RuntimeError(
                "adaptive sampling context was requested outside the current reference motion"
            )
        # With explicit ``motion_time_out`` termination, command updates advance
        # a freshly reset environment after the reset phase.  A reset sampled at
        # the final frame is therefore represented temporarily by
        # ``time_steps == motion_lengths`` while its observation and cached
        # reference still correspond to the final frame.  Match the clamping
        # already used by command observations, exposure, and quality metrics.
        time_steps = torch.minimum(time_steps, motion_lengths - 1)
        current_bins = self._bucket_ids(self.motion_ids[env_ids], time_steps)
        coverage = (
            self.adaptive_sampler.base_weights[current_bins]
            / self._active_coverage_probability_mass
        )
        adaptive = (
            self.global_sampling_probabilities[current_bins].double()
            / self._active_adaptive_probability_mass
        )
        ratios = adaptive / coverage.clamp_min(torch.finfo(coverage.dtype).tiny)
        # ``global_sampling_probabilities`` is stored in float32 while coverage
        # is accumulated in float64. Snap numerical round-off at the neutral
        # ratio so a cold coverage distribution is not split arbitrarily into
        # high/remaining groups.
        ratios = torch.where(
            torch.isclose(ratios, torch.ones_like(ratios), rtol=1.0e-5, atol=1.0e-7),
            torch.ones_like(ratios),
            ratios,
        )
        if not torch.all(torch.isfinite(ratios)) or torch.any(ratios <= 0.0):
            raise RuntimeError("adaptive sampling context produced invalid probability ratios")

        bin_ids[env_ids] = current_bins
        probability_ratios[env_ids] = ratios.float()
        return bin_ids, probability_ratios

    def _rebuild_global_sampling_distribution(self) -> None:
        self.global_sampling_probabilities = self.adaptive_sampler.distribution()
        motion_sampling_probabilities = torch.zeros(self.global_num_motions, dtype=torch.float32, device=self.device)
        motion_sampling_probabilities.index_add_(0, self.bin_motion_ids, self.global_sampling_probabilities)
        self._motion_sampling_probabilities = motion_sampling_probabilities / motion_sampling_probabilities.sum()

    def _rebuild_sampling_distribution(self) -> None:
        active_probabilities = self.global_sampling_probabilities[self._active_bin_ids].double()
        active_coverage = self.adaptive_sampler.base_weights[self._active_bin_ids]
        self._active_adaptive_probability_mass = active_probabilities.sum()
        self._active_coverage_probability_mass = active_coverage.sum()
        if (
            not torch.isfinite(self._active_adaptive_probability_mass)
            or self._active_adaptive_probability_mass <= 0.0
            or not torch.isfinite(self._active_coverage_probability_mass)
            or self._active_coverage_probability_mass <= 0.0
        ):
            raise RuntimeError("active adaptive-sampling normalizers must be finite and positive")
        active_coverage /= self._active_coverage_probability_mass
        self.active_sampling_probabilities = self.adaptive_sampler._normalize_or(
            active_probabilities,
            active_coverage,
        ).float()

        entropy = -(
            self.active_sampling_probabilities * self.active_sampling_probabilities.clamp_min(1.0e-12).log()
        ).sum()
        entropy_normalized = (
            entropy / math.log(len(self.active_sampling_probabilities))
            if len(self.active_sampling_probabilities) > 1
            else torch.ones_like(entropy)
        )
        probability_max = self.active_sampling_probabilities.max()
        motion_probabilities = torch.zeros(self.global_num_motions, dtype=torch.float32, device=self.device)
        motion_probabilities.index_add_(
            0,
            self.bin_motion_ids[self._active_bin_ids],
            self.active_sampling_probabilities,
        )
        self._sampling_diagnostics = {
            "sampling_entropy": float(entropy_normalized.item()),
            "sampling_top1_prob": float(probability_max.item()),
            "sampling_top1_motion_prob": float(motion_probabilities.max().item()),
            "sampling_coverage_mass": self.adaptive_sampler.last_coverage_mass,
            "sampling_hard_mass": self.adaptive_sampler.last_hard_mass,
            "sampling_learnable_mass": self.adaptive_sampler.last_learnable_mass,
            "sampling_learnability_gate": self.adaptive_sampler.last_gate,
            "sampling_remaining_learnability": self.adaptive_sampler.last_remaining_learnability,
            "sampling_difficulty_mean": self.adaptive_sampler.last_difficulty_mean,
        }
        if "sampling_entropy" in self.metrics:
            self._restore_sampling_diagnostics(slice(None))

    def _restore_sampling_diagnostics(self, env_ids) -> None:
        for name, value in self._sampling_diagnostics.items():
            self.metrics[name][env_ids] = value

    def _update_adaptive_exposure(self) -> None:
        active_env_ids = torch.where(self._has_sampled)[0]
        time_steps = torch.minimum(self.time_steps[active_env_ids], self.motion_lengths[active_env_ids] - 1)
        bin_ids = self._bucket_ids(self.motion_ids[active_env_ids], time_steps)
        self.adaptive_sampler.record_exposure(bin_ids)

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

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        previous_env_ids = env_ids[self._has_sampled[env_ids]]
        previous_time_steps = torch.minimum(
            self.time_steps[previous_env_ids], self.motion_lengths[previous_env_ids] - 1
        )
        previous_bins = self._bucket_ids(self.motion_ids[previous_env_ids], previous_time_steps)
        episode_failed = self._env.termination_manager.terminated[previous_env_ids]
        self.adaptive_sampler.record_exposure(previous_bins)
        self.adaptive_sampler.record_failures(
            previous_bins[episode_failed],
            event_ids=previous_env_ids[episode_failed],
        )

        if self._fixed_motion_ids is None:
            sampled_active_bin_ids = torch.multinomial(
                self.active_sampling_probabilities,
                num_samples=len(env_ids),
                replacement=True,
            )
            sampled_bins = self._active_bin_ids[sampled_active_bin_ids]
            sampled_motion_ids = self._active_local_motion_ids[sampled_active_bin_ids]
        else:
            sampled_motion_ids = self._fixed_motion_ids[env_ids]
            sampled_global_motion_ids = self.motion.global_ids[sampled_motion_ids]
            sampled_motion_bin_counts = self.motion_bin_counts[sampled_global_motion_ids]
            sampled_local_bin_ids = (torch.rand(len(env_ids), device=self.device) * sampled_motion_bin_counts).long()
            sampled_bins = self.motion_bin_offsets[sampled_global_motion_ids] + sampled_local_bin_ids
        self.motion_ids[env_ids] = sampled_motion_ids
        self.motion_lengths[env_ids] = self.motion.lengths(sampled_motion_ids)
        bin_lengths = self.bin_ends[sampled_bins] - self.bin_starts[sampled_bins]
        if self.cfg.playback_start_frame is not None:
            start_frame = int(self.cfg.playback_start_frame)
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
            sampled_time_steps = (
                self.bin_starts[sampled_bins] + (torch.rand(len(env_ids), device=self.device) * bin_lengths).long()
            )
            if self.cfg.adaptive_sampling.pre_failure_window > 0:
                pre_failure_offsets = torch.randint(
                    self.cfg.adaptive_sampling.pre_failure_window,
                    (len(env_ids),),
                    device=self.device,
                )
                sampled_time_steps = (sampled_time_steps - pre_failure_offsets).clamp_min(0)
        self.time_steps[env_ids] = sampled_time_steps
        self._has_sampled[env_ids] = True
        self._quality_valid[env_ids] = False
        self._restore_sampling_diagnostics(env_ids)

    def _check_adaptive_distributed_layout(self) -> None:
        """Fail coherently before any variable-size adaptive collective."""

        if self._adaptive_layout_checked:
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            # The environment is constructed before the runner initializes
            # the process group. Preserve the future cross-rank check.
            self._adaptive_layout_checked = self.motion.world_size == 1
            return
        assignment_matches = torch.tensor(
            [
                int(
                    self.motion.world_size == torch.distributed.get_world_size()
                    and self.motion.rank == torch.distributed.get_rank()
                )
            ],
            dtype=torch.long,
            device=self.device,
        )
        torch.distributed.all_reduce(assignment_matches, op=torch.distributed.ReduceOp.MIN)
        if not bool(assignment_matches.item()):
            raise RuntimeError("Motion shard rank/world-size metadata disagrees with torch.distributed")
        layout_header = torch.tensor(
            [self.global_num_motions, self.bin_count, *self.motion.manifest_fingerprint_words],
            dtype=torch.long,
            device=self.device,
        )
        layout = torch.cat((layout_header, self._motion_schema_descriptor(), self._global_time_totals))
        layout_min = layout.clone()
        layout_max = layout.clone()
        torch.distributed.all_reduce(layout_min, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(layout_max, op=torch.distributed.ReduceOp.MAX)
        recipe = self.adaptive_sampler.recipe_tensor().to(self.device)
        recipe_min = recipe.clone()
        recipe_max = recipe.clone()
        torch.distributed.all_reduce(recipe_min, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(recipe_max, op=torch.distributed.ReduceOp.MAX)
        if not torch.equal(layout_min, layout_max) or not torch.equal(recipe_min, recipe_max):
            raise RuntimeError("Distributed workers disagree on the motion layout or adaptive sampling recipe")
        self._adaptive_layout_checked = True

    def _motion_schema_descriptor(self) -> torch.Tensor:
        """Encode the process-local decoded schema in a fixed-width integer vector."""

        max_trailing_dims = 2
        schema = [
            int(round(float(self.motion.fps) * 1_000_000)),
            int(self.motion.world_size),
            len(self.motion.fields),
        ]
        for field in self.motion.fields:
            shape = tuple(int(dim) for dim in self.motion.field_shapes[field])
            if len(shape) > max_trailing_dims:
                raise RuntimeError(
                    f"Motion field {field!r} has unsupported trailing rank {len(shape)}; "
                    f"expected at most {max_trailing_dims}"
                )
            schema.extend((len(shape), *shape, *([-1] * (max_trailing_dims - len(shape)))))
        group_names = sorted(self.motion._group_widths)
        schema.append(len(group_names))
        schema.extend(int(self.motion._group_widths[name]) for name in group_names)
        return torch.tensor(schema, dtype=torch.long, device=self.device)

    def sync_and_compute_adaptive_sampling(self, *, sync_across_ranks: bool) -> None:
        if sync_across_ranks:
            self._check_adaptive_distributed_layout()
        if not self.adaptive_sampler.update(sync_across_ranks=sync_across_ranks):
            return
        self._rebuild_global_sampling_distribution()
        self._rebuild_sampling_distribution()

    def get_adaptive_sampling_state(self) -> dict[str, torch.Tensor]:
        state = self.adaptive_sampler.state_dict()
        state["bin_size"] = torch.tensor(self.cfg.adaptive_sampling.bin_size, dtype=torch.long)
        state["manifest_fingerprint_words"] = torch.tensor(
            self.motion.manifest_fingerprint_words,
            dtype=torch.long,
        )
        state["motion_lengths"] = self._global_time_totals.detach().cpu().clone()
        state["body_names_utf8"] = torch.tensor(
            list("\0".join(self.cfg.body_names).encode("utf-8")),
            dtype=torch.uint8,
        )
        return state

    def load_adaptive_sampling_state(self, state_dict: dict[str, torch.Tensor]) -> bool:
        saved_bin_size = state_dict.get("bin_size")
        saved_fingerprint = state_dict.get("manifest_fingerprint_words")
        saved_motion_lengths = state_dict.get("motion_lengths")
        saved_body_names = state_dict.get("body_names_utf8")
        expected_fingerprint = torch.tensor(self.motion.manifest_fingerprint_words, dtype=torch.long)
        expected_body_names = torch.tensor(
            list("\0".join(self.cfg.body_names).encode("utf-8")),
            dtype=torch.uint8,
        )
        layout_matches = (
            isinstance(saved_bin_size, torch.Tensor)
            and saved_bin_size.numel() == 1
            and saved_bin_size.dtype in _INTEGER_DTYPES
            and int(saved_bin_size.item()) == self.cfg.adaptive_sampling.bin_size
            and isinstance(saved_fingerprint, torch.Tensor)
            and saved_fingerprint.dtype in _INTEGER_DTYPES
            and torch.equal(saved_fingerprint.cpu().long(), expected_fingerprint)
            and isinstance(saved_motion_lengths, torch.Tensor)
            and saved_motion_lengths.dtype in _INTEGER_DTYPES
            and torch.equal(saved_motion_lengths.cpu().long(), self._global_time_totals.cpu())
            and isinstance(saved_body_names, torch.Tensor)
            and saved_body_names.dtype == torch.uint8
            and torch.equal(saved_body_names.cpu(), expected_body_names)
        )
        if not layout_matches or not self.adaptive_sampler.load_state_dict(state_dict):
            print("[WARN] Sampler checkpoint does not match command_v1; starting with fresh statistics.")
            self._has_sampled.zero_()
            self._quality_valid.zero_()
            return False
        self._has_sampled.zero_()
        self._quality_valid.zero_()
        self._rebuild_global_sampling_distribution()
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
        self._quality_valid.zero_()
        self._fixed_motion_ids = None
        self.motion_ids.zero_()
        self.motion_lengths.zero_()
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
        joint_vel = self.joint_vel[env_ids].clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
        self._apply_fixed_joint_state(joint_pos, joint_vel)
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)
        # ManagerBasedEnv.reset() writes scene data once after command reset.  The
        # delayed actuators have just cleared their histories at that point, so
        # seed their first position command with the reset pose instead of the
        # zero/stale target left by the previous episode.  The next policy step
        # will overwrite this target normally.
        self.robot.actuators.target_command.set_position_index(value=joint_pos, env_ids=env_ids)
        # Auto-reset does not write scene commands immediately.  Seed each
        # delayed actuator group directly so its first policy command is
        # appended after the reset pose and observes the configured lag.  The
        # collection mapping exposes group owners and their articulation-order
        # joint indices; subset seeding leaves every other environment's
        # history untouched.
        for actuator in self.robot.actuators.values():
            if isinstance(actuator, DelayedImplicitActuator):
                actuator.seed_position_history(joint_pos[:, actuator.joint_indices], env_ids)
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

        self.refresh_relative_body_targets()

    def refresh_relative_body_targets(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Refresh robot-aligned reference poses without advancing motion time.

        ``ManagerBasedEnv.reset()`` resamples the command and forwards the
        articulation, but it does not call ``CommandManager.compute()`` before
        the first policy step.  Terminations and rewards run before that first
        command update, so these derived targets must be synchronized explicitly
        after a manual reset to avoid reading their zero/stale initialization.
        """

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        anchor_pos_w = self.anchor_pos_w[env_ids, None, :]
        anchor_quat_w = self.anchor_quat_w[env_ids, None, :]
        robot_anchor_pos_w = self.robot_anchor_pos_w[env_ids, None, :]
        robot_anchor_quat_w = self.robot_anchor_quat_w[env_ids, None, :]

        delta_pos_w = robot_anchor_pos_w.expand(-1, len(self.cfg.body_names), -1).clone()
        delta_pos_w[..., 2] = anchor_pos_w[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w))).expand(
            -1, len(self.cfg.body_names), -1
        )

        self.body_quat_relative_w[env_ids] = quat_mul(delta_ori_w, self.body_quat_w[env_ids])
        self.body_pos_relative_w[env_ids] = delta_pos_w + quat_apply(
            delta_ori_w,
            self.body_pos_w[env_ids] - anchor_pos_w,
        )

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
class MotionCommandV1Cfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommandV1

    asset_name: str = MISSING

    motion_file: str = MISSING
    motion_data_device: str = "auto"
    # Optional manifest catalog sidecar containing the resolved paths and frame
    # counts. None preserves uncached YAML/NPZ discovery.
    motion_catalog_cache: str | None = None
    # Resolve the complete global layout on every process, but decode and retain
    # only the deterministic files[global_rank::world_size] slice. Small
    # datasets remain replicated so no rank receives an empty motion collection.
    # Each rank conditions the global adaptive weights on its local slice, so
    # equal-sized DDP batches give every rank shard equal aggregate mass.
    motion_shard_across_ranks: bool = False
    # Load all motions when they fit. Otherwise draw a process-local working set.
    max_num_load_motions: int | None = None
    # Disable replacement to keep every process-local working set unique.
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
    fixed_joint_positions: dict[str, float] = {}
    """Joint positions [rad] that override motion data and reset randomization."""

    # When disabled, the environment is expected to terminate at the final
    # motion frame instead of silently switching to another motion.
    resample_at_motion_end: bool = True
    # Playback/debug option. Training keeps random motion-time initialization
    # unless this is explicitly enabled by a caller such as play.py.
    start_at_motion_beginning: bool = False
    playback_start_frame: int | None = None

    adaptive_sampling: AdaptiveSamplerV1Cfg = AdaptiveSamplerV1Cfg()

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
