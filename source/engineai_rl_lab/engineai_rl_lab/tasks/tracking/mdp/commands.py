from __future__ import annotations

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

from .motion_data import MotionCollection, resolve_motion_files
from .motion_data import MotionLoader as MotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        motion_files = resolve_motion_files(self.cfg.motion_file)
        storage_device = self.cfg.motion_data_device
        if storage_device == "auto":
            storage_device = "cpu" if len(motion_files) > 1 else self.device
        self.motion = MotionCollection(
            motion_files,
            self.body_indexes,
            device=self.device,
            storage_device=storage_device,
            joint_names=self.robot.joint_names,
            body_names=self.cfg.body_names,
            shard_by_rank=self.cfg.motion_shard_across_ranks,
            max_chunk_frames=self.cfg.motion_chunk_frames,
            max_workers=self.cfg.motion_load_workers,
        )
        print(
            "[INFO] Motion YAML/NPZ assignment: "
            f"rank={self.motion.rank}/{self.motion.world_size}, "
            f"files={self.motion.rank_num_files}/{self.motion.global_num_motions}, "
            f"frames={self.motion.rank_num_frames:,}, chunks={self.motion.num_chunks}, "
            f"resident={self.motion.resident_bytes / 2**30:.2f} GiB, "
            f"storage={self.motion.storage_device}."
        )
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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

        if self.cfg.adaptive_bin_size <= 0:
            raise ValueError("adaptive_bin_size must be positive")
        if self.cfg.adaptive_max_bins <= 0:
            raise ValueError("adaptive_max_bins must be positive")
        motion_lengths = self.motion.time_totals
        motion_bin_counts = torch.div(
            motion_lengths + self.cfg.adaptive_bin_size - 1,
            self.cfg.adaptive_bin_size,
            rounding_mode="floor",
        )
        requested_bin_count = int(motion_bin_counts.sum().item())
        if requested_bin_count <= self.cfg.adaptive_max_bins:
            self._adaptive_granularity = "bin"
            self.bin_count = requested_bin_count
            self.motion_bin_offsets = torch.zeros(self.motion.num_motions, dtype=torch.long, device=self.device)
            if self.motion.num_motions > 1:
                self.motion_bin_offsets[1:] = torch.cumsum(motion_bin_counts[:-1], dim=0)
            self.bin_motion_ids = torch.repeat_interleave(
                torch.arange(self.motion.num_motions, device=self.device), motion_bin_counts
            )
            bin_indexes = torch.arange(self.bin_count, device=self.device)
            local_bin_indexes = bin_indexes - self.motion_bin_offsets[self.bin_motion_ids]
            self.bin_starts = local_bin_indexes * self.cfg.adaptive_bin_size
            self.bin_ends = torch.minimum(
                self.bin_starts + self.cfg.adaptive_bin_size,
                motion_lengths[self.bin_motion_ids],
            )
            self.bin_prior_weights = (self.bin_ends - self.bin_starts).float()
            if self.cfg.adaptive_sequence_length_agnostic:
                self.bin_prior_weights /= motion_bin_counts[self.bin_motion_ids]
        else:
            # Keep adaptive state and reset-time sampling bounded for future
            # corpora whose temporal-bin table would be too large.
            self._adaptive_granularity = "motion"
            self.bin_count = self.motion.num_motions
            self.bin_motion_ids = torch.arange(self.motion.num_motions, device=self.device)
            self.bin_starts = torch.zeros(self.bin_count, dtype=torch.long, device=self.device)
            self.bin_ends = motion_lengths.clone()
            self.motion_bin_offsets = torch.arange(self.motion.num_motions, device=self.device)
            self.bin_prior_weights = motion_lengths.float()
            if self.cfg.adaptive_sequence_length_agnostic:
                self.bin_prior_weights.fill_(1.0)
        self.bin_prior_weights /= self.bin_prior_weights.sum()
        initial_count = float(self.cfg.adaptive_prior_count)
        self.bin_episode_count = torch.full((self.bin_count,), initial_count, device=self.device)
        self.bin_failed_count = torch.full((self.bin_count,), initial_count, device=self.device)
        self._current_bin_episodes = torch.zeros(self.bin_count, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, device=self.device)
        self._adaptive_sync_counter = 0
        self._adaptive_layout_checked = False
        if not 0.0 <= self.cfg.adaptive_uniform_ratio <= 1.0:
            raise ValueError("adaptive_uniform_ratio must be in [0, 1]")
        if (
            self.cfg.adaptive_failure_rate_max_over_mean is not None
            and self.cfg.adaptive_failure_rate_max_over_mean <= 0.0
        ):
            raise ValueError("adaptive_failure_rate_max_over_mean must be positive or None")
        if self.cfg.adaptive_pre_failure_sample_window < 0:
            raise ValueError("adaptive_pre_failure_sample_window must be non-negative")
        if self.cfg.adaptive_sync_interval <= 0:
            raise ValueError("adaptive_sync_interval must be positive")

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self._rebuild_sampling_distribution()

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
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _bucket_ids(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        if self._adaptive_granularity == "motion":
            return motion_ids
        return self.motion_bin_offsets[motion_ids] + torch.div(
            time_steps, self.cfg.adaptive_bin_size, rounding_mode="floor"
        )

    def _rebuild_sampling_distribution(self) -> None:
        """Rebuild the inverse CDF only after accumulated statistics change."""

        failure_rate = self.bin_failed_count / self.bin_episode_count.clamp_min(1.0)
        if self.cfg.adaptive_failure_rate_max_over_mean is not None:
            failure_rate_upper_bound = failure_rate.mean() * self.cfg.adaptive_failure_rate_max_over_mean
            failure_rate = failure_rate.clamp(max=failure_rate_upper_bound)
        weighted_failure_rate = failure_rate * self.bin_prior_weights
        weighted_sum = weighted_failure_rate.sum()
        if float(weighted_sum) > 0.0:
            failure_probabilities = weighted_failure_rate / weighted_sum
        else:
            failure_probabilities = self.bin_prior_weights
        sampling_probabilities = (
            1.0 - self.cfg.adaptive_uniform_ratio
        ) * failure_probabilities + self.cfg.adaptive_uniform_ratio * self.bin_prior_weights
        sampling_probabilities /= sampling_probabilities.sum().clamp_min(1.0e-12)
        if not torch.all(torch.isfinite(sampling_probabilities)):
            raise RuntimeError("Adaptive motion sampling produced non-finite probabilities")
        self._sampling_probabilities = sampling_probabilities
        self._sampling_cdf = torch.cumsum(sampling_probabilities, dim=0)
        self._sampling_cdf[-1] = 1.0

        entropy = -(sampling_probabilities * sampling_probabilities.clamp_min(1.0e-12).log()).sum()
        entropy_normalized = entropy / math.log(self.bin_count) if self.bin_count > 1 else torch.ones_like(entropy)
        probability_max, index_max = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = entropy_normalized
        self.metrics["sampling_top1_prob"][:] = probability_max
        self.metrics["sampling_top1_bin"][:] = index_max.float() / self.bin_count

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        previously_sampled = self._has_sampled[env_ids]
        if torch.any(previously_sampled):
            previous_env_ids = env_ids[previously_sampled]
            previous_time_steps = torch.minimum(
                self.time_steps[previous_env_ids], self.motion_lengths[previous_env_ids] - 1
            )
            previous_bins = self._bucket_ids(self.motion_ids[previous_env_ids], previous_time_steps)
            self._current_bin_episodes.index_add_(
                0,
                previous_bins,
                torch.ones_like(previous_bins, dtype=self._current_bin_episodes.dtype),
            )
            episode_failed = self._env.termination_manager.terminated[previous_env_ids]
            if torch.any(episode_failed):
                failed_bins = previous_bins[episode_failed]
                self._current_bin_failed.index_add_(
                    0,
                    failed_bins,
                    torch.ones_like(failed_bins, dtype=self._current_bin_failed.dtype),
                )

        uniforms = torch.rand(len(env_ids), device=self.device)
        sampled_bins = torch.searchsorted(self._sampling_cdf, uniforms, right=True).clamp_max(self.bin_count - 1)
        sampled_motion_ids = self.bin_motion_ids[sampled_bins]
        self.motion_ids[env_ids] = sampled_motion_ids
        self.motion_lengths[env_ids] = self.motion.lengths(sampled_motion_ids)
        bin_lengths = self.bin_ends[sampled_bins] - self.bin_starts[sampled_bins]
        sampled_time_steps = (
            self.bin_starts[sampled_bins] + (torch.rand(len(env_ids), device=self.device) * bin_lengths).long()
        )
        if self.cfg.adaptive_pre_failure_sample_window > 0:
            pre_failure_offsets = torch.randint(
                self.cfg.adaptive_pre_failure_sample_window,
                (len(env_ids),),
                device=self.device,
            )
            sampled_time_steps = (sampled_time_steps - pre_failure_offsets).clamp_min(0)
        self.time_steps[env_ids] = sampled_time_steps
        self._has_sampled[env_ids] = True

    def _check_adaptive_distributed_layout(self) -> None:
        """Fail coherently before any variable-size adaptive collective."""

        if self._adaptive_layout_checked:
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            # The environment is constructed before the runner initializes
            # the process group. Preserve the future cross-rank check.
            self._adaptive_layout_checked = self.motion.world_size == 1
            return
        layout = torch.tensor(
            [
                int(self.motion.is_distributed_shard),
                self.motion.global_num_motions,
                0 if self.motion.is_distributed_shard else self.bin_count,
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
                "Distributed workers disagree on motion sharding/global count/bin layout or manifest fingerprint: "
                f"local={layout.tolist()}, min={layout_min.tolist()}, max={layout_max.tolist()}"
            )
        self._adaptive_layout_checked = True

    def _sync_adaptive_stats(self):
        """Merge counts locally for shards, or globally for replicated data."""

        self._check_adaptive_distributed_layout()
        episodes = self._current_bin_episodes
        failures = self._current_bin_failed
        if (
            not self.motion.is_distributed_shard
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            packed = torch.cat((episodes, failures))
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            episodes, failures = packed.chunk(2)
        self.bin_episode_count += episodes
        self.bin_failed_count += failures * self.cfg.adaptive_failure_multiplier
        self._current_bin_episodes.zero_()
        self._current_bin_failed.zero_()
        self._rebuild_sampling_distribution()

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

        self._adaptive_sync_counter += 1
        if self._adaptive_sync_counter % self.cfg.adaptive_sync_interval == 0:
            self._sync_adaptive_stats()

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
    # Resolve the complete YAML manifest on every worker, then load only the
    # deterministic global-rank slice. Small datasets remain replicated.
    motion_shard_across_ranks: bool = False
    # NPZ files are decoded concurrently but consumed in manifest order.
    motion_load_workers: int = 4
    # Upper bound for one process-local construction/runtime chunk. The scale
    # task raises this so its expected 32-rank shard is normally contiguous.
    motion_chunk_frames: int = 262_144
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # When disabled, the environment is expected to terminate at the final
    # motion frame instead of silently switching to another motion.
    resample_at_motion_end: bool = True

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_bin_size: int = 50
    adaptive_prior_count: float = 1.0
    adaptive_failure_multiplier: float = 1.0
    adaptive_failure_rate_max_over_mean: float | None = None
    adaptive_pre_failure_sample_window: int = 0
    adaptive_sync_interval: int = 200
    adaptive_sequence_length_agnostic: bool = True
    # Fall back to one adaptive bucket per motion before a temporal-bin table
    # becomes large enough to dominate device memory and reset-time work.
    adaptive_max_bins: int = 5_000_000

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
