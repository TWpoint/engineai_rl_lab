"""V26 STAR-lite PPO layered on the V25 grouped-reward critic.

V26 is a standalone scratch experiment.  It shares the grouped-reward critic
implementation and keeps the baseline scalar PPO objective, while changing
how current-rollout samples are normalized and reused:

* policy advantages are normalized separately for bins sampled above and at or
  below their frame-coverage baseline;
* positive-advantage fragments from the high-difficulty group are selected
  within broad difficulty bands; and
* a bounded fraction of each PPO mini-batch is drawn from those fragments.

All prioritized samples remain in the current rollout.  This is deliberate:
V26 is a sample-reuse experiment, not an off-policy replay buffer.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Generator, Sequence
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from .v25_compact_multi_critic_ppo import V25CompactMultiCriticPPO
from .v25_multi_critic_ppo import (
    V25RolloutStorage,
    _validate_distributed_contract,
    compute_grouped_gae,
)

_SOURCE_PATH = Path(__file__).resolve()
_FROZEN_SOURCE = _SOURCE_PATH.read_text()
_FROZEN_SOURCE_SHA256 = hashlib.sha256(_FROZEN_SOURCE.encode()).hexdigest()


def _text_sha256(value: object) -> str | None:
    return hashlib.sha256(value.encode()).hexdigest() if isinstance(value, str) else None


def _state_schema(value: object) -> tuple[tuple[str, tuple[int, ...], str], ...] | None:
    if not isinstance(value, dict):
        return None
    schema = []
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            return None
        schema.append((name, tuple(tensor.shape), str(tensor.dtype)))
    return tuple(schema)


def _validate_star_parameters(
    *,
    priority_fraction: float,
    high_difficulty_threshold: float,
    top_fraction: float,
    difficulty_boundaries: Sequence[float],
    reuse_cap: int,
    priority_weight_cap: float,
) -> tuple[float, ...]:
    if not 0.0 <= priority_fraction < 1.0:
        raise ValueError(f"star_priority_fraction must be in [0, 1), got {priority_fraction}.")
    if high_difficulty_threshold <= 0.0:
        raise ValueError(
            f"star_high_difficulty_threshold must be positive, got {high_difficulty_threshold}."
        )
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"star_top_fraction must be in (0, 1], got {top_fraction}.")
    boundaries = tuple(float(value) for value in difficulty_boundaries)
    if any(not math.isfinite(value) for value in boundaries):
        raise ValueError(f"star_difficulty_boundaries must be finite, got {boundaries}.")
    if any(value <= high_difficulty_threshold for value in boundaries):
        raise ValueError(
            "star_difficulty_boundaries must all exceed the high-difficulty threshold: "
            f"threshold={high_difficulty_threshold}, boundaries={boundaries}."
        )
    if tuple(sorted(set(boundaries))) != boundaries:
        raise ValueError(f"star_difficulty_boundaries must be strictly increasing, got {boundaries}.")
    if reuse_cap < 1:
        raise ValueError(f"star_reuse_cap must be positive, got {reuse_cap}.")
    if priority_weight_cap < high_difficulty_threshold:
        raise ValueError(
            "star_priority_weight_cap must be at least the high-difficulty threshold: "
            f"cap={priority_weight_cap}, threshold={high_difficulty_threshold}."
        )
    return boundaries


def difficulty_conditioned_normalize(
    raw_advantages: torch.Tensor,
    probability_ratios: torch.Tensor,
    *,
    high_difficulty_threshold: float,
    distributed: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize scalar advantages independently above/below coverage baseline."""

    if raw_advantages.shape != probability_ratios.shape or raw_advantages.shape[-1:] != (1,):
        raise ValueError(
            "raw advantages and probability ratios must share shape [steps, envs, 1], got "
            f"{tuple(raw_advantages.shape)} and {tuple(probability_ratios.shape)}."
        )
    if not torch.all(torch.isfinite(raw_advantages)):
        raise RuntimeError("V26 received non-finite raw policy advantages.")
    if not torch.all(torch.isfinite(probability_ratios)) or torch.any(probability_ratios <= 0.0):
        raise RuntimeError("V26 received invalid adaptive/coverage probability ratios.")

    raw = raw_advantages.double()
    high = probability_ratios > high_difficulty_threshold
    easy = ~high
    statistics = []
    for mask in (high, easy):
        values = raw[mask]
        statistics.extend(
            (
                values.sum(),
                values.square().sum(),
                raw.new_tensor(values.numel()),
            )
        )
    packed = torch.stack(statistics)
    if distributed:
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)

    normalized = torch.empty_like(raw_advantages)
    diagnostics: dict[str, torch.Tensor] = {}
    for group_index, (name, mask) in enumerate((("high", high), ("remaining", easy))):
        total, square_total, count = packed[group_index * 3 : group_index * 3 + 3]
        if count.item() <= 0.0:
            # No local values can exist when the global count is zero.
            diagnostics[f"{name}_count"] = count
            diagnostics[f"{name}_mean"] = total
            diagnostics[f"{name}_std"] = total
            continue
        mean = total / count
        # Match the baseline PPO's torch.std(correction=1), including when the
        # neutral group contains the entire cold-start rollout.
        variance = (square_total - total.square() / count) / (count - 1.0).clamp_min(1.0)
        variance = variance.clamp_min(0.0)
        std = variance.sqrt()
        normalized[mask] = ((raw[mask] - mean) / (std + 1.0e-8)).to(raw_advantages.dtype)
        diagnostics[f"{name}_count"] = count
        diagnostics[f"{name}_mean"] = mean
        diagnostics[f"{name}_std"] = std

    return normalized, diagnostics


class V26StarRolloutStorage(V25RolloutStorage):
    """V25 vector-value storage with per-transition STAR context and batching."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        num_reward_groups: int,
        *,
        star_priority_fraction: float,
        star_high_difficulty_threshold: float,
        star_top_fraction: float,
        star_difficulty_boundaries: Sequence[float],
        star_reuse_cap: int,
        star_priority_weight_cap: float,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            num_reward_groups,
            device,
        )
        self.star_priority_fraction = float(star_priority_fraction)
        self.star_high_difficulty_threshold = float(star_high_difficulty_threshold)
        self.star_top_fraction = float(star_top_fraction)
        self.star_difficulty_boundaries = _validate_star_parameters(
            priority_fraction=self.star_priority_fraction,
            high_difficulty_threshold=self.star_high_difficulty_threshold,
            top_fraction=self.star_top_fraction,
            difficulty_boundaries=star_difficulty_boundaries,
            reuse_cap=star_reuse_cap,
            priority_weight_cap=star_priority_weight_cap,
        )
        self.star_reuse_cap = int(star_reuse_cap)
        self.star_priority_weight_cap = float(star_priority_weight_cap)

        transition_shape = (num_transitions_per_env, num_envs, 1)
        self.raw_policy_advantages = torch.zeros(transition_shape, device=self.device)
        self.star_bin_ids = torch.full(transition_shape, -1, dtype=torch.long, device=self.device)
        self.star_probability_ratios = torch.ones(transition_shape, device=self.device)
        self._priority_pool_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self._priority_pool_weights = torch.empty(0, dtype=torch.float32, device=self.device)
        self._star_prepare_statistics = torch.zeros(
            17 + len(self.star_difficulty_boundaries) + 1,
            dtype=torch.float64,
            device=self.device,
        )
        self._star_generator_statistics = torch.zeros(4, dtype=torch.float64, device=self.device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        bin_ids = getattr(transition, "star_bin_ids", None)
        probability_ratios = getattr(transition, "star_probability_ratios", None)
        expected_shape = (self.num_envs,)
        if not isinstance(bin_ids, torch.Tensor) or bin_ids.shape != expected_shape:
            raise ValueError(
                f"V26 transition star_bin_ids must have shape {expected_shape}, got "
                f"{None if bin_ids is None else tuple(bin_ids.shape)}."
            )
        if bin_ids.dtype != torch.long or torch.any(bin_ids < 0):
            raise RuntimeError("V26 transition requires non-negative int64 global motion-bin ids.")
        if not isinstance(probability_ratios, torch.Tensor) or probability_ratios.shape != expected_shape:
            raise ValueError(
                f"V26 transition star_probability_ratios must have shape {expected_shape}, got "
                f"{None if probability_ratios is None else tuple(probability_ratios.shape)}."
            )
        if not torch.all(torch.isfinite(probability_ratios)) or torch.any(probability_ratios <= 0.0):
            raise RuntimeError("V26 transition contains invalid adaptive/coverage probability ratios.")

        step = self.step
        super().add_transition(transition)
        self.star_bin_ids[step, :, 0].copy_(bin_ids)
        self.star_probability_ratios[step, :, 0].copy_(probability_ratios)

    def prepare_star_pool(self) -> None:
        """Select positive (difficulty band, fragment) pairs and pool full fragments."""

        steps, envs = self.dones.shape[:2]
        raw = self.raw_policy_advantages[..., 0]
        ratios = self.star_probability_ratios[..., 0]
        valid = self.star_bin_ids[..., 0] >= 0
        if not torch.all(valid):
            raise RuntimeError("V26 rollout contains transitions without a global motion-bin id.")
        high = valid & (ratios > self.star_high_difficulty_threshold)

        # A done at transition t ends its fragment; transition t+1 starts the
        # next fragment.  Offset each environment's ids into a disjoint range.
        starts = torch.zeros((steps, envs), dtype=torch.long, device=self.device)
        starts[0] = 1
        if steps > 1:
            starts[1:] = self.dones[:-1, :, 0].long()
        local_fragment_ids = starts.cumsum(dim=0) - 1
        fragment_ids = local_fragment_ids + torch.arange(envs, device=self.device) * (steps + 1)
        flat_fragment_ids = fragment_ids.flatten()
        capacity = envs * (steps + 1)

        raw_flat = raw.flatten()
        ratio_flat = ratios.flatten()
        fragment_counts = torch.zeros(capacity, dtype=torch.float64, device=self.device)
        fragment_advantage_sums = torch.zeros_like(fragment_counts)
        fragment_ratio_sums = torch.zeros_like(fragment_counts)
        ones = torch.ones_like(flat_fragment_ids, dtype=torch.float64)
        fragment_counts.scatter_add_(0, flat_fragment_ids, ones)
        fragment_advantage_sums.scatter_add_(0, flat_fragment_ids, raw_flat.double())
        fragment_ratio_sums.scatter_add_(0, flat_fragment_ids, ratio_flat.double())
        fragment_mean_advantage = fragment_advantage_sums / fragment_counts.clamp_min(1.0)
        fragment_mean_ratio = fragment_ratio_sums / fragment_counts.clamp_min(1.0)

        selected_fragments = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        high_fragments = torch.zeros_like(selected_fragments)
        high_pair_count = 0
        candidate_pair_count = 0
        selected_pair_count = 0
        selected_pair_score_sum = 0.0
        unselected_candidate_score_sum = 0.0
        unselected_candidate_count = 0
        lower = self.star_high_difficulty_threshold
        band_counts: list[float] = []
        for upper in (*self.star_difficulty_boundaries, math.inf):
            band_steps = (ratio_flat > lower) & (ratio_flat <= upper)
            band_fragment_ids = flat_fragment_ids[band_steps]
            band_advantage_sums = torch.zeros(capacity, dtype=torch.float64, device=self.device)
            band_counts_by_fragment = torch.zeros_like(band_advantage_sums)
            if len(band_fragment_ids) > 0:
                band_advantage_sums.scatter_add_(
                    0,
                    band_fragment_ids,
                    raw_flat[band_steps].double(),
                )
                band_counts_by_fragment.scatter_add_(
                    0,
                    band_fragment_ids,
                    torch.ones_like(band_fragment_ids, dtype=torch.float64),
                )
            band_pairs = band_counts_by_fragment > 0.0
            high_fragments |= band_pairs
            pair_scores = band_advantage_sums / band_counts_by_fragment.clamp_min(1.0)
            candidates = band_pairs & (pair_scores > 0.0)
            candidate_indices = torch.where(candidates)[0]
            high_pair_count += int(band_pairs.sum().item())
            candidate_pair_count += len(candidate_indices)
            if len(candidate_indices) > 0:
                keep = max(int(math.ceil(self.star_top_fraction * len(candidate_indices))), 1)
                top = torch.topk(pair_scores[candidate_indices], k=keep, sorted=False).indices
                selected_indices = candidate_indices[top]
                selected_fragments[selected_indices] = True
                selected_pair_count += keep
                selected_pair_score_sum += float(pair_scores[selected_indices].sum().item())
                unselected = candidates.clone()
                unselected[selected_indices] = False
                unselected_scores = pair_scores[unselected]
                unselected_candidate_score_sum += float(unselected_scores.sum().item())
                unselected_candidate_count += unselected_scores.numel()
                band_counts.append(float(keep))
            else:
                band_counts.append(0.0)
            lower = upper

        selected_transition_mask = selected_fragments[flat_fragment_ids]
        self._priority_pool_indices = torch.where(selected_transition_mask)[0]
        if len(self._priority_pool_indices) > 0:
            # Eq. 29: every transition in a retained fragment receives that
            # fragment's mean difficulty over all of its rollout transitions.
            selected_weights = fragment_mean_ratio[flat_fragment_ids[self._priority_pool_indices]]
            self._priority_pool_weights = selected_weights.clamp(
                min=torch.finfo(torch.float32).tiny,
                max=self.star_priority_weight_cap,
            ).float()
        else:
            self._priority_pool_weights = torch.empty(0, dtype=torch.float32, device=self.device)

        actual_fragments = starts.sum().double()
        selected_fragment_advantages = fragment_mean_advantage[selected_fragments]
        selected_fragment_ratios = fragment_mean_ratio[selected_fragments]
        self._star_prepare_statistics.copy_(
            torch.tensor(
                [
                    float(steps * envs),
                    float(high.sum().item()),
                    float(actual_fragments.item()),
                    float(high_fragments.sum().item()),
                    float(high_pair_count),
                    float(candidate_pair_count),
                    float(selected_pair_count),
                    float(selected_fragments.sum().item()),
                    float(len(self._priority_pool_indices)),
                    float(selected_pair_score_sum),
                    float(selected_pair_count),
                    float(unselected_candidate_score_sum),
                    float(unselected_candidate_count),
                    float(selected_fragment_ratios.sum().item()),
                    float(selected_fragment_ratios.numel()),
                    float(selected_fragment_advantages.sum().item()),
                    float(selected_fragment_advantages.numel()),
                    *band_counts,
                ],
                dtype=torch.float64,
                device=self.device,
            )
        )

    def _batch(self, batch_idx: torch.Tensor) -> RolloutStorage.Batch:
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(parameter.flatten(0, 1) for parameter in self.distribution_params)  # type: ignore[union-attr]
        return RolloutStorage.Batch(
            observations=observations[batch_idx],
            actions=actions[batch_idx],
            values=values[batch_idx],
            advantages=advantages[batch_idx],
            returns=returns[batch_idx],
            old_actions_log_prob=old_actions_log_prob[batch_idx],
            old_distribution_params=tuple(parameter[batch_idx] for parameter in old_distribution_params),
        )

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """Yield fixed-size batches with bounded current-rollout prioritization."""

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        used_size = mini_batch_size * num_mini_batches
        requested_per_batch = int(math.floor(mini_batch_size * self.star_priority_fraction))
        requested_per_epoch = requested_per_batch * num_mini_batches

        pool_indices = self._priority_pool_indices
        pool_weights = self._priority_pool_weights
        if requested_per_epoch > 0 and len(pool_indices) > 0:
            slot_indices = pool_indices.repeat_interleave(self.star_reuse_cap)
            slot_weights = pool_weights.repeat_interleave(self.star_reuse_cap)
            available_per_epoch = len(slot_indices) // num_epochs
            actual_per_epoch = min(requested_per_epoch, available_per_epoch)
            total_priority = actual_per_epoch * num_epochs
            if total_priority > 0:
                chosen_slots = torch.multinomial(slot_weights, total_priority, replacement=False)
                priority_schedule = slot_indices[chosen_slots].reshape(num_epochs, actual_per_epoch)
            else:
                priority_schedule = torch.empty((num_epochs, 0), dtype=torch.long, device=self.device)
        else:
            actual_per_epoch = 0
            priority_schedule = torch.empty((num_epochs, 0), dtype=torch.long, device=self.device)

        flat_schedule = priority_schedule.flatten()
        if len(flat_schedule) > 0:
            _, reuse_counts = torch.unique(flat_schedule, return_counts=True)
            unique_priority = reuse_counts.numel()
            max_reuse = reuse_counts.max().item()
        else:
            unique_priority = 0
            max_reuse = 0
        self._star_generator_statistics.copy_(
            torch.tensor(
                [
                    float(requested_per_epoch * num_epochs),
                    float(len(flat_schedule)),
                    float(unique_priority),
                    float(max_reuse),
                ],
                dtype=torch.float64,
                device=self.device,
            )
        )

        for epoch in range(num_epochs):
            base_indices = torch.randperm(used_size, requires_grad=False, device=self.device)
            priority_indices = priority_schedule[epoch]
            base_cursor = 0
            priority_cursor = 0
            quotient, remainder = divmod(actual_per_epoch, num_mini_batches)
            for batch_number in range(num_mini_batches):
                priority_count = quotient + int(batch_number < remainder)
                base_count = mini_batch_size - priority_count
                batch_idx = torch.cat(
                    (
                        base_indices[base_cursor : base_cursor + base_count],
                        priority_indices[priority_cursor : priority_cursor + priority_count],
                    )
                )
                base_cursor += base_count
                priority_cursor += priority_count
                permutation = torch.randperm(len(batch_idx), device=self.device)
                yield self._batch(batch_idx[permutation])


class V26StarMultiCriticPPO(V25CompactMultiCriticPPO):
    """V25 compact multi-critic PPO with conservative STAR-lite reuse and logs."""

    def __init__(
        self,
        *args,
        star_priority_fraction: float = 0.125,
        star_high_difficulty_threshold: float = 1.0,
        star_top_fraction: float = 0.05,
        star_difficulty_boundaries: Sequence[float] = (1.5, 2.0, 4.0),
        star_reuse_cap: int = 2,
        star_priority_weight_cap: float = 8.0,
        **kwargs,
    ) -> None:
        self.star_priority_fraction = float(star_priority_fraction)
        self.star_high_difficulty_threshold = float(star_high_difficulty_threshold)
        self.star_top_fraction = float(star_top_fraction)
        self.star_difficulty_boundaries = _validate_star_parameters(
            priority_fraction=self.star_priority_fraction,
            high_difficulty_threshold=self.star_high_difficulty_threshold,
            top_fraction=self.star_top_fraction,
            difficulty_boundaries=star_difficulty_boundaries,
            reuse_cap=star_reuse_cap,
            priority_weight_cap=star_priority_weight_cap,
        )
        self.star_reuse_cap = int(star_reuse_cap)
        self.star_priority_weight_cap = float(star_priority_weight_cap)
        super().__init__(*args, **kwargs)
        if self.normalize_advantage_per_mini_batch:
            raise ValueError("V26 requires rollout-level H/E advantage normalization.")

    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> V26StarMultiCriticPPO:
        star_cfg = {
            key: cfg["algorithm"].get(key)
            for key in (
                "star_priority_fraction",
                "star_high_difficulty_threshold",
                "star_top_fraction",
                "star_difficulty_boundaries",
                "star_reuse_cap",
                "star_priority_weight_cap",
            )
        }
        _validate_distributed_contract(
            {
                "v26_star_source_sha256": _FROZEN_SOURCE_SHA256,
                "v26_star_config": star_cfg,
            }
        )
        algorithm = V25CompactMultiCriticPPO.construct_algorithm(obs, env, cfg, device)
        if not isinstance(algorithm, V26StarMultiCriticPPO):
            raise TypeError(
                "V26 constructor resolved an unexpected algorithm class: "
                f"{type(algorithm).__module__}:{type(algorithm).__qualname__}."
            )
        if algorithm.actor.is_recurrent or algorithm.critic.is_recurrent:
            raise ValueError("V26 STAR-lite currently supports feed-forward actor/critic models only.")

        old_storage = algorithm.storage
        if old_storage.step != 0:
            raise RuntimeError("V26 can only replace an empty construction-time rollout storage.")
        algorithm.storage = V26StarRolloutStorage(
            "rl",
            env.num_envs,
            old_storage.num_transitions_per_env,
            obs,
            [env.num_actions],
            algorithm.num_reward_groups,
            star_priority_fraction=algorithm.star_priority_fraction,
            star_high_difficulty_threshold=algorithm.star_high_difficulty_threshold,
            star_top_fraction=algorithm.star_top_fraction,
            star_difficulty_boundaries=algorithm.star_difficulty_boundaries,
            star_reuse_cap=algorithm.star_reuse_cap,
            star_priority_weight_cap=algorithm.star_priority_weight_cap,
            device=device,
        )
        return algorithm

    def _motion_command(self):
        command_manager = getattr(getattr(self.env, "unwrapped", self.env), "command_manager", None)
        if command_manager is None:
            raise RuntimeError("V26 requires a motion command manager on the training environment.")
        try:
            command = command_manager.get_term("motion")
        except (KeyError, ValueError) as error:
            raise RuntimeError("V26 could not resolve the motion command term.") from error
        if not hasattr(command, "snapshot_transition_sampling_context"):
            raise RuntimeError("V26 motion command does not expose adaptive sampling context.")
        return command

    def act(self, obs: TensorDict) -> torch.Tensor:
        bin_ids, probability_ratios = self._motion_command().snapshot_transition_sampling_context()
        bin_ids = bin_ids.to(self.device).detach().clone()
        probability_ratios = probability_ratios.to(self.device).detach().clone()
        actions = super().act(obs)
        self.transition.star_bin_ids = bin_ids
        self.transition.star_probability_ratios = probability_ratios
        return actions

    def compute_returns(self, obs: TensorDict) -> None:
        storage = self.storage
        if not isinstance(storage, V26StarRolloutStorage):
            raise TypeError(f"V26 requires V26StarRolloutStorage, got {type(storage).__name__}.")
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach()
        self.critic.reset(hidden_state=critic_hidden_state)
        returns, group_advantages, raw_policy_advantages = compute_grouped_gae(
            storage.rewards,
            storage.values,
            storage.dones,
            last_values,
            self.gamma,
            self.lam,
        )
        storage.returns.copy_(returns)
        storage.group_advantages.copy_(group_advantages)
        storage.raw_policy_advantages.copy_(raw_policy_advantages)
        normalized, _ = difficulty_conditioned_normalize(
            raw_policy_advantages,
            storage.star_probability_ratios,
            high_difficulty_threshold=self.star_high_difficulty_threshold,
            distributed=self.is_multi_gpu,
        )
        storage.advantages.copy_(normalized)
        storage.prepare_star_pool()

    @staticmethod
    def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        return (numerator / denominator.clamp_min(1.0)).item()

    def _compact_star_metrics(self) -> dict[str, float]:
        """Publish only the three STAR signals needed on the training dashboard.

        Detailed selection statistics remain in rollout storage for validation,
        but V26 follows V25's compact logging contract instead of exposing every
        intermediate count, score, band, and normalization statistic.
        """
        storage = self.storage
        if not isinstance(storage, V26StarRolloutStorage):
            return {}

        # Sum the population and intervention counts globally. Reuse is a
        # per-transition safety bound, so report the worst rank instead.
        counts = torch.stack(
            (
                storage._star_prepare_statistics[0],
                storage._star_prepare_statistics[1],
                storage._star_generator_statistics[1],
            )
        )
        max_reuse = storage._star_generator_statistics[3].clone()
        if self.is_multi_gpu:
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(max_reuse, op=torch.distributed.ReduceOp.MAX)

        transition_count, high_transition_count, prioritized_sample_count = counts
        return {
            "star/high_difficulty_transition_fraction": self._safe_ratio(
                high_transition_count, transition_count
            ),
            "star/actual_priority_fraction": self._safe_ratio(
                prioritized_sample_count, transition_count * self.num_learning_epochs
            ),
            "star/max_priority_reuse": max_reuse.item(),
        }

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict.update(self._compact_star_metrics())
        return loss_dict

    def _v26_metadata(self) -> dict:
        return {
            "format_version": 1,
            "implementation_sha256": _FROZEN_SOURCE_SHA256,
            "base_algorithm": "grouped_reward_compact_multi_critic",
            "advantage_normalization": "adaptive_over_coverage_high_vs_remaining",
            "fragment_score": "positive_mean_raw_summed_policy_advantage_per_band_fragment_pair",
            "priority_pool": "union_of_selected_full_done_delimited_fragments",
            "priority_weight": "whole_fragment_mean_adaptive_over_coverage_ratio",
            "priority_scope": "shared_actor_and_critic_minibatch",
            "difficulty_bands": self.star_difficulty_boundaries,
            "priority_fraction": self.star_priority_fraction,
            "high_difficulty_threshold": self.star_high_difficulty_threshold,
            "top_fraction": self.star_top_fraction,
            "reuse_cap": self.star_reuse_cap,
            "reuse_cap_scope": "extra_priority_lane_per_transition_per_update",
            "priority_weight_cap": self.star_priority_weight_cap,
            "resume_iteration": "saved_completed_iteration_plus_one",
            "logging_contract": "v25_compact_plus_three_star_metrics",
        }

    def save(self) -> dict:
        saved = super().save()
        saved["v26_star"] = self._v26_metadata()
        saved["v26_star_source"] = _FROZEN_SOURCE
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Resume an exact standalone V26 checkpoint."""

        optimizer = loaded_dict.get("optimizer_state_dict")
        optimizer_groups = optimizer.get("param_groups") if isinstance(optimizer, dict) else None
        _validate_distributed_contract(
            {
                "v26_load_cfg": load_cfg,
                "v26_checkpoint_iteration": loaded_dict.get("iter"),
                "v26_checkpoint_metadata": loaded_dict.get("v26_star"),
                "v26_grouped_critic_metadata": loaded_dict.get("v25_multi_critic"),
                "v26_archived_source_sha256": tuple(
                    _text_sha256(loaded_dict.get(key))
                    for key in (
                        "v26_star_source",
                        "v25_multi_critic_source",
                        "v25_compact_multi_critic_source",
                        "v25_rsl_rl_ppo_source",
                    )
                ),
                "v26_actor_state_schema": _state_schema(loaded_dict.get("actor_state_dict")),
                "v26_critic_state_schema": _state_schema(loaded_dict.get("critic_state_dict")),
                "v26_optimizer_group_count": (
                    len(optimizer_groups) if isinstance(optimizer_groups, list) else None
                ),
            }
        )

        load_critic = load_cfg is None or load_cfg.get("critic", False) or load_cfg.get("optimizer", False)
        if load_critic:
            if "v26_star" not in loaded_dict:
                raise ValueError("V26 is an independent scratch experiment and only resumes V26 checkpoints.")
            if loaded_dict.get("v26_star") != self._v26_metadata():
                raise ValueError("V26 STAR checkpoint metadata does not match this process.")
            if loaded_dict.get("v26_star_source") != _FROZEN_SOURCE:
                raise ValueError("V26 STAR checkpoint source does not match this process.")

        if load_cfg is None or load_cfg.get("optimizer", False):
            if "optimizer_state_dict" not in loaded_dict:
                raise ValueError("V26 full continuation requires an optimizer state.")

        load_iteration_requested = load_cfg is None or load_cfg.get("iteration", False)
        if load_iteration_requested and load_critic:
            saved_iteration = loaded_dict.get("iter")
            if not isinstance(saved_iteration, int) or saved_iteration < 0:
                raise ValueError(f"V26 checkpoint iteration must be a non-negative integer, got {saved_iteration!r}.")

        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration and load_critic:
            # RSL-RL stores the just-completed update index.  Its runner treats
            # the loaded value as the next index, so advance only in memory.
            loaded_dict["iter"] += 1
        return load_iteration
