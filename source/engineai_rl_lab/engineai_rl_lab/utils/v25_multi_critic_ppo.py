"""V25 grouped-reward multi-critic PPO.

The critic predicts one value per reward group, but the actor still optimizes
the original scalar reward objective: raw per-group advantages are summed
before the single normalization and single PPO clipping operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict


def _implementation_sha256() -> str:
    """Hash the exact V25 algorithm source used to create a run/checkpoint."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _config_sha256(cfg: Mapping) -> str:
    """Hash rank-invariant model/algorithm fields from the resolved runner config."""
    contract_cfg = {
        "actor": cfg["actor"],
        "critic": cfg["critic"],
        "algorithm": cfg["algorithm"],
        "obs_groups": cfg["obs_groups"],
        "num_steps_per_env": cfg["num_steps_per_env"],
        "torch_compile_mode": cfg.get("torch_compile_mode"),
    }
    serialized = json.dumps(contract_cfg, sort_keys=True, separators=(",", ":"), default=repr).encode()
    return hashlib.sha256(serialized).hexdigest()


def _validate_distributed_contract(local_contract: Mapping) -> None:
    """Make every rank accept or reject the same V25 launch contract."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        error = local_contract.get("reward_schema_error")
        if error is not None:
            raise RuntimeError(f"Unable to inspect the V25 reward schema: {error}.")
        return

    gathered_contracts: list[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered_contracts, dict(local_contract))
    reference = gathered_contracts[0]
    mismatched_ranks = [rank for rank, contract in enumerate(gathered_contracts) if contract != reference]
    if mismatched_ranks:
        # Every rank receives the same ordered list and therefore raises here
        # together, instead of leaving matching ranks blocked in a later
        # collective when just one stale rank exits.
        raise RuntimeError(
            "V25 launch contract differs across ranks; "
            f"mismatched_ranks={mismatched_ranks}, contracts={gathered_contracts!r}."
        )
    error = local_contract.get("reward_schema_error")
    if error is not None:
        raise RuntimeError(f"Unable to inspect the V25 reward schema on every rank: {error}.")


def resolve_reward_groups(
    reward_terms: Sequence[str],
    configured_groups: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Validate an ordered, mutually-exclusive and exhaustive reward partition."""
    ordered_terms = tuple(reward_terms)
    if not ordered_terms:
        raise ValueError("V25 multi-critic PPO requires at least one active reward term.")
    if len(set(ordered_terms)) != len(ordered_terms):
        raise ValueError(f"Active reward terms contain duplicates: {ordered_terms}.")
    if not configured_groups:
        raise ValueError("V25 multi-critic PPO requires at least one configured reward group.")

    group_names = tuple(configured_groups)
    if any(not name for name in group_names):
        raise ValueError(f"Reward group names must be non-empty, got {group_names}.")

    available = set(ordered_terms)
    owners: dict[str, str] = {}
    groups: dict[str, tuple[str, ...]] = {}
    unknown: dict[str, list[str]] = {}
    duplicates: dict[str, tuple[str, str]] = {}
    for group_name, terms in configured_groups.items():
        group_terms = tuple(terms)
        if not group_terms:
            raise ValueError(f"Reward group {group_name!r} must contain at least one term.")
        groups[group_name] = group_terms
        for term in group_terms:
            if term not in available:
                unknown.setdefault(group_name, []).append(term)
            if term in owners:
                duplicates[term] = (owners[term], group_name)
            else:
                owners[term] = group_name

    if unknown:
        raise ValueError(f"Reward groups contain unknown terms {unknown}; active terms are {ordered_terms}.")
    if duplicates:
        raise ValueError(f"Reward terms must belong to exactly one group; duplicates: {duplicates}.")

    uncovered = tuple(term for term in ordered_terms if term not in owners)
    if uncovered:
        raise ValueError(f"Reward groups do not cover active terms: {uncovered}.")
    return group_names, groups


def aggregate_grouped_rewards(
    reward_terms: Mapping[str, torch.Tensor],
    group_names: Sequence[str],
    reward_groups: Mapping[str, Sequence[str]],
    *,
    num_envs: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Aggregate already-weighted per-term rewards into ``[num_envs, num_groups]``."""
    missing = {
        group_name: tuple(term for term in reward_groups[group_name] if term not in reward_terms)
        for group_name in group_names
    }
    missing = {name: terms for name, terms in missing.items() if terms}
    if missing:
        raise RuntimeError(f"Current RewardManager output is missing configured terms: {missing}.")

    columns: list[torch.Tensor] = []
    for group_name in group_names:
        group_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
        for term_name in reward_groups[group_name]:
            term_reward = reward_terms[term_name].to(device=device, dtype=torch.float32).reshape(num_envs)
            group_reward.add_(term_reward)
        columns.append(group_reward)
    return torch.stack(columns, dim=-1)


def compute_grouped_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-group GAE and the raw summed scalar policy advantage."""
    if rewards.shape != values.shape:
        raise ValueError(f"Reward/value shapes must match, got {rewards.shape} and {values.shape}.")
    if last_values.shape != values.shape[1:]:
        raise ValueError(f"Last-value shape must be {values.shape[1:]}, got {last_values.shape}.")
    if dones.shape != (*rewards.shape[:2], 1):
        raise ValueError(f"Done shape must be {(*rewards.shape[:2], 1)}, got {dones.shape}.")

    returns = torch.zeros_like(rewards)
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(rewards.shape[0])):
        next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        next_is_not_terminal = 1.0 - dones[step].float()
        delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        returns[step] = advantage + values[step]

    group_advantages = returns - values
    policy_advantages = group_advantages.sum(dim=-1, keepdim=True)
    return returns, group_advantages, policy_advantages


def bootstrap_grouped_timeouts(
    rewards: torch.Tensor,
    values: torch.Tensor,
    time_outs: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Add the matching value head to each timed-out reward channel."""
    if rewards.shape != values.shape:
        raise ValueError(f"Timeout reward/value shapes must match, got {rewards.shape} and {values.shape}.")
    expected_time_out_shape = (*rewards.shape[:-1], 1)
    if time_outs.shape != expected_time_out_shape:
        raise ValueError(f"Timeout shape must be {expected_time_out_shape}, got {time_outs.shape}.")
    return rewards + gamma * values * time_outs.to(device=rewards.device, dtype=rewards.dtype)


class V25RolloutStorage(RolloutStorage):
    """Rollout storage with vector rewards/values and scalar policy advantages."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        num_reward_groups: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        if training_type != "rl":
            raise ValueError("V25RolloutStorage only supports reinforcement learning.")
        if num_reward_groups < 1:
            raise ValueError(f"num_reward_groups must be positive, got {num_reward_groups}.")
        self.num_reward_groups = int(num_reward_groups)
        grouped_shape = (num_transitions_per_env, num_envs, self.num_reward_groups)
        self.rewards = torch.zeros(grouped_shape, device=self.device)
        self.values = torch.zeros(grouped_shape, device=self.device)
        self.returns = torch.zeros(grouped_shape, device=self.device)
        self.group_advantages = torch.zeros(grouped_shape, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

    @staticmethod
    def _require_shape(tensor: torch.Tensor, expected: torch.Size, name: str) -> torch.Tensor:
        if tensor.shape != expected:
            raise ValueError(f"Transition {name} must have shape {tuple(expected)}, got {tuple(tensor.shape)}.")
        return tensor

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Store one transition without collapsing its reward/value head dimension."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! Call clear() before adding more transitions.")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)  # type: ignore[arg-type]
        rewards = self._require_shape(transition.rewards, self.rewards[self.step].shape, "rewards")  # type: ignore[arg-type]
        values = self._require_shape(transition.values, self.values[self.step].shape, "values")  # type: ignore[arg-type]
        self.rewards[self.step].copy_(rewards)
        self.values[self.step].copy_(values)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))  # type: ignore[union-attr]
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))  # type: ignore[union-attr]

        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(self.num_transitions_per_env, *parameter.shape, device=self.device)
                for parameter in transition.distribution_params  # type: ignore[union-attr]
            )
        for index, parameter in enumerate(transition.distribution_params):  # type: ignore[union-attr]
            self.distribution_params[index][self.step].copy_(parameter)

        self._save_hidden_states(transition.hidden_states)
        self.step += 1


class V25MultiCriticPPO(PPO):
    """Multi-head value decomposition with one unchanged scalar PPO objective."""

    def __init__(
        self,
        *args,
        env: VecEnv,
        reward_group_names: Sequence[str],
        reward_groups: Mapping[str, Sequence[str]],
        value_loss_reduction: str = "mean",
        reward_consistency_atol: float = 2.0e-6,
        reward_consistency_rtol: float = 1.0e-5,
        **kwargs,
    ) -> None:
        if value_loss_reduction != "mean":
            raise ValueError(
                "V25 uses mean-over-head value loss to keep the critic loss scale independent of head count; "
                f"got {value_loss_reduction!r}."
            )
        if reward_consistency_atol <= 0.0:
            raise ValueError(f"reward_consistency_atol must be positive, got {reward_consistency_atol}.")
        if reward_consistency_rtol < 0.0:
            raise ValueError(f"reward_consistency_rtol must be non-negative, got {reward_consistency_rtol}.")
        super().__init__(*args, **kwargs)
        self.env = env
        self.reward_group_names = tuple(reward_group_names)
        self.reward_groups = {name: tuple(reward_groups[name]) for name in self.reward_group_names}
        self.value_loss_reduction = value_loss_reduction
        self.reward_consistency_atol = float(reward_consistency_atol)
        self.reward_consistency_rtol = float(reward_consistency_rtol)
        self.num_reward_groups = len(self.reward_group_names)
        self._pure_reward_sum = torch.zeros(self.num_reward_groups, dtype=torch.float64, device=self.device)
        self._pure_reward_count = torch.zeros((), dtype=torch.float64, device=self.device)
        self._reward_sum_error_max = torch.zeros((), dtype=torch.float64, device=self.device)
        self._reward_sum_error_ratio_max = torch.zeros((), dtype=torch.float64, device=self.device)

    @staticmethod
    def _reward_manager(env: VecEnv):
        reward_manager = getattr(env, "reward_manager", None)
        if reward_manager is None:
            reward_manager = getattr(getattr(env, "unwrapped", None), "reward_manager", None)
        if reward_manager is None:
            raise RuntimeError("V25 multi-critic PPO requires an Isaac Lab RewardManager on env or env.unwrapped.")
        return reward_manager

    @classmethod
    def _active_reward_terms(cls, env: VecEnv) -> tuple[str, ...]:
        reward_manager = cls._reward_manager(env)
        term_names = getattr(reward_manager, "_term_names", None)
        if term_names is None:
            term_names = getattr(reward_manager, "active_terms", None)
        if term_names is None:
            raise RuntimeError("RewardManager exposes neither _term_names nor active_terms.")
        return tuple(term_names)

    def _reward_terms_for_step(self) -> dict[str, torch.Tensor]:
        reward_manager = self._reward_manager(self.env)
        term_names = getattr(reward_manager, "_term_names", None)
        step_reward = getattr(reward_manager, "_step_reward", None)
        base_env = getattr(self.env, "unwrapped", self.env)
        step_dt = getattr(base_env, "step_dt", None)
        if term_names is None or step_reward is None or step_dt is None:
            raise RuntimeError(
                "V25 requires RewardManager._term_names, RewardManager._step_reward and env.step_dt "
                "to reconstruct weighted per-step rewards."
            )
        if step_reward.shape != (self.storage.num_envs, len(term_names)):
            raise RuntimeError(
                "RewardManager._step_reward shape does not match the active reward schema: "
                f"got {tuple(step_reward.shape)}, expected {(self.storage.num_envs, len(term_names))}."
            )
        return {name: step_reward[:, index] * step_dt for index, name in enumerate(term_names)}

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Store grouped rewards and bootstrap timeout rewards head by head."""
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            raise RuntimeError("V25 multi-critic PPO does not support rnd_cfg.")

        grouped_rewards = aggregate_grouped_rewards(
            self._reward_terms_for_step(),
            self.reward_group_names,
            self.reward_groups,
            num_envs=self.storage.num_envs,
            device=self.device,
        )
        # ``FiniteResetRslRlVecEnvWrapper`` zeroes a non-finite scalar reward
        # on terminal reset steps. Mirror that behavior for every reward head;
        # non-terminal non-finite rewards remain untouched and are rejected by
        # the runner's normal finite-value guard.
        done_mask = dones.to(device=self.device, dtype=torch.bool).reshape(self.storage.num_envs, 1)
        terminal_nonfinite = done_mask & ~torch.isfinite(grouped_rewards).all(dim=-1, keepdim=True)
        grouped_rewards = torch.where(terminal_nonfinite, torch.zeros_like(grouped_rewards), grouped_rewards)
        scalar_rewards = rewards.to(device=self.device, dtype=torch.float32).reshape(self.storage.num_envs, -1)
        if scalar_rewards.shape[1] != 1:
            raise ValueError(
                f"Environment scalar rewards must reshape to ({self.storage.num_envs}, 1), got {tuple(rewards.shape)}."
            )
        reward_errors = (grouped_rewards.sum(dim=-1, keepdim=True) - scalar_rewards).abs()
        reward_tolerances = self.reward_consistency_atol + self.reward_consistency_rtol * scalar_rewards.abs()
        # The runner calls this method under ``torch.inference_mode()``. Keep
        # the persistent accumulators as the normal tensors allocated in
        # ``__init__`` instead of rebinding them to inference tensors, which
        # cannot later be reset from the gradient-enabled update phase.
        self._reward_sum_error_max.copy_(torch.maximum(self._reward_sum_error_max, reward_errors.max().double()))
        self._reward_sum_error_ratio_max.copy_(
            torch.maximum(
                self._reward_sum_error_ratio_max,
                (reward_errors / reward_tolerances).max().double(),
            )
        )

        self._pure_reward_sum.add_(grouped_rewards.double().sum(dim=0))
        self._pure_reward_count.add_(grouped_rewards.shape[0])
        self.transition.rewards = grouped_rewards.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(self.device, dtype=torch.float32).reshape(self.storage.num_envs, -1)
            if time_outs.shape[1] != 1:
                raise ValueError(
                    f"extras['time_outs'] must reshape to ({self.storage.num_envs}, 1), "
                    f"got {tuple(extras['time_outs'].shape)}."
                )
            self.transition.rewards = bootstrap_grouped_timeouts(
                self.transition.rewards,
                self.transition.values,  # type: ignore[arg-type]
                time_outs,
                self.gamma,
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Fit one return per head, then form one raw summed policy advantage."""
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach()
        self.critic.reset(hidden_state=critic_hidden_state)
        returns, group_advantages, policy_advantages = compute_grouped_gae(
            self.storage.rewards,
            self.storage.values,
            self.storage.dones,
            last_values,
            self.gamma,
            self.lam,
        )
        self.storage.returns.copy_(returns)
        self.storage.group_advantages.copy_(group_advantages)
        self.storage.advantages.copy_(policy_advantages)
        if not self.normalize_advantage_per_mini_batch:
            self.storage.advantages.copy_(self._normalize_advantages(self.storage.advantages))

    def _clip_gradients(self) -> torch.Tensor:
        """Clip actor and critic separately so value heads cannot shrink the actor update."""
        actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        return torch.maximum(actor_norm, critic_norm)

    def _group_diagnostics(self) -> dict[str, float]:
        storage_count = float(self.storage.num_transitions_per_env * self.storage.num_envs)
        group_advantages = self.storage.group_advantages.double()
        statistics = torch.cat(
            (
                self._pure_reward_sum,
                self.storage.returns.double().sum(dim=(0, 1)),
                self.storage.values.double().sum(dim=(0, 1)),
                group_advantages.sum(dim=(0, 1)),
                group_advantages.square().sum(dim=(0, 1)),
                (self.storage.values.double() - self.storage.returns.double()).square().sum(dim=(0, 1)),
                self._pure_reward_count.reshape(1),
                torch.tensor([storage_count], dtype=torch.float64, device=self.device),
            )
        )
        reward_error_metrics = torch.stack((self._reward_sum_error_max, self._reward_sum_error_ratio_max))
        if self.is_multi_gpu:
            torch.distributed.all_reduce(statistics, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reward_error_metrics, op=torch.distributed.ReduceOp.MAX)
        reward_error_max, reward_error_ratio_max = reward_error_metrics.unbind()
        reward_error_value = reward_error_max.item()
        reward_error_ratio = reward_error_ratio_max.item()
        if not math.isfinite(reward_error_ratio) or reward_error_ratio > 1.0:
            raise RuntimeError(
                "Grouped rewards do not reproduce the environment scalar reward: "
                f"max_abs_error={reward_error_value:.3e}, max_tolerance_ratio={reward_error_ratio:.3e}, "
                f"atol={self.reward_consistency_atol:.3e}, rtol={self.reward_consistency_rtol:.3e}."
            )

        width = self.num_reward_groups
        reward_sum, return_sum, value_sum, advantage_sum, advantage_square_sum, value_error_sum = (
            statistics[index * width : (index + 1) * width] for index in range(6)
        )
        reward_count = statistics[6 * width].clamp_min(1.0)
        rollout_count = statistics[6 * width + 1].clamp_min(1.0)
        advantage_mean = advantage_sum / rollout_count
        advantage_variance = advantage_square_sum / rollout_count - advantage_mean.square()

        diagnostics: dict[str, float] = {
            "reward_group_sum_error_max": reward_error_value,
            "reward_group_sum_error_tolerance_ratio_max": reward_error_ratio,
        }
        for index, group_name in enumerate(self.reward_group_names):
            diagnostics[f"group_reward/{group_name}"] = (reward_sum[index] / reward_count).item()
            diagnostics[f"group_return/{group_name}"] = (return_sum[index] / rollout_count).item()
            diagnostics[f"group_value/{group_name}"] = (value_sum[index] / rollout_count).item()
            diagnostics[f"group_advantage_mean/{group_name}"] = advantage_mean[index].item()
            diagnostics[f"group_advantage_std/{group_name}"] = advantage_variance[index].clamp_min(0.0).sqrt().item()
            diagnostics[f"group_value_mse_before_update/{group_name}"] = (value_error_sum[index] / rollout_count).item()
        return diagnostics

    def update(self) -> dict[str, float]:
        diagnostics = self._group_diagnostics()
        loss_dict = super().update()
        loss_dict.update(diagnostics)
        self._pure_reward_sum.zero_()
        self._pure_reward_count.zero_()
        self._reward_sum_error_max.zero_()
        self._reward_sum_error_ratio_max.zero_()
        return loss_dict

    def _checkpoint_metadata(self) -> dict:
        return {
            "format_version": 1,
            "implementation_sha256": _implementation_sha256(),
            "reward_group_names": self.reward_group_names,
            "reward_groups": self.reward_groups,
            "critic_layout": "shared_trunk_multi_output",
            "policy_advantage_aggregation": "raw_sum_then_normalize_once",
            "value_loss_reduction": self.value_loss_reduction,
            "gradient_clipping": "separate_actor_critic",
        }

    def save(self) -> dict:
        saved = super().save()
        saved["v25_multi_critic"] = self._checkpoint_metadata()
        # V25 is currently experiment-local code. Archive the exact source so
        # an untracked workspace file cannot make a successful run impossible
        # to reconstruct later.
        saved["v25_multi_critic_source"] = Path(__file__).read_text()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_critic = load_cfg is None or load_cfg.get("critic", False) or load_cfg.get("optimizer", False)
        if load_critic:
            expected = self._checkpoint_metadata()
            actual = loaded_dict.get("v25_multi_critic")
            if actual != expected:
                raise ValueError(
                    "V25 critic/optimizer checkpoint implementation or reward contract does not match this run: "
                    f"checkpoint={actual!r}, expected={expected!r}."
                )
        return super().load(loaded_dict, load_cfg, strict)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> V25MultiCriticPPO:
        """Construct a shared-trunk critic with one output per reward group."""
        # RSL-RL passes the runner's live config dictionary here. Resolve and
        # pop constructor-only fields from a private copy so logging and
        # checkpoint metadata keep the complete V25 reward schema.
        cfg = deepcopy(cfg)
        config_sha256 = _config_sha256(cfg)
        configured_groups = deepcopy(cfg.get("algorithm", {}).get("reward_groups"))
        try:
            active_reward_terms = V25MultiCriticPPO._active_reward_terms(env)
            reward_schema_error = None
        except Exception as error:  # noqa: BLE001 - synchronize this failure across distributed ranks
            active_reward_terms = ()
            reward_schema_error = f"{type(error).__name__}: {error}"

        # Validate the immutable contract before resolving local callables or
        # observation groups. A stale rank therefore fails with every other
        # rank here instead of exiting alone before a later collective.
        _validate_distributed_contract(
            {
                "implementation_sha256": _implementation_sha256(),
                "config_sha256": config_sha256,
                "active_reward_terms": active_reward_terms,
                "configured_reward_groups": configured_groups,
                "reward_schema_error": reward_schema_error,
            }
        )

        alg_class: type[V25MultiCriticPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore[assignment]
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[assignment]

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("V25 multi-critic PPO does not support rnd_cfg; configure rnd_cfg=None.")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        cfg["algorithm"].pop("reward_groups", None)
        group_names, reward_groups = resolve_reward_groups(active_reward_terms, configured_groups)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]
        critic = critic_class(obs, cfg["obs_groups"], "critic", len(group_names), **cfg["critic"]).to(device)
        print(f"V25 Multi-Critic Model {group_names}: {critic}")

        storage = V25RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            len(group_names),
            device,
        )
        algorithm = alg_class(
            actor,
            critic,
            storage,
            env=env,
            reward_group_names=group_names,
            reward_groups=reward_groups,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm
