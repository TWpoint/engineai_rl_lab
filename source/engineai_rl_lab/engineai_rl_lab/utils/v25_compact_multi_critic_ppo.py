"""Compact logging wrapper for the V25 grouped-reward multi-critic PPO.

The optimization objective is inherited unchanged from :class:`V25MultiCriticPPO`.
This wrapper only replaces its verbose rollout diagnostics with the effective
training value loss for every reward head.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from . import v25_multi_critic_ppo as _v25_base_module
from .v25_multi_critic_ppo import V25MultiCriticPPO


def _read_source_once(path: Path) -> str:
    """Read source at module import so checkpoints never archive a later edit."""
    return path.read_text()


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


_BASE_SOURCE_PATH = Path(_v25_base_module.__file__).resolve()
_COMPACT_SOURCE_PATH = Path(__file__).resolve()
_RSL_PPO_MODULE = importlib.import_module("rsl_rl.algorithms.ppo")
_RSL_PPO_SOURCE_PATH = Path(_RSL_PPO_MODULE.__file__).resolve()
_FROZEN_BASE_SOURCE = _read_source_once(_BASE_SOURCE_PATH)
_FROZEN_COMPACT_SOURCE = _read_source_once(_COMPACT_SOURCE_PATH)
_FROZEN_RSL_PPO_SOURCE = _read_source_once(_RSL_PPO_SOURCE_PATH)
_FROZEN_BASE_SHA256 = _sha256(_FROZEN_BASE_SOURCE)
_FROZEN_COMPACT_SHA256 = _sha256(_FROZEN_COMPACT_SOURCE)
_FROZEN_RSL_PPO_SHA256 = _sha256(_FROZEN_RSL_PPO_SOURCE)
_FROZEN_IMPLEMENTATION_SHA256 = hashlib.sha256(
    (_FROZEN_BASE_SHA256 + ":" + _FROZEN_COMPACT_SHA256 + ":" + _FROZEN_RSL_PPO_SHA256).encode()
).hexdigest()


class V25CompactMultiCriticPPO(V25MultiCriticPPO):
    """V25 multi-critic PPO with V24 logs plus one value loss per head."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._head_value_loss_sum = torch.zeros(
            self.num_reward_groups,
            dtype=torch.float64,
            device=self.device,
        )
        self._head_value_loss_count = torch.zeros((), dtype=torch.float64, device=self.device)

    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> V25CompactMultiCriticPPO:
        """Validate wrapper/RSL source parity before the base construction collectives."""
        _v25_base_module._validate_distributed_contract(
            {
                "compact_implementation_sha256": _FROZEN_COMPACT_SHA256,
                "rsl_ppo_implementation_sha256": _FROZEN_RSL_PPO_SHA256,
            }
        )
        algorithm = V25MultiCriticPPO.construct_algorithm(obs, env, cfg, device)
        if not isinstance(algorithm, V25CompactMultiCriticPPO):
            raise TypeError(
                "V25 compact constructor resolved an unexpected algorithm class: "
                f"{type(algorithm).__module__}:{type(algorithm).__qualname__}."
            )
        return algorithm

    def _record_value_loss(self, effective_value_loss: torch.Tensor) -> None:
        """Accumulate the exact clipped loss tensor used by PPO for every head."""
        if effective_value_loss.ndim < 1 or effective_value_loss.shape[-1] != self.num_reward_groups:
            raise RuntimeError(
                "V25 effective value loss must end in the configured reward-head dimension: "
                f"got shape={tuple(effective_value_loss.shape)}, heads={self.num_reward_groups}."
            )
        detached_loss = effective_value_loss.detach()
        reduction_dims = tuple(range(detached_loss.ndim - 1))
        if reduction_dims:
            # Accumulate directly into float64 without materializing a full
            # float64 copy of every mini-batch loss tensor.
            per_head_sum = detached_loss.sum(dim=reduction_dims, dtype=torch.float64)
        else:
            per_head_sum = detached_loss.to(dtype=torch.float64)
        self._head_value_loss_sum.add_(per_head_sum)
        self._head_value_loss_count.add_(detached_loss.numel() // self.num_reward_groups)

    def _validate_reward_consistency(self) -> None:
        """Keep the V25 grouped/scalar reward guard without publishing diagnostics."""
        error_metrics = torch.stack((self._reward_sum_error_max, self._reward_sum_error_ratio_max))
        if self.is_multi_gpu:
            torch.distributed.all_reduce(error_metrics, op=torch.distributed.ReduceOp.MAX)
        reward_error_max, reward_error_ratio_max = error_metrics.unbind()
        reward_error_value = reward_error_max.item()
        reward_error_ratio = reward_error_ratio_max.item()
        if not math.isfinite(reward_error_ratio) or reward_error_ratio > 1.0:
            raise RuntimeError(
                "Grouped rewards do not reproduce the environment scalar reward: "
                f"max_abs_error={reward_error_value:.3e}, max_tolerance_ratio={reward_error_ratio:.3e}, "
                f"atol={self.reward_consistency_atol:.3e}, rtol={self.reward_consistency_rtol:.3e}."
            )

    def _global_head_value_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return globally weighted head means and their all-head mean."""
        statistics = torch.cat((self._head_value_loss_sum, self._head_value_loss_count.reshape(1)))
        if self.is_multi_gpu:
            torch.distributed.all_reduce(statistics, op=torch.distributed.ReduceOp.SUM)
        per_head_count = statistics[-1]
        if not torch.isfinite(per_head_count) or per_head_count.item() <= 0.0:
            raise RuntimeError(f"V25 recorded an invalid value-loss sample count: {per_head_count.item()}.")
        head_losses = statistics[:-1] / per_head_count
        return head_losses, head_losses.mean()

    def _reset_update_statistics(self) -> None:
        # The base process_env_step still fills the pure-reward accumulators.
        # They are intentionally not logged, but must be bounded across updates.
        self._pure_reward_sum.zero_()
        self._pure_reward_count.zero_()
        self._reward_sum_error_max.zero_()
        self._reward_sum_error_ratio_max.zero_()
        self._head_value_loss_sum.zero_()
        self._head_value_loss_count.zero_()

    def update(self) -> dict[str, float]:
        """Run base PPO directly and append only dynamic per-head value losses."""
        # Bypass V25MultiCriticPPO.update(), whose _group_diagnostics() method
        # produces the verbose rollout statistics this wrapper replaces.
        self._head_value_loss_sum.zero_()
        self._head_value_loss_count.zero_()
        try:
            self._validate_reward_consistency()
            loss_dict = PPO.update(self)
            head_losses, total_value_loss = self._global_head_value_losses()
            # PPO's original value is rank-local. Replace it with the exact
            # element-weighted global mean used by all value heads.
            loss_dict["value"] = total_value_loss.item()
            for index, group_name in enumerate(self.reward_group_names):
                loss_dict[f"value/{group_name}"] = head_losses[index].item()
            return loss_dict
        finally:
            self._reset_update_statistics()

    def _checkpoint_metadata(self) -> dict:
        """Describe the frozen base+wrapper implementation and training contract."""
        return {
            "format_version": 2,
            "implementation_sha256": _FROZEN_IMPLEMENTATION_SHA256,
            "base_implementation_sha256": _FROZEN_BASE_SHA256,
            "compact_implementation_sha256": _FROZEN_COMPACT_SHA256,
            "rsl_ppo_implementation_sha256": _FROZEN_RSL_PPO_SHA256,
            "reward_group_names": self.reward_group_names,
            "reward_groups": self.reward_groups,
            "critic_layout": "shared_trunk_multi_output",
            "policy_advantage_aggregation": "raw_sum_then_normalize_once",
            "value_loss_reduction": self.value_loss_reduction,
            "gradient_clipping": "separate_actor_critic",
            "logging_contract": "v24_plus_dynamic_per_head_effective_value_loss",
        }

    def save(self) -> dict:
        """Archive the sources frozen when this training process imported them."""
        # Calling PPO directly avoids the old V25 save implementation, which
        # reads its source file at checkpoint time and can therefore archive a
        # mid-run edit instead of the code actually imported by this process.
        saved = PPO.save(self)
        saved["v25_multi_critic"] = self._checkpoint_metadata()
        saved["v25_multi_critic_source"] = _FROZEN_BASE_SOURCE
        saved["v25_compact_multi_critic_source"] = _FROZEN_COMPACT_SOURCE
        saved["v25_rsl_rl_ppo_source"] = _FROZEN_RSL_PPO_SOURCE
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Reject full resumes whose archived frozen implementation differs."""
        load_critic = load_cfg is None or load_cfg.get("critic", False) or load_cfg.get("optimizer", False)
        if load_critic:
            archived_sources = (
                loaded_dict.get("v25_multi_critic_source"),
                loaded_dict.get("v25_compact_multi_critic_source"),
                loaded_dict.get("v25_rsl_rl_ppo_source"),
            )
            expected_sources = (_FROZEN_BASE_SOURCE, _FROZEN_COMPACT_SOURCE, _FROZEN_RSL_PPO_SOURCE)
            if archived_sources != expected_sources:
                raise ValueError("V25 checkpoint frozen base/compact/RSL sources do not match this process.")
        return super().load(loaded_dict, load_cfg, strict)
