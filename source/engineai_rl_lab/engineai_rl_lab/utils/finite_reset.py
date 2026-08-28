from __future__ import annotations

import math
from typing import Any

import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _is_finite_log_value(value: Any) -> bool:
    """Return whether a scalar or tensor log value is entirely finite."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def drop_nonfinite_episode_reward_logs(extras: dict[str, Any]) -> int:
    """Drop invalid episodic reward summaries without inventing replacement values.

    Isaac Lab accumulates reward terms before it resets a terminal environment.
    When physics diverges, those internal episode sums can therefore contain
    NaN/Inf even though the terminal scalar reward is sanitized for PPO.  A bad
    scalar poisons the iteration-wide logger average, so omit only those samples
    and expose a diagnostic count instead.
    """
    log = extras.get("log")
    if not isinstance(log, dict):
        return 0

    invalid_keys = [
        key for key, value in log.items() if key.startswith("Episode_Reward/") and not _is_finite_log_value(value)
    ]
    for key in invalid_keys:
        del log[key]
    if invalid_keys:
        log["Diagnostics/nonfinite_episode_reward_terms"] = float(len(invalid_keys))
    return len(invalid_keys)


class FiniteResetRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Keep diverged terminal transitions finite and make their logs truthful."""

    def step(self, actions: torch.Tensor):
        observations, rewards, dones, extras = super().step(actions)
        # ManagerBasedRLEnv retains its previous ``extras['log']`` dictionary on
        # steps without a reset. Return a fresh object so RSL-RL neither samples
        # stale values nor observes later in-place mutations through old refs.
        extras = dict(extras)
        log = extras.get("log")
        extras["log"] = dict(log) if bool(dones.bool().any().item()) and isinstance(log, dict) else {}
        # Reward computation precedes same-step reset. Sanitize only terminal
        # values; a non-terminal non-finite reward must still fail the runner's
        # environment-output check.
        rewards = torch.where(dones.bool() & ~torch.isfinite(rewards), torch.zeros_like(rewards), rewards)
        drop_nonfinite_episode_reward_logs(extras)
        return observations, rewards, dones, extras
