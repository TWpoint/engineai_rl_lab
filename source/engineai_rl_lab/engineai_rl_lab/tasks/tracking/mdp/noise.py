from __future__ import annotations

import inspect
import re

import torch

from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg


def joint_uniform_noise(data: torch.Tensor, cfg: Unoise) -> torch.Tensor:
    """Apply uniform noise with optional per-joint ranges."""

    if cfg.joint_noise_scales:
        _resolve_joint_noise_ranges(data, cfg)

    if isinstance(cfg.n_max, torch.Tensor):
        cfg.n_max = cfg.n_max.to(data.device)
    if isinstance(cfg.n_min, torch.Tensor):
        cfg.n_min = cfg.n_min.to(data.device)

    noise = torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    if cfg.operation == "add":
        return data + noise
    elif cfg.operation == "scale":
        return data * noise
    elif cfg.operation == "abs":
        return noise
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def _resolve_joint_noise_ranges(data: torch.Tensor, cfg: Unoise) -> None:
    if isinstance(cfg.n_min, torch.Tensor) and isinstance(cfg.n_max, torch.Tensor):
        return

    env = _find_env_from_call_stack()
    if env is None:
        raise RuntimeError(
            "Joint-wise Unoise requires access to the active env. "
            "Use it from an IsaacLab observation term after the ObservationManager is initialized."
        )

    asset = env.scene[cfg.asset_name]
    if cfg.joint_names is None:
        joint_names = list(asset.joint_names)
    else:
        _, joint_names = asset.find_joints(cfg.joint_names, preserve_order=True)
    if data.shape[-1] != len(joint_names):
        raise ValueError(
            f"Joint-wise Unoise expected the observation last dimension to match "
            f"{cfg.asset_name}.joint_names ({len(joint_names)}), but got {data.shape[-1]}."
        )

    n_min = torch.full((len(joint_names),), float(cfg.default_n_min), dtype=data.dtype, device=data.device)
    n_max = torch.full((len(joint_names),), float(cfg.default_n_max), dtype=data.dtype, device=data.device)

    for name_id, joint_name in enumerate(joint_names):
        for joint_pattern, scale in cfg.joint_noise_scales.items():
            if re.fullmatch(joint_pattern, joint_name):
                n_min[name_id] = -float(scale)
                n_max[name_id] = float(scale)
                break

    cfg.n_min = n_min
    cfg.n_max = n_max


def _find_env_from_call_stack():
    frame = inspect.currentframe()
    while frame is not None:
        maybe_self = frame.f_locals.get("self")
        maybe_env = getattr(maybe_self, "_env", None)
        if maybe_env is not None and hasattr(maybe_env, "scene"):
            return maybe_env
        frame = frame.f_back
    return None


@configclass
class Unoise(UniformNoiseCfg):
    """Uniform noise with optional per-joint noise ranges.

    When ``joint_noise_scales`` is provided, values are matched against
    ``env.scene[asset_name].joint_names`` in order. Matched joints receive
    ``[-scale, scale]`` noise and all other joints use ``default_n_min/max``.
    """

    func = joint_uniform_noise

    asset_name: str = "robot"
    joint_names: list[str] | None = None
    """Joint names represented by the observation, or all asset joints when unset."""
    default_n_min: float = -0.5
    default_n_max: float = 0.5
    joint_noise_scales: dict[str, float] | None = None
