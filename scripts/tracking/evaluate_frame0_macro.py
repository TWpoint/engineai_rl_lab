"""Evaluate checkpoints with equal per-motion trials starting at frame zero."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import pathlib
import pickle
import random
import sys
import types
from collections import defaultdict
from datetime import datetime

import engineai_rl_lab.tasks  # noqa: F401, E402

from isaaclab.app import add_launcher_args

from isaaclab_tasks.utils import setup_preset_cli

import cli_args as cli_args  # isort: skip


class _PositionTargetCompatibilityAdapter:
    """Expose the removed collection API through the articulation public API."""

    def __init__(self, robot):
        self._robot = robot

    def set_position_index(self, *, value, env_ids) -> None:
        self._robot.set_joint_position_target_index(target=value, env_ids=env_ids)


class _ActuatorGroupsCompatibilityDict(dict):
    """Preserve actuator-group dict behavior while serving legacy target writes."""

    def __init__(self, groups, robot):
        super().__init__(groups)
        self.target_command = _PositionTargetCompatibilityAdapter(robot)


def _install_actuator_target_compatibility(env) -> bool:
    """Install an evaluation-local bridge before the wrapper's first reset."""
    robot = env.unwrapped.scene["robot"]
    groups = robot.actuators
    if hasattr(groups, "target_command"):
        return False
    if not isinstance(groups, dict):
        raise TypeError(f"unsupported actuator collection type: {type(groups)!r}")
    robot.actuators = _ActuatorGroupsCompatibilityDict(groups, robot)
    return True


parser = argparse.ArgumentParser(description="Frame-zero per-motion macro checkpoint evaluation.")
parser.add_argument("--task", type=str, required=True, help="Registered tracking task name.")
parser.add_argument("--checkpoints", type=str, nargs="+", required=True, help="Checkpoint paths to compare.")
parser.add_argument("--motion_file", type=str, required=True, help="Source motion manifest.")
parser.add_argument("--num_motions", type=int, default=8192, help="Number of unique motions to evaluate.")
parser.add_argument("--num_envs", type=int, default=4096, help="Parallel evaluation environments.")
parser.add_argument("--trials_per_motion", type=int, default=3, help="Equal frame-zero trials per motion.")
parser.add_argument(
    "--parallel_trials",
    action="store_true",
    help="Run the equal trials for each motion in separate environments in the same batch.",
)
parser.add_argument("--eval_seed", type=int, default=20260818, help="Evaluation seed.")
parser.add_argument("--output", type=str, required=True, help="Human-readable output log path.")
parser.add_argument(
    "--env_cfg_snapshot",
    type=str,
    default=None,
    help="Optional trusted env.pkl used as the environment base for reproducible evaluation.",
)
parser.add_argument(
    "--agent_cfg_snapshot",
    type=str,
    default=None,
    help="Optional trusted agent.pkl used to restore the checkpoint's model architecture.",
)
parser.add_argument(
    "--invalid_state_snapshot_dir",
    type=str,
    default=None,
    help="Opt-in directory for atomic invalid-state and finite-runaway diagnostic snapshots.",
)
parser.add_argument(
    "--snapshot_first_control_step",
    action="store_true",
    help="Also save one deterministic first-control-step actuator/state snapshot per checkpoint.",
)
parser.add_argument(
    "--newton_num_substeps",
    type=int,
    default=None,
    help="Diagnostic override for Newton solver substeps.",
)
parser.add_argument(
    "--newton_contact_margin",
    type=float,
    default=None,
    help="Diagnostic override for Newton's default shape margin.",
)
parser.add_argument(
    "--eval_action_clip",
    type=float,
    default=None,
    help="Diagnostic override for the evaluation action clip.",
)
parser.add_argument(
    "--disable_reset_randomization",
    action="store_true",
    help="Disable startup/reset/interval randomization for a diagnostic replay only.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser, agent_library="rsl_rl")
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import numpy as np
import torch
import yaml
from engineai_rl_lab.tasks.tracking.mdp.motion_data import resolve_motion_files
from engineai_rl_lab.tasks.tracking.mdp.recorders import InvalidRobotStateRecorderManagerCfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import launch_simulation
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils.hydra import hydra_task_config


class _EnvSnapshotUnpickler(pickle.Unpickler):
    """Load saved env configs even if legacy container classes were renamed.

    RewardTermCfg and TerminationTermCfg instances still resolve to their real
    classes.  Only the two V13 dataclass-like containers are replaced with a
    namespace, since evaluation only needs their stored attributes.
    """

    _LEGACY_CONTAINERS = {
        (
            "engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v13",
            "T800SonicRewardsCfg",
        ),
        (
            "engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v13",
            "T800SonicTerminationsCfg",
        ),
    }

    def find_class(self, module: str, name: str):
        if (module, name) in self._LEGACY_CONTAINERS:
            return types.SimpleNamespace
        return super().find_class(module, name)


def _backfill_motion_defaults(motion_cfg) -> list[str]:
    """Restore no-op command defaults added after a trusted snapshot was saved.

    Pickle restores an instance's stored ``__dict__`` without adding fields
    introduced later on its config class.  Keep every saved value intact and
    fill only absent MotionCommandV1Cfg and AdaptiveSamplerV1Cfg fields from
    fresh current default instances.  The v22 snapshots predate optional
    catalog/sharding and multi-error difficulty fields; their defaults disable
    those features and therefore preserve the snapshot's original behavior.
    """

    added = []
    motion_defaults = type(motion_cfg)()
    for name, value in vars(motion_defaults).items():
        if name == "adaptive_sampling":
            continue
        if not hasattr(motion_cfg, name):
            setattr(motion_cfg, name, copy.deepcopy(value))
            added.append(f"motion.{name}")

    sampler_cfg = motion_cfg.adaptive_sampling
    sampler_defaults = type(sampler_cfg)()
    for name, value in vars(sampler_defaults).items():
        if not hasattr(sampler_cfg, name):
            setattr(sampler_cfg, name, copy.deepcopy(value))
            added.append(f"adaptive_sampling.{name}")
    return added


def _sanitize_rsl_rl_cfg(cfg: dict) -> dict:
    legacy_model_keys = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    for model_name in ("actor", "critic"):
        model_cfg = cfg.get(model_name)
        if isinstance(model_cfg, dict):
            for key in legacy_model_keys:
                model_cfg.pop(key, None)
    algorithm_cfg = cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict) and "sonic_kl_adaptation" in algorithm_cfg:
        # The v22 snapshot predates the local rsl_rl rename performed after
        # training launched.  Preserve the saved boolean under its current
        # equivalent name so OnPolicyRunner can construct PPO for inference.
        algorithm_cfg.setdefault("shared_kl_adaptation", algorithm_cfg["sonic_kl_adaptation"])
        algorithm_cfg.pop("sonic_kl_adaptation")
    return cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_concrete_newton_overrides(physics_cfg, *, num_substeps: int | None, contact_margin: float | None) -> None:
    """Apply replay overrides to Hydra's already-selected concrete physics config."""
    physics_type = f"{type(physics_cfg).__module__}.{type(physics_cfg).__qualname__}"
    if num_substeps is not None:
        if num_substeps <= 0:
            raise ValueError("--newton_num_substeps must be positive")
        if not hasattr(physics_cfg, "num_substeps"):
            raise ValueError(
                "--newton_num_substeps requires physics=newton_mjwarp; "
                f"resolved physics config {physics_type!r} has no num_substeps"
            )
        physics_cfg.num_substeps = num_substeps
    if contact_margin is not None:
        if contact_margin < 0.0:
            raise ValueError("--newton_contact_margin must be non-negative")
        shape_cfg = getattr(physics_cfg, "default_shape_cfg", None)
        if shape_cfg is None or not hasattr(shape_cfg, "margin"):
            raise ValueError(
                "--newton_contact_margin requires physics=newton_mjwarp; "
                f"resolved physics config {physics_type!r} has no default_shape_cfg.margin"
            )
        shape_cfg.margin = contact_margin


def _apply_diagnostic_overrides(env_cfg, agent_cfg) -> None:
    """Apply opt-in replay controls without changing the saved training MDP."""
    _apply_concrete_newton_overrides(
        env_cfg.sim.physics,
        num_substeps=args_cli.newton_num_substeps,
        contact_margin=args_cli.newton_contact_margin,
    )
    if args_cli.eval_action_clip is not None:
        if args_cli.eval_action_clip <= 0.0:
            raise ValueError("--eval_action_clip must be positive")
        agent_cfg.clip_actions = args_cli.eval_action_clip
    if args_cli.disable_reset_randomization:
        env_cfg.commands.motion.pose_range = {}
        env_cfg.commands.motion.velocity_range = {}
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
        for event_name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
    if args_cli.snapshot_first_control_step and args_cli.invalid_state_snapshot_dir is None:
        raise ValueError("--snapshot_first_control_step requires --invalid_state_snapshot_dir")
    if args_cli.invalid_state_snapshot_dir is not None:
        recorder_cfg = InvalidRobotStateRecorderManagerCfg()
        recorder_cfg.invalid_robot_state.snapshot_dir = args_cli.invalid_state_snapshot_dir
        recorder_cfg.invalid_robot_state.capture_first_control_step = args_cli.snapshot_first_control_step
        env_cfg.recorders = recorder_cfg


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"p{percent}": float(np.quantile(values, percent / 100.0)) for percent in (10, 50, 90)}


def _source_name(path: str) -> str:
    parts = pathlib.Path(path).parts
    for marker in ("t800_datasets", "motions"):
        try:
            return parts[parts.index(marker) + 1]
        except (ValueError, IndexError):
            continue
    return "other"


def _load_deleted_v24_multi_critic_compat() -> None:
    """Load the exact locally-versioned v24 classes if their source was removed mid-evaluation."""
    module_names = (
        "engineai_rl_lab.tasks.tracking.multi_critic_env",
        "engineai_rl_lab.utils.multi_critic_ppo",
    )
    try:
        importlib.import_module(module_names[-1])
        return
    except ModuleNotFoundError as exc:
        if exc.name not in module_names:
            raise

    repo_root = pathlib.Path(__file__).resolve().parents[2]

    # The same workspace cleanup that removed the v24 algorithm also reverted
    # RolloutStorage's multi-value-head allocation.  Load that exact historical
    # class only into this evaluation process before importing MultiCriticPPO.
    storage_archive = (
        repo_root.parent
        / "rsl_rl"
        / ".stversions"
        / "rsl_rl"
        / "storage"
        / "rollout_storage~20260826-185932.py"
    )
    if not storage_archive.is_file():
        raise ModuleNotFoundError(f"Missing v24 RolloutStorage compatibility source: {storage_archive}")
    storage_spec = importlib.util.spec_from_file_location("_frame0_v24_rollout_storage", storage_archive)
    if storage_spec is None or storage_spec.loader is None:
        raise ImportError(f"Cannot load v24 RolloutStorage from {storage_archive}")
    storage_module = importlib.util.module_from_spec(storage_spec)
    storage_spec.loader.exec_module(storage_module)
    import rsl_rl.storage as storage_package

    storage_package.RolloutStorage = storage_module.RolloutStorage
    print(f"[FRAME0] Loaded v24 multi-head RolloutStorage from local history: {storage_archive}")

    archive_root = repo_root / ".stversions" / "source" / "engineai_rl_lab" / "engineai_rl_lab"
    patterns = (
        archive_root / "tasks" / "tracking" / "multi_critic_env~*.py",
        archive_root / "utils" / "multi_critic_ppo~*.py",
    )
    for module_name, pattern in zip(module_names, patterns, strict=True):
        candidates = sorted(pattern.parent.glob(pattern.name))
        if not candidates:
            raise ModuleNotFoundError(
                f"Missing {module_name} and no local version-history copy exists at {pattern}"
            )
        source = candidates[-1]
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load compatibility module {module_name} from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print(f"[FRAME0] Loaded missing v24 compatibility module from local history: {source}")


def _evaluate_checkpoint(
    env,
    agent_cfg,
    checkpoint: pathlib.Path,
    trials_per_motion: int,
    seed: int,
    parallel_trials: bool = False,
) -> dict:
    unwrapped = env.unwrapped
    # Recorder filenames and per-context first-step budgets remain unambiguous
    # when several checkpoints share one evaluator/environment process.
    unwrapped.invalid_state_snapshot_context = checkpoint.stem
    command = unwrapped.command_manager.get_term("motion")
    termination_manager = unwrapped.termination_manager
    device = unwrapped.device
    num_envs = unwrapped.num_envs
    num_motions = command.motion.num_motions
    error_names = tuple(name for name in command.metrics if name.startswith("error_"))
    termination_names = tuple(termination_manager.active_terms)
    success_termination_names = tuple(name for name in ("motion_time_out", "time_out") if name in termination_names)
    failure_termination_names = tuple(name for name in termination_names if name not in success_termination_names)
    if not success_termination_names:
        raise RuntimeError(f"No success timeout termination found in active terms: {termination_names}")

    sanitized_agent_cfg = _sanitize_rsl_rl_cfg(agent_cfg.to_dict())
    algorithm_class = str(sanitized_agent_cfg.get("algorithm", {}).get("class_name", ""))
    if algorithm_class == "engineai_rl_lab.utils.multi_critic_ppo:MultiCriticPPO":
        _load_deleted_v24_multi_critic_compat()

    runner = OnPolicyRunner(
        env,
        sanitized_agent_cfg,
        log_dir=None,
        device=device,
    )
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=device)

    attempts = torch.zeros(num_motions, dtype=torch.long, device=device)
    successes = torch.zeros_like(attempts)
    failed_attempts = torch.zeros_like(attempts)
    termination_counts = {name: torch.zeros_like(attempts) for name in termination_names}
    episode_return_sums = torch.zeros(num_motions, dtype=torch.float64, device=device)
    episode_length_sums = torch.zeros(num_motions, dtype=torch.float64, device=device)
    step_counts = torch.zeros(num_motions, dtype=torch.long, device=device)
    step_reward_sums = torch.zeros(num_motions, dtype=torch.float64, device=device)
    # Accumulate every tracking metric in one 2-D tensor.  The previous loop
    # launched one index_add kernel per metric and per simulation step, which
    # is especially expensive for the low-environment, long-motion shards.
    error_sums = torch.zeros((num_motions, len(error_names)), dtype=torch.float64, device=device)
    body_pos_failure_details = []

    envs_per_motion = trials_per_motion if parallel_trials else 1
    motions_per_batch = num_envs // envs_per_motion
    if motions_per_batch <= 0:
        raise ValueError(
            f"num_envs={num_envs} is too small for {envs_per_motion} parallel trial environments per motion"
        )
    batch_count = (num_motions + motions_per_batch - 1) // motions_per_batch
    for batch_index, batch_start in enumerate(range(0, num_motions, motions_per_batch)):
        batch_end = min(batch_start + motions_per_batch, num_motions)
        batch_motion_count = batch_end - batch_start
        batch_size = batch_motion_count * envs_per_motion
        assigned_motion_ids = torch.full((num_envs,), batch_start, dtype=torch.long, device=device)
        assigned_motion_ids[:batch_size] = torch.arange(
            batch_start, batch_end, device=device
        ).repeat_interleave(envs_per_motion)
        command.set_fixed_motion_ids(assigned_motion_ids)

        _seed_everything(seed + batch_index)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        # ManagerBasedEnv.reset() forwards the articulation but does not run a
        # command-manager update.  Synchronize robot-aligned reference caches
        # before the first physics step, since v23 rewards/terminations consume
        # them before CommandManager.compute() runs at the end of that step.
        refresh_relative_targets = getattr(command, "refresh_relative_body_targets", None)
        if refresh_relative_targets is not None:
            refresh_relative_targets()
        elif "end_effector_height" in termination_names:
            raise RuntimeError(
                "The active end_effector_height termination requires a command implementation "
                "that can refresh frame-zero relative body targets after reset"
            )
        if hasattr(policy, "reset"):
            policy.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

        active = torch.arange(num_envs, device=device) < batch_size
        trial_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        episode_returns = torch.zeros(num_envs, dtype=torch.float64, device=device)
        episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
        per_trial_step_limits = []
        if "motion_time_out" in success_termination_names:
            batch_motion_ids = torch.arange(batch_start, batch_end, device=device)
            per_trial_step_limits.append(int(command.motion.lengths(batch_motion_ids).max().item()))
        if "time_out" in success_termination_names:
            per_trial_step_limits.append(int(unwrapped.max_episode_length))
        max_trial_steps = min(per_trial_step_limits) + 2
        required_trials_per_env = 1 if parallel_trials else trials_per_motion
        max_batch_steps = required_trials_per_env * max_trial_steps + 2
        print(
            f"[FRAME0] {checkpoint.name} batch {batch_index + 1}/{batch_count} starting: "
            f"motions={batch_motion_count}, active_envs={batch_size}, "
            f"max_trial_steps={max_trial_steps}, max_batch_steps={max_batch_steps}"
        )

        with torch.inference_mode():
            for batch_step in range(max_batch_steps):
                if not torch.any(active):
                    break
                active_before_step = active.clone()
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                rewards = rewards.view(-1).double()
                dones = dones.view(-1).bool()
                if hasattr(policy, "reset"):
                    policy.reset(dones)

                active_env_ids = torch.where(active_before_step)[0]
                active_motion_ids = assigned_motion_ids[active_env_ids]
                ones = torch.ones_like(active_motion_ids, dtype=torch.long)
                step_counts.index_add_(0, active_motion_ids, ones)
                step_reward_sums.index_add_(0, active_motion_ids, rewards[active_env_ids])
                episode_returns[active_env_ids] += rewards[active_env_ids]
                episode_lengths[active_env_ids] += 1
                step_errors = torch.stack(
                    [command.metrics[name][active_env_ids] for name in error_names],
                    dim=1,
                ).double()
                error_sums.index_add_(0, active_motion_ids, step_errors)

                done_env_ids = torch.where(dones & active_before_step)[0]
                if done_env_ids.numel() == 0:
                    continue
                done_motion_ids = assigned_motion_ids[done_env_ids]
                done_terms = {
                    name: termination_manager.get_term(name)[done_env_ids].bool() for name in termination_names
                }
                if "body_pos" in done_terms and hasattr(command, "last_global_body_pos_errors"):
                    for env_id in done_env_ids[done_terms["body_pos"]].tolist():
                        errors = command.last_global_body_pos_errors[env_id]
                        max_index = int(torch.argmax(errors).item())
                        body_pos_failure_details.append(
                            {
                                "motion_id": int(assigned_motion_ids[env_id].item()),
                                "body": command.last_global_body_pos_error_names[max_index],
                                "error": float(errors[max_index].item()),
                                "error_xyz": [
                                    float(value)
                                    for value in command.last_global_body_pos_error_vectors[env_id, max_index].tolist()
                                ],
                                "motion_step": int(command.last_global_body_pos_error_time_steps[env_id].item()),
                            }
                        )
                failed = torch.zeros_like(done_env_ids, dtype=torch.bool)
                for name in failure_termination_names:
                    failed |= done_terms[name]
                timed_out = torch.zeros_like(done_env_ids, dtype=torch.bool)
                for name in success_termination_names:
                    timed_out |= done_terms[name]
                succeeded = timed_out & ~failed

                attempts.index_add_(0, done_motion_ids, torch.ones_like(done_motion_ids))
                successes.index_add_(0, done_motion_ids, succeeded.long())
                failed_attempts.index_add_(0, done_motion_ids, failed.long())
                for name in termination_names:
                    termination_counts[name].index_add_(0, done_motion_ids, done_terms[name].long())
                episode_return_sums.index_add_(0, done_motion_ids, episode_returns[done_env_ids])
                episode_length_sums.index_add_(0, done_motion_ids, episode_lengths[done_env_ids].double())

                trial_counts[done_env_ids] += 1
                episode_returns[done_env_ids] = 0.0
                episode_lengths[done_env_ids] = 0
                finished_env_ids = done_env_ids[
                    trial_counts[done_env_ids] >= required_trials_per_env
                ]
                active[finished_env_ids] = False
            else:
                raise RuntimeError(
                    f"Batch {batch_index + 1}/{batch_count} did not finish within {max_batch_steps} steps; "
                    f"active={int(active.sum().item())}, trial_count_range="
                    f"[{int(trial_counts[active].min().item())}, {int(trial_counts[active].max().item())}]"
                )

        print(
            f"[FRAME0] {checkpoint.name} batch {batch_index + 1}/{batch_count}: "
            f"motions={batch_motion_count}, steps={batch_step + 1}"
        )

    command.clear_fixed_motion_ids()
    attempts_cpu = attempts.cpu().numpy()
    if not np.all(attempts_cpu == trials_per_motion):
        raise RuntimeError(
            f"Expected exactly {trials_per_motion} attempts per motion, got "
            f"range [{attempts_cpu.min()}, {attempts_cpu.max()}]"
        )

    successes_cpu = successes.cpu().numpy()
    rates = successes_cpu / attempts_cpu
    step_counts_cpu = step_counts.cpu().numpy()
    error_sums_cpu = error_sums.cpu().numpy()
    per_motion_errors = {
        name: error_sums_cpu[:, index] / step_counts_cpu for index, name in enumerate(error_names)
    }
    paths = [command._motion_files[int(global_id)] for global_id in command.motion.global_ids.cpu().tolist()]
    lengths = command.motion.lengths(torch.arange(num_motions, device=device)).cpu().numpy()
    failed_attempts_cpu = failed_attempts.cpu().numpy()
    termination_counts_cpu = {name: values.cpu().numpy() for name, values in termination_counts.items()}
    episode_return_sums_cpu = episode_return_sums.cpu().numpy()
    episode_length_sums_cpu = episode_length_sums.cpu().numpy()
    step_reward_sums_cpu = step_reward_sums.cpu().numpy()

    per_motion = []
    for motion_id in range(num_motions):
        per_motion.append(
            {
                "motion_id": motion_id,
                "file": paths[motion_id],
                "source": _source_name(paths[motion_id]),
                "frames": int(lengths[motion_id]),
                "attempts": int(attempts_cpu[motion_id]),
                "successes": int(successes_cpu[motion_id]),
                "success_rate": float(rates[motion_id]),
                "failed_attempts": int(failed_attempts_cpu[motion_id]),
                "termination_counts": {name: int(values[motion_id]) for name, values in termination_counts_cpu.items()},
                "mean_episode_reward": float(episode_return_sums_cpu[motion_id] / attempts_cpu[motion_id]),
                "mean_episode_length": float(episode_length_sums_cpu[motion_id] / attempts_cpu[motion_id]),
                "mean_step_reward": float(step_reward_sums_cpu[motion_id] / step_counts_cpu[motion_id]),
                "tracking_errors": {name: float(per_motion_errors[name][motion_id]) for name in error_names},
            }
        )

    source_ids: dict[str, list[int]] = defaultdict(list)
    for motion_id, path in enumerate(paths):
        source_ids[_source_name(path)].append(motion_id)
    source_summary = {}
    for source, ids in sorted(source_ids.items()):
        source_rates = rates[np.asarray(ids)]
        source_summary[source] = {
            "motions": len(ids),
            "macro_success_rate": float(source_rates.mean()),
            "zero_success_motion_fraction": float(np.mean(source_rates == 0.0)),
            "all_success_motion_fraction": float(np.mean(source_rates == 1.0)),
            "success_rate_quantiles": _quantiles(source_rates),
        }

    error_summary = {}
    for index, (name, values) in enumerate(per_motion_errors.items()):
        error_summary[name] = {
            "macro_mean": float(values.mean()),
            "transition_weighted_mean": float(error_sums[:, index].sum().item() / step_counts.sum().item()),
            **_quantiles(values),
        }

    worst = sorted(
        per_motion,
        key=lambda item: (
            item["success_rate"],
            -item["failed_attempts"],
            item["mean_episode_length"],
        ),
    )[:50]
    total_attempts = attempts.sum().item()
    termination_rates = {
        name: float(values.sum().item() / total_attempts) for name, values in termination_counts.items()
    }
    result = {
        "checkpoint": str(checkpoint),
        "motions": num_motions,
        "trials_per_motion": trials_per_motion,
        "episodes": int(attempts.sum().item()),
        "macro_success_rate": float(rates.mean()),
        "micro_success_rate": float(successes.sum().item() / attempts.sum().item()),
        "success_rate_quantiles": _quantiles(rates),
        "zero_success_motions": int(np.sum(rates == 0.0)),
        "zero_success_motion_fraction": float(np.mean(rates == 0.0)),
        "all_success_motions": int(np.sum(rates == 1.0)),
        "all_success_motion_fraction": float(np.mean(rates == 1.0)),
        "partial_success_motions": int(np.sum((rates > 0.0) & (rates < 1.0))),
        "failure_rate": float(failed_attempts.sum().item() / total_attempts),
        "failure_termination_names": list(failure_termination_names),
        "success_termination_names": list(success_termination_names),
        "termination_rates": termination_rates,
        "macro_mean_episode_reward": float((episode_return_sums_cpu / attempts_cpu).mean()),
        "macro_mean_episode_length": float((episode_length_sums_cpu / attempts_cpu).mean()),
        "macro_mean_step_reward": float((step_reward_sums_cpu / step_counts_cpu).mean()),
        "tracking_error_summary": error_summary,
        "source_summary": source_summary,
        "worst_motions": worst,
        "per_motion": per_motion,
        "body_pos_failure_details": body_pos_failure_details,
    }

    del policy, runner
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _format_result(result: dict) -> list[str]:
    quantiles = result["success_rate_quantiles"]
    lines = [
        f"Checkpoint: {result['checkpoint']}",
        f"  motions: {result['motions']:,}",
        f"  equal trials per motion: {result['trials_per_motion']}",
        f"  episodes: {result['episodes']:,}",
        f"  macro success rate: {result['macro_success_rate']:.6f}",
        f"  per-motion success P10/P50/P90: {quantiles['p10']:.6f}/{quantiles['p50']:.6f}/{quantiles['p90']:.6f}",
        f"  zero-success motions: {result['zero_success_motions']:,} ({result['zero_success_motion_fraction']:.6f})",
        f"  partial-success motions: {result['partial_success_motions']:,}",
        f"  all-success motions: {result['all_success_motions']:,} ({result['all_success_motion_fraction']:.6f})",
        f"  episode failure rate (union): {result['failure_rate']:.6f}",
        f"  macro mean episode length: {result['macro_mean_episode_length']:.3f}",
        f"  macro mean episode reward: {result['macro_mean_episode_reward']:.6f}",
        f"  macro mean step reward: {result['macro_mean_step_reward']:.6f}",
        "  termination rates (may overlap):",
    ]
    for name, rate in result["termination_rates"].items():
        role = "success" if name in result["success_termination_names"] else "failure"
        lines.append(f"    {name} [{role}]: {rate:.6f}")
    lines.append("  tracking errors (macro mean / P50 / P90):")
    for name, summary in result["tracking_error_summary"].items():
        lines.append(f"    {name}: {summary['macro_mean']:.6f} / {summary['p50']:.6f} / {summary['p90']:.6f}")
    lines.append("  source summary:")
    for source, summary in result["source_summary"].items():
        lines.append(
            f"    {source}: motions={summary['motions']}, macro={summary['macro_success_rate']:.6f}, "
            f"zero={summary['zero_success_motion_fraction']:.6f}"
        )
    lines.append("  worst motions:")
    for item in result["worst_motions"][:20]:
        failure_terms = ", ".join(
            f"{name}={count}"
            for name, count in item["termination_counts"].items()
            if name in result["failure_termination_names"] and count
        )
        lines.append(
            f"    success={item['success_rate']:.4f} failures={item['failed_attempts']}/{item['attempts']} "
            f"length={item['mean_episode_length']:.1f} [{failure_terms}] {item['file']}"
        )
    return lines


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point", play_mode=True)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.num_motions <= 0 or args_cli.num_envs <= 0 or args_cli.trials_per_motion <= 0:
        raise ValueError("num_motions, num_envs, and trials_per_motion must be positive")
    env_cfg_snapshot = None
    agent_cfg_snapshot = None
    if args_cli.env_cfg_snapshot is not None:
        snapshot_path = pathlib.Path(args_cli.env_cfg_snapshot).resolve()
        if not snapshot_path.is_file():
            raise FileNotFoundError(snapshot_path)
        with snapshot_path.open("rb") as snapshot_file:
            # The path is explicitly supplied and must be a trusted local artifact.
            snapshot_cfg = _EnvSnapshotUnpickler(snapshot_file).load()  # noqa: S301
        for config_name in (
            "observations",
            "actions",
            "commands",
            "events",
            "rewards",
            "terminations",
            "curriculum",
        ):
            setattr(env_cfg, config_name, getattr(snapshot_cfg, config_name))
        for group_cfg in vars(env_cfg.observations).values():
            if hasattr(group_cfg, "enable_corruption"):
                group_cfg.enable_corruption = False
        added_motion_fields = _backfill_motion_defaults(env_cfg.commands.motion)
        if added_motion_fields:
            print(
                "[FRAME0] Backfilled post-snapshot motion defaults: "
                + ", ".join(added_motion_fields)
            )
        env_cfg_snapshot = str(snapshot_path)
        print(f"[FRAME0] Restored environment config from {snapshot_path}")
    if args_cli.agent_cfg_snapshot is not None:
        snapshot_path = pathlib.Path(args_cli.agent_cfg_snapshot).resolve()
        if not snapshot_path.is_file():
            raise FileNotFoundError(snapshot_path)
        with snapshot_path.open("rb") as snapshot_file:
            # The path is explicitly supplied and must be a trusted local artifact.
            agent_cfg = pickle.load(snapshot_file)  # noqa: S301
        agent_cfg_snapshot = str(snapshot_path)
        print(f"[FRAME0] Restored agent config from {snapshot_path}")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    _apply_diagnostic_overrides(env_cfg, agent_cfg)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg.seed = args_cli.eval_seed
    agent_cfg.device = env_cfg.sim.device

    output_path = pathlib.Path(args_cli.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_motion_file = os.path.abspath(args_cli.motion_file)
    source_motion_files = resolve_motion_files(source_motion_file)
    if len(source_motion_files) > args_cli.num_motions:
        selected_motion_files = random.Random(args_cli.eval_seed).sample(source_motion_files, args_cli.num_motions)
        evaluation_manifest = output_path.with_name(output_path.stem + "_motions.yaml")
        evaluation_manifest.write_text(
            yaml.safe_dump({"files": selected_motion_files}, sort_keys=False),
            encoding="utf-8",
        )
    else:
        selected_motion_files = source_motion_files
        evaluation_manifest = pathlib.Path(source_motion_file)
    effective_num_envs = args_cli.num_envs
    if not args_cli.parallel_trials:
        # Sequential trials need at most one environment per motion.
        effective_num_envs = min(effective_num_envs, len(selected_motion_files))
    env_cfg.scene.num_envs = effective_num_envs
    if len(selected_motion_files) > effective_num_envs and len(selected_motion_files) % effective_num_envs != 0:
        print("[WARN] Final motion batch will contain inactive padding environments.")

    env_cfg.commands.motion.motion_file = str(evaluation_manifest)
    env_cfg.commands.motion.uniform_sampling_rate = 1.0
    env_cfg.commands.motion.pre_failure_sample_window = 0
    env_cfg.commands.motion.max_num_load_motions = len(selected_motion_files)
    # Trusted snapshots created before fixed-joint support do not carry this
    # newly-added config field.  An empty mapping exactly preserves the old
    # command behavior while allowing the current command implementation to
    # consume the historical snapshot.
    if not hasattr(env_cfg.commands.motion, "fixed_joint_positions"):
        env_cfg.commands.motion.fixed_joint_positions = {}
    # Older trusted config snapshots predate the explicit playback override.
    # Preserve their original behavior before forcing frame-zero starts below.
    if not hasattr(env_cfg.commands.motion, "playback_start_frame"):
        env_cfg.commands.motion.playback_start_frame = None
    env_cfg.commands.motion.start_at_motion_beginning = True
    env_cfg.commands.motion.debug_vis = False
    for model_name in ("actor", "critic"):
        model_cfg = getattr(agent_cfg, model_name, None)
        if model_cfg is not None and not hasattr(model_cfg, "stochastic"):
            model_cfg.stochastic = False
    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    checkpoints = [pathlib.Path(path).resolve() for path in args_cli.checkpoints]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    _seed_everything(args_cli.eval_seed)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        if _install_actuator_target_compatibility(env):
            print("[FRAME0] Installed evaluation-local actuator target compatibility bridge.")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        command = env.unwrapped.command_manager.get_term("motion")
        if command.motion.num_motions != len(selected_motion_files):
            raise RuntimeError(f"Loaded {command.motion.num_motions} motions, expected {len(selected_motion_files)}")

        results = []
        for checkpoint in checkpoints:
            print(f"[FRAME0] Evaluating {checkpoint.name}...")
            results.append(
                _evaluate_checkpoint(
                    env,
                    agent_cfg,
                    checkpoint,
                    args_cli.trials_per_motion,
                    args_cli.eval_seed,
                    args_cli.parallel_trials,
                )
            )

        report = [
            "Frame-zero per-motion macro checkpoint evaluation",
            f"Generated: {datetime.now().astimezone().isoformat()}",
            f"Task: {args_cli.task}",
            f"Environment config snapshot: {env_cfg_snapshot}",
            f"Agent config snapshot: {agent_cfg_snapshot}",
            f"Source motion file: {source_motion_file}",
            f"Evaluation manifest: {evaluation_manifest}",
            f"Seed: {args_cli.eval_seed}",
            f"Motions: {len(selected_motion_files)}",
            f"Environments: {effective_num_envs}",
            f"Equal trials per motion: {args_cli.trials_per_motion}",
            f"Trial execution: {'parallel environments' if args_cli.parallel_trials else 'sequential per environment'}",
            "Frame-zero reset synchronization: relative body targets refreshed before first physics step (v1)",
            "Episode start: frame zero",
            "Success: reaches motion end or task horizon without any active failure termination",
            "Reset/event randomization: task defaults retained",
            "Observation corruption: disabled by play_mode",
            "Aggregation: equal-weight per-motion macro average",
            "",
        ]
        for result in results:
            report.extend(_format_result(result))
            report.append("")
        output_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        output_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "protocol": "frame_zero_equal_trials_per_motion",
                    "task": args_cli.task,
                    "environment_config_snapshot": env_cfg_snapshot,
                    "agent_config_snapshot": agent_cfg_snapshot,
                    "source_motion_file": source_motion_file,
                    "evaluation_manifest": str(evaluation_manifest),
                    "seed": args_cli.eval_seed,
                    "num_motions": len(selected_motion_files),
                    "num_envs": effective_num_envs,
                    "trials_per_motion": args_cli.trials_per_motion,
                    "parallel_trials": args_cli.parallel_trials,
                    "frame_zero_reset_sync_version": 1,
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[FRAME0] Wrote {output_path}")
        env.close()


if __name__ == "__main__":
    main()
