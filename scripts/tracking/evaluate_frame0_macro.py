"""Evaluate checkpoints with equal per-motion trials starting at frame zero."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as metadata
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


parser = argparse.ArgumentParser(description="Frame-zero per-motion macro checkpoint evaluation.")
parser.add_argument("--task", type=str, required=True, help="Registered tracking task name.")
parser.add_argument("--checkpoints", type=str, nargs="+", required=True, help="Checkpoint paths to compare.")
parser.add_argument("--motion_file", type=str, required=True, help="Source motion manifest.")
parser.add_argument("--num_motions", type=int, default=8192, help="Number of unique motions to evaluate.")
parser.add_argument("--num_envs", type=int, default=4096, help="Parallel evaluation environments.")
parser.add_argument("--trials_per_motion", type=int, default=3, help="Equal frame-zero trials per motion.")
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
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser, agent_library="rsl_rl")
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import numpy as np
import torch
import yaml
from engineai_rl_lab.tasks.tracking.mdp.motion_data import resolve_motion_files
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


def _sanitize_rsl_rl_cfg(cfg: dict) -> dict:
    legacy_model_keys = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    for model_name in ("actor", "critic"):
        model_cfg = cfg.get(model_name)
        if isinstance(model_cfg, dict):
            for key in legacy_model_keys:
                model_cfg.pop(key, None)
    return cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _evaluate_checkpoint(env, agent_cfg, checkpoint: pathlib.Path, trials_per_motion: int, seed: int) -> dict:
    unwrapped = env.unwrapped
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

    runner = OnPolicyRunner(
        env,
        _sanitize_rsl_rl_cfg(agent_cfg.to_dict()),
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
    error_sums = {name: torch.zeros(num_motions, dtype=torch.float64, device=device) for name in error_names}
    body_pos_failure_details = []

    batch_count = (num_motions + num_envs - 1) // num_envs
    for batch_index, batch_start in enumerate(range(0, num_motions, num_envs)):
        batch_end = min(batch_start + num_envs, num_motions)
        batch_size = batch_end - batch_start
        assigned_motion_ids = torch.full((num_envs,), batch_start, dtype=torch.long, device=device)
        assigned_motion_ids[:batch_size] = torch.arange(batch_start, batch_end, device=device)
        command.set_fixed_motion_ids(assigned_motion_ids)

        _seed_everything(seed + batch_index)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
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
        max_batch_steps = trials_per_motion * max_trial_steps + 2
        print(
            f"[FRAME0] {checkpoint.name} batch {batch_index + 1}/{batch_count} starting: "
            f"motions={batch_size}, max_trial_steps={max_trial_steps}, max_batch_steps={max_batch_steps}"
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
                for name in error_names:
                    error_sums[name].index_add_(
                        0,
                        active_motion_ids,
                        command.metrics[name][active_env_ids].double(),
                    )

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
                finished_env_ids = done_env_ids[trial_counts[done_env_ids] >= trials_per_motion]
                active[finished_env_ids] = False
            else:
                raise RuntimeError(
                    f"Batch {batch_index + 1}/{batch_count} did not finish within {max_batch_steps} steps; "
                    f"active={int(active.sum().item())}, trial_count_range="
                    f"[{int(trial_counts[active].min().item())}, {int(trial_counts[active].max().item())}]"
                )

        print(
            f"[FRAME0] {checkpoint.name} batch {batch_index + 1}/{batch_count}: "
            f"motions={batch_size}, steps={batch_step + 1}"
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
    per_motion_errors = {name: error_sums[name].cpu().numpy() / step_counts_cpu for name in error_names}
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
    for name, values in per_motion_errors.items():
        error_summary[name] = {
            "macro_mean": float(values.mean()),
            "transition_weighted_mean": float(error_sums[name].sum().item() / step_counts.sum().item()),
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
    env_cfg.scene.num_envs = args_cli.num_envs
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
    if len(selected_motion_files) > args_cli.num_envs and len(selected_motion_files) % args_cli.num_envs != 0:
        print("[WARN] Final motion batch will contain inactive padding environments.")

    env_cfg.commands.motion.motion_file = str(evaluation_manifest)
    env_cfg.commands.motion.uniform_sampling_rate = 1.0
    env_cfg.commands.motion.pre_failure_sample_window = 0
    env_cfg.commands.motion.max_num_load_motions = len(selected_motion_files)
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
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        command = env.unwrapped.command_manager.get_term("motion")
        if command.motion.num_motions != len(selected_motion_files):
            raise RuntimeError(f"Loaded {command.motion.num_motions} motions, expected {len(selected_motion_files)}")

        results = []
        for checkpoint in checkpoints:
            print(f"[FRAME0] Evaluating {checkpoint.name}...")
            results.append(
                _evaluate_checkpoint(env, agent_cfg, checkpoint, args_cli.trials_per_motion, args_cli.eval_seed)
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
            f"Environments: {args_cli.num_envs}",
            f"Equal trials per motion: {args_cli.trials_per_motion}",
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
                    "num_envs": args_cli.num_envs,
                    "trials_per_motion": args_cli.trials_per_motion,
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
