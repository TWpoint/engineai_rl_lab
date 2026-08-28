"""Compare tracking checkpoints on a fixed, uniform motion distribution."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as metadata
import json
import os
import pathlib
import random
import sys
from collections import defaultdict
from datetime import datetime

import engineai_rl_lab.tasks  # noqa: F401, E402

from isaaclab.app import add_launcher_args

from isaaclab_tasks.utils import setup_preset_cli

import cli_args as cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Compare RSL-RL checkpoints with uniform motion sampling.")
parser.add_argument("--task", type=str, required=True, help="Registered tracking task name.")
parser.add_argument("--checkpoints", type=str, nargs="+", required=True, help="Checkpoint paths to compare.")
parser.add_argument("--motion_file", type=str, default=None, help="Optional evaluation motion manifest.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel evaluation environments.")
parser.add_argument("--eval_steps", type=int, default=2000, help="Policy steps evaluated per checkpoint.")
parser.add_argument("--working_set_size", type=int, default=1024, help="Fixed motion working-set size.")
parser.add_argument("--eval_seed", type=int, default=20260818, help="Seed used for the fixed evaluation set.")
parser.add_argument(
    "--start_at_motion_beginning",
    action="store_true",
    help="Reset every episode to frame zero instead of sampling a random motion time.",
)
parser.add_argument("--output", type=str, required=True, help="Path for the human-readable evaluation log.")
parser.add_argument(
    "--invalid_state_snapshot_dir",
    type=str,
    default=None,
    help="Opt-in directory for invalid_robot_state and finite-runaway diagnostic snapshots.",
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


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _evaluate_checkpoint(env, agent_cfg, checkpoint: pathlib.Path, eval_steps: int, seed: int) -> dict:
    _seed_everything(seed)
    env.unwrapped.invalid_state_snapshot_context = checkpoint.stem
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    runner = OnPolicyRunner(
        env,
        _sanitize_rsl_rl_cfg(agent_cfg.to_dict()),
        log_dir=None,
        device=env.unwrapped.device,
    )
    # Evaluation only consumes the actor.  Loading the whole algorithm makes
    # historical multi-critic checkpoints depend on their deleted critic/PPO
    # implementation even though none of it participates in inference.
    checkpoint_data = torch.load(checkpoint, weights_only=False, map_location=env.unwrapped.device)
    runner.alg.get_policy().load_state_dict(checkpoint_data["actor_state_dict"], strict=True)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    command = unwrapped.command_manager.get_term("motion")
    termination_manager = unwrapped.termination_manager
    term_names = tuple(termination_manager.active_terms)
    metric_names = tuple(command.metrics)

    term_counts = {name: 0 for name in term_names}
    metric_sums = {name: 0.0 for name in metric_names}
    reward_sum = 0.0
    transition_count = 0
    completed_episode_returns: list[float] = []
    completed_episode_lengths: list[float] = []
    episode_returns = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
    episode_lengths = torch.zeros(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
    per_motion_episodes: dict[int, int] = defaultdict(int)
    per_motion_body_pos_failures: dict[int, int] = defaultdict(int)

    with torch.inference_mode():
        for _ in range(eval_steps):
            motion_ids_before_step = command.motion.global_ids[command.motion_ids].clone()
            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

            rewards = rewards.view(-1)
            dones = dones.bool().view(-1)
            episode_returns += rewards
            episode_lengths += 1
            reward_sum += rewards.double().sum().item()
            transition_count += rewards.numel()

            for name in term_names:
                term_counts[name] += int(termination_manager.get_term(name).sum().item())
            for name in metric_names:
                metric_sums[name] += float(command.metrics[name].double().mean().item())

            done_ids = torch.where(dones)[0]
            if done_ids.numel() > 0:
                completed_episode_returns.extend(episode_returns[done_ids].cpu().tolist())
                completed_episode_lengths.extend(episode_lengths[done_ids].cpu().tolist())
                ended_motion_ids = motion_ids_before_step[done_ids].cpu().tolist()
                body_pos_done = termination_manager.get_term("body_pos")[done_ids].cpu().tolist()
                for motion_id, failed in zip(ended_motion_ids, body_pos_done, strict=True):
                    per_motion_episodes[int(motion_id)] += 1
                    per_motion_body_pos_failures[int(motion_id)] += int(failed)
                episode_returns[done_ids] = 0.0
                episode_lengths[done_ids] = 0

    total_episodes = int(sum(per_motion_episodes.values()))
    body_pos_failures = int(term_counts.get("body_pos", 0))
    invalid_failures = int(term_counts.get("invalid_robot_state", 0))
    successful_episodes = max(total_episodes - body_pos_failures - invalid_failures, 0)
    worst_motions = []
    for motion_id, episodes in per_motion_episodes.items():
        if episodes < 3:
            continue
        failures = per_motion_body_pos_failures[motion_id]
        worst_motions.append(
            {
                "global_motion_id": motion_id,
                "file": command._motion_files[motion_id],
                "episodes": episodes,
                "body_pos_failures": failures,
                "body_pos_failure_rate": failures / episodes,
            }
        )
    worst_motions.sort(key=lambda item: (item["body_pos_failure_rate"], item["episodes"]), reverse=True)

    result = {
        "checkpoint": str(checkpoint),
        "eval_steps": eval_steps,
        "transitions": transition_count,
        "episodes": total_episodes,
        "success_rate": successful_episodes / total_episodes if total_episodes else float("nan"),
        "body_pos_failure_rate": body_pos_failures / total_episodes if total_episodes else float("nan"),
        "mean_step_reward": reward_sum / transition_count,
        "mean_episode_reward": _safe_mean(completed_episode_returns),
        "mean_episode_length": _safe_mean(completed_episode_lengths),
        "termination_counts": term_counts,
        "motion_metrics": {name: value / eval_steps for name, value in metric_sums.items()},
        "worst_body_pos_motions": worst_motions[:10],
    }

    del policy, runner
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _format_result(result: dict) -> list[str]:
    metrics = result["motion_metrics"]
    lines = [
        f"Checkpoint: {result['checkpoint']}",
        f"  transitions: {result['transitions']:,}",
        f"  completed episodes: {result['episodes']:,}",
        f"  success rate: {result['success_rate']:.6f}",
        f"  body_pos failure rate: {result['body_pos_failure_rate']:.6f}",
        f"  mean step reward: {result['mean_step_reward']:.6f}",
        f"  mean episode reward: {result['mean_episode_reward']:.6f}",
        f"  mean episode length: {result['mean_episode_length']:.3f}",
        f"  termination counts: {json.dumps(result['termination_counts'], sort_keys=True)}",
    ]
    for name in (
        "error_anchor_pos",
        "error_anchor_rot",
        "error_body_pos",
        "error_body_rot",
        "error_body_lin_vel",
        "error_body_ang_vel",
        "error_joint_pos",
        "error_joint_vel",
    ):
        if name in metrics:
            lines.append(f"  {name}: {metrics[name]:.6f}")
    lines.append("  worst body_pos motions:")
    for item in result["worst_body_pos_motions"]:
        lines.append(
            f"    {item['body_pos_failure_rate']:.4f} ({item['body_pos_failures']}/{item['episodes']}) {item['file']}"
        )
    return lines


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point", play_mode=True)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"
    agent_cfg.device = env_cfg.sim.device
    env_cfg.seed = args_cli.eval_seed
    output_path = pathlib.Path(args_cli.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_motion_file = (
        os.path.abspath(args_cli.motion_file)
        if args_cli.motion_file is not None
        else env_cfg.commands.motion.motion_file
    )
    source_motion_files = resolve_motion_files(source_motion_file)
    if len(source_motion_files) > args_cli.working_set_size:
        manifest_rng = random.Random(args_cli.eval_seed)
        selected_motion_files = manifest_rng.sample(source_motion_files, args_cli.working_set_size)
        eval_manifest_path = output_path.with_name(output_path.stem + "_motions.yaml")
        eval_manifest_path.write_text(
            yaml.safe_dump({"files": selected_motion_files}, sort_keys=False),
            encoding="utf-8",
        )
        env_cfg.commands.motion.motion_file = str(eval_manifest_path)
    else:
        selected_motion_files = source_motion_files
        env_cfg.commands.motion.motion_file = source_motion_file

    env_cfg.commands.motion.uniform_sampling_rate = 1.0
    env_cfg.commands.motion.pre_failure_sample_window = 0
    env_cfg.commands.motion.max_num_load_motions = args_cli.working_set_size
    env_cfg.commands.motion.start_at_motion_beginning = args_cli.start_at_motion_beginning
    env_cfg.commands.motion.debug_vis = False
    if args_cli.invalid_state_snapshot_dir is not None:
        recorder_cfg = InvalidRobotStateRecorderManagerCfg()
        recorder_cfg.invalid_robot_state.snapshot_dir = os.path.abspath(args_cli.invalid_state_snapshot_dir)
        env_cfg.recorders = recorder_cfg
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
        active_global_ids = command.motion.global_ids.cpu().tolist()
        results = []
        for checkpoint in checkpoints:
            print(f"[EVAL] Evaluating {checkpoint.name} for {args_cli.eval_steps} steps...")
            result = _evaluate_checkpoint(env, agent_cfg, checkpoint, args_cli.eval_steps, args_cli.eval_seed)
            results.append(result)
            print(
                f"[EVAL] {checkpoint.name}: success={result['success_rate']:.4f}, "
                f"body_pos={result['body_pos_failure_rate']:.4f}, "
                f"episode_length={result['mean_episode_length']:.1f}"
            )

        report = [
            "Fixed uniform checkpoint evaluation",
            f"Generated: {datetime.now().astimezone().isoformat()}",
            f"Task: {args_cli.task}",
            f"Source motion file: {source_motion_file}",
            f"Fixed evaluation manifest: {env_cfg.commands.motion.motion_file}",
            f"Seed: {args_cli.eval_seed}",
            f"Environments: {args_cli.num_envs}",
            f"Evaluation steps per checkpoint: {args_cli.eval_steps}",
            f"Working set requested/loaded: {args_cli.working_set_size}/{len(active_global_ids)}",
            "Sampling: 100% uniform, no pre-failure offset, no working-set rotation",
            "Reset/event randomization: task defaults retained",
            "Observation corruption: disabled by play_mode",
            "Success definition: completed episode without body_pos or invalid_robot_state termination",
            "Episode start: " + ("frame zero" if args_cli.start_at_motion_beginning else "random sampled motion time"),
            "",
        ]
        for result in results:
            report.extend(_format_result(result))
            report.append("")

        if len(results) == 2:
            first, second = results
            report.extend(
                [
                    f"Delta ({pathlib.Path(second['checkpoint']).name} - {pathlib.Path(first['checkpoint']).name}):",
                    f"  success rate: {second['success_rate'] - first['success_rate']:+.6f}",
                    f"  body_pos failure rate: {second['body_pos_failure_rate'] - first['body_pos_failure_rate']:+.6f}",
                    f"  mean episode length: {second['mean_episode_length'] - first['mean_episode_length']:+.3f}",
                    f"  mean episode reward: {second['mean_episode_reward'] - first['mean_episode_reward']:+.6f}",
                ]
            )

        output_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        output_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "task": args_cli.task,
                    "source_motion_file": source_motion_file,
                    "fixed_evaluation_manifest": env_cfg.commands.motion.motion_file,
                    "seed": args_cli.eval_seed,
                    "num_envs": args_cli.num_envs,
                    "eval_steps": args_cli.eval_steps,
                    "start_at_motion_beginning": args_cli.start_at_motion_beginning,
                    "active_global_motion_ids": active_global_ids,
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[EVAL] Wrote {output_path}")
        env.close()


if __name__ == "__main__":
    main()
