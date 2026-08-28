# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

import argparse
import contextlib
import importlib.metadata as metadata
import math
import pathlib
import pickle
import sys

# Register the downstream tasks before preset-aware argument parsing so --help can enumerate their variants.
import engineai_rl_lab.tasks  # noqa: F401, E402

from isaaclab.app import add_launcher_args

from isaaclab_tasks.utils import setup_preset_cli

# local imports
import cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Path to a local motion .npz file or YAML motion manifest.",
)
parser.add_argument(
    "--overfit_single_motion",
    action="store_true",
    default=False,
    help="Disable randomization and sampling adaptation to overfit one --motion_file .npz trajectory.",
)
parser.add_argument(
    "--invalid_state_snapshot_dir",
    type=str,
    default=None,
    help=(
        "Opt-in directory for invalid_robot_state and finite-runaway diagnostic snapshots "
        "(zero recorder overhead when omitted)."
    ),
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append simulation launcher and backend-preset arguments
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser, agent_library="rsl_rl")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

import os
from datetime import datetime

import gymnasium as gym
import torch
from engineai_rl_lab.tasks.tracking.mdp.recorders import InvalidRobotStateRecorderManagerCfg
from engineai_rl_lab.utils.finite_reset import FiniteResetRslRlVecEnvWrapper
from engineai_rl_lab.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

from isaaclab.app import launch_simulation
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.entrypoints.common import apply_video_recording, pre_launch_video_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def configure_rank_cpu_runtime() -> None:
    """Partition host CPUs across local workers without requiring NUMA tools."""

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if not args_cli.distributed or local_world_size <= 1 or not hasattr(os, "sched_getaffinity"):
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= local_rank < local_world_size:
        raise ValueError(f"Invalid LOCAL_RANK/LOCAL_WORLD_SIZE: {local_rank}/{local_world_size}")
    available_cpus = sorted(os.sched_getaffinity(0))
    if not available_cpus:
        return

    def _gpu_numa_node(gpu_index: int) -> int | None:
        if gpu_index >= torch.cuda.device_count():
            return None
        try:
            properties = torch.cuda.get_device_properties(gpu_index)
            pci_address = f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0"
            value = int(pathlib.Path("/sys/bus/pci/devices", pci_address, "numa_node").read_text().strip())
        except (AttributeError, OSError, ValueError):
            return None
        return value if value >= 0 else None

    rank_numa_nodes = [_gpu_numa_node(index) for index in range(local_world_size)]
    rank_numa_node = rank_numa_nodes[local_rank]
    node_cpus: list[int] = []
    if rank_numa_node is not None:
        try:
            node_cpu_text = pathlib.Path(f"/sys/devices/system/node/node{rank_numa_node}/cpulist").read_text().strip()
            for group in node_cpu_text.split(","):
                bounds = [int(value) for value in group.split("-")]
                node_cpus.extend(range(bounds[0], bounds[-1] + 1))
        except (OSError, ValueError):
            node_cpus = []
        node_cpus = sorted(set(node_cpus).intersection(available_cpus))

    if node_cpus:
        peer_ranks = [index for index, node in enumerate(rank_numa_nodes) if node == rank_numa_node]
        peer_index = peer_ranks.index(local_rank)
        cores_per_rank = max(1, math.ceil(len(node_cpus) / len(peer_ranks)))
        begin = peer_index * cores_per_rank
        rank_cpus = node_cpus[begin : begin + cores_per_rank]
    else:
        cores_per_rank = max(1, math.ceil(len(available_cpus) / local_world_size))
        begin = local_rank * cores_per_rank
        rank_cpus = available_cpus[begin : begin + cores_per_rank]
    if not rank_cpus:
        rank_cpus = [available_cpus[local_rank % len(available_cpus)]]

    if os.environ.get("ENGINEAI_RANK_CPU_AFFINITY", "1") != "0":
        with contextlib.suppress(OSError):
            os.sched_setaffinity(0, rank_cpus)
    requested_threads = os.environ.get("ENGINEAI_TORCH_THREADS_PER_RANK")
    thread_count = int(requested_threads) if requested_threads else len(rank_cpus)
    torch.set_num_threads(max(1, min(thread_count, len(rank_cpus))))
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def dump_pickle(filename: str, data: object):
    """Compatibility shim for Isaac Lab versions without `dump_pickle`."""
    if not filename.endswith("pkl"):
        filename += ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def sanitize_rsl_rl_cfg(cfg: dict) -> dict:
    """Remove only legacy model keys, preserving CNN/RNN and custom-model options."""
    legacy_model_keys = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    for model_name in ("actor", "critic"):
        model_cfg = cfg.get(model_name)
        if isinstance(model_cfg, dict):
            for key in legacy_model_keys:
                model_cfg.pop(key, None)
    return cfg


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    configure_rank_cpu_runtime()
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # An iteration limit is opt-in for tracking runs. Without the CLI flag the
    # runner trains indefinitely and reports a projected duration for 10k iterations.
    agent_cfg.max_iterations = args_cli.max_iterations

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # load the motion file
    registry_name = args_cli.registry_name
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
    elif registry_name is not None:
        if ":" not in registry_name:
            registry_name += ":latest"
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    elif not isinstance(env_cfg.commands.motion.motion_file, str):
        raise ValueError("Provide --motion_file/--registry_name or configure commands.motion.motion_file in the task.")

    if args_cli.overfit_single_motion:
        if args_cli.motion_file is None or pathlib.Path(args_cli.motion_file).suffix.lower() != ".npz":
            raise ValueError("--overfit_single_motion requires one local .npz file via --motion_file")
        configure_overfit = getattr(env_cfg, "enable_single_motion_overfit", None)
        if configure_overfit is None:
            raise ValueError(f"Task {args_cli.task!r} does not provide a single-motion overfit configuration")
        configure_overfit()
        agent_cfg.run_name = f"{agent_cfg.run_name}_single_motion_overfit"
        print("[INFO]: Single-motion overfit mode enabled (randomization, noise, and adaptive sampling disabled).")

    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    global_rank = int(os.getenv("RANK", "0")) if args_cli.distributed else 0

    if args_cli.invalid_state_snapshot_dir is not None:
        recorder_cfg = InvalidRobotStateRecorderManagerCfg()
        recorder_cfg.invalid_robot_state.snapshot_dir = os.path.abspath(args_cli.invalid_state_snapshot_dir)
        recorder_cfg.invalid_robot_state.snapshot_context = pathlib.Path(log_dir).name
        env_cfg.recorders = recorder_cfg
        print(
            "[INFO] Raw invalid-state snapshots enabled at "
            f"{recorder_cfg.invalid_robot_state.snapshot_dir}"
        )

    pre_launch_video_config(env_cfg, args_cli=args_cli)
    with launch_simulation(env_cfg, args_cli):
        # ``launch_simulation`` resolves the simulation device from LOCAL_RANK for
        # distributed runs. RSL-RL must use the same device, and each rank needs a
        # distinct seed to avoid collecting identical rollouts.
        if args_cli.distributed:
            agent_cfg.device = env_cfg.sim.device
            env_cfg.seed = agent_cfg.seed + global_rank
            agent_cfg.seed = env_cfg.seed

        env_cfg.log_dir = log_dir
        apply_video_recording(env_cfg, log_dir, args_cli)
        env = None
        try:
            # create isaac environment after the requested physics backend has been launched
            env = gym.make(args_cli.task, cfg=env_cfg)

            # convert to single-agent instance if required by the RL algorithm
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)

            # wrap around environment for rsl-rl
            env = FiniteResetRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            # create runner from rsl-rl
            train_cfg = sanitize_rsl_rl_cfg(agent_cfg.to_dict())
            runner = OnPolicyRunner(
                env,
                train_cfg,
                log_dir=log_dir,
                device=agent_cfg.device,
                registry_name=registry_name,
            )
            # write git state to logs
            runner.add_git_repo_to_log(__file__)
            # save resume path before creating a new log_dir
            if agent_cfg.resume:
                # get path to previous checkpoint
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
                print(f"[INFO]: Loading model checkpoint from: {resume_path}")
                # load previously trained model
                runner.load(resume_path)

            # Only the logging rank owns the canonical run directory. This
            # prevents all distributed workers from concurrently overwriting
            # the same YAML/pickle files when their timestamps coincide.
            if global_rank == 0:
                dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
                dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
                dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
                dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

            # run training
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
            )
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if args_cli.distributed and torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
