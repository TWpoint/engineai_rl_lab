"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import importlib.metadata as metadata
import importlib.util
import os
import pathlib
import re
import subprocess
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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--export_only", action="store_true", default=False, help="Export the policy and exit.")
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

import gymnasium as gym
import torch
from engineai_rl_lab.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    get_actor_obs_normalizer,
)
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import launch_simulation
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.entrypoints.common import apply_video_recording, pre_launch_video_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def sanitize_rsl_rl_cfg(cfg: dict) -> dict:
    """Remove only legacy model keys, preserving CNN/RNN and custom-model options."""
    legacy_model_keys = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    for model_name in ("actor", "critic"):
        model_cfg = cfg.get(model_name)
        if isinstance(model_cfg, dict):
            for key in legacy_model_keys:
                model_cfg.pop(key, None)
    return cfg


def get_mnn_filename(resume_path: str, run_name: str | None) -> str:
    """Build an MNN filename from the loaded run and checkpoint names."""
    run_dir_name = pathlib.Path(resume_path).parent.name
    timestamped_run = re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)$", run_dir_name)
    resolved_run_name = timestamped_run.group(1) if timestamped_run else run_name
    if not resolved_run_name:
        resolved_run_name = run_dir_name

    checkpoint_stem = pathlib.Path(resume_path).stem
    model_number = re.search(r"(\d+)(?!.*\d)", checkpoint_stem)
    model_id = model_number.group(1) if model_number else checkpoint_stem
    if model_number is None:
        print(
            f"[WARN] Checkpoint '{pathlib.Path(resume_path).name}' has no model number; "
            f"using '{model_id}' in the MNN filename."
        )

    return f"policy_{resolved_run_name}_{model_id}.mnn"


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point", play_mode=True)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)

    log_dir = os.path.dirname(resume_path)

    pre_launch_video_config(env_cfg, args_cli=args_cli)
    with launch_simulation(env_cfg, args_cli):
        env_cfg.log_dir = log_dir
        apply_video_recording(env_cfg, log_dir, args_cli, subdir="play")
        # create isaac environment after the requested physics backend has been launched
        env = gym.make(args_cli.task, cfg=env_cfg)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # load previously trained model
        ppo_runner = OnPolicyRunner(
            env, sanitize_rsl_rl_cfg(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device
        )
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # export policy to ONNX
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        export_motion_policy_as_onnx(
            env.unwrapped,
            ppo_runner.alg.get_policy(),
            normalizer=get_actor_obs_normalizer(ppo_runner),
            path=export_model_dir,
            filename="policy.onnx",
        )
        # Validate the exported actor inputs and write deployment metadata
        # before converting the model. This prevents publishing an MNN model
        # whose observation groups cannot be reproduced by the native SDK.
        attach_onnx_metadata(
            env.unwrapped,
            args_cli.wandb_path if args_cli.wandb_path else "none",
            export_model_dir,
            actor_obs_groups=agent_cfg.obs_groups["actor"],
        )
        # Convert the exported policy to MNN when the optional converter is installed.
        onnx_file = os.path.join(export_model_dir, "policy.onnx")
        mnn_file = os.path.join(export_model_dir, get_mnn_filename(resume_path, agent_cfg.run_name))

        if os.path.exists(onnx_file):
            if importlib.util.find_spec("MNN") is None:
                print("[WARN] MNN is not installed in the current Python environment. Skipping MNN conversion.")
            else:
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "MNN.tools.mnnconvert",
                            "-f",
                            "ONNX",
                            "--modelFile",
                            onnx_file,
                            "--MNNModel",
                            mnn_file,
                            "--bizCode",
                            "MNN",
                        ],
                        check=True,
                    )
                    print(f"Successfully converted to MNN: {mnn_file}")
                except subprocess.CalledProcessError as err:
                    print(f"[WARN] Failed to convert ONNX to MNN: {err}")
        else:
            print(f"ONNX file not found: {onnx_file}")

        if args_cli.export_only:
            env.close()
            return

        obs = env.get_observations()
        timestep = 0
        try:
            while env.unwrapped.sim.is_headless_or_exist_active_visualizer():
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if hasattr(policy, "reset"):
                        policy.reset(dones)
                if args_cli.video:
                    timestep += 1
                    if timestep >= args_cli.video_length:
                        break
        except KeyboardInterrupt:
            pass

        env.close()


if __name__ == "__main__":
    main()
