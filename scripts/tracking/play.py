"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import importlib.metadata as metadata
import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import tempfile

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
parser.add_argument(
    "--latent_output", type=str, default=None, help="Save one rollout's policy latents to this NPZ file."
)
parser.add_argument(
    "--latent_node",
    type=str,
    default="attention_blocks",
    help="Actor graph node to capture (default: attention_blocks).",
)
parser.add_argument(
    "--max_steps", type=int, default=None, help="Stop playback/latent collection after this many steps."
)
parser.add_argument(
    "--follow_camera",
    action="store_true",
    default=False,
    help="Continuously move the visualizer camera to follow the robot root.",
)
parser.add_argument(
    "--start_at_motion_beginning",
    action="store_true",
    default=False,
    help="Start every playback episode at frame zero instead of sampling a random motion time.",
)
parser.add_argument(
    "--start_frame",
    "--start-frame",
    type=int,
    default=None,
    help="Start every playback episode at this zero-based reference-motion frame.",
)
parser.add_argument(
    "--ghost_reference",
    action="store_true",
    default=False,
    help="Overlay a collision-free translucent robot driven by the aligned reference motion.",
)
parser.add_argument(
    "--ghost_opacity",
    type=float,
    default=0.3,
    help="Opacity of the reference robot (default: 0.3).",
)
parser.add_argument(
    "--ghost_offset",
    type=float,
    default=1.0,
    help="Lateral distance in meters between the policy and reference robots (default: 1.0).",
)
parser.add_argument(
    "--disable_body_pos_termination",
    "--disable-body-pos-termination",
    "--disable_termination",
    "--disable-termination",
    action="store_true",
    default=False,
    help="Disable pose/position tracking-error terminations during playback.",
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

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.app import launch_simulation
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.math import quat_apply, yaw_quat

from isaaclab_rl.entrypoints.common import apply_video_recording, pre_launch_video_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, handle_deprecated_rsl_rl_cfg

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
def main(  # noqa: C901
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    if args_cli.start_frame is not None:
        if args_cli.start_frame < 0:
            raise ValueError("--start_frame must be non-negative.")
        if args_cli.start_at_motion_beginning:
            raise ValueError("--start_frame cannot be combined with --start_at_motion_beginning.")
        env_cfg.commands.motion.playback_start_frame = args_cli.start_frame
        print(f"[INFO]: Starting every playback episode at motion frame {args_cli.start_frame}.")
    elif args_cli.start_at_motion_beginning:
        env_cfg.commands.motion.start_at_motion_beginning = True
        print("[INFO]: Starting every playback episode at the beginning of the motion.")
    if args_cli.disable_body_pos_termination:
        tracking_termination_names = (
            "body_pos",
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
            "anchor_height",
            "anchor_orientation",
            "end_effector_height",
        )
        disabled_terminations = []
        for name in tracking_termination_names:
            if hasattr(env_cfg.terminations, name) and getattr(env_cfg.terminations, name) is not None:
                setattr(env_cfg.terminations, name, None)
                disabled_terminations.append(name)
        if not disabled_terminations:
            raise ValueError("This task does not define any pose/position tracking termination terms.")
        print(
            "[INFO]: Disabled tracking terminations for playback: "
            + ", ".join(disabled_terminations)
        )
    if args_cli.ghost_reference:
        if not 0.0 < args_cli.ghost_opacity <= 1.0:
            raise ValueError("--ghost_opacity must be in the interval (0, 1].")
        if args_cli.ghost_offset < 0.0:
            raise ValueError("--ghost_offset must be non-negative.")
        ghost_cfg = env_cfg.scene.robot.replace(prim_path="{ENV_REGEX_NS}/GhostReference")
        ghost_cfg.spawn = ghost_cfg.spawn.replace(
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.65, 1.0),
                emissive_color=(0.02, 0.08, 0.12),
                roughness=0.35,
                opacity=args_cli.ghost_opacity,
            ),
            make_uninstanceable=True,
        )
        env_cfg.scene.ghost_reference = ghost_cfg
        print(f"[INFO]: Enabling aligned reference robot (opacity={args_cli.ghost_opacity:g}).")

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
            env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
        else:
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is None:
                print("[WARN] No motion artifact found in the run.")
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
        # These modules load protobuf/gRPC. Import them only after Kit has started;
        # importing them earlier conflicts with omni.grpc during PhysX startup.
        from engineai_rl_lab.utils.exporter import (
            attach_onnx_metadata,
            export_motion_policy_as_onnx,
            get_actor_obs_normalizer,
            rewrite_silu_for_mnn_2_9_5,
        )
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

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

        latent_samples = []
        latent_hook = None
        if args_cli.latent_output:
            actor = ppo_runner.alg.get_policy()
            if not hasattr(actor, "nodes") or args_cli.latent_node not in actor.nodes:
                available = list(getattr(actor, "nodes", {}).keys())
                raise ValueError(f"Actor node '{args_cli.latent_node}' not found; available nodes: {available}")

            def capture_latent(_module, _inputs, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                latent_samples.append(tensor[0].detach().float().cpu().reshape(-1).numpy())

            latent_hook = actor.nodes[args_cli.latent_node].register_forward_hook(capture_latent)
            print(f"[INFO]: Capturing latent node '{args_cli.latent_node}' to: {args_cli.latent_output}")

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
        deploy_export_valid = True
        try:
            attach_onnx_metadata(
                env.unwrapped,
                args_cli.wandb_path if args_cli.wandb_path else "none",
                export_model_dir,
                actor_obs_groups=agent_cfg.obs_groups["actor"],
            )
        except ValueError as err:
            # Export is ancillary to interactive playback.  In particular, a
            # flat MLP may concatenate several actor observation groups into a
            # single ONNX ``obs`` tensor, which schema-v2 deployment metadata
            # cannot currently describe without losing the group boundaries.
            # Do not publish/convert that model, but still allow the loaded
            # checkpoint to be evaluated in simulation.
            if args_cli.export_only:
                raise
            deploy_export_valid = False
            print(f"[WARN] Deployment export validation failed: {err}")
            print("[WARN] Skipping MNN conversion and continuing interactive playback.")
        # Convert the exported policy to MNN when the optional converter is installed.
        onnx_file = os.path.join(export_model_dir, "policy.onnx")
        mnn_file = os.path.join(export_model_dir, get_mnn_filename(resume_path, agent_cfg.run_name))

        if os.path.exists(onnx_file) and deploy_export_valid:
            if importlib.util.find_spec("MNN") is None:
                print("[WARN] MNN is not installed in the current Python environment. Skipping MNN conversion.")
            else:
                try:
                    # The deployment SDK uses MNN 2.9.5. MNN 3.6.1's
                    # converter otherwise fuses ONNX x*sigmoid(x) into a SILU
                    # opcode that the older runtime cannot execute, even with
                    # --targetVersion alone.
                    with tempfile.TemporaryDirectory(prefix="engineai_mnn_2_9_5_") as temporary_dir:
                        compatible_onnx_file = os.path.join(temporary_dir, "policy.onnx")
                        rewritten_silu_count = rewrite_silu_for_mnn_2_9_5(onnx_file, compatible_onnx_file)
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "MNN.tools.mnnconvert",
                                "-f",
                                "ONNX",
                                "--modelFile",
                                compatible_onnx_file,
                                "--MNNModel",
                                mnn_file,
                                "--bizCode",
                                "MNN",
                                "--targetVersion",
                                "2.9.5",
                            ],
                            check=True,
                        )
                    print(
                        "[INFO]: MNN 2.9.5 compatibility lowering rewrote "
                        f"{rewritten_silu_count} ONNX SiLU pattern(s)."
                    )
                    print(f"Successfully converted to MNN: {mnn_file}")
                except subprocess.CalledProcessError as err:
                    print(f"[WARN] Failed to convert ONNX to MNN: {err}")
        elif not os.path.exists(onnx_file):
            print(f"ONNX file not found: {onnx_file}")

        if args_cli.export_only:
            env.close()
            return

        obs = env.get_observations()
        ghost_robot = None
        motion_command = env.unwrapped.command_manager.get_term("motion")
        if args_cli.ghost_reference:
            ghost_robot = env.unwrapped.scene["ghost_reference"]
            if ghost_robot.joint_names != motion_command.robot.joint_names:
                raise RuntimeError("Reference robot joint ordering does not match the policy robot.")

        def update_ghost_reference():
            if ghost_robot is None or motion_command is None:
                return
            anchor_index = motion_command.motion_anchor_body_index
            root_pose = torch.cat(
                (
                    motion_command.body_pos_relative_w[:, anchor_index],
                    motion_command.body_quat_relative_w[:, anchor_index],
                ),
                dim=-1,
            )
            lateral_offset = torch.zeros_like(root_pose[:, :3])
            lateral_offset[:, 1] = args_cli.ghost_offset
            root_pose[:, :3] += quat_apply(yaw_quat(motion_command.robot_anchor_quat_w), lateral_offset)
            ghost_robot.write_root_pose_to_sim_index(root_pose=root_pose)
            ghost_robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((root_pose.shape[0], 6), device=root_pose.device)
            )
            ghost_robot.write_joint_state_to_sim_index(
                position=motion_command.joint_pos,
                velocity=torch.zeros_like(motion_command.joint_vel),
            )

        update_ghost_reference()
        timestep = 0
        episode_index = 1
        episode_steps = 0
        episode_total_steps = max(
            int((motion_command.motion_lengths[0] - motion_command.time_steps[0] - 1).item()), 1
        )
        try:
            while env.unwrapped.sim.is_headless_or_exist_active_visualizer():
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    episode_steps += 1
                    update_ghost_reference()
                    if bool(dones[0].item()):
                        termination_manager = env.unwrapped.termination_manager
                        reasons = [
                            name
                            for name in termination_manager.active_terms
                            if bool(termination_manager.get_term(name)[0].item())
                        ]
                        reason_text = ", ".join(reasons) if reasons else "unknown"
                        print(
                            f"[TERMINATION] Episode {episode_index}: {reason_text} "
                            f"(steps {episode_steps}/{episode_total_steps})",
                            flush=True,
                        )
                        episode_index += 1
                        episode_steps = 0
                        episode_total_steps = max(
                            int(
                                (
                                    motion_command.motion_lengths[0]
                                    - motion_command.time_steps[0]
                                    - 1
                                ).item()
                            ),
                            1,
                        )
                    if hasattr(policy, "reset"):
                        policy.reset(dones)
                if args_cli.follow_camera:
                    robot_root_pos = env.unwrapped.scene["robot"].data.root_pos_w.torch[0]
                    camera_lookat = robot_root_pos.detach().cpu().numpy()
                    env.unwrapped.sim.set_camera_view(camera_lookat + [2.0, 2.0, 0.5], camera_lookat)
                if args_cli.video:
                    timestep += 1
                    if timestep >= args_cli.video_length:
                        break
                elif args_cli.latent_output or args_cli.max_steps is not None:
                    timestep += 1
                if args_cli.latent_output and bool(dones[0].item()):
                    break
                if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if latent_hook is not None:
                latent_hook.remove()
            if args_cli.latent_output:
                import numpy as np

                if not latent_samples:
                    raise RuntimeError("No latent samples were captured.")
                latent_path = pathlib.Path(args_cli.latent_output)
                latent_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    latent_path,
                    latent=np.stack(latent_samples),
                    motion_file=os.path.abspath(args_cli.motion_file or ""),
                    checkpoint=os.path.abspath(resume_path),
                    node=args_cli.latent_node,
                )
                print(f"[INFO]: Saved {len(latent_samples)} latent samples to: {latent_path}")

        env.close()


if __name__ == "__main__":
    main()
