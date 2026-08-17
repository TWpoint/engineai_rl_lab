"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python scripts/replay_npz.py --robot pm01 --input_file <path_to_motion.npz>
    python scripts/replay_npz.py --robot t800 --input_file <path_to_motion.npz>
    python scripts/replay_npz.py --robot t800 --visualize_body_poses --input_file <path_to_motion.npz>
    python scripts/replay_npz.py --robot t800 --visualize_key_body_poses --input_file <path_to_motion.npz>
"""

"""Parse CLI arguments before selecting an Isaac Lab physics backend."""

import argparse
import time

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument("--input_file", type=str, default=None, help="Path to a local .npz motion file.")
parser.add_argument("--robot", type=str, default="pm01", choices=["pm01", "t800"], help="Robot type to use.")
pose_visualization_group = parser.add_mutually_exclusive_group()
pose_visualization_group.add_argument(
    "--visualize_body_poses",
    action="store_true",
    help="Visualize the position and orientation of every robot body stored in the NPZ.",
)
pose_visualization_group.add_argument(
    "--visualize_key_body_poses",
    action="store_true",
    help="Visualize key-body positions and orientations stored in the NPZ as coordinate frames.",
)
parser.add_argument(
    "--follow_camera",
    action="store_true",
    help="Continuously move the camera to follow the robot root. By default the camera remains user-controlled.",
)
parser.add_argument(
    "--physics",
    default="isaacsim_physx",
    choices=["isaacsim_physx", "newton_mjwarp"],
    help="Physics backend used to load and replay the robot.",
)

# append simulation launcher cli args
add_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

from engineai_rl_lab.tasks.tracking.mdp.commands import MotionLoader

##
# Pre-defined configs
##
from engineai_rl_lab.tasks.tracking.robots.pm01 import PM01_CYLINDER_CFG
from engineai_rl_lab.tasks.tracking.robots.t800 import T800_CYLINDER_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass

ROBOT_CFGS = {
    "pm01": PM01_CYLINDER_CFG,
    "t800": T800_CYLINDER_CFG,
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.CuboidCfg(size=(100.0, 100.0, 0.1)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )

    robot: ArticulationCfg = ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    if args_cli.input_file is not None:
        motion_file = args_cli.input_file
    elif args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        raise ValueError("Either --input_file or --registry_name must be provided.")

    motion = MotionLoader(
        motion_file,
        joint_names=robot.joint_names,
        body_names=robot.body_names,
        body_indexes=torch.arange(len(robot.body_names), dtype=torch.long, device=sim.device),
        device=sim.device,
    )

    key_body_names = None
    key_body_pos_w = None
    key_body_quat_w = None
    key_body_visualizer = None
    if args_cli.visualize_key_body_poses:
        with np.load(motion_file) as data:
            required_fields = {"key_body_names", "key_body_pos_w", "key_body_quat_w"}
            missing_fields = sorted(required_fields.difference(data.files))
            if missing_fields:
                raise ValueError(
                    f"Motion file {motion_file!r} does not contain key-body poses. Missing fields: {missing_fields}. "
                    "Regenerate it with csv_to_npz.py --include_key_body_poses."
                )
            key_body_names = data["key_body_names"].astype(str).tolist()
            key_body_pos_w = torch.tensor(data["key_body_pos_w"], dtype=torch.float32, device=sim.device)
            key_body_quat_w = torch.tensor(data["key_body_quat_w"], dtype=torch.float32, device=sim.device)

        if key_body_pos_w.shape != (motion.time_step_total, len(key_body_names), 3):
            raise ValueError(
                f"Invalid key_body_pos_w shape {tuple(key_body_pos_w.shape)} for "
                f"{motion.time_step_total} frames and {len(key_body_names)} key bodies."
            )
        if key_body_quat_w.shape != (motion.time_step_total, len(key_body_names), 4):
            raise ValueError(
                f"Invalid key_body_quat_w shape {tuple(key_body_quat_w.shape)} for "
                f"{motion.time_step_total} frames and {len(key_body_names)} key bodies."
            )

        key_body_visualizer_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/KeyBodyPoses")
        key_body_visualizer_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
        key_body_visualizer = VisualizationMarkers(key_body_visualizer_cfg)

    body_visualizer = None
    if args_cli.visualize_body_poses:
        body_visualizer_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/BodyPoses")
        body_visualizer_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
        body_visualizer = VisualizationMarkers(body_visualizer_cfg)
    playback_fps = float(np.asarray(motion.fps).reshape(-1)[0])
    if playback_fps <= 0.0:
        raise ValueError(f"Motion FPS must be positive, got {playback_fps}.")
    frame_period = 1.0 / playback_fps
    next_frame_time = time.perf_counter()

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    print(f"[INFO]: Replaying {len(robot.joint_names)} joints and {len(robot.body_names)} bodies by name.")
    if key_body_names is not None:
        print(f"[INFO]: Visualizing {len(key_body_names)} key-body pose frames: {key_body_names}")
    if body_visualizer is not None:
        print(f"[INFO]: Visualizing {len(robot.body_names)} body pose frames: {robot.body_names}")
    print(f"[INFO]: Playback rate: {playback_fps:g} Hz ({frame_period * 1000.0:.2f} ms per frame).")

    # Simulation loop
    while sim.is_headless_or_exist_active_visualizer():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_pose[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_velocity = robot.data.default_root_vel.torch.clone()
        root_velocity[:, :3] = motion.body_lin_vel_w[time_steps][:, 0]
        root_velocity[:, 3:] = motion.body_ang_vel_w[time_steps][:, 0]

        joint_pos = robot.data.default_joint_pos.torch.clone()
        joint_vel = robot.data.default_joint_vel.torch.clone()
        joint_pos[:] = motion.joint_pos[time_steps]
        joint_vel[:] = motion.joint_vel[time_steps]

        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
        robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        if key_body_visualizer is not None:
            marker_positions = key_body_pos_w[time_steps[0]] + scene.env_origins[0]
            marker_orientations = key_body_quat_w[time_steps[0]]
            key_body_visualizer.visualize(marker_positions, marker_orientations)

        if body_visualizer is not None:
            marker_positions = motion.body_pos_w[time_steps[0]] + scene.env_origins[0]
            marker_orientations = motion.body_quat_w[time_steps[0]]
            body_visualizer.visualize(marker_positions, marker_orientations)

        if args_cli.follow_camera:
            pos_lookat = root_pose[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        # Pace motion frames against wall-clock time. If rendering takes longer
        # than one frame period, rebase the deadline to avoid a catch-up burst.
        next_frame_time += frame_period
        sleep_duration = next_frame_time - time.perf_counter()
        if sleep_duration > 0.0:
            time.sleep(sleep_duration)
        else:
            next_frame_time = time.perf_counter()


def main():
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.02, physics=physics_cfg)
        sim = SimulationContext(sim_cfg)

        scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
