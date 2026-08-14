"""Replay a CSV motion file and output it to an NPZ file.

.. code-block:: bash

    python scripts/csv_to_npz.py --robot pm01 --input_fps 30 -f <path_to_input.csv>
    python scripts/csv_to_npz.py --robot t800 --input_fps 30 -f <path_to_input.csv>
    python scripts/csv_to_npz.py --robot t800 --include_key_body_poses -f <path_to_input.csv>
    python scripts/csv_to_npz.py --robot pm01 --input_quaternion_order wxyz -f <legacy_motion.csv>

The CSV columns are ``root_x, root_y, root_z, quaternion, joint_positions``.
Input quaternions can be XYZW or WXYZ; output NPZ files always use Isaac Lab 3.x XYZW.
"""

"""Parse CLI arguments before selecting an Isaac Lab physics backend."""

import argparse
import os

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument(
    "--robot",
    type=str,
    default="pm01",
    choices=["pm01", "t800"],
    help="The robot configuration to use.",
)
parser.add_argument("--input_file", "-f", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--input_quaternion_order",
    choices=["xyzw", "wxyz"],
    default="xyzw",
    help=(
        "Quaternion component order in columns 4-7 of the input CSV. Defaults to 'xyzw', which matches the CSV "
        "files shipped with this repository. Values are converted internally to Isaac Lab 3.x XYZW."
    ),
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument(
    "--physics",
    default="isaacsim_physx",
    choices=["isaacsim_physx", "newton_mjwarp"],
    help="Physics backend used to load and replay the robot.",
)
parser.add_argument(
    "--include_key_body_poses",
    action="store_true",
    help=(
        "Additionally save a 14-link T800 key-body pose subset. "
        "This is optional and does not change the existing full-body arrays."
    ),
)

# append simulation launcher cli args
add_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if not args_cli.output_name:
    # generate at the same location as input file
    output_suffix = "_key_body_poses.npz" if args_cli.include_key_body_poses else ".npz"
    args_cli.output_name = os.path.splitext(args_cli.input_file)[0] + output_suffix


import torch

##
# Pre-defined configs
##
from engineai_rl_lab.tasks.tracking.robots.pm01 import PM01_CYLINDER_CFG
from engineai_rl_lab.tasks.tracking.robots.t800 import T800_CYLINDER_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

ROBOT_CFGS = {
    "pm01": PM01_CYLINDER_CFG,
    "t800": T800_CYLINDER_CFG,
}

# Key bodies used by the T800 tracking task. Keep this order stable because
# downstream task observations flatten the body axis in order.
T800_KEY_BODY_NAMES = [
    "LINK_BASE",
    "LINK_HIP_ROLL_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_WAIST_YAW",
    "LINK_SHOULDER_ROLL_L",
    "LINK_ELBOW_YAW_L",
    "LINK_WRIST_END_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_ELBOW_YAW_R",
    "LINK_WRIST_END_R",
]


def quaternion_to_xyzw(quaternions: torch.Tensor, input_order: str) -> torch.Tensor:
    """Return quaternions in Isaac Lab 3.x XYZW order.

    Args:
        quaternions: Tensor whose final dimension contains four quaternion components.
        input_order: Component order in ``quaternions``; either ``"xyzw"`` or ``"wxyz"``.
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with four components, received shape {tuple(quaternions.shape)}.")
    if input_order == "xyzw":
        return quaternions
    if input_order == "wxyz":
        return torch.roll(quaternions, shifts=-1, dims=-1)
    raise ValueError(f"Unsupported input quaternion order {input_order!r}; expected 'xyzw' or 'wxyz'.")


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.CuboidCfg(size=(100.0, 100.0, 0.1)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        input_quaternion_order: str = "xyzw",
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.input_quaternion_order = input_quaternion_order
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = quaternion_to_xyzw(motion[:, 3:7], self.input_quaternion_order)
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        if rotations.shape[0] == 1:
            return torch.zeros((1, 3), dtype=rotations.dtype, device=rotations.device)
        if rotations.shape[0] == 2:
            q_rel = quat_mul(rotations[1:], quat_conjugate(rotations[:-1]))
            omega = axis_angle_from_quat(q_rel) / dt
            return torch.cat([omega, omega], dim=0)

        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B-2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B-2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        input_quaternion_order=args_cli.input_quaternion_order,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(scene.cfg.robot.joint_sdk_names, preserve_order=True)[0]

    key_body_indexes = None
    if args_cli.include_key_body_poses:
        if args_cli.robot != "t800":
            raise ValueError("--include_key_body_poses currently supports only --robot t800.")
        missing_body_names = [name for name in T800_KEY_BODY_NAMES if name not in robot.body_names]
        if missing_body_names:
            raise ValueError(f"T800 articulation is missing key bodies: {missing_body_names}.")
        key_body_indexes = robot.find_bodies(T800_KEY_BODY_NAMES, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "joint_names": robot.joint_names,
        "body_names": robot.body_names,
        "quaternion_order": "xyzw",
    }
    if key_body_indexes is not None:
        log.update(
            {
                "key_body_names": T800_KEY_BODY_NAMES,
                "key_body_pos_w": [],
                "key_body_quat_w": [],
            }
        )
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while sim.is_headless_or_exist_active_visualizer():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] = motion_base_pos
        root_pose[:, :2] += scene.env_origins[:, :2]
        root_pose[:, 3:7] = motion_base_rot
        root_velocity = robot.data.default_root_vel.torch.clone()
        root_velocity[:, :3] = motion_base_lin_vel
        root_velocity[:, 3:] = motion_base_ang_vel
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)

        # set joint state
        joint_pos = robot.data.default_joint_pos.torch.clone()
        joint_vel = robot.data.default_joint_vel.torch.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_pose[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos.torch[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel.torch[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w.torch[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w.torch[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w.torch[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w.torch[0, :].cpu().numpy().copy())
            if key_body_indexes is not None:
                log["key_body_pos_w"].append(
                    robot.data.body_pos_w.torch[0, key_body_indexes].cpu().numpy().copy()
                )
                log["key_body_quat_w"].append(
                    robot.data.body_quat_w.torch[0, key_body_indexes].cpu().numpy().copy()
                )

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)
            if key_body_indexes is not None:
                log["key_body_pos_w"] = np.stack(log["key_body_pos_w"], axis=0)
                log["key_body_quat_w"] = np.stack(log["key_body_quat_w"], axis=0)

            np.savez(args_cli.output_name, **log)
            print("[INFO]: Motion npz file saved to", args_cli.output_name)
            return


def main():
    """Main function."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(
            device=args_cli.device,
            dt=1.0 / args_cli.output_fps,
            physics=physics_cfg,
        )
        sim = SimulationContext(sim_cfg)
        scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
