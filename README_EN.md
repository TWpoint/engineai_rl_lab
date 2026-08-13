# engineai_rl_lab
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0_beta2+-silver)](https://github.com/isaac-sim/IsaacLab/releases)
[![Newton](https://img.shields.io/badge/Physics-Newton_MJWarp-blue)](https://github.com/newton-physics/newton)
[![Isaac Sim](https://img.shields.io/badge/Optional-Isaac_Sim_6.0.1-silver)](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/index.html)

[中文](README.md)

## Overview
This project provides a set of reinforcement learning environments based on Isaac Lab. It currently supports EngineAI PM01 and T800 robots, with implemented tasks including whole-body tracking.

|Robot|Training| Sim2Sim |Deploy|
|:--------:|:--------:|:--------:|:--------:|
|||**whole body tracking**|||
|**T800**|<img src="./docs/train.gif" height="180"/>|<img src="./docs/sim2sim.gif" height="180"/>|<img src="./docs/deploy.gif" height="180"/>|
|**PM01**|<img src="./docs/train_pm.gif" height="180"/>|<img src="./docs/sim2sim_pm.gif" height="180"/>|<img src="./docs/deploy_pm.gif" height="180"/>|

## Compatibility

The current code targets Isaac Lab 3.0 Beta 2. The environment was validated against commit `ef4611e0152d3422ac1aa88fe0f0a3922fe8bd7e` on `develop`. Pin this commit during installation instead of following the moving `develop` HEAD.

- Newton MJWarp runs without Isaac Sim; select it with `physics=newton_mjwarp`.
- Isaac Sim PhysX/Kit is optional and requires Isaac Sim 6.0.1; select it with `physics=isaacsim_physx --viz kit`.
- Isaac Lab 3.0 requires Python 3.12 and changes quaternion ordering to XYZW and asset data to Warp-backed arrays.
- MJWarp in the current baseline does not enforce actuator `velocity_limit`/`velocity_limit_sim`. Do not treat those fields as training or real-robot safety limits; keep policy-side clipping and deployment safety checks.
- Keep `SimulationCfg.use_newton_actuators=False` (the current configuration pins it explicitly); the Newton-native actuator path would otherwise bypass T800's Python delay actuator.

## Installation

### Prepare Git LFS

The robot USD assets and example checkpoints are managed by Git LFS. Install and initialize Git LFS before fetching the repository; otherwise, the USD files remain roughly 130-byte text pointers and the simulator cannot load the robots. On Ubuntu, run:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs python3.12-venv
git lfs install
```

Set the workspace directory containing `IsaacLab`, this repository, and the virtual environment. On the current machine, use:

```bash
export ENGINEAI_WORKSPACE=/mnt/workspace/lpz/engineai
cd "$ENGINEAI_WORKSPACE"
```

If the repository already exists, fetch its LFS objects:

```bash
git -C engineai_rl_lab lfs pull
```

For a fresh workspace, clone only after `git lfs install`:

```bash
cd "$ENGINEAI_WORKSPACE"
git clone https://github.com/engineai-robotics/engineai_rl_lab.git
git -C engineai_rl_lab lfs pull
```

### Create and activate the Python environment

Isaac Lab 3.0 requires Python 3.12. The virtual environment on the current machine is named `engineai`:

```bash
cd "$ENGINEAI_WORKSPACE"
python3.12 -m venv engineai
source engineai/bin/activate
python -m pip install --upgrade pip
python --version  # must be 3.12.x
```

If the environment already exists, only run `source engineai/bin/activate`.

### Install Isaac Lab 3.0 and Newton

Clone and pin the validated Isaac Lab revision if it is not present yet:

```bash
cd "$ENGINEAI_WORKSPACE"
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout ef4611e0152d3422ac1aa88fe0f0a3922fe8bd7e
```

Install Newton, its built-in Newton GL visualizer, and RSL-RL in the activated environment:

```bash
source "$ENGINEAI_WORKSPACE/engineai/bin/activate"

cd "$ENGINEAI_WORKSPACE/IsaacLab"
./isaaclab.sh -i 'newton,rl[rsl-rl]'
```

The Newton GL visualizer is a base dependency in this revision; `visualizer[newton]` is invalid. To also install Isaac Sim PhysX/Kit, run:

```bash
./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[kit]'
```

Do not use the old `isaacsim` install token. This CLI installs Isaac Sim 6.0.1 through `visualizer[kit]`. Run `./isaaclab.sh --help` to verify selectors supported by the pinned revision.

### Install engineai_rl_lab

After `isaaclab.sh` finishes, the current directory is still `IsaacLab`. Use the workspace variable to return to this repository, then install it in the same activated environment:

```bash
cd "$ENGINEAI_WORKSPACE/engineai_rl_lab"
python -m pip install -e 'source/engineai_rl_lab[export]'
python -c "import MNN, gymnasium, isaaclab, isaaclab_newton, newton, onnx, rsl_rl, torch, trimesh, wandb, warp, yaml, engineai_rl_lab; print('environment ready')"
```

Direct project dependencies are declared in `source/engineai_rl_lab/setup.py`. Isaac Lab and Newton must be installed as a matched source stack by `isaaclab.sh`, so they are intentionally not duplicated in a regular `requirements.txt`. The recommended command includes the `[export]` extra, which pins `MNN==3.6.1` to convert `policy.onnx` into the deployable `policy.mnn`. The import check covers the modules needed for training, export, and logging.

Dependencies are grouped by workflow:

- Base training: `numpy`, `torch`, `gymnasium`, `trimesh`, `PyYAML`, `onnx`, `wandb`, and `rsl-rl-lib==5.4.1`.
- RSL-RL transitive dependencies: the pinned `rsl-rl-lib==5.4.1` installs `tensorboard`, `onnxscript`, `torchvision`, `tensordict`, and `GitPython`; this project does not duplicate them.
- Isaac Lab components: the code directly uses `isaaclab`, `isaaclab_newton`, `isaaclab_physx`, `isaaclab_ov`, `isaaclab_rl`, and `isaaclab_tasks`. They are installed as a matched stack by `isaaclab.sh` at the pinned revision and are also declared in `config/extension.toml`.
- MNN export: install `[export]` (`MNN==3.6.1`). ONNX export works without it, but `policy.mnn` will not be generated.
- Video recording: optionally run `python -m pip install -e 'source/engineai_rl_lab[video]'`.
- Neptune logging: optionally run `python -m pip install -e 'source/engineai_rl_lab[neptune]'`.
- Every project extra: run `python -m pip install -e 'source/engineai_rl_lab[all]'`.

`engineai_robotics_native_sdk`, the robot-side MNN C++ runtime, MuJoCo, and the virtual gamepad are SDK/container dependencies rather than dependencies of this Python package. Install them with the SDK repository scripts. Keep the Python converter compatible with the deployment runtime; the currently validated MNN version is 3.6.1.

Validated key versions are Python 3.12, optional Isaac Sim 6.0.1, Newton 1.5.0, Warp 1.16.0, PyTorch 2.11.0, and RSL-RL 5.4.1. If the installer resolves different major versions, first verify the Isaac Lab commit.

### Common installation failures

- `python3.12 -m venv` is unavailable: install `python3.12-venv` on Ubuntu.
- USD files are about 130 bytes or fail to load: run `git -C engineai_rl_lab lfs pull`.
- `isaaclab_newton`, `newton`, or `rsl_rl` cannot be imported: activate `engineai` before running the installer from `IsaacLab`.
- Do not upgrade `torch`, `warp-lang`, or `newton` separately; rerun the installer at the pinned Isaac Lab revision to repair the environment. Isaac Lab intentionally overrides several strict dependency pins from the optional Isaac Sim wheels, so a generic `pip check` may report known metadata conflicts after Kit is installed and should not be the only readiness test.

## Training
### Whole-Body Tracking

Run every command below from `$ENGINEAI_WORKSPACE/engineai_rl_lab` with the environment above still activated. All four scripts use Isaac Sim PhysX when their physics selector is omitted, so keep the selector shown below for Newton. The selector syntax intentionally differs by entry point:

- `train.py` / `play.py` use the Hydra preset token `physics=newton_mjwarp`, without a leading `--`.
- `csv_to_npz.py` / `replay_npz.py` use the argparse option `--physics newton_mjwarp`.

1. Convert CSV files to NPZ files:
```bash
# The bundled CSV files use XYZW; declare the input order explicitly. Output NPZ files always use Isaac Lab 3.0 XYZW.
python scripts/csv_to_npz.py --robot pm01 --input_fps 30 --input_quaternion_order xyzw -f datasets/tracking/pm01/dance.csv --physics newton_mjwarp
python scripts/csv_to_npz.py --robot t800 --input_fps 30 --input_quaternion_order xyzw -f datasets/tracking/t800/dance_t800.csv --physics newton_mjwarp

# For an external CSV whose columns 4–7 use WXYZ, run:
python scripts/csv_to_npz.py --robot pm01 --input_fps 30 --input_quaternion_order wxyz -f path/to/motion.csv --physics newton_mjwarp

# Replay NPZ files with the Newton visualizer.
python scripts/replay_npz.py --robot pm01 --input_file datasets/tracking/pm01/dance.npz --physics newton_mjwarp --viz newton_gl
python scripts/replay_npz.py --robot t800 --input_file datasets/tracking/t800/dance_t800.npz --physics newton_mjwarp --viz newton_gl
```

The converter accepts `--input_quaternion_order {xyzw,wxyz}` and normalizes both formats to XYZW. The legacy NPZ files shipped in this repository do not contain `quaternion_order` or `body_names`; the loader treats their quaternions as legacy WXYZ data and converts them to XYZW automatically. Body mapping falls back to the fixed PhysX ordering in the robot configuration. Do not rewrite those binary files in bulk.

2. Train:
```bash
# PM01
python scripts/tracking/train.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --num_envs 4096 --motion_file datasets/tracking/pm01/dance.npz physics=newton_mjwarp

# T800
python scripts/tracking/train.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --num_envs 4096 --motion_file datasets/tracking/t800/dance_t800.npz physics=newton_mjwarp

# View training logs.
python -m tensorboard.main --logdir logs
```

3. Evaluate the trained policy and export it:
```bash
# PM01
python scripts/tracking/play.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/pm01/dance.npz --load_run 2026-06-23_09-58-43 --checkpoint dance.pt physics=newton_mjwarp --viz newton_gl

# T800
python scripts/tracking/play.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/t800/dance_t800.npz --load_run 2026-06-28_20-47-15 --checkpoint dance.pt physics=newton_mjwarp --viz newton_gl
```

`--viz newton_gl` opens the Newton GL visualizer. The current project config launches no visualizer when `--viz` is omitted; `--viz none` explicitly disables every visualizer. This baseline removed `--headless` and `--enable_cameras` from the command-line parser, so do not pass those legacy flags. Use the `HEADLESS=1` environment variable only when Kit itself must run without a host window; visualizer selection is still controlled by `--viz`. Run `python scripts/tracking/train.py --task <TASK_ID> --help` to list the physics presets declared by a task.

## Deployment
### Install engineai_robotics_native_sdk
Both simulation deployment and real-robot deployment depend on `engineai_robotics_native_sdk`. For installation instructions, please refer to [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk).

### Sim2Sim
#### Whole-Body Tracking
1. Prepare the data:
- Copy `logs/rsl_rl/xx_flat/xxxx-xx-xx/exported/policy.mnn` to the `assets/config/xxx/rl_dance_example/policies` directory.
- Copy the NPZ motion file to the `assets/config/xxx/rl_dance_example/trajectories` directory.
- Edit `policy_file` and `trajectory_file_npz` in `assets/config/xxx/rl_dance_example/default.yaml`.

> Please make sure that the robot joint positions in the first and last frames of the motion data are basically consistent with those in PD stand. This helps ensure smooth motion when switching policies or modes.

2. Run the simulation:
```bash
# Terminal 1: run the MuJoCo simulation environment.
# Enter the container.
engineai_robotics_env
./scripts/run_mujoco.sh pm01_edu
# ./scripts/run_mujoco.sh t800

# Terminal 2: run the controller.
# Enter the container.
engineai_robotics_env
./run.sh pm01_edu
# ./run.sh t800

# Terminal 3: start the virtual gamepad or use a remote controller.
# Enter the container first, then start the Python program.
engineai_robotics_env
python3 tools/virtual_gamepad/virtual_gamepad.py
```

`play.py` first writes `exported/policy.onnx`, then invokes `MNN.tools.mnnconvert` to produce `exported/policy.mnn`. With only the minimal dependencies it warns and skips the MNN step. Before deployment, install the `[export]` extra and verify that both files exist.

![Gamepad control interface](docs/gamepad.png)

For remote controller key mappings, please refer to [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk).

> After entering MuJoCo, the robot will automatically fall down. Switch to PD stand mode to restore the robot joints to the initial positions. This operation is only recommended in simulation. Press Enter on the keyboard to reset the MuJoCo environment and place the robot in a standing state. You can then enter dance mode. After the robot finishes the motion, it will automatically switch to walk mode.

### Sim2Real
#### Whole-Body Tracking
1. Edit the deployment target robot parameters in `install.sh` from [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk):
```bash
remote_user="user"
remote_host="192.168.0.163"
remote_dir="~/projects/engineai_robotics"
```

2. Run the installation:
```bash
# Enter the container.
engineai_robotics_env

# Install.
./install pm01_edu robot
# ./install t800 robot
```

3. Run on the real robot:

> **Safety Notice:**
> - Make sure the area is open and that all people keep a safe distance from the robot.
> - If the robot behaves abnormally, stop it immediately by pressing the emergency stop button or switching back to `passive` mode.
> - It is recommended to suspend the robot with a support frame first, enter `pd_stand` mode, place it on the ground, and then switch to walking mode.

**Preparation Before Running:**
- Enable the robot motor system using the emergency stop button for PM01 or the remote controller for T800.
- Connect to the robot hotspot.

**Startup Steps:**
```bash
# 1. SSH into the robot (Nezha).
ssh user@192.168.0.163

# 2. Stop the auto-started motion-control program; otherwise native_sdk cannot be started.
sudo systemctl stop robotics.service

# 3. Start native_sdk. Make sure the motor system has been enabled.
cd ~/projects/engineai_robotics
sudo ./run_robot.sh pm01_edu
# sudo ./run_robot.sh t800

# 4. Use the remote controller to switch modes and enter dance mode.
```

## TODO
- [ ] Add AMP humanoid walking.
- [ ] Add walking training based on Direct RL Environment.
- [ ] Improve documentation and tutorials.

## Technical Support
If you encounter any issues while using this project, please submit an issue in this project's GitHub repository and we will reply as soon as possible. 

## License
This project is open-sourced under the **BSD 3-Clause License**. See the [LICENSE](LICENSE.txt) file for details.

## Acknowledgements
This project benefits from the support and contributions of the following open-source projects. We sincerely thank them:

- **[IsaacLab](https://github.com/isaac-sim/IsaacLab)** - The base framework for training and running simulation experiments.
- **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)** - A high-performance reinforcement learning library for legged robots.
- **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** - Inspiration for the project structure and reference implementation of valuable features.
- **[MNN](https://github.com/alibaba/mnn)** - A lightweight, high-performance inference engine for edge deployment.
