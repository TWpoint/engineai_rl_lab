# engineai_rl_lab
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0_beta2+-silver)](https://github.com/isaac-sim/IsaacLab/releases)
[![Newton](https://img.shields.io/badge/Physics-Newton_MJWarp-blue)](https://github.com/newton-physics/newton)
[![Isaac Sim](https://img.shields.io/badge/Optional-Isaac_Sim_6.0.1-silver)](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/index.html)

[English](README_EN.md)

## 概览
本项目提供了一套基于Isaac Lab的强化学习环境，目前已经支持众擎PM01、T800机器人，实现的任务包括whole body tracking。
|Robot|Training| Sim2Sim |Deploy|
|:--------:|:--------:|:--------:|:--------:|
|||**whole body tracking**|||
|**T800**|<img src="./docs/train.gif" height="180"/>|<img src="./docs/sim2sim.gif" height="180"/>|<img src="./docs/deploy.gif" height="180"/>|
|**PM01**|<img src="./docs/train_pm.gif" height="180"/>|<img src="./docs/sim2sim_pm.gif" height="180"/>|<img src="./docs/deploy_pm.gif" height="180"/>|

## 兼容性

当前代码面向 Isaac Lab 3.0，迁移和验证基线为 `develop` 分支提交 `891116e68dac3d417cb5de2d722d16d44ae3f06d`，并使用 Isaac Lab 3.0 Beta 2 引入的多物理后端和 preset CLI。3.0 仍在快速迭代；升级 Isaac Lab 后请先阅读[官方发布说明](https://github.com/isaac-sim/IsaacLab/releases)和[迁移指南](https://isaac-sim.github.io/IsaacLab/develop/source/migration/migrating_to_isaaclab_3-0.html)。

- Newton MJWarp 可在不安装 Isaac Sim 的情况下运行；训练时选择 `physics=newton_mjwarp`。
- Isaac Sim PhysX/Kit 是可选后端，需要 Isaac Sim 6.0.1；选择 `physics=isaacsim_physx --viz kit`。
- Isaac Lab 3.0 要求 Python 3.12，并将四元数顺序改为 XYZW、资产数据改为 Warp-backed 数组。
- 当前基线中的 MJWarp 不强制执行 actuator 的 `velocity_limit`/`velocity_limit_sim`；不要把它当作训练或真机安全限速，策略侧限幅与部署侧安全检查仍然必须保留。
- `SimulationCfg.use_newton_actuators` 必须保持为 `False`（当前配置已显式固定），否则 Newton 原生 actuator 路径会绕过 T800 的 Python 延迟执行器。

## 安装

### 准备 Git LFS

机器人 USD、示例 checkpoint 等文件由 Git LFS 管理。必须先安装并初始化 Git LFS，再拉取仓库内容；否则 USD 文件只是约 130 字节的文本指针，仿真无法加载机器人。在 Ubuntu 上运行：

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
```

先将变量设为包含 `IsaacLab` 和本仓库的工作区目录。当前机器可使用
`export ENGINEAI_WORKSPACE=/home/ubuntu/engineai-newton`；其他机器请替换示例路径：

```bash
export ENGINEAI_WORKSPACE=/path/to/your/workspace
cd "$ENGINEAI_WORKSPACE"
```

如果工作区中已经有仓库，直接补拉 LFS 文件：

```bash
git -C engineai_rl_lab lfs pull
```

全新工作区则应在 `git lfs install` 之后克隆：

```bash
cd "$ENGINEAI_WORKSPACE"
git clone https://github.com/engineai-robotics/engineai_rl_lab.git
git -C engineai_rl_lab lfs pull
```

### 安装 Isaac Lab 3.0 和 Newton

下面假设 `IsaacLab`、Python 3.12 虚拟环境 `engineai-newton` 和本仓库都位于
`$ENGINEAI_WORKSPACE` 下：

```bash
cd "$ENGINEAI_WORKSPACE"
source "$ENGINEAI_WORKSPACE/engineai-newton/bin/activate"

cd IsaacLab
./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'
```

这会安装 Newton、RSL-RL 和 Newton visualizer，但不会安装 Isaac Sim。若还需要 PhysX/Kit，请改用：

```bash
./isaaclab.sh -i 'isaacsim,newton,rl[rsl-rl],visualizer[kit]'
```

全新环境请按 Isaac Lab 3.0 官方流程创建 Python 3.12 环境，并在 IsaacLab 根目录运行 `./isaaclab.sh -i`。详细步骤见 [Isaac Lab 安装指南](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html)。

### 安装 engineai_rl_lab

`isaaclab.sh` 执行结束后当前目录仍是 `IsaacLab`，因此用工作区变量回到本仓库，再在同一个已激活环境中安装：

```bash
cd "$ENGINEAI_WORKSPACE/engineai_rl_lab"
python -m pip install -e source/engineai_rl_lab
python -m pip check
python -c "import isaaclab, isaaclab_newton, newton, warp, rsl_rl, engineai_rl_lab; print('environment ready')"
```

仅看到 `isaaclab` 的 editable 安装并不代表环境完整；上面的导入检查还会确认 Newton、Warp 相关依赖和 RSL-RL 已实际安装。

## 训练
### whole body tracking

以下命令都应从仓库根目录 `$ENGINEAI_WORKSPACE/engineai_rl_lab` 运行，并保持上述虚拟环境处于激活状态。当前四个脚本省略 physics selector 时都会使用 Isaac Sim PhysX；使用 Newton 时请保留示例中的 selector。两类脚本的写法不同：

- `train.py` / `play.py` 使用 Hydra preset token：`physics=newton_mjwarp`，前面没有 `--`。
- `csv_to_npz.py` / `replay_npz.py` 使用 argparse 选项：`--physics newton_mjwarp`。

1. 将csv文件转换npz文件
```bash
# 仓库自带 CSV 为 XYZW；显式声明输入顺序。生成的 NPZ 始终保存为 Isaac Lab 3.0 的 XYZW
python scripts/csv_to_npz.py --robot pm01 --input_fps 30 --input_quaternion_order xyzw -f datasets/tracking/pm01/dance.csv --physics newton_mjwarp
python scripts/csv_to_npz.py --robot t800 --input_fps 30 --input_quaternion_order xyzw -f datasets/tracking/t800/dance_t800.csv --physics newton_mjwarp

# 如果外部 CSV 的第 4–7 列是 WXYZ，请改用：
python scripts/csv_to_npz.py --robot pm01 --input_fps 30 --input_quaternion_order wxyz -f path/to/motion.csv --physics newton_mjwarp

# 使用 Newton visualizer 重放 NPZ
python scripts/replay_npz.py --robot pm01 --input_file datasets/tracking/pm01/dance.npz --physics newton_mjwarp --viz newton_gl
python scripts/replay_npz.py --robot t800 --input_file datasets/tracking/t800/dance_t800.npz --physics newton_mjwarp --viz newton_gl
```

转换器支持 `--input_quaternion_order {xyzw,wxyz}`，内部会统一转为 XYZW。仓库内已有的旧 NPZ 没有 `quaternion_order` 和 `body_names` 元数据；加载器会按旧版 WXYZ 读取并自动转为 XYZW，body 顺序则依靠机器人配置中固定的 PhysX 顺序兼容。请勿手动批量改写这些二进制数据。

2. 训练
```bash
# PM01
python scripts/tracking/train.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --num_envs 4096 --motion_file datasets/tracking/pm01/dance.npz physics=newton_mjwarp

# T800
python scripts/tracking/train.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --num_envs 4096 --motion_file datasets/tracking/t800/dance_t800.npz physics=newton_mjwarp

# 查看训练日志
python -m tensorboard.main --logdir logs
```

3. 验证训练效果并导出策略
```bash
# PM01
python scripts/tracking/play.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/pm01/dance.npz --load_run 2026-06-23_09-58-43 --checkpoint dance.pt physics=newton_mjwarp --viz newton_gl

# T800
python scripts/tracking/play.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/t800/dance_t800.npz --load_run 2026-06-28_20-47-15 --checkpoint dance.pt physics=newton_mjwarp --viz newton_gl
```

`--viz newton_gl` 会打开 Newton GL visualizer。本项目当前配置省略 `--viz` 时不会启动 visualizer；`--viz none` 可显式禁用所有 visualizer。此基线已从命令行解析器移除 `--headless` 和 `--enable_cameras`，不要再传这两个旧参数；若需控制 Kit 的无窗口运行模式，可使用 `HEADLESS=1` 环境变量，但 visualizer 的启停仍由 `--viz` 决定。可用 `python scripts/tracking/train.py --task <TASK_ID> --help` 查看任务声明的 physics presets。

## 部署
### 安装engineai_robotics_native_sdk
仿真与真机部署均依赖安装engineai_robotics_native_sdk，具体安装教程请参阅[engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk)

### Sim2Sim
#### whole body tracking
1. 数据准备
- 将`logs/rsl_rl/xx_flat/xxxx-xx-xx/exported/policy.mnn`复制到`assets/config/xxx/rl_dance_example/policies`目录下
- 将npz动作文件复制到`assets/config/xxx/rl_dance_example/trajectories`目录下
- 修改`assets/config/xxx/rl_dance_example/default.yaml`文件内的`policy_file`、`trajectory_file_npz`的文件名

> 请保证动作数据的第一帧和最后一帧的机器人关节位置和pd stand下的基本一致，这有利于切换策略(模式)时的动作流畅性。

2. 运行仿真
```bash
# 终端1：运行mujoco仿真环境
# 进入容器
engineai_robotics_env
./scripts/run_mujoco.sh pm01_edu
# ./scripts/run_mujoco.sh t800

# 终端2：运行控制程序
# 进入容器
engineai_robotics_env
./run.sh pm01_edu
# ./run.sh t800

# 终端3：启动虚拟手柄或使用遥控器
# 进入容器后再启动python程序
engineai_robotics_env
python3 tools/virtual_gamepad/virtual_gamepad.py
```
![手柄控制界面](docs/gamepad.png)

遥控器操作请参阅[engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk)中的键位。
> 进入mujoco后机器人会自动倒地，此时应切换到pd stand模式使机器人关节恢复到初始位置(仅仿真环境可以这样操作),按下键盘中的Enter即可重置mujoco环境，使机器人处于站立状态，这时可进入dance模式。机器人执行完动作之后自动会切换到walk状态。

### Sim2Real
#### whole body tracking
1. 编辑[engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk)中的`install.sh` 中的要部署目标机器人参数：
```bash
remote_user="user"
remote_host="192.168.0.163"
remote_dir="~/projects/engineai_robotics"
```

2. 执行安装：
```bash
# 进入容器
engineai_robotics_env
# 安装程序
./install pm01_edu robot
# ./install t800 robot
```

3. 真机运行
>  **安全提示：** 
> - 确保场地空旷，所有人员与机器人保持安全距离
> - 若机器人动作异常，随时快速停止（按急停键或切回 `passive` 模式）
> - 建议先用吊架吊起机器人，在进入 `pd_stand` 模式之后放到地上，再切入行走模式

**运行前准备：**
- 利用急停按键(pm01)或遥控器(t800)使能机器人的电机系统
- 连接机器人热点

**启动步骤：**
```bash
# 1. SSH 连接机器人（Nezha）
ssh user@192.168.0.163

# 2. 暂停自启动的运控程序，否则无法启动native_sdk
sudo systemctl stop robotics.service

# 3. 启动 native_sdk(确保已经使能电机系统)
cd ~/projects/engineai_robotics
sudo ./run_robot.sh pm01_edu
# sudo ./run_robot.sh t800

# 4. 使用遥控器切换模式，进入dance模式
```
## TODO
- [ ] 添加AMP拟人行走
- [ ] 添加基于Direct RL Environment的行走训练
- [ ] 完善文档与教程

## 技术支持
如果您在使用本项目过程中遇到任何问题，请在本项目的GitHub仓库中提交Issue，我们会尽快回复。同时，也欢迎您加入我们的众擎机器人开发者交流微信群。
<div align="center"> <img src="docs/weixin.png" height="300" alt="微信交流群"/> <br> <em>扫码加入众擎机器人开发者交流微信群</em> </div>

## 许可证
本项目采用 **BSD 3-Clause License** 开源协议。详见 [LICENSE](LICENSE.txt) 文件。

## 致谢
本项目得益于以下开源项目的支持和贡献，在此表示衷心的感谢：
- **[IsaacLab](https://github.com/isaac-sim/IsaacLab)** — 训练和运行仿真实验的基础框架。
- **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)** — 适用于足式机器人的高性能强化学习库。
- **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** — 项目结构启发及有价值的功能实现参考。
- **[MNN](https://github.com/alibaba/mnn)** — 轻量级高性能推理引擎，用于端侧部署。
