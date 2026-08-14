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

当前代码面向 Isaac Lab 3.0 Beta 2，当前环境验证基线为 `develop` 分支提交 `ef4611e0152d3422ac1aa88fe0f0a3922fe8bd7e`。该分支仍在快速迭代，建议安装时固定此提交，不要直接使用不断变化的 `develop` HEAD。

- Newton MJWarp 可在不安装 Isaac Sim 的情况下运行；训练时选择 `physics=newton_mjwarp`。
- Isaac Sim PhysX/Kit 是可选后端，需要 Isaac Sim 6.0.1；选择 `physics=isaacsim_physx --viz kit`。
- Isaac Lab 3.0 要求 Python 3.12，并将四元数顺序改为 XYZW、资产数据改为 Warp-backed 数组。
- 当前基线中的 MJWarp 不强制执行 actuator 的 `velocity_limit`/`velocity_limit_sim`；不要把它当作训练或真机安全限速，策略侧限幅与部署侧安全检查仍然必须保留。
- `SimulationCfg.use_newton_actuators` 必须保持为 `False`（当前配置已显式固定），否则 Newton 原生 actuator 路径会绕过 T800 的 Python 延迟执行器。

## 安装

### 默认开发环境

本工作区默认使用 `$ENGINEAI_WORKSPACE/engineai` 虚拟环境。执行训练、测试、格式化、依赖检查或其他 Python 命令前，必须先激活它：

```bash
cd "$ENGINEAI_WORKSPACE"
source engineai/bin/activate
```

在当前标准工作区中，对应命令为：

```bash
cd /mnt/workspace/lpz/engineai
source engineai/bin/activate
```

除非明确要求重建环境，否则不要调用系统 Python、不要另外创建 `.venv`，也不要通过 `uv run --with ...` 临时解析并下载另一套 PyTorch/CUDA 依赖。激活后统一使用 `python`、`python -m pytest` 和 `python -m pip`；缺少依赖时也应安装到该 `engineai` 环境中。

### 准备 Git LFS

机器人 USD、示例 checkpoint 等文件由 Git LFS 管理。必须先安装并初始化 Git LFS，再拉取仓库内容；否则 USD 文件只是约 130 字节的文本指针，仿真无法加载机器人。在 Ubuntu 上运行：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs python3.12-venv
git lfs install
```

先将变量设为包含 `IsaacLab`、本仓库和虚拟环境的工作区目录。当前机器使用：

```bash
export ENGINEAI_WORKSPACE=/mnt/workspace/lpz/engineai
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

### 创建并激活 Python 环境

Isaac Lab 3.0 只支持 Python 3.12。当前机器的虚拟环境目录名为 `engineai`：

```bash
cd "$ENGINEAI_WORKSPACE"
python3.12 -m venv engineai
source engineai/bin/activate
python -m pip install --upgrade pip
python --version  # 应为 3.12.x
```

环境已经存在时，不要重复创建，只需运行 `source engineai/bin/activate`。

### 安装 Isaac Lab 3.0 和 Newton

若还没有 Isaac Lab 源码，克隆并固定到已验证提交：

```bash
cd "$ENGINEAI_WORKSPACE"
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout ef4611e0152d3422ac1aa88fe0f0a3922fe8bd7e
```

在已激活的 `engineai` 环境中安装 Newton、Newton GL visualizer 和 RSL-RL：

```bash
source "$ENGINEAI_WORKSPACE/engineai/bin/activate"

cd "$ENGINEAI_WORKSPACE/IsaacLab"
./isaaclab.sh -i 'newton,rl[rsl-rl]'
```

Newton GL visualizer 属于当前 Isaac Lab 的基础依赖，`visualizer[newton]` 是无效 selector，不要使用。若还需要 Isaac Sim PhysX/Kit，在上述命令完成后追加安装：

```bash
./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[kit]'
```

不要使用旧文档中的 `isaacsim` 安装 token；当前 CLI 通过 `visualizer[kit]` 安装 Isaac Sim 6.0.1。可先运行 `./isaaclab.sh --help` 核对当前提交支持的 selector。

### 安装 engineai_rl_lab

`isaaclab.sh` 执行结束后当前目录仍是 `IsaacLab`，因此用工作区变量回到本仓库，再在同一个已激活环境中安装：

```bash
cd "$ENGINEAI_WORKSPACE/engineai_rl_lab"
python -m pip install -e 'source/engineai_rl_lab[export]'
python -c "import MNN, gymnasium, isaaclab, isaaclab_newton, newton, onnx, rsl_rl, torch, trimesh, wandb, warp, yaml, engineai_rl_lab; print('environment ready')"
```

`engineai_rl_lab` 的直接运行时依赖由 `source/engineai_rl_lab/setup.py` 管理；Isaac Lab/Newton 由于必须与指定源码提交成套安装，仍由 `isaaclab.sh` 管理，不应再用普通 `requirements.txt` 重复解析。推荐命令安装了 `[export]` 依赖，其中固定 `MNN==3.6.1`，用于把 `policy.onnx` 转为部署所需的 `policy.mnn`。仅看到 `isaaclab` 的 editable 安装并不代表环境完整，上面的导入检查会覆盖训练、导出和日志所需模块。

依赖按用途划分如下：

- 基础训练：`numpy`、`torch`、`gymnasium`、`trimesh`、`PyYAML`、`onnx`、`wandb`，以及基于上游 5.4.1 的 [TWpoint RSL-RL fork](https://github.com/TWpoint/rsl_rl/tree/engineai/custom-networks)。该 fork 提供 EngineAI 所需的 model graph/custom network 支持。
- RSL-RL 传递依赖：该 TWpoint RSL-RL fork 会安装 `tensorboard`、`onnxscript`、`torchvision`、`tensordict` 和 `GitPython`，无需在本项目中重复声明。
- Isaac Lab 组件：代码直接使用 `isaaclab`、`isaaclab_newton`、`isaaclab_physx`、`isaaclab_ov`、`isaaclab_rl` 和 `isaaclab_tasks`；它们由固定提交下的 `isaaclab.sh` 成套安装，并已同步写入扩展清单 `config/extension.toml`。
- MNN 导出：安装 `[export]`，即 `MNN==3.6.1`；它不是普通 ONNX 导出的必需项，但没有它就不会生成 `policy.mnn`。
- 视频录制：按需运行 `python -m pip install -e 'source/engineai_rl_lab[video]'`。
- Neptune 日志：按需运行 `python -m pip install -e 'source/engineai_rl_lab[neptune]'`。
- 所有项目可选依赖：运行 `python -m pip install -e 'source/engineai_rl_lab[all]'`。

`engineai_robotics_native_sdk`、机器人端 MNN C++ runtime、MuJoCo 和虚拟手柄属于部署 SDK/容器依赖，不是本 Python 包的依赖，仍需按照 SDK 仓库的安装脚本安装。用于转换模型的 Python MNN 与部署端 runtime 应保持兼容；当前验证版本为 3.6.1。

当前验证环境的关键版本为 Python 3.12、Isaac Sim 6.0.1（可选）、Newton 1.5.0、Warp 1.16.0、PyTorch 2.11.0，以及基于 RSL-RL 5.4.1 的 TWpoint fork。若安装器解析出不同的大版本，请优先检查 Isaac Lab 的固定提交和 RSL-RL 的 `engineai/custom-networks` 分支。

### 常见安装问题

- `python3.12 -m venv` 不可用：Ubuntu 上先安装 `python3.12-venv`。
- USD 文件只有约 130 字节或加载失败：运行 `git -C engineai_rl_lab lfs pull`。
- `No module named isaaclab_newton/newton/rsl_rl`：确认先激活 `engineai`，再从 `IsaacLab` 目录执行安装命令。
- 不要单独升级 `torch`、`warp-lang` 或 `newton`；需要修复环境时，在固定的 Isaac Lab 提交重新运行安装器。Isaac Lab 会有意覆盖部分 Isaac Sim wheel 的严格依赖版本，因此安装了可选 Isaac Sim 后，通用的 `pip check` 可能报告已知的元数据冲突，不应把它作为此环境唯一的成功标准。

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

对于多个 actor observation group 的策略（例如 T800 v2/v3/v4），`play.py` 还会生成
`exported/deploy_config.yaml`。部署时应同时复制 `.mnn`、动作 `.npz` 和该配置内容；根据 SDK 中的目标目录修改
`policy_file` 与 `trajectory_file_npz`，并为配置分配独立的 `param_tag`。不要把分组输入配置改写成旧版单一
`observation_names` 向量，否则输入的时间维度和 tensor name 会丢失。

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

`play.py` 会先导出 `exported/policy.onnx` 和 `exported/deploy_config.yaml`，验证模型输入与训练 observation group
一致后，再调用 `MNN.tools.mnnconvert` 生成 `exported/policy.mnn`。如果只安装了最小依赖，脚本会提示缺少 MNN
并跳过第二步；部署前应使用上面的 `[export]` 安装方式，并确认三个文件都已生成。
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
- **[TWpoint/rsl_rl](https://github.com/TWpoint/rsl_rl/tree/engineai/custom-networks)** — 基于官方 RSL-RL 的 EngineAI fork，增加 model graph/custom network 支持。
- **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** — 项目结构启发及有价值的功能实现参考。
- **[MNN](https://github.com/alibaba/mnn)** — 轻量级高性能推理引擎，用于端侧部署。
