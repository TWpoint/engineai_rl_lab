#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT] [NUM_ENVS] [MAX_ITERATIONS]" >&2
    exit 2
fi

v22_node_rank="$1"
v22_master_addr="${2:-10.66.17.163}"
v22_master_port="${3:-29517}"
v22_num_envs="${4:-8192}"
v22_max_iterations="${5:-}"
v22_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v22_python="/mnt/workspace/lpz/engineai/engineai/bin/python"

case "$v22_node_rank" in
    0) v22_node_name="master" ;;
    1) v22_node_name="worker0" ;;
    2) v22_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0 (Master), 1 (Worker-0), or 2 (Worker-1)." >&2
        exit 2
        ;;
esac

if [[ ! "$v22_master_port" =~ ^[0-9]+$ ]] || ((v22_master_port < 1 || v22_master_port > 65535)); then
    echo "MASTER_PORT must be an integer in [1, 65535], got: ${v22_master_port}" >&2
    exit 2
fi
if [[ ! "$v22_num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got: ${v22_num_envs}" >&2
    exit 2
fi
if [[ -n "$v22_max_iterations" && ! "$v22_max_iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_ITERATIONS must be a positive integer, got: ${v22_max_iterations}" >&2
    exit 2
fi

v22_session="v22_scratch_p${v22_master_port}_r${v22_node_rank}"
if [[ -z "${TMUX:-}" && "${V22_SCRATCH_TMUX_LAUNCHED:-0}" != "1" ]]; then
    if tmux has-session -t "$v22_session" 2>/dev/null; then
        echo "tmux session already exists: ${v22_session}" >&2
        echo "Attach with: tmux attach -t ${v22_session}" >&2
        exit 1
    fi

    v22_script_path="$(realpath "$0")"
    printf -v v22_tmux_command 'V22_SCRATCH_TMUX_LAUNCHED=1 bash %q' "$v22_script_path"
    for v22_arg in "$@"; do
        printf -v v22_tmux_command '%s %q' "$v22_tmux_command" "$v22_arg"
    done
    tmux new-session -d -s "$v22_session" "$v22_tmux_command"
    echo "Started tmux session: ${v22_session}"
    echo "Attach with: tmux attach -t ${v22_session}"
    exit 0
fi

v22_launch_stamp="$(date +%Y%m%d_%H%M%S)"
v22_iteration_label="${v22_max_iterations:-unlimited}"
v22_log="${v22_project_dir}/v22_scratch_3node_${v22_num_envs}env_${v22_iteration_label}iter_p${v22_master_port}_${v22_node_name}_${v22_launch_stamp}.log"

cd "$v22_project_dir"
exec > >(tee "$v22_log") 2>&1

echo "[preflight] mode=scratch node=${v22_node_name} rank=${v22_node_rank} master=${v22_master_addr}:${v22_master_port} envs_per_rank=${v22_num_envs} max_iterations=${v22_max_iterations:-unlimited}"
[[ -x "$v22_python" ]] || { echo "Missing Python: ${v22_python}" >&2; exit 1; }

v22_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$v22_gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v22_node_name}, found ${v22_gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

"$v22_python" - <<'PY'
from pathlib import Path

import gymnasium as gym

import engineai_rl_lab.tasks  # noqa: F401
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v22 import (
    T800FlatV22ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v22 import (
    T800FlatWoStateEstimationEnvCfgV22Scale,
)
from engineai_rl_lab.tasks.tracking.mdp import (
    motion_local_body_position_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import MotionCommandV1, MotionCommandV1Cfg
from engineai_rl_lab.tasks.tracking.robots.t800 import T800_HEAD_JOINT_POSITIONS, T800_POLICY_JOINT_NAMES

spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v22-scale")
assert spec.kwargs["env_cfg_entry_point"].endswith(
    ".flat_env_cfg_v22:T800FlatWoStateEstimationEnvCfgV22Scale"
)
assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
    ".rsl_rl_ppo_cfg_v22:T800FlatV22ScalePPORunnerCfg"
)

runtime_cfg = T800FlatWoStateEstimationEnvCfgV22Scale()
runner_cfg = T800FlatV22ScalePPORunnerCfg()
motion_cfg = runtime_cfg.commands.motion
assert isinstance(motion_cfg, MotionCommandV1Cfg)
assert motion_cfg.class_type is MotionCommandV1
assert runtime_cfg.actions.joint_pos.joint_names == T800_POLICY_JOINT_NAMES
assert len(runtime_cfg.actions.joint_pos.scale) == 23
assert motion_cfg.fixed_joint_positions == T800_HEAD_JOINT_POSITIONS
assert runner_cfg.run_name == "v22-scale"
assert not runner_cfg.resume
assert runner_cfg.num_steps_per_env == 32
assert runner_cfg.algorithm.num_learning_epochs == 2
assert runner_cfg.algorithm.num_mini_batches == 16
assert runner_cfg.algorithm.max_grad_norm == 0.1
assert runner_cfg.algorithm.entropy_coef == 0.01
assert runner_cfg.algorithm.critic_learning_rate == 1.0e-3
assert runner_cfg.algorithm.shared_kl_adaptation is True
assert runner_cfg.actor.distribution_cfg.init_std == 0.05
assert runner_cfg.actor.distribution_cfg.std_range == (0.001, 0.5)
assert runtime_cfg.rewards.motion_local_end_effector_pos.func is motion_local_body_position_error_exp
assert runtime_cfg.rewards.motion_local_end_effector_pos.weight == 1.0
assert runtime_cfg.rewards.motion_relative_body_pos.func is motion_relative_body_position_error_exp
assert runtime_cfg.rewards.motion_relative_body_pos.weight == 0.5
assert runtime_cfg.rewards.motion_relative_body_ori.func is motion_relative_body_orientation_error_exp
assert runtime_cfg.rewards.motion_relative_body_ori.weight == 0.5
assert Path(motion_cfg.motion_file).is_file(), f"Missing motion manifest: {motion_cfg.motion_file}"
print("[preflight] V22 scratch config is correct; no checkpoint will be loaded.")
PY

# Remove stale launcher variables inherited from an earlier torchrun shell.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

v22_iteration_args=()
if [[ -n "$v22_max_iterations" ]]; then
    v22_iteration_args=(--max_iterations "$v22_max_iterations")
fi

echo "[launch] mode=scratch world_size=24 log=${v22_log}"
exec "$v22_python" -m torch.distributed.run \
    --nnodes=3 \
    --nproc-per-node=8 \
    --node-rank="$v22_node_rank" \
    --master-addr="$v22_master_addr" \
    --master-port="$v22_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v22-scale \
    --distributed \
    --num_envs "$v22_num_envs" \
    "${v22_iteration_args[@]}" \
    --seed 42 \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
