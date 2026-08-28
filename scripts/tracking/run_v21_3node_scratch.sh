#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT] [NUM_ENVS] [MAX_ITERATIONS]" >&2
    exit 2
fi

v21_node_rank="$1"
v21_master_addr="${2:-10.66.17.163}"
v21_master_port="${3:-29516}"
v21_num_envs="${4:-8192}"
v21_max_iterations="${5:-}"
v21_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v21_python="/mnt/workspace/lpz/engineai/engineai/bin/python"

case "$v21_node_rank" in
    0) v21_node_name="master" ;;
    1) v21_node_name="worker0" ;;
    2) v21_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0 (Master), 1 (Worker-0), or 2 (Worker-1)." >&2
        exit 2
        ;;
esac

if [[ ! "$v21_master_port" =~ ^[0-9]+$ ]] || ((v21_master_port < 1 || v21_master_port > 65535)); then
    echo "MASTER_PORT must be an integer in [1, 65535], got: ${v21_master_port}" >&2
    exit 2
fi
if [[ ! "$v21_num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got: ${v21_num_envs}" >&2
    exit 2
fi
if [[ -n "$v21_max_iterations" && ! "$v21_max_iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_ITERATIONS must be a positive integer, got: ${v21_max_iterations}" >&2
    exit 2
fi

v21_session="v21_scratch_p${v21_master_port}_r${v21_node_rank}"
if [[ -z "${TMUX:-}" && "${V21_SCRATCH_TMUX_LAUNCHED:-0}" != "1" ]]; then
    if tmux has-session -t "$v21_session" 2>/dev/null; then
        echo "tmux session already exists: ${v21_session}" >&2
        echo "Attach with: tmux attach -t ${v21_session}" >&2
        exit 1
    fi

    v21_script_path="$(realpath "$0")"
    printf -v v21_tmux_command 'V21_SCRATCH_TMUX_LAUNCHED=1 bash %q' "$v21_script_path"
    for v21_arg in "$@"; do
        printf -v v21_tmux_command '%s %q' "$v21_tmux_command" "$v21_arg"
    done
    tmux new-session -d -s "$v21_session" "$v21_tmux_command"
    echo "Started tmux session: ${v21_session}"
    echo "Attach with: tmux attach -t ${v21_session}"
    exit 0
fi

v21_launch_stamp="$(date +%Y%m%d_%H%M%S)"
v21_iteration_label="${v21_max_iterations:-unlimited}"
v21_log="${v21_project_dir}/v21_scratch_3node_${v21_num_envs}env_${v21_iteration_label}iter_p${v21_master_port}_${v21_node_name}_${v21_launch_stamp}.log"

cd "$v21_project_dir"
exec > >(tee "$v21_log") 2>&1

echo "[preflight] mode=scratch node=${v21_node_name} rank=${v21_node_rank} master=${v21_master_addr}:${v21_master_port} envs_per_rank=${v21_num_envs} max_iterations=${v21_max_iterations:-unlimited}"
[[ -x "$v21_python" ]] || { echo "Missing Python: ${v21_python}" >&2; exit 1; }

v21_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$v21_gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v21_node_name}, found ${v21_gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

"$v21_python" - <<'PY'
from pathlib import Path

import gymnasium as gym

import engineai_rl_lab.tasks  # noqa: F401
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v21 import (
    T800FlatV21ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v21 import (
    T800FlatWoStateEstimationEnvCfgV21Scale,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import MotionCommandV1, MotionCommandV1Cfg
from engineai_rl_lab.tasks.tracking.robots.t800 import T800_HEAD_JOINT_POSITIONS, T800_POLICY_JOINT_NAMES

spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v21-scale")
assert spec.kwargs["env_cfg_entry_point"].endswith(
    ".flat_env_cfg_v21:T800FlatWoStateEstimationEnvCfgV21Scale"
)
assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
    ".rsl_rl_ppo_cfg_v21:T800FlatV21ScalePPORunnerCfg"
)

runtime_cfg = T800FlatWoStateEstimationEnvCfgV21Scale()
runner_cfg = T800FlatV21ScalePPORunnerCfg()
motion_cfg = runtime_cfg.commands.motion
assert isinstance(motion_cfg, MotionCommandV1Cfg)
assert motion_cfg.class_type is MotionCommandV1
assert runtime_cfg.actions.joint_pos.joint_names == T800_POLICY_JOINT_NAMES
assert len(runtime_cfg.actions.joint_pos.scale) == 23
assert motion_cfg.fixed_joint_positions == T800_HEAD_JOINT_POSITIONS
assert runner_cfg.run_name == "v21-scale"
assert not runner_cfg.resume
assert runner_cfg.num_steps_per_env == 32
assert runner_cfg.algorithm.num_learning_epochs == 2
assert runner_cfg.algorithm.num_mini_batches == 16
assert runner_cfg.algorithm.max_grad_norm == 0.1
assert runner_cfg.algorithm.critic_learning_rate == 2.0e-5
assert runner_cfg.actor.distribution_cfg.init_std == 0.05
assert runner_cfg.actor.distribution_cfg.std_range == (0.001, 0.5)
assert Path(motion_cfg.motion_file).is_file(), f"Missing motion manifest: {motion_cfg.motion_file}"
print("[preflight] V21 scratch config is correct; no checkpoint will be loaded.")
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

v21_iteration_args=()
if [[ -n "$v21_max_iterations" ]]; then
    v21_iteration_args=(--max_iterations "$v21_max_iterations")
fi

echo "[launch] mode=scratch world_size=24 log=${v21_log}"
exec "$v21_python" -m torch.distributed.run \
    --nnodes=3 \
    --nproc-per-node=8 \
    --node-rank="$v21_node_rank" \
    --master-addr="$v21_master_addr" \
    --master-port="$v21_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v21-scale \
    --distributed \
    --num_envs "$v21_num_envs" \
    "${v21_iteration_args[@]}" \
    --seed 42 \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
