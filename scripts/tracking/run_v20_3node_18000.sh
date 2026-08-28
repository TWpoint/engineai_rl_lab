#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT] [NUM_ENVS]" >&2
    exit 2
fi

v20_node_rank="$1"
v20_master_addr="${2:-10.66.17.163}"
v20_master_port="${3:-29511}"
v20_num_envs="${4:-11264}"
v20_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v20_python="/mnt/workspace/lpz/engineai/engineai/bin/python"
v20_load_run="2026-08-24_00-01-57_v19-scale"
v20_checkpoint="model_18000.pt"
v20_checkpoint_path="${v20_project_dir}/logs/rsl_rl/tracking_t800/${v20_load_run}/${v20_checkpoint}"

case "$v20_node_rank" in
    0) v20_node_name="master" ;;
    1) v20_node_name="worker0" ;;
    2) v20_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0 (Master), 1 (Worker-0), or 2 (Worker-1)." >&2
        exit 2
        ;;
esac

if [[ ! "$v20_num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got: ${v20_num_envs}" >&2
    exit 2
fi

v20_session="v20_18000_r${v20_node_rank}"
if [[ -z "${TMUX:-}" && "${V20_TMUX_LAUNCHED:-0}" != "1" ]]; then
    if tmux has-session -t "$v20_session" 2>/dev/null; then
        echo "tmux session already exists: ${v20_session}" >&2
        echo "Attach with: tmux attach -t ${v20_session}" >&2
        exit 1
    fi

    v20_script_path="$(realpath "$0")"
    printf -v v20_tmux_command 'V20_TMUX_LAUNCHED=1 bash %q' "$v20_script_path"
    for v20_arg in "$@"; do
        printf -v v20_tmux_command '%s %q' "$v20_tmux_command" "$v20_arg"
    done
    tmux new-session -d -s "$v20_session" "$v20_tmux_command"
    echo "Started tmux session: ${v20_session}"
    echo "Attach with: tmux attach -t ${v20_session}"
    exit 0
fi

v20_log="${v20_project_dir}/v20_resume_18000_3node_${v20_num_envs}_${v20_node_name}.log"

cd "$v20_project_dir"
exec > >(tee "$v20_log") 2>&1

echo "[preflight] node=${v20_node_name} rank=${v20_node_rank} master=${v20_master_addr}:${v20_master_port} envs=${v20_num_envs}"
[[ -x "$v20_python" ]] || { echo "Missing Python: ${v20_python}" >&2; exit 1; }
[[ -f "$v20_checkpoint_path" ]] || { echo "Missing checkpoint: ${v20_checkpoint_path}" >&2; exit 1; }

v20_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$v20_gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v20_node_name}, found ${v20_gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

V20_CHECKPOINT_PATH="$v20_checkpoint_path" "$v20_python" - <<'PY'
import os

import gymnasium as gym
import torch

import engineai_rl_lab.tasks  # noqa: F401
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v20 import (
    T800FlatWoStateEstimationEnvCfgV20Scale,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import MotionCommandV1, MotionCommandV1Cfg

spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v20-scale")
assert spec.kwargs["env_cfg_entry_point"].endswith(
    ".flat_env_cfg_v20:T800FlatWoStateEstimationEnvCfgV20Scale"
)
assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
    ".rsl_rl_ppo_cfg_v20:T800FlatV20ScalePPORunnerCfg"
)
runtime_cfg = T800FlatWoStateEstimationEnvCfgV20Scale()
assert isinstance(runtime_cfg.commands.motion, MotionCommandV1Cfg)
assert runtime_cfg.commands.motion.class_type is MotionCommandV1
assert runtime_cfg.commands.motion.adaptive_sampling.slow_half_life == 64.0
assert runtime_cfg.commands.motion.adaptive_sampling.probability_cap_ratio == 200.0
assert runtime_cfg.rewards.joint_acc_l2 is None

checkpoint = torch.load(os.environ["V20_CHECKPOINT_PATH"], map_location="cpu", weights_only=False)
assert int(checkpoint["iter"]) == 18000
for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
    assert key in checkpoint
print("[preflight] V20 config and V19/model_18000 policy checkpoint are correct.")
PY

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[launch] log=${v20_log}"
exec "$v20_python" -m torch.distributed.run \
    --nnodes=3 \
    --nproc-per-node=8 \
    --node-rank="$v20_node_rank" \
    --master-addr="$v20_master_addr" \
    --master-port="$v20_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v20-scale \
    --distributed \
    --num_envs "$v20_num_envs" \
    --seed 42 \
    --resume \
    --load_run "$v20_load_run" \
    --checkpoint "$v20_checkpoint" \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
