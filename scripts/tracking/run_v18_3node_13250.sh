#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT] [NUM_ENVS]" >&2
    exit 2
fi

v18_node_rank="$1"
v18_master_addr="${2:-10.66.17.163}"
v18_master_port="${3:-29508}"
v18_num_envs="${4:-11264}"
v18_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v18_python="/mnt/workspace/lpz/engineai/engineai/bin/python"
v18_load_run="2026-08-22_20-17-35_v17-scale"
v18_checkpoint="model_13250.pt"
v18_checkpoint_path="${v18_project_dir}/logs/rsl_rl/tracking_t800/${v18_load_run}/${v18_checkpoint}"

case "$v18_node_rank" in
    0) v18_node_name="master" ;;
    1) v18_node_name="worker0" ;;
    2) v18_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0 (Master), 1 (Worker-0), or 2 (Worker-1)." >&2
        exit 2
        ;;
esac

if [[ ! "$v18_num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got: ${v18_num_envs}" >&2
    exit 2
fi

v18_log="${v18_project_dir}/v18_scale_resume_13250_3node_${v18_num_envs}_${v18_node_name}.log"

cd "$v18_project_dir"
exec > >(tee "$v18_log") 2>&1

echo "[preflight] node=${v18_node_name} rank=${v18_node_rank} master=${v18_master_addr}:${v18_master_port} envs=${v18_num_envs}"
[[ -x "$v18_python" ]] || { echo "Missing Python: ${v18_python}" >&2; exit 1; }
[[ -f "$v18_checkpoint_path" ]] || { echo "Missing checkpoint: ${v18_checkpoint_path}" >&2; exit 1; }

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v18_node_name}, found ${gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

"$v18_python" - <<'PY'
import gymnasium as gym

import engineai_rl_lab.tasks  # noqa: F401

spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v18-scale")
assert spec.kwargs["env_cfg_entry_point"].endswith(
    ".flat_env_cfg_v18:T800FlatWoStateEstimationEnvCfgV18Scale"
)
assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
    ".rsl_rl_ppo_cfg_v18:T800FlatV18ScalePPORunnerCfg"
)
print("[preflight] V18 task registration is correct.")
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

echo "[launch] log=${v18_log}"
exec "$v18_python" -m torch.distributed.run \
    --nnodes=3 \
    --nproc-per-node=8 \
    --node-rank="$v18_node_rank" \
    --master-addr="$v18_master_addr" \
    --master-port="$v18_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v18-scale \
    --distributed \
    --num_envs "$v18_num_envs" \
    --seed 42 \
    --resume \
    --load_run "$v18_load_run" \
    --checkpoint "$v18_checkpoint" \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
