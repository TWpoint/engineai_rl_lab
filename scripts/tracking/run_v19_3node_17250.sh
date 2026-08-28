#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT] [NUM_ENVS]" >&2
    exit 2
fi

v19_node_rank="$1"
v19_master_addr="${2:-10.66.17.163}"
v19_master_port="${3:-29509}"
v19_num_envs="${4:-11264}"
v19_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v19_python="/mnt/workspace/lpz/engineai/engineai/bin/python"
v19_load_run="2026-08-23_01-23-21_v18-scale"
v19_checkpoint="model_17250.pt"
v19_checkpoint_path="${v19_project_dir}/logs/rsl_rl/tracking_t800/${v19_load_run}/${v19_checkpoint}"

case "$v19_node_rank" in
    0) v19_node_name="master" ;;
    1) v19_node_name="worker0" ;;
    2) v19_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0 (Master), 1 (Worker-0), or 2 (Worker-1)." >&2
        exit 2
        ;;
esac

if [[ ! "$v19_num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got: ${v19_num_envs}" >&2
    exit 2
fi

v19_log="${v19_project_dir}/v19_scale_resume_17250_3node_${v19_num_envs}_${v19_node_name}.log"

cd "$v19_project_dir"
exec > >(tee "$v19_log") 2>&1

echo "[preflight] node=${v19_node_name} rank=${v19_node_rank} master=${v19_master_addr}:${v19_master_port} envs=${v19_num_envs}"
[[ -x "$v19_python" ]] || { echo "Missing Python: ${v19_python}" >&2; exit 1; }
[[ -f "$v19_checkpoint_path" ]] || { echo "Missing checkpoint: ${v19_checkpoint_path}" >&2; exit 1; }

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v19_node_name}, found ${gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

V19_CHECKPOINT_PATH="$v19_checkpoint_path" "$v19_python" - <<'PY'
import os

import gymnasium as gym
import torch

import engineai_rl_lab.tasks  # noqa: F401

spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v19-scale")
assert spec.kwargs["env_cfg_entry_point"].endswith(
    ".flat_env_cfg_v19:T800FlatWoStateEstimationEnvCfgV19Scale"
)
assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
    ".rsl_rl_ppo_cfg_v19:T800FlatV19ScalePPORunnerCfg"
)
checkpoint = torch.load(os.environ["V19_CHECKPOINT_PATH"], map_location="cpu", weights_only=False)
adaptive = checkpoint["infos"]["motion_adaptive_sampling"]
assert int(checkpoint["iter"]) == 17250
assert int(adaptive["curriculum_state_schema_version"].item()) == 7
probabilities = adaptive["curriculum_smoothed_probabilities"].float()
assert torch.all(torch.isfinite(probabilities))
assert torch.all(probabilities >= 0.0)
assert torch.isclose(probabilities.sum(), torch.tensor(1.0), rtol=0.0, atol=1.0e-5)
print("[preflight] V19 registration and V18/model_17250 schema-7 checkpoint are correct.")
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

echo "[launch] log=${v19_log}"
exec "$v19_python" -m torch.distributed.run \
    --nnodes=3 \
    --nproc-per-node=8 \
    --node-rank="$v19_node_rank" \
    --master-addr="$v19_master_addr" \
    --master-port="$v19_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v19-scale \
    --distributed \
    --num_envs "$v19_num_envs" \
    --seed 42 \
    --resume \
    --load_run "$v19_load_run" \
    --checkpoint "$v19_checkpoint" \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
