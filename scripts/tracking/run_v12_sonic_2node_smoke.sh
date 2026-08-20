#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT]" >&2
    exit 2
fi

smoke_node_rank="$1"
smoke_master_addr="${2:-10.66.17.104}"
smoke_master_port="${3:-29501}"
smoke_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
smoke_python="/mnt/workspace/lpz/engineai/engineai/bin/python"
smoke_log="${smoke_project_dir}/v12_sonic_2node_smoke_node${smoke_node_rank}.log"

if [[ "$smoke_node_rank" != "0" && "$smoke_node_rank" != "1" ]]; then
    echo "NODE_RANK must be 0 (Master) or 1 (Worker)." >&2
    exit 2
fi

cd "$smoke_project_dir"

# Capture preflight failures as well as the training output. Without this, a
# detached tmux session can disappear before leaving any diagnostics behind.
exec > >(tee "$smoke_log") 2>&1

# Verify the node's NVIDIA kernel driver before launching. Do not override
# NVML with LD_LIBRARY_PATH: mixing driver userspace components can make NCCL
# abort inside native code even when nvidia-smi alone appears to work.
smoke_driver_version="$(grep -oE '[0-9]{3}\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -n 1 || true)"
if [[ -z "$smoke_driver_version" ]]; then
    echo "[preflight] Could not read the active NVIDIA kernel driver version." >&2
    exit 1
fi
echo "[preflight] kernel driver=${smoke_driver_version}"

echo "[preflight] node_rank=${smoke_node_rank} master=${smoke_master_addr}:${smoke_master_port}"
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$smoke_python" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 8:
    raise RuntimeError(f"Expected 8 visible GPUs, found {count}")
for index in range(count):
    print(f"[preflight] cuda:{index} {torch.cuda.get_device_name(index)}")
PY

# JoyBuilder shells may retain variables from an earlier torchrun. Remove them
# before the new launcher creates the correct 16-rank environment.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
# These nodes do not have a usable NCCL IB plugin; use the working socket path.
export NCCL_IB_DISABLE=1

echo "[launch] writing ${smoke_log}"
"$smoke_python" -m torch.distributed.run \
    --nnodes=2 \
    --nproc-per-node=8 \
    --node-rank="$smoke_node_rank" \
    --master-addr="$smoke_master_addr" \
    --master-port="$smoke_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v12-scale-lafan-sonic \
    --distributed \
    --num_envs 256 \
    --max_iterations 4 \
    physics=newton_mjwarp \
    agent.motion_resample_frequency=2 \
    agent.sync_adaptive_sampling_all_gpus_freq=2
