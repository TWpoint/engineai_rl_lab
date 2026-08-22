#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT]" >&2
    exit 2
fi

v17_node_rank="$1"
v17_master_addr="${2:-10.66.17.163}"
v17_master_port="${3:-29507}"
v17_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v17_python="/mnt/workspace/lpz/engineai/engineai/bin/python"
v17_load_run="2026-08-21_22-41-15_v16-scale"
v17_checkpoint="model_11750.pt"
v17_checkpoint_path="${v17_project_dir}/logs/rsl_rl/tracking_t800/${v17_load_run}/${v17_checkpoint}"

case "$v17_node_rank" in
    0) v17_node_name="master" ;;
    1) v17_node_name="worker0" ;;
    2) v17_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0 (Master), 1 (Worker-0), or 2 (Worker-1)." >&2
        exit 2
        ;;
esac

v17_log="${v17_project_dir}/v17_scale_resume_11750_3node_12000_${v17_node_name}.log"

cd "$v17_project_dir"
exec > >(tee "$v17_log") 2>&1

echo "[preflight] node=${v17_node_name} rank=${v17_node_rank} master=${v17_master_addr}:${v17_master_port}"
[[ -x "$v17_python" ]] || { echo "Missing Python: ${v17_python}" >&2; exit 1; }
[[ -f "$v17_checkpoint_path" ]] || { echo "Missing checkpoint: ${v17_checkpoint_path}" >&2; exit 1; }

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v17_node_name}, found ${gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

"$v17_python" - <<'PY'
import gymnasium as gym

import engineai_rl_lab.tasks  # noqa: F401

spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v17-scale")
assert spec.kwargs["env_cfg_entry_point"].endswith(
    ".flat_env_cfg_v17:T800FlatWoStateEstimationEnvCfgV17Scale"
)
assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
    ".rsl_rl_ppo_cfg_v17:T800FlatV17ScalePPORunnerCfg"
)
print("[preflight] V17 task registration is correct.")
PY

# Remove stale launcher variables inherited from an earlier torchrun shell.
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

echo "[launch] log=${v17_log}"
exec "$v17_python" -m torch.distributed.run \
    --nnodes=3 \
    --nproc-per-node=8 \
    --node-rank="$v17_node_rank" \
    --master-addr="$v17_master_addr" \
    --master-port="$v17_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v17-scale \
    --distributed \
    --num_envs 12000 \
    --seed 42 \
    --resume \
    --load_run "$v17_load_run" \
    --checkpoint "$v17_checkpoint" \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
