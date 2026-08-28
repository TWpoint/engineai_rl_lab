#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT] [NUM_ENVS] [MAX_ITERATIONS]" >&2
    exit 2
fi

v20_node_rank="$1"
v20_master_addr="${2:-10.66.17.163}"
v20_master_port="${3:-29512}"
v20_num_envs="${4:-8192}"
v20_max_iterations="${5:-}"
v20_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v20_python="/mnt/workspace/lpz/engineai/engineai/bin/python"

case "$v20_node_rank" in
    0) v20_node_name="master" ;;
    1) v20_node_name="worker0" ;;
    2) v20_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0 (Master), 1 (Worker-0), or 2 (Worker-1)." >&2
        exit 2
        ;;
esac

if [[ ! "$v20_master_port" =~ ^[0-9]+$ ]] || ((v20_master_port < 1 || v20_master_port > 65535)); then
    echo "MASTER_PORT must be an integer in [1, 65535], got: ${v20_master_port}" >&2
    exit 2
fi
if [[ ! "$v20_num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got: ${v20_num_envs}" >&2
    exit 2
fi
if [[ -n "$v20_max_iterations" && ! "$v20_max_iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_ITERATIONS must be a positive integer, got: ${v20_max_iterations}" >&2
    exit 2
fi

v20_session="v20_scratch_p${v20_master_port}_r${v20_node_rank}"
if [[ -z "${TMUX:-}" && "${V20_SCRATCH_TMUX_LAUNCHED:-0}" != "1" ]]; then
    if tmux has-session -t "$v20_session" 2>/dev/null; then
        echo "tmux session already exists: ${v20_session}" >&2
        echo "Attach with: tmux attach -t ${v20_session}" >&2
        exit 1
    fi

    v20_script_path="$(realpath "$0")"
    printf -v v20_tmux_command 'V20_SCRATCH_TMUX_LAUNCHED=1 bash %q' "$v20_script_path"
    for v20_arg in "$@"; do
        printf -v v20_tmux_command '%s %q' "$v20_tmux_command" "$v20_arg"
    done
    tmux new-session -d -s "$v20_session" "$v20_tmux_command"
    echo "Started tmux session: ${v20_session}"
    echo "Attach with: tmux attach -t ${v20_session}"
    exit 0
fi

v20_launch_stamp="$(date +%Y%m%d_%H%M%S)"
v20_iteration_label="${v20_max_iterations:-unlimited}"
v20_log="${v20_project_dir}/v20_scratch_3node_${v20_num_envs}env_${v20_iteration_label}iter_p${v20_master_port}_${v20_node_name}_${v20_launch_stamp}.log"

cd "$v20_project_dir"
exec > >(tee "$v20_log") 2>&1

echo "[preflight] mode=scratch node=${v20_node_name} rank=${v20_node_rank} master=${v20_master_addr}:${v20_master_port} envs_per_rank=${v20_num_envs} max_iterations=${v20_max_iterations:-unlimited}"
[[ -x "$v20_python" ]] || { echo "Missing Python: ${v20_python}" >&2; exit 1; }

v20_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$v20_gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v20_node_name}, found ${v20_gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

"$v20_python" - <<'PY'
from pathlib import Path

import gymnasium as gym

import engineai_rl_lab.tasks  # noqa: F401
import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v20 import (
    T800FlatV20ScalePPORunnerCfg,
)
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
runner_cfg = T800FlatV20ScalePPORunnerCfg()
motion_cfg = runtime_cfg.commands.motion
assert isinstance(motion_cfg, MotionCommandV1Cfg)
assert motion_cfg.class_type is MotionCommandV1
assert motion_cfg.adaptive_sampling.fast_half_life == 32.0
assert motion_cfg.adaptive_sampling.slow_half_life == 64.0
assert motion_cfg.adaptive_sampling.probability_cap_ratio == 200.0
assert runtime_cfg.rewards.joint_acc_l2 is None
assert runtime_cfg.observations.command.link_pose_b.func is mdp.motion_body_pose_b_window_by_entity_xz
assert runner_cfg.run_name == "v20-scale"
assert not runner_cfg.resume
assert runner_cfg.num_steps_per_env == 32
assert runner_cfg.algorithm.num_learning_epochs == 2
assert runner_cfg.algorithm.num_mini_batches == 16
assert runner_cfg.max_iterations is None
assert runner_cfg.algorithm.schedule == "adaptive"
assert runner_cfg.algorithm.critic_learning_rate == 2.0e-5
assert runner_cfg.algorithm.optimizer_fused is False
assert runner_cfg.algorithm.learning_rate_min == 1.0e-5
assert runner_cfg.algorithm.learning_rate_max == 2.0e-4
assert runner_cfg.actor.distribution_cfg.init_std == 0.05
assert runner_cfg.actor.distribution_cfg.std_range == (0.001, 0.5)
assert runner_cfg.critic.hidden_dims == [2048, 1024, 512]
assert Path(motion_cfg.motion_file).is_file(), f"Missing motion manifest: {motion_cfg.motion_file}"
print("[preflight] V20 scratch config is correct; no checkpoint will be loaded.")
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

v20_iteration_args=()
if [[ -n "$v20_max_iterations" ]]; then
    v20_iteration_args=(--max_iterations "$v20_max_iterations")
fi

echo "[launch] mode=scratch world_size=24 log=${v20_log}"
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
    "${v20_iteration_args[@]}" \
    --seed 42 \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
