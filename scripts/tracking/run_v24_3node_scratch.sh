#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT] [NUM_ENVS] [MAX_ITERATIONS]" >&2
    exit 2
fi

v24_node_rank="$1"
v24_master_addr="${2:-10.66.17.163}"
v24_master_port="${3:-29524}"
v24_num_envs="${4:-8192}"
v24_max_iterations="${5:-}"
v24_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v24_python="/mnt/workspace/lpz/engineai/engineai/bin/python"

case "$v24_node_rank" in
    0) v24_node_name="master" ;;
    1) v24_node_name="worker0" ;;
    2) v24_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0, 1, or 2." >&2
        exit 2
        ;;
esac

if [[ ! "$v24_master_port" =~ ^[0-9]+$ ]] || ((v24_master_port < 1 || v24_master_port > 65535)); then
    echo "MASTER_PORT must be an integer in [1, 65535], got: ${v24_master_port}" >&2
    exit 2
fi
if [[ ! "$v24_num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got: ${v24_num_envs}" >&2
    exit 2
fi
if [[ -n "$v24_max_iterations" && ! "$v24_max_iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_ITERATIONS must be a positive integer, got: ${v24_max_iterations}" >&2
    exit 2
fi

v24_session="v24_scratch_p${v24_master_port}_r${v24_node_rank}"
if [[ -z "${TMUX:-}" && "${V24_SCRATCH_TMUX_LAUNCHED:-0}" != "1" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is required but was not found in PATH." >&2
        exit 1
    fi
    if tmux has-session -t "$v24_session" 2>/dev/null; then
        echo "tmux session already exists: ${v24_session}" >&2
        echo "Attach with: tmux attach -t ${v24_session}" >&2
        exit 1
    fi

    v24_script_path="$(realpath "$0")"
    printf -v v24_tmux_command 'V24_SCRATCH_TMUX_LAUNCHED=1 bash %q' "$v24_script_path"
    for v24_arg in "$@"; do
        printf -v v24_tmux_command '%s %q' "$v24_tmux_command" "$v24_arg"
    done
    # Start an interactive shell first so an early preflight failure remains
    # visible instead of making the tmux session disappear immediately.
    tmux new-session -d -s "$v24_session"
    tmux set-option -t "$v24_session" remain-on-exit on
    tmux send-keys -t "${v24_session}:0.0" -l "$v24_tmux_command"
    tmux send-keys -t "${v24_session}:0.0" Enter
    echo "Submitted V24 launch in tmux session: ${v24_session}"
    echo "Attach with: tmux attach -t ${v24_session}"
    exit 0
fi

cd "$v24_project_dir"
v24_launch_stamp="$(date +%Y%m%d_%H%M%S)"
v24_iteration_label="${v24_max_iterations:-unlimited}"
v24_log="${v24_project_dir}/v24_scratch_3node_${v24_num_envs}env_${v24_iteration_label}iter_p${v24_master_port}_${v24_node_name}_${v24_launch_stamp}.log"
exec > >(tee "$v24_log") 2>&1

echo "[preflight] node=${v24_node_name} rank=${v24_node_rank} master=${v24_master_addr}:${v24_master_port} envs_per_rank=${v24_num_envs} max_iterations=${v24_max_iterations:-unlimited}"
[[ -x "$v24_python" ]] || { echo "Missing Python: ${v24_python}" >&2; exit 1; }

v24_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$v24_gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v24_node_name}, found ${v24_gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

echo "[preflight] Running the V24 command/single-critic contract tests."
"$v24_python" -m pytest -q --disable-warnings tests/test_v24_config.py

"$v24_python" - <<'PY'
from pathlib import Path

import gymnasium as gym

import engineai_rl_lab.tasks  # noqa: F401
import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v24 import (
    T800FlatV24ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v23 import (
    T800FlatV23ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v24 import (
    T800FlatWoStateEstimationEnvCfgV24Scale,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import MotionCommandV1, MotionCommandV1Cfg
from engineai_rl_lab.tasks.tracking.mdp.motion_data import resolve_motion_catalog
spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v24-scale")
assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
assert spec.kwargs["env_cfg_entry_point"].endswith(
    ".flat_env_cfg_v24:T800FlatWoStateEstimationEnvCfgV24Scale"
)
assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
    ".rsl_rl_ppo_cfg_v24:T800FlatV24ScalePPORunnerCfg"
)
runtime_cfg = T800FlatWoStateEstimationEnvCfgV24Scale()
runner_cfg = T800FlatV24ScalePPORunnerCfg()
v23_runner_cfg = T800FlatV23ScalePPORunnerCfg()
motion_cfg = runtime_cfg.commands.motion

assert len(runtime_cfg.commands.motion.body_names) == 14
assert "LINK_HEAD_YAW" not in runtime_cfg.commands.motion.body_names
assert runtime_cfg.observations.command.link_pose_b.func is mdp.motion_body_pose_reference_anchor_window_by_entity_xz
assert runtime_cfg.observations.critic.command.func is mdp.motion_body_pose_and_error_b_window_flat
assert runtime_cfg.observations.critic.local_command.func is mdp.motion_body_pose_reference_anchor_window_xz_flat
assert runtime_cfg.observations.critic.local_command.params["frame_offsets"] == [0]
assert runtime_cfg.rewards.action_rate_l2.weight == -0.1
assert runner_cfg.algorithm.class_name == "PPO"
assert runner_cfg.algorithm.to_dict() == v23_runner_cfg.algorithm.to_dict()
assert runner_cfg.actor.to_dict() == v23_runner_cfg.actor.to_dict()
assert runner_cfg.critic.to_dict() == v23_runner_cfg.critic.to_dict()
assert runner_cfg.algorithm.rnd_cfg is None
assert runner_cfg.run_name == "v24-scale"
assert isinstance(motion_cfg, MotionCommandV1Cfg)
assert motion_cfg.class_type is MotionCommandV1
assert motion_cfg.motion_shard_across_ranks is True
assert motion_cfg.motion_catalog_cache is not None
assert Path(motion_cfg.motion_file).is_file(), f"Missing motion manifest: {motion_cfg.motion_file}"
motion_files, motion_lengths = resolve_motion_catalog(
    motion_cfg.motion_file,
    cache_path=motion_cfg.motion_catalog_cache,
    max_workers=motion_cfg.motion_load_workers,
)
assert len(motion_files) == len(motion_lengths) > 0
assert Path(motion_cfg.motion_catalog_cache).is_file(), (
    f"Motion catalog was not published: {motion_cfg.motion_catalog_cache}"
)
print(
    "[preflight] Motion catalog ready: "
    f"files={len(motion_files):,}, frames={sum(motion_lengths):,}, "
    f"cache={motion_cfg.motion_catalog_cache}"
)
print("[preflight] V24 local command and single-critic PPO config are correct.")
PY

# Avoid inheriting rank metadata from an older torchrun shell.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

v24_iteration_args=()
if [[ -n "$v24_max_iterations" ]]; then
    v24_iteration_args=(--max_iterations "$v24_max_iterations")
fi

echo "[launch] mode=scratch world_size=24 log=${v24_log}"
exec "$v24_python" -m torch.distributed.run \
    --nnodes=3 \
    --nproc-per-node=8 \
    --node-rank="$v24_node_rank" \
    --master-addr="$v24_master_addr" \
    --master-port="$v24_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v24-scale \
    --distributed \
    --num_envs "$v24_num_envs" \
    "${v24_iteration_args[@]}" \
    --seed 42 \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
