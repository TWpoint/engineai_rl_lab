#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 6 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT] [NUM_ENVS] [MAX_ITERATIONS] [SNAPSHOT_ROOT]" >&2
    exit 2
fi

v26_node_rank="$1"
v26_master_addr="${2:-10.66.17.163}"
v26_master_port="${3:-29532}"
v26_num_envs="${4:-8192}"
v26_max_iterations="${5:-}"
v26_snapshot_root="${6:-}"
v26_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v26_python="/mnt/workspace/lpz/engineai/engineai/bin/python"

case "$v26_node_rank" in
    0) v26_node_name="master" ;;
    1) v26_node_name="worker0" ;;
    2) v26_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0, 1, or 2." >&2
        exit 2
        ;;
esac

if [[ ! "$v26_master_port" =~ ^[0-9]+$ ]] || ((v26_master_port < 1 || v26_master_port > 65535)); then
    echo "MASTER_PORT must be an integer in [1, 65535], got: ${v26_master_port}" >&2
    exit 2
fi
if [[ ! "$v26_num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got: ${v26_num_envs}" >&2
    exit 2
fi

if [[ "$v26_node_rank" == "0" ]] && ss -H -ltn | awk -v port=":${v26_master_port}" '$4 ~ port "$" { found=1 } END { exit !found }'; then
    echo "Refusing to launch because master port ${v26_master_port} is already listening." >&2
    exit 1
fi
if [[ -n "$v26_max_iterations" && ! "$v26_max_iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_ITERATIONS must be a positive integer, got: ${v26_max_iterations}" >&2
    exit 2
fi

v26_session="v26_scratch_p${v26_master_port}_r${v26_node_rank}"
if [[ -z "${TMUX:-}" && "${V26_SCRATCH_TMUX_LAUNCHED:-0}" != "1" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is required but was not found in PATH." >&2
        exit 1
    fi
    if tmux has-session -t "$v26_session" 2>/dev/null; then
        echo "tmux session already exists: ${v26_session}" >&2
        echo "Attach with: tmux attach -t ${v26_session}" >&2
        exit 1
    fi

    v26_script_path="$(realpath "$0")"
    printf -v v26_tmux_command 'V26_SCRATCH_TMUX_LAUNCHED=1 bash %q' "$v26_script_path"
    for v26_arg in "$@"; do
        printf -v v26_tmux_command '%s %q' "$v26_tmux_command" "$v26_arg"
    done
    tmux new-session -d -s "$v26_session"
    tmux set-option -t "$v26_session" remain-on-exit on
    tmux send-keys -t "${v26_session}:0.0" -l "$v26_tmux_command"
    tmux send-keys -t "${v26_session}:0.0" Enter
    echo "Submitted V26 scratch launch in tmux session: ${v26_session}"
    echo "Attach with: tmux attach -t ${v26_session}"
    exit 0
fi

cd "$v26_project_dir"
v26_launch_stamp="$(date +%Y%m%d_%H%M%S)"
v26_iteration_label="${v26_max_iterations:-unlimited}"
v26_log="${v26_project_dir}/v26_scratch_3node_${v26_num_envs}env_${v26_iteration_label}iter_p${v26_master_port}_${v26_node_name}_${v26_launch_stamp}.log"
exec > >(tee "$v26_log") 2>&1

echo "[preflight] node=${v26_node_name} rank=${v26_node_rank} master=${v26_master_addr}:${v26_master_port} envs_per_rank=${v26_num_envs} max_iterations=${v26_max_iterations:-unlimited} snapshots=${v26_snapshot_root:-disabled}"
[[ -x "$v26_python" ]] || { echo "Missing Python: ${v26_python}" >&2; exit 1; }

v26_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$v26_gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v26_node_name}, found ${v26_gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# Never overlap this scratch run with an older torchrun or simulator.  Report
# existing GPU users and leave termination of those jobs to the operator.
mapfile -t v26_active_gpu_pids < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ { gsub(/[[:space:]]/, ""); print }' \
        | sort -u
)
if (( ${#v26_active_gpu_pids[@]} > 0 )); then
    echo "Refusing to launch because GPU compute processes are already active on ${v26_node_name}:" >&2
    printf '  pid=%s\n' "${v26_active_gpu_pids[@]}" >&2
    echo "Stop or account for those jobs, then rerun this command." >&2
    exit 1
fi

echo "[preflight] Running the V26 STAR and simulation-safety contract tests."
"$v26_python" -m pytest -q --disable-warnings \
    tests/test_v26_config.py \
    tests/test_delayed_implicit_actuator.py \
    tests/test_newton_com_randomization.py \
    tests/test_simulation_safety.py \
    tests/test_invalid_state_recorder.py

"$v26_python" - <<'PY'
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym

import engineai_rl_lab.tasks  # noqa: F401
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v26 import (
    T800FlatV26ScalePPORunnerCfg,
    V26_ACTOR_NUM_BLOCKS,
    V26_REWARD_GROUPS,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v25 import (
    T800FlatWoStateEstimationEnvCfgV25Scale,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v26 import (
    T800FlatWoStateEstimationEnvCfgV26Scale,
    T800_V26_FRAME_OFFSETS,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import MotionCommandV1, MotionCommandV1Cfg
from engineai_rl_lab.tasks.tracking.mdp.events import _require_safe_newton_com_randomization
from engineai_rl_lab.tasks.tracking.mdp.motion_data import resolve_motion_catalog

spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v26-scale")
assert spec.kwargs["env_cfg_entry_point"].endswith(
    ".flat_env_cfg_v26:T800FlatWoStateEstimationEnvCfgV26Scale"
)
assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
    ".rsl_rl_ppo_cfg_v26:T800FlatV26ScalePPORunnerCfg"
)

env_cfg = T800FlatWoStateEstimationEnvCfgV26Scale()
runner_cfg = T800FlatV26ScalePPORunnerCfg()
motion_cfg = env_cfg.commands.motion
algorithm = runner_cfg.algorithm
newton_cfg = env_cfg.sim.physics.newton_mjwarp
assert runner_cfg.run_name == "v26-scale"
assert runner_cfg.resume is False
assert algorithm.class_name.endswith(":V26StarMultiCriticPPO")
assert runner_cfg.actor.nodes["attention_blocks"]["cell"]["num_blocks"] == V26_ACTOR_NUM_BLOCKS == 3
assert env_cfg.to_dict() == T800FlatWoStateEstimationEnvCfgV25Scale().to_dict()
assert algorithm.reward_groups == V26_REWARD_GROUPS
assert algorithm.star_priority_fraction == 0.125
assert algorithm.star_high_difficulty_threshold == 1.0
assert algorithm.star_top_fraction == 0.05
assert algorithm.star_difficulty_boundaries == (1.5, 2.0, 4.0)
assert algorithm.star_reuse_cap == 2
assert algorithm.star_priority_weight_cap == 8.0
assert env_cfg.rewards.action_rate_l2.weight == -0.05
assert env_cfg.observations.command.link_pose_b.params["frame_offsets"] == T800_V26_FRAME_OFFSETS
assert newton_cfg.num_substeps == 1
assert newton_cfg.default_shape_cfg.margin == 0.01
assert env_cfg.events.base_com is not None
assert env_cfg.events.base_com.params["com_range"] == {
    "x": (-0.025, 0.025),
    "y": (-0.05, 0.05),
    "z": (-0.05, 0.05),
}
_require_safe_newton_com_randomization("NewtonManager")
assert isinstance(motion_cfg, MotionCommandV1Cfg)
assert motion_cfg.class_type is MotionCommandV1
assert motion_cfg.adaptive_sampling.equal_motion_weighting is False
assert motion_cfg.adaptive_sampling.learnability_full_scale == 0.05
assert motion_cfg.start_at_motion_beginning is False
assert motion_cfg.motion_shard_across_ranks is True
assert Path(motion_cfg.motion_file).is_file(), f"Missing motion manifest: {motion_cfg.motion_file}"
motion_files, motion_lengths = resolve_motion_catalog(
    motion_cfg.motion_file,
    cache_path=motion_cfg.motion_catalog_cache,
    max_workers=motion_cfg.motion_load_workers,
)
assert len(motion_files) == len(motion_lengths) > 0
print(
    "[preflight] V26 contract passed: V25 MDP/three-block actor/three-head critic "
    "with only STAR-lite adaptive rollout sampling added. "
    f"Newton={version('newton')}, MuJoCo Warp={version('mujoco-warp')}."
)
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
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export PYTHONUNBUFFERED=1

v26_iteration_args=()
if [[ -n "$v26_max_iterations" ]]; then
    v26_iteration_args=(--max_iterations "$v26_max_iterations")
fi

v26_snapshot_args=()
if [[ -n "$v26_snapshot_root" ]]; then
    v26_snapshot_dir="${v26_snapshot_root%/}/${v26_node_name}"
    mkdir -p "$v26_snapshot_dir"
    v26_snapshot_args=(--invalid_state_snapshot_dir "$v26_snapshot_dir")
    echo "[launch] invalid/finite-runaway snapshots=${v26_snapshot_dir}"
fi

echo "[launch] mode=scratch world_size=24 log=${v26_log}"
exec "$v26_python" -m torch.distributed.run \
    --nnodes=3 \
    --nproc-per-node=8 \
    --node-rank="$v26_node_rank" \
    --master-addr="$v26_master_addr" \
    --master-port="$v26_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v26-scale \
    --distributed \
    --num_envs "$v26_num_envs" \
    "${v26_iteration_args[@]}" \
    "${v26_snapshot_args[@]}" \
    --seed 42 \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
