#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 6 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT] [NUM_ENVS] [MAX_ITERATIONS] [SNAPSHOT_ROOT]" >&2
    exit 2
fi

v25_node_rank="$1"
v25_master_addr="${2:-10.66.17.163}"
v25_master_port="${3:-29528}"
v25_num_envs="${4:-8192}"
v25_max_iterations="${5:-}"
v25_snapshot_root="${6:-}"
v25_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v25_python="/mnt/workspace/lpz/engineai/engineai/bin/python"

case "$v25_node_rank" in
    0) v25_node_name="master" ;;
    1) v25_node_name="worker0" ;;
    2) v25_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0, 1, or 2." >&2
        exit 2
        ;;
esac

if [[ ! "$v25_master_port" =~ ^[0-9]+$ ]] || ((v25_master_port < 1 || v25_master_port > 65535)); then
    echo "MASTER_PORT must be an integer in [1, 65535], got: ${v25_master_port}" >&2
    exit 2
fi
if [[ ! "$v25_num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got: ${v25_num_envs}" >&2
    exit 2
fi
if [[ -n "$v25_max_iterations" && ! "$v25_max_iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_ITERATIONS must be a positive integer, got: ${v25_max_iterations}" >&2
    exit 2
fi

v25_session="v25_scratch_p${v25_master_port}_r${v25_node_rank}"
if [[ -z "${TMUX:-}" && "${V25_SCRATCH_TMUX_LAUNCHED:-0}" != "1" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is required but was not found in PATH." >&2
        exit 1
    fi
    if tmux has-session -t "$v25_session" 2>/dev/null; then
        echo "tmux session already exists: ${v25_session}" >&2
        echo "Attach with: tmux attach -t ${v25_session}" >&2
        exit 1
    fi

    v25_script_path="$(realpath "$0")"
    printf -v v25_tmux_command 'V25_SCRATCH_TMUX_LAUNCHED=1 bash %q' "$v25_script_path"
    for v25_arg in "$@"; do
        printf -v v25_tmux_command '%s %q' "$v25_tmux_command" "$v25_arg"
    done
    # Start an interactive shell first so an early preflight failure remains
    # visible instead of making the tmux session disappear immediately.
    tmux new-session -d -s "$v25_session"
    tmux set-option -t "$v25_session" remain-on-exit on
    tmux send-keys -t "${v25_session}:0.0" -l "$v25_tmux_command"
    tmux send-keys -t "${v25_session}:0.0" Enter
    echo "Submitted V25 launch in tmux session: ${v25_session}"
    echo "Attach with: tmux attach -t ${v25_session}"
    exit 0
fi

cd "$v25_project_dir"
v25_launch_stamp="$(date +%Y%m%d_%H%M%S)"
v25_iteration_label="${v25_max_iterations:-unlimited}"
v25_log="${v25_project_dir}/v25_scratch_3node_${v25_num_envs}env_${v25_iteration_label}iter_p${v25_master_port}_${v25_node_name}_${v25_launch_stamp}.log"
exec > >(tee "$v25_log") 2>&1

echo "[preflight] node=${v25_node_name} rank=${v25_node_rank} master=${v25_master_addr}:${v25_master_port} envs_per_rank=${v25_num_envs} max_iterations=${v25_max_iterations:-unlimited} snapshots=${v25_snapshot_root:-disabled}"
[[ -x "$v25_python" ]] || { echo "Missing Python: ${v25_python}" >&2; exit 1; }

v25_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$v25_gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v25_node_name}, found ${v25_gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# Never silently overlap a scratch run with an older torchrun or simulator.
# This is deliberately read-only: the operator decides which old job to stop.
mapfile -t v25_active_gpu_pids < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ { gsub(/[[:space:]]/, ""); print }' \
        | sort -u
)
if (( ${#v25_active_gpu_pids[@]} > 0 )); then
    echo "Refusing to launch because GPU compute processes are already active on ${v25_node_name}:" >&2
    printf '  pid=%s\n' "${v25_active_gpu_pids[@]}" >&2
    echo "Stop or account for those jobs, then rerun this command." >&2
    exit 1
fi

echo "[preflight] Running the V25 config and simulation-safety contract tests."
"$v25_python" -m pytest -q --disable-warnings \
    tests/test_v25_config.py \
    tests/test_delayed_implicit_actuator.py \
    tests/test_simulation_safety.py \
    tests/test_invalid_state_recorder.py

"$v25_python" - <<'PY'
from copy import deepcopy
from pathlib import Path

import gymnasium as gym

import engineai_rl_lab.tasks  # noqa: F401
import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v24 import (
    T800FlatV24ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v25 import (
    T800FlatV25ScalePPORunnerCfg,
    V25_ACTOR_NUM_BLOCKS,
    V25_REWARD_GROUPS,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v25 import (
    T800FlatWoStateEstimationEnvCfgV25Scale,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import MotionCommandV1, MotionCommandV1Cfg
from engineai_rl_lab.tasks.tracking.mdp.motion_data import resolve_motion_catalog

spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v25-scale")
assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
assert spec.kwargs["env_cfg_entry_point"].endswith(
    ".flat_env_cfg_v25:T800FlatWoStateEstimationEnvCfgV25Scale"
)
assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
    ".rsl_rl_ppo_cfg_v25:T800FlatV25ScalePPORunnerCfg"
)

runtime_cfg = T800FlatWoStateEstimationEnvCfgV25Scale()
runner_cfg = T800FlatV25ScalePPORunnerCfg()
v24_runner_cfg = T800FlatV24ScalePPORunnerCfg()
motion_cfg = runtime_cfg.commands.motion

# V25 changes the actor depth, run name, and grouped value-learning fields.
# Every remaining runner, PPO, actor, and critic setting must match V24.
assert runner_cfg.run_name == "v25-scale"
v25_actor = runner_cfg.actor.to_dict()
v24_actor = v24_runner_cfg.actor.to_dict()
assert v24_actor["nodes"]["attention_blocks"]["cell"]["num_blocks"] == 2
assert v25_actor["nodes"]["attention_blocks"]["cell"]["num_blocks"] == V25_ACTOR_NUM_BLOCKS == 3
v25_actor["nodes"]["attention_blocks"]["cell"]["num_blocks"] = 2
assert v25_actor == v24_actor
assert runner_cfg.critic.to_dict() == v24_runner_cfg.critic.to_dict()
v25_runner_dict = runner_cfg.to_dict()
v24_runner_dict = v24_runner_cfg.to_dict()
v25_runner_dict["run_name"] = v24_runner_dict["run_name"]
v25_runner_dict["actor"]["nodes"]["attention_blocks"]["cell"]["num_blocks"] = 2
v25_algorithm = deepcopy(v25_runner_dict["algorithm"])
assert v25_algorithm.pop("class_name") == (
    "engineai_rl_lab.utils.v25_compact_multi_critic_ppo:V25CompactMultiCriticPPO"
)
assert v25_algorithm.pop("value_loss_reduction") == "mean"
configured_groups = v25_algorithm.pop("reward_groups")
v25_algorithm["class_name"] = v24_runner_dict["algorithm"]["class_name"]
v25_runner_dict["algorithm"] = v25_algorithm
assert v25_runner_dict == v24_runner_dict

# The three heads form an ordered, complete, non-overlapping partition of all
# active V25 reward terms. This catches silent reward loss and double counting.
assert list(configured_groups) == ["global", "local", "regularization"]
assert configured_groups == V25_REWARD_GROUPS
grouped_terms = [term for terms in configured_groups.values() for term in terms]
active_reward_terms = {
    name for name, term_cfg in runtime_cfg.rewards.to_dict().items() if term_cfg is not None
}
assert len(configured_groups) == 3
assert len(grouped_terms) == len(set(grouped_terms))
assert set(grouped_terms) == active_reward_terms

# The actor command remains exactly 14 entity tokens x 99 features
# (eleven temporal frames x nine pose features), using the ScaleBFM-local
# current-reference-anchor translation. The critic retains both global/error
# and ScaleBFM-local command views in its single privileged observation group.
command_term = runtime_cfg.observations.command.link_pose_b
assert len(motion_cfg.body_names) == 14
assert "LINK_HEAD_YAW" not in motion_cfg.body_names
assert command_term.func is mdp.motion_body_pose_reference_anchor_window_by_entity_xz
assert command_term.params["frame_offsets"] == [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
assert len(command_term.params["frame_offsets"]) * 9 == 99
assert runtime_cfg.observations.critic.command.func is mdp.motion_body_pose_and_error_b_window_flat
assert runtime_cfg.observations.critic.local_command.func is (
    mdp.motion_body_pose_reference_anchor_window_xz_flat
)
assert runtime_cfg.observations.critic.local_command.params["frame_offsets"] == [0]
assert runner_cfg.obs_groups["critic"] == ["critic"]
assert runtime_cfg.rewards.action_rate_l2.weight == -0.05
assert runner_cfg.algorithm.rnd_cfg is None

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
print(
    "[preflight] V25 static config contract passed: standard env registration, "
    "action_rate_l2=-0.05, 3 actor blocks, 14x99 local command config, global+local critic config, "
    "complete three-head reward partition, and compact per-head value-loss logging."
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

v25_iteration_args=()
if [[ -n "$v25_max_iterations" ]]; then
    v25_iteration_args=(--max_iterations "$v25_max_iterations")
fi

v25_snapshot_args=()
if [[ -n "$v25_snapshot_root" ]]; then
    v25_snapshot_dir="${v25_snapshot_root%/}/${v25_node_name}"
    mkdir -p "$v25_snapshot_dir"
    v25_snapshot_args=(--invalid_state_snapshot_dir "$v25_snapshot_dir")
    echo "[launch] invalid/finite-runaway snapshots=${v25_snapshot_dir}"
fi

echo "[launch] mode=scratch world_size=24 log=${v25_log}"
exec "$v25_python" -m torch.distributed.run \
    --nnodes=3 \
    --nproc-per-node=8 \
    --node-rank="$v25_node_rank" \
    --master-addr="$v25_master_addr" \
    --master-port="$v25_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v25-scale \
    --distributed \
    --num_envs "$v25_num_envs" \
    "${v25_iteration_args[@]}" \
    "${v25_snapshot_args[@]}" \
    --seed 42 \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
