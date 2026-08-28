#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
    echo "Usage: $0 NODE_RANK [MASTER_ADDR] [MASTER_PORT] [NUM_ENVS] [MAX_ITERATIONS]" >&2
    exit 2
fi

v23_node_rank="$1"
v23_master_addr="${2:-10.66.17.163}"
v23_master_port="${3:-29518}"
v23_num_envs="${4:-8192}"
v23_max_iterations="${5:-}"
v23_project_dir="/mnt/workspace/lpz/engineai/engineai_rl_lab"
v23_python="/mnt/workspace/lpz/engineai/engineai/bin/python"

case "$v23_node_rank" in
    0) v23_node_name="master" ;;
    1) v23_node_name="worker0" ;;
    2) v23_node_name="worker1" ;;
    *)
        echo "NODE_RANK must be 0 (Master), 1 (Worker-0), or 2 (Worker-1)." >&2
        exit 2
        ;;
esac

if [[ ! "$v23_master_port" =~ ^[0-9]+$ ]] || ((v23_master_port < 1 || v23_master_port > 65535)); then
    echo "MASTER_PORT must be an integer in [1, 65535], got: ${v23_master_port}" >&2
    exit 2
fi
if [[ ! "$v23_num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got: ${v23_num_envs}" >&2
    exit 2
fi
if [[ -n "$v23_max_iterations" && ! "$v23_max_iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_ITERATIONS must be a positive integer, got: ${v23_max_iterations}" >&2
    exit 2
fi

v23_session="v23_scratch_p${v23_master_port}_r${v23_node_rank}"
if [[ -z "${TMUX:-}" && "${V23_SCRATCH_TMUX_LAUNCHED:-0}" != "1" ]]; then
    if tmux has-session -t "$v23_session" 2>/dev/null; then
        echo "tmux session already exists: ${v23_session}" >&2
        echo "Attach with: tmux attach -t ${v23_session}" >&2
        exit 1
    fi

    v23_script_path="$(realpath "$0")"
    printf -v v23_tmux_command 'V23_SCRATCH_TMUX_LAUNCHED=1 bash %q' "$v23_script_path"
    for v23_arg in "$@"; do
        printf -v v23_tmux_command '%s %q' "$v23_tmux_command" "$v23_arg"
    done
    tmux new-session -d -s "$v23_session" "$v23_tmux_command"
    echo "Started tmux session: ${v23_session}"
    echo "Attach with: tmux attach -t ${v23_session}"
    exit 0
fi

v23_launch_stamp="$(date +%Y%m%d_%H%M%S)"
v23_iteration_label="${v23_max_iterations:-unlimited}"
v23_log="${v23_project_dir}/v23_scratch_3node_${v23_num_envs}env_${v23_iteration_label}iter_p${v23_master_port}_${v23_node_name}_${v23_launch_stamp}.log"

cd "$v23_project_dir"
exec > >(tee "$v23_log") 2>&1

echo "[preflight] mode=scratch node=${v23_node_name} rank=${v23_node_rank} master=${v23_master_addr}:${v23_master_port} envs_per_rank=${v23_num_envs} max_iterations=${v23_max_iterations:-unlimited}"
[[ -x "$v23_python" ]] || { echo "Missing Python: ${v23_python}" >&2; exit 1; }

v23_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$v23_gpu_count" -ne 8 ]]; then
    echo "Expected 8 GPUs on ${v23_node_name}, found ${v23_gpu_count}." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

"$v23_python" - <<'PY'
from pathlib import Path

import gymnasium as gym

import engineai_rl_lab.tasks  # noqa: F401
from engineai_rl_lab.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg_v23 import (
    T800FlatV23ScalePPORunnerCfg,
)
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v23 import (
    T800FlatWoStateEstimationEnvCfgV23Scale,
)
from engineai_rl_lab.tasks.tracking.mdp import (
    motion_joint_position_error_exp,
    motion_joint_position_target_and_error,
    motion_joint_velocity_error_exp,
    motion_joint_velocity_target_and_error,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import MotionCommandV1, MotionCommandV1Cfg
from engineai_rl_lab.tasks.tracking.mdp.motion_data import resolve_motion_catalog
from engineai_rl_lab.tasks.tracking.robots.t800 import T800_HEAD_JOINT_POSITIONS, T800_POLICY_JOINT_NAMES

spec = gym.spec("Tracking-Flat-T800-Wo-State-Estimation-v23-scale")
assert spec.kwargs["env_cfg_entry_point"].endswith(
    ".flat_env_cfg_v23:T800FlatWoStateEstimationEnvCfgV23Scale"
)
assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
    ".rsl_rl_ppo_cfg_v23:T800FlatV23ScalePPORunnerCfg"
)

runtime_cfg = T800FlatWoStateEstimationEnvCfgV23Scale()
runner_cfg = T800FlatV23ScalePPORunnerCfg()
motion_cfg = runtime_cfg.commands.motion
assert isinstance(motion_cfg, MotionCommandV1Cfg)
assert motion_cfg.class_type is MotionCommandV1
assert motion_cfg.motion_shard_across_ranks is True
assert motion_cfg.motion_catalog_cache is not None
assert motion_cfg.motion_load_workers == 4
assert runtime_cfg.actions.joint_pos.joint_names == T800_POLICY_JOINT_NAMES
assert len(runtime_cfg.actions.joint_pos.scale) == 23
assert motion_cfg.fixed_joint_positions == T800_HEAD_JOINT_POSITIONS
assert runner_cfg.run_name == "v23-scale"
assert not runner_cfg.resume
assert runner_cfg.num_steps_per_env == 32
assert runner_cfg.algorithm.num_learning_epochs == 2
assert runner_cfg.algorithm.num_mini_batches == 16
assert runner_cfg.algorithm.max_grad_norm == 0.1
assert runner_cfg.algorithm.entropy_coef == 0.01
assert runner_cfg.algorithm.critic_learning_rate == 1.0e-3
assert runner_cfg.algorithm.shared_kl_adaptation is True
assert runner_cfg.actor.distribution_cfg.init_std == 0.05
assert runner_cfg.actor.distribution_cfg.std_range == (1.0e-6, 1.0e6)
assert runner_cfg.critic.hidden_dims == [1024, 1024, 512, 512]

joint_pos_reward = runtime_cfg.rewards.motion_joint_pos
assert joint_pos_reward.func is motion_joint_position_error_exp
assert joint_pos_reward.weight == 0.5
assert joint_pos_reward.params["std"] == 0.5
assert joint_pos_reward.params["joint_names"] == T800_POLICY_JOINT_NAMES

joint_vel_reward = runtime_cfg.rewards.motion_joint_vel
assert joint_vel_reward.func is motion_joint_velocity_error_exp
assert joint_vel_reward.weight == 0.25
assert joint_vel_reward.params["std"] == 3.0
assert joint_vel_reward.params["joint_names"] == T800_POLICY_JOINT_NAMES

critic_joint_pos = runtime_cfg.observations.critic.motion_joint_pos
assert critic_joint_pos.func is motion_joint_position_target_and_error
assert critic_joint_pos.params["joint_names"] == T800_POLICY_JOINT_NAMES
critic_joint_vel = runtime_cfg.observations.critic.motion_joint_vel
assert critic_joint_vel.func is motion_joint_velocity_target_and_error
assert critic_joint_vel.params["joint_names"] == T800_POLICY_JOINT_NAMES

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
print("[preflight] V23 scratch config is correct; no checkpoint will be loaded.")
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

v23_iteration_args=()
if [[ -n "$v23_max_iterations" ]]; then
    v23_iteration_args=(--max_iterations "$v23_max_iterations")
fi

echo "[launch] mode=scratch world_size=24 log=${v23_log}"
exec "$v23_python" -m torch.distributed.run \
    --nnodes=3 \
    --nproc-per-node=8 \
    --node-rank="$v23_node_rank" \
    --master-addr="$v23_master_addr" \
    --master-port="$v23_master_port" \
    scripts/tracking/train.py \
    --task Tracking-Flat-T800-Wo-State-Estimation-v23-scale \
    --distributed \
    --num_envs "$v23_num_envs" \
    "${v23_iteration_args[@]}" \
    --seed 42 \
    --logger wandb \
    --log_project_name tracking_t800 \
    physics=newton_mjwarp
