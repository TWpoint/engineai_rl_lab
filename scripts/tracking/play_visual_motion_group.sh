#!/usr/bin/env bash

set -u -o pipefail

repo_dir="/home/ubuntu/engineai/engineai_rl_lab"
engineai_python="/home/ubuntu/engineai/engineai/bin/python"
default_group="$repo_dir/logs/rsl_rl/tracking_t800/2026-08-20_02-24-19_v13-scale/evaluation/visual_motion_group_v13_12.yaml"

task="Tracking-Flat-T800-Wo-State-Estimation-v24-scale"
run="2026-08-26_19-05-49_v24-scale"
checkpoint="model_.*.pt"
group="$default_group"
single_motion=""
start_index=1
start_frame=0
mode="policy"
physics="newton"
follow_camera=1
ghost_reference=0
ghost_opacity="0.3"
ghost_offset="1.0"
disable_body_pos_termination=0
no_visualization=0
auto_launch=0

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "" \
        "Options:" \
        "  --run NAME          Run folder containing the checkpoint (default: $run)" \
        "  --checkpoint FILE   Checkpoint filename or regex (default: $checkpoint, selects latest)" \
        "  --start INDEX       Start at the 1-based group index (default: 1)" \
        "  --start-frame N     Start motion playback at zero-based frame N (default: $start_frame)" \
        "  --group FILE        YAML motion group" \
        "  --motion FILE       Play only this NPZ motion (ignores --group)" \
        "  --reference         Replay raw NPZ references instead of running the policy" \
        "  --physics BACKEND   Physics backend: newton or physx (default: $physics)" \
        "  --ghost-reference   Overlay a translucent aligned reference robot in policy mode" \
        "  --ghost-opacity N   Reference robot opacity in (0, 1] (default: $ghost_opacity)" \
        "  --ghost-offset M    Lateral separation in meters (default: $ghost_offset)" \
        "  --disable-termination  Disable pose/position tracking-error terminations" \
        "  --no-visualization  Run headlessly without opening a visualizer" \
        "  --auto              Launch immediately without waiting for Enter" \
        "  --no-follow-camera  Keep the camera user-controlled" \
        "  --list              List the group and exit" \
        "  -h, --help          Show this help"
}

list_only=0
while (($#)); do
    case "$1" in
        --run)
            run="$2"
            shift 2
            ;;
        --checkpoint)
            checkpoint="$2"
            shift 2
            ;;
        --start)
            start_index="$2"
            shift 2
            ;;
        --start-frame)
            start_frame="$2"
            shift 2
            ;;
        --group)
            group="$2"
            shift 2
            ;;
        --motion)
            single_motion="$2"
            shift 2
            ;;
        --reference)
            mode="reference"
            shift
            ;;
        --physics)
            physics="$2"
            shift 2
            ;;
        --ghost-reference)
            ghost_reference=1
            shift
            ;;
        --ghost-opacity)
            ghost_opacity="$2"
            shift 2
            ;;
        --ghost-offset)
            ghost_offset="$2"
            shift 2
            ;;
        --disable-termination|--disable_termination|--disable-body-pos-termination)
            disable_body_pos_termination=1
            shift
            ;;
        --no-visualization|--headless)
            no_visualization=1
            follow_camera=0
            shift
            ;;
        --auto)
            auto_launch=1
            shift
            ;;
        --no-follow-camera)
            follow_camera=0
            shift
            ;;
        --list)
            list_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$single_motion" ]]; then
    if [[ ! -f "$single_motion" ]]; then
        printf 'Motion file does not exist: %s\n' "$single_motion" >&2
        exit 1
    fi
    motions=("$single_motion")
else
    if [[ ! -f "$group" ]]; then
        printf 'Motion group does not exist: %s\n' "$group" >&2
        exit 1
    fi

    mapfile -t motions < <(sed -n 's/^[[:space:]]*-[[:space:]]*//p' "$group")
    if ((${#motions[@]} == 0)); then
        printf 'No motion paths found in: %s\n' "$group" >&2
        exit 1
    fi
fi

for index in "${!motions[@]}"; do
    printf '%2d  %s\n' "$((index + 1))" "${motions[$index]}"
done

if ((list_only)); then
    exit 0
fi

if ! [[ "$start_index" =~ ^[0-9]+$ ]] || ((start_index < 1 || start_index > ${#motions[@]})); then
    printf 'Invalid --start index %s; expected 1..%d\n' "$start_index" "${#motions[@]}" >&2
    exit 2
fi

if ! [[ "$start_frame" =~ ^[0-9]+$ ]]; then
    printf 'Invalid --start-frame %s; expected a non-negative integer.\n' "$start_frame" >&2
    exit 2
fi

if ((ghost_reference)) && [[ "$mode" == "reference" ]]; then
    printf '%s\n' '--ghost-reference cannot be combined with --reference.' >&2
    exit 2
fi

case "$physics" in
    newton)
        physics_preset="newton_mjwarp"
        visualizer="newton_gl"
        ;;
    physx)
        physics_preset="isaacsim_physx"
        visualizer="kit"
        ;;
    *)
        printf 'Invalid --physics %s; expected newton or physx.\n' "$physics" >&2
        exit 2
        ;;
esac

if ((no_visualization)); then
    visualizer="none"
fi

cd "$repo_dir" || exit 1

for ((index = start_index - 1; index < ${#motions[@]}; index++)); do
    motion="${motions[$index]}"
    if [[ ! -f "$motion" ]]; then
        printf '\n[%d/%d] Missing file, skipped: %s\n' "$((index + 1))" "${#motions[@]}" "$motion" >&2
        continue
    fi

    printf '\n[%d/%d] %s\n' "$((index + 1))" "${#motions[@]}" "$(basename "$motion")"
    printf 'Path: %s\n' "$motion"
    printf 'Mode: %s' "$mode"
    if [[ "$mode" == "policy" ]]; then
        printf ', checkpoint: %s' "$checkpoint"
    fi
    printf ', physics: %s\n' "$physics"

    if ((auto_launch == 0)); then
        read -r -p 'Enter=launch, s=skip, q=quit: ' answer
        case "$answer" in
            q|Q)
                exit 0
                ;;
            s|S)
                continue
                ;;
        esac
    fi

    if [[ "$mode" == "reference" ]]; then
        command=(
            "$engineai_python"
        )
        if [[ "$physics" == "physx" ]]; then
            command+=(scripts/kit_bootstrap.py scripts/replay_npz.py)
        else
            command+=(scripts/replay_npz.py)
        fi
        command+=(
            --robot t800
            --input_file "$motion"
            --physics "$physics_preset"
            --viz "$visualizer"
        )
    else
        command=(
            "$engineai_python"
        )
        if [[ "$physics" == "physx" ]]; then
            command+=(scripts/kit_bootstrap.py scripts/tracking/play.py)
        else
            command+=(scripts/tracking/play.py)
        fi
        command+=(
            --task "$task"
            --num_envs 1
            --motion_file "$motion"
            --load_run "$run"
            --checkpoint "$checkpoint"
            --start_frame "$start_frame"
            physics="$physics_preset"
            --viz "$visualizer"
        )
        if ((ghost_reference)); then
            command+=(--ghost_reference --ghost_opacity "$ghost_opacity" --ghost_offset "$ghost_offset")
        fi
        if ((disable_body_pos_termination)); then
            command+=(--disable_body_pos_termination)
        fi
    fi

    if ((follow_camera)); then
        command+=(--follow_camera)
    fi

    "${command[@]}"
    status=$?
    if ((status != 0)); then
        printf 'Playback exited with status %d; continuing to the next motion.\n' "$status" >&2
    fi
done
