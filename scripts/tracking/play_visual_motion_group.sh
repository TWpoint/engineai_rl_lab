#!/usr/bin/env bash

set -u -o pipefail

repo_dir="/home/ubuntu/engineai/engineai_rl_lab"
engineai_python="/home/ubuntu/engineai/engineai/bin/python"
default_group="$repo_dir/logs/rsl_rl/tracking_t800/2026-08-20_02-24-19_v13-scale/evaluation/visual_motion_group_v13_12.yaml"

task="Tracking-Flat-T800-Wo-State-Estimation-v13-scale"
run="2026-08-20_13-35-00_v13-scale"
checkpoint="model_2500.pt"
group="$default_group"
start_index=1
mode="policy"
follow_camera=1
ghost_reference=0
ghost_opacity="0.3"
ghost_offset="1.0"
auto_launch=0

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "" \
        "Options:" \
        "  --run NAME          Run folder containing the checkpoint (default: $run)" \
        "  --checkpoint FILE   Checkpoint filename (default: $checkpoint)" \
        "  --start INDEX       Start at the 1-based group index (default: 1)" \
        "  --group FILE        YAML motion group" \
        "  --reference         Replay raw NPZ references instead of running the policy" \
        "  --ghost-reference   Overlay a translucent aligned reference robot in policy mode" \
        "  --ghost-opacity N   Reference robot opacity in (0, 1] (default: $ghost_opacity)" \
        "  --ghost-offset M    Lateral separation in meters (default: $ghost_offset)" \
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
        --group)
            group="$2"
            shift 2
            ;;
        --reference)
            mode="reference"
            shift
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

if [[ ! -f "$group" ]]; then
    printf 'Motion group does not exist: %s\n' "$group" >&2
    exit 1
fi

mapfile -t motions < <(sed -n 's/^[[:space:]]*-[[:space:]]*//p' "$group")
if ((${#motions[@]} == 0)); then
    printf 'No motion paths found in: %s\n' "$group" >&2
    exit 1
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

if ((ghost_reference)) && [[ "$mode" == "reference" ]]; then
    printf '%s\n' '--ghost-reference cannot be combined with --reference.' >&2
    exit 2
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
    printf '\n'

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
            "$engineai_python" scripts/replay_npz.py
            --robot t800
            --input_file "$motion"
            --physics newton_mjwarp
            --viz newton_gl
        )
    else
        command=(
            "$engineai_python" scripts/tracking/play.py
            --task "$task"
            --num_envs 1
            --motion_file "$motion"
            --load_run "$run"
            --checkpoint "$checkpoint"
            --start_at_motion_beginning
            physics=newton_mjwarp
            --viz newton_gl
        )
        if ((ghost_reference)); then
            command+=(--ghost_reference --ghost_opacity "$ghost_opacity" --ghost_offset "$ghost_offset")
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
