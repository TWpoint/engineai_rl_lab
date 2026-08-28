"""Continuously evaluate every 500th checkpoint of one explicitly scoped run.

This is intentionally run-specific.  It never discovers checkpoints or results
outside RUN_DIR, and it serializes the seven fixed length shards on one GPU.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from engineai_rl_lab.tasks.tracking.mdp.motion_data import resolve_motion_files


DEFAULT_RUN_DIR = Path(
    "/home/ubuntu/engineai/engineai_rl_lab/logs/rsl_rl/tracking_t800/"
    "2026-08-24_20-19-16_v21-scale"
)
RUN_DIR = Path(os.environ.get("ENGINEAI_FRAME0_RUN_DIR", str(DEFAULT_RUN_DIR))).resolve()
REPO_DIR = Path("/home/ubuntu/engineai/engineai_rl_lab")
PYTHON = Path("/home/ubuntu/engineai/engineai/bin/python")
SCRIPT_DIR = REPO_DIR / "scripts/tracking"
EVALUATION_DIR = RUN_DIR / "evaluation"
PROTOCOL_DIR = EVALUATION_DIR / "protocol_length_shards_seed20260818"
RESULTS_DIR = PROTOCOL_DIR / "results"
PROGRESS_CSV = EVALUATION_DIR / "checkpoint_progress.csv"
SOURCE_MANIFEST = Path("/mnt/data-1/lpz/t800_datasets/t800_v0.yaml")
ENV_SNAPSHOT = RUN_DIR / "params/env.pkl"
AGENT_SNAPSHOT = RUN_DIR / "params/agent.pkl"
TASK = os.environ.get(
    "ENGINEAI_FRAME0_TASK",
    "Tracking-Flat-T800-Wo-State-Estimation-v21-scale",
)
SEED = 20260818
NUM_MOTIONS = 8192
TRIALS = 3
PHYSICS = "newton_mjwarp"
CHECKPOINT_INTERVAL = 500
POLL_SECONDS = 30
RETRY_SECONDS = 300
MAX_INVALID_RETRIES = 3


class EvaluationAttemptError(RuntimeError):
    """An evaluation attempt failed, optionally with fresh invalid-state evidence."""

    def __init__(self, message: str, *, invalid_robot_state: int | None = None) -> None:
        super().__init__(message)
        self.invalid_robot_state = invalid_robot_state
TRACKING_ERROR_COUNT = 10
RESET_SYNC_VERSION = 1
MAX_PARALLEL_ENVS = 4096
SHARD_LABELS = (
    "le250",
    "251_500",
    "501_1000",
    "1001_2000",
    "2001_5000",
    "5001_10000",
    "gt10000",
)


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"[{stamp}] {message}", flush=True)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def upsert_progress(checkpoint: str, status: str, **values: object) -> None:
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    if PROGRESS_CSV.is_file():
        with PROGRESS_CSV.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    by_checkpoint = {row.get("checkpoint", ""): row for row in rows}
    row = by_checkpoint.setdefault(checkpoint, {"checkpoint": checkpoint})
    row["status"] = status
    for key, value in values.items():
        row[key] = str(value)
    for required in ("checkpoint", "status"):
        if required not in fields:
            fields.append(required)
    for item in by_checkpoint.values():
        for key in item:
            if key not in fields:
                fields.append(key)
    PROGRESS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=PROGRESS_CSV.parent,
        prefix=f".{PROGRESS_CSV.name}.", delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in sorted(
            (name for name in by_checkpoint if name),
            key=lambda value: int(value.rsplit("_", 1)[-1]),
        ):
            writer.writerow(by_checkpoint[name])
        temporary = Path(handle.name)
    os.replace(temporary, PROGRESS_CSV)


def progress_rows() -> dict[str, dict[str, str]]:
    if not PROGRESS_CSV.is_file():
        return {}
    with PROGRESS_CSV.open(newline="", encoding="utf-8") as handle:
        return {
            row.get("checkpoint", ""): row
            for row in csv.DictReader(handle)
            if row.get("checkpoint")
        }


def checkpoint_invalid_count(checkpoint: Path) -> int:
    """Count protocol-valid invalid-state attempts in any materialized shard."""
    checkpoint_dir = RESULTS_DIR / checkpoint.stem
    total = 0
    for path in checkpoint_dir.glob("shard_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = payload["results"][0]
            if not (
                len(payload["results"]) == 1
                and Path(result["checkpoint"]).resolve() == checkpoint.resolve()
                and payload.get("seed") == SEED
                and payload.get("physics") == PHYSICS
                and payload.get("frame_zero_reset_sync_version") == RESET_SYNC_VERSION
            ):
                continue
            total += sum(
                int(item.get("termination_counts", {}).get("invalid_robot_state", 0))
                for item in result.get("per_motion", [])
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError):
            continue
    return total


def shard_invalid_count(path: Path, checkpoint: Path) -> int | None:
    """Return invalid attempts only when *path* is valid evidence for checkpoint."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload.get("results", [])) != 1:
            return None
        result = payload["results"][0]
        if not (
            Path(result["checkpoint"]).resolve() == checkpoint.resolve()
            and payload.get("seed") == SEED
            and payload.get("physics") == PHYSICS
            and payload.get("frame_zero_reset_sync_version") == RESET_SYNC_VERSION
        ):
            return None
        return sum(
            int(item.get("termination_counts", {}).get("invalid_robot_state", 0))
            for item in result.get("per_motion", [])
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError):
        return None


def protocol_sha(files: list[str]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_protocol() -> dict:
    metadata_path = PROTOCOL_DIR / "protocol.json"
    manifest_path = PROTOCOL_DIR / "fixed_8192_seed20260818.yaml"
    plan_path = PROTOCOL_DIR / "shards.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list) or len(files) != NUM_MOTIONS or len(set(files)) != NUM_MOTIONS:
        raise ValueError("fixed protocol manifest is not exactly 8,192 unique motions")
    expected = {
        "seed": SEED,
        "num_motions": NUM_MOTIONS,
        "trials_per_motion": TRIALS,
        "start_frame": 0,
        "physics": PHYSICS,
        "source_manifest": str(SOURCE_MANIFEST),
        "env_cfg_snapshot": str(ENV_SNAPSHOT),
        "agent_cfg_snapshot": str(AGENT_SNAPSHOT),
        "selection_sha256": protocol_sha(files),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"protocol metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}")
    shards = plan.get("shards", [])
    if [item.get("label") for item in shards] != list(SHARD_LABELS):
        raise ValueError("the seven length-shard definitions/order changed")
    if sum(int(item["num_motions"]) for item in shards) != NUM_MOTIONS:
        raise ValueError("length shards do not sum to 8,192 motions")
    if plan.get("ordered_files") != files:
        raise ValueError("shard plan motion ordering differs from fixed manifest")
    for item in shards:
        shard_manifest = Path(item["manifest"])
        if not shard_manifest.is_file() or int(item["num_motions"]) <= 0:
            raise ValueError(f"missing or empty shard: {item}")
    return plan


def ensure_protocol() -> dict:
    try:
        return validate_protocol()
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log(f"fixed protocol needs preparation: {exc}")
    for required in (SOURCE_MANIFEST, ENV_SNAPSHOT, AGENT_SNAPSHOT):
        if not required.is_file():
            raise FileNotFoundError(required)
    all_files = resolve_motion_files(str(SOURCE_MANIFEST))
    if len(all_files) < NUM_MOTIONS or len(all_files) != len(set(all_files)):
        raise ValueError(
            f"source manifest must resolve to at least {NUM_MOTIONS} unique motions; "
            f"got total={len(all_files)}, unique={len(set(all_files))}"
        )
    length_checkpoint = RUN_DIR / "model_500.pt"
    if not stable_checkpoint(length_checkpoint):
        raise RuntimeError("model_500.pt must be stable before the fixed protocol is prepared")
    checkpoint = torch.load(length_checkpoint, map_location="cpu", weights_only=False)
    sampler = checkpoint["infos"]["motion_adaptive_sampling"]
    all_lengths = sampler["motion_lengths"].long().tolist()
    fingerprint_words = sampler["manifest_fingerprint_words"].long().tolist()
    del checkpoint
    if len(all_lengths) != len(all_files):
        raise ValueError(
            "checkpoint motion_lengths do not align with the target run's source manifest: "
            f"lengths={len(all_lengths)}, files={len(all_files)}"
        )
    # Cross-check deterministic positions against the original NPZ arrays before
    # trusting the checkpoint's manifest-order length vector for all 8,192 bins.
    anchor_rng = random.Random(SEED ^ 0x51A7)
    anchor_indices = sorted(anchor_rng.sample(range(len(all_files)), 32))
    for index in anchor_indices:
        with np.load(all_files[index], allow_pickle=False) as data:
            actual_length = int(data["joint_pos"].shape[0])
        if actual_length != int(all_lengths[index]):
            raise ValueError(
                f"checkpoint/source manifest order mismatch at {index}: "
                f"checkpoint={all_lengths[index]}, npz={actual_length}, file={all_files[index]}"
            )
    selected_indices = random.Random(SEED).sample(range(len(all_files)), NUM_MOTIONS)
    files = [all_files[index] for index in selected_indices]
    lengths = [int(all_lengths[index]) for index in selected_indices]
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROTOCOL_DIR / "fixed_8192_seed20260818.yaml"
    atomic_yaml(manifest_path, {"files": files})
    motion_lengths_path = PROTOCOL_DIR / "motion_lengths.json"
    atomic_json(
        motion_lengths_path,
        {"results": [{"per_motion": [
            {"file": path, "frames": frames} for path, frames in zip(files, lengths, strict=True)
        ]}]},
    )
    subprocess.run(
        [
            str(PYTHON), str(SCRIPT_DIR / "merge_frame0_macro_shards.py"),
            "prepare", "--reference_json", str(motion_lengths_path),
            "--output_dir", str(PROTOCOL_DIR),
        ],
        cwd=REPO_DIR,
        check=True,
    )
    atomic_json(
        PROTOCOL_DIR / "protocol.json",
        {
            "seed": SEED,
            "num_motions": NUM_MOTIONS,
            "trials_per_motion": TRIALS,
            "episodes": NUM_MOTIONS * TRIALS,
            "start_frame": 0,
            "physics": PHYSICS,
            "length_buckets": [
                "≤250", "251–500", "501–1000", "1001–2000",
                "2001–5000", "5001–10000", ">10000",
            ],
            "source_manifest": str(SOURCE_MANIFEST),
            "env_cfg_snapshot": str(ENV_SNAPSHOT),
            "agent_cfg_snapshot": str(AGENT_SNAPSHOT),
            "selection_sha256": protocol_sha(files),
            "motion_length_source": str(length_checkpoint.resolve()) + ":infos.motion_adaptive_sampling.motion_lengths",
            "manifest_fingerprint_words": fingerprint_words,
            "motion_length_anchor_checks": len(anchor_indices),
        },
    )
    plan = validate_protocol()
    log("fixed 8,192-motion seven-shard protocol prepared and validated")
    return plan


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(path.name)
    return int(match.group(1))


def eligible_checkpoints() -> list[Path]:
    checkpoints = []
    for path in RUN_DIR.glob("model_*.pt"):
        try:
            step = checkpoint_step(path)
        except ValueError:
            continue
        if step > 0 and step % CHECKPOINT_INTERVAL == 0:
            checkpoints.append(path)
    return sorted(checkpoints, key=checkpoint_step)


def stable_checkpoint(path: Path) -> bool:
    step = checkpoint_step(path)
    first = path.stat()
    if time.time() - first.st_mtime < 30:
        return False
    time.sleep(5)
    second = path.stat()
    if (first.st_size, first.st_mtime_ns) != (second.st_size, second.st_mtime_ns):
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - a partially written torch zip has varied errors.
        log(f"{path.name} is not loadable yet: {exc}")
        return False
    required_sampler = {
        "recipe", "bin_size", "manifest_fingerprint_words", "motion_lengths",
        "body_names_utf8", "difficulty_fast", "difficulty_slow", "total_exposure",
    }
    sampler = checkpoint.get("infos", {}).get("motion_adaptive_sampling")
    valid = (
        checkpoint.get("iter") == step
        and "actor_state_dict" in checkpoint
        and "critic_state_dict" in checkpoint
        and isinstance(sampler, dict)
        and required_sampler.issubset(sampler)
    )
    del checkpoint
    if not valid:
        log(f"{path.name} loaded but required iteration/model/sampler data are incomplete")
    return valid


def result_paths(checkpoint: Path) -> tuple[Path, Path, Path]:
    step = checkpoint_step(checkpoint)
    merged = RESULTS_DIR / f"model_{step}" / f"model_{step}_frame0_merged.json"
    report = EVALUATION_DIR / f"model_{step}_comprehensive.md"
    raw = report.with_suffix(".json")
    return merged, report, raw


def validate_evaluation(path: Path, checkpoint: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload.get("results", [])) != 1:
        raise ValueError("expected exactly one checkpoint result")
    result = payload["results"][0]
    motions = result.get("per_motion", [])
    files = [item.get("file") for item in motions]
    attempts = [item.get("attempts") for item in motions]
    invalid = sum(
        int(item.get("termination_counts", {}).get("invalid_robot_state", 0))
        for item in motions
    )
    expected_checkpoint = str(checkpoint.resolve())
    checks = {
        "checkpoint": str(Path(result.get("checkpoint", "")).resolve()) == expected_checkpoint,
        "seed": payload.get("seed") == SEED,
        "physics": payload.get("physics") == PHYSICS,
        "frame_zero_reset_sync_version": (
            payload.get("frame_zero_reset_sync_version") == RESET_SYNC_VERSION
        ),
        "env_snapshot": payload.get("environment_config_snapshot") == str(ENV_SNAPSHOT.resolve()),
        "agent_snapshot": payload.get("agent_config_snapshot") == str(AGENT_SNAPSHOT.resolve()),
        "motions": payload.get("num_motions") == result.get("motions") == len(motions) == NUM_MOTIONS,
        "episodes": result.get("episodes") == sum(attempts) == NUM_MOTIONS * TRIALS,
        "attempts": all(value == TRIALS for value in attempts),
        "unique_files": len(set(files)) == NUM_MOTIONS,
        "invalid_robot_state": invalid == 0,
        "tracking_errors": len(result.get("tracking_error_summary", {})) == TRACKING_ERROR_COUNT,
    }
    if not all(checks.values()):
        raise ValueError(f"strict merged validation failed: {checks}")
    return payload


def validate_comprehensive(checkpoint: Path) -> bool:
    merged, report, raw = result_paths(checkpoint)
    if not (merged.is_file() and report.is_file() and raw.is_file()):
        return False
    try:
        evaluation = validate_evaluation(merged, checkpoint)
        result = evaluation["results"][0]
        payload = json.loads(raw.read_text(encoding="utf-8"))
        report_text = report.read_text(encoding="utf-8")
        chart = report.with_name(f"{checkpoint.stem}_success_trend.png")
        mastered_chart = chart
        validation = payload.get("validation", {})
        if not (
            Path(payload.get("checkpoint", "")).resolve() == checkpoint.resolve()
            and Path(payload.get("success_trend_chart", "")).resolve() == chart.resolve()
            and chart.is_file()
            and chart.stat().st_size > 0
            and chart.name in report_text
            and payload.get("mastered_proxy_trend_chart")
            and Path(payload["mastered_proxy_trend_chart"]).resolve() == mastered_chart.resolve()
            and mastered_chart.is_file()
            and mastered_chart.stat().st_size > 0
            and mastered_chart.name in report_text
            and validation.get("complete") is True
            and validation.get("motions") == NUM_MOTIONS
            and validation.get("episodes") == NUM_MOTIONS * TRIALS
            and validation.get("attempts") == NUM_MOTIONS * TRIALS
            and validation.get("attempts_per_motion") == TRIALS
            and validation.get("unique_files") == NUM_MOTIONS
            and validation.get("invalid_robot_state") == 0
            and validation.get("tracking_errors") == TRACKING_ERROR_COUNT
        ):
            return False

        # Every actual termination reason except the invalid-state gate must be
        # explicit in both the comprehensive raw artifact and Chinese report.
        expected_terminations = {
            name for name in result["termination_rates"] if name != "invalid_robot_state"
        }
        termination_reasons = payload.get("termination_reasons", {})
        if set(termination_reasons) != expected_terminations:
            return False
        attempts = sum(item["attempts"] for item in result["per_motion"])
        for name in expected_terminations:
            count = sum(
                item.get("termination_counts", {}).get(name, 0)
                for item in result["per_motion"]
            )
            item = termination_reasons[name]
            if (
                item.get("count") != count
                or abs(float(item.get("rate", -1.0)) - count / attempts) > 1e-12
                or name not in report_text
            ):
                return False

        sampler_comparison = payload.get("sampler_comparison")
        frame0_comparison = payload.get("frame0_comparison")
        if sampler_comparison is not None or frame0_comparison is not None:
            if not isinstance(sampler_comparison, dict) or not isinstance(frame0_comparison, dict):
                return False
            if frame0_comparison.get("protocol", {}).get("bootstrap_draws") != 100_000:
                return False
            comparison_terminations = frame0_comparison.get("termination_reasons", {})
            if set(comparison_terminations) != expected_terminations:
                return False
            bins = sampler_comparison.get("current", {}).get("bins")
            transitions = sampler_comparison.get("transitions", {})
            matrix = transitions.get("category_matrix", {})
            interpretation = transitions.get("interpretation", {})
            matrix_total = sum(
                sum(int(value) for value in row.values())
                for row in matrix.values()
                if isinstance(row, dict)
            )
            interpretation_total = sum(
                int(interpretation.get(name, -bins if isinstance(bins, int) else -1))
                for name in ("逐渐掌握", "重新开始学习", "困难停滞", "发生退化", "其他")
            )
            if not isinstance(bins, int) or matrix_total != bins or interpretation_total != bins:
                return False
        else:
            sampler = payload.get("sampler", {})
            if sampler.get("source") != "checkpoint infos.motion_adaptive_sampling":
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def same_checkpoint_process(checkpoint: Path) -> bool:
    needle = str(checkpoint.resolve()).encode()
    evaluator_name = b"evaluate_frame0_macro.py"
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # Match real argv entries, not a diagnostic shell command whose text
        # happens to mention both the evaluator and checkpoint path.
        argv = [item for item in command.split(b"\0") if item]
        has_evaluator = any(item.rsplit(b"/", 1)[-1] == evaluator_name for item in argv)
        if has_evaluator and needle in argv:
            return True
    return False


def validate_shard(path: Path, checkpoint: Path, expected_motions: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        motions = result["per_motion"]
        attempts = [item["attempts"] for item in motions]
        invalid = sum(
            item.get("termination_counts", {}).get("invalid_robot_state", 0)
            for item in motions
        )
        return (
            len(payload["results"]) == 1
            and str(Path(result["checkpoint"]).resolve()) == str(checkpoint.resolve())
            and payload["seed"] == SEED
            and payload.get("physics") == PHYSICS
            and payload["environment_config_snapshot"] == str(ENV_SNAPSHOT.resolve())
            and payload["agent_config_snapshot"] == str(AGENT_SNAPSHOT.resolve())
            and payload.get("frame_zero_reset_sync_version") == RESET_SYNC_VERSION
            and payload["num_motions"] == result["motions"] == len(motions) == expected_motions
            and len({item["file"] for item in motions}) == expected_motions
            and result["episodes"] == sum(attempts) == expected_motions * TRIALS
            and all(value == TRIALS for value in attempts)
            and invalid == 0
            and len(result["tracking_error_summary"]) == TRACKING_ERROR_COUNT
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError):
        return False


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=REPO_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def evaluate_checkpoint(checkpoint: Path, plan: dict) -> None:
    step = checkpoint_step(checkpoint)
    checkpoint_name = checkpoint.stem
    checkpoint_dir = RESULTS_DIR / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shard_jsons = []
    for index, shard in enumerate(plan["shards"]):
        label = shard["label"]
        expected_motions = int(shard["num_motions"])
        output_txt = checkpoint_dir / f"shard_{index:02d}_{label}.txt"
        output_json = output_txt.with_suffix(".json")
        shard_jsons.append(output_json)
        if validate_shard(output_json, checkpoint, expected_motions):
            log(f"{checkpoint_name} shard {label}: complete result already exists")
            continue
        if same_checkpoint_process(checkpoint):
            raise RuntimeError(f"same-checkpoint evaluation process appeared for {checkpoint_name}")
        command = [
            str(PYTHON), str(SCRIPT_DIR / "evaluate_frame0_macro.py"),
            "--task", TASK,
            "--checkpoints", str(checkpoint.resolve()),
            "--motion_file", shard["manifest"],
            "--num_motions", str(expected_motions),
            "--num_envs", str(min(expected_motions * TRIALS, MAX_PARALLEL_ENVS)),
            "--trials_per_motion", str(TRIALS),
            "--eval_seed", str(SEED),
            "--output", str(output_txt),
            "--env_cfg_snapshot", str(ENV_SNAPSHOT.resolve()),
            "--agent_cfg_snapshot", str(AGENT_SNAPSHOT.resolve()),
            "--device", "cuda:0",
            "--parallel_trials",
            f"physics={PHYSICS}",
        ]
        # The reset-sync fix makes trials independent across environments.  Large
        # buckets retain their exact membership and are batched inside the evaluator.
        log(f"{checkpoint_name} shard {label}: starting {expected_motions} motions")
        previous_signature = None
        if output_json.is_file():
            stat = output_json.stat()
            previous_signature = (stat.st_size, stat.st_mtime_ns)
        attempt_started_ns = time.time_ns()
        try:
            run_command(command, checkpoint_dir / f"shard_{index:02d}_{label}.stdout.log")
        except subprocess.CalledProcessError as exc:
            fresh = output_json.is_file() and output_json.stat().st_mtime_ns >= attempt_started_ns
            invalid = shard_invalid_count(output_json, checkpoint) if fresh else None
            raise EvaluationAttemptError(
                f"evaluator exited {exc.returncode} without an acceptable fresh result for {output_json}",
                invalid_robot_state=invalid,
            ) from exc
        if not output_json.is_file():
            raise EvaluationAttemptError(f"evaluator produced no JSON for {output_json}")
        stat = output_json.stat()
        current_signature = (stat.st_size, stat.st_mtime_ns)
        if stat.st_mtime_ns < attempt_started_ns or current_signature == previous_signature:
            raise EvaluationAttemptError(f"evaluator did not refresh stale JSON for {output_json}")
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        payload["physics"] = PHYSICS
        payload["fixed_protocol"] = str((PROTOCOL_DIR / "protocol.json").resolve())
        atomic_json(output_json, payload)
        if not validate_shard(output_json, checkpoint, expected_motions):
            raise EvaluationAttemptError(
                f"strict validation failed for {output_json}",
                invalid_robot_state=shard_invalid_count(output_json, checkpoint),
            )
        log(f"{checkpoint_name} shard {label}: complete and validated")

    merged_txt = checkpoint_dir / f"model_{step}_frame0_merged.txt"
    merge_command = [
        str(PYTHON), str(SCRIPT_DIR / "merge_frame0_macro_shards.py"), "merge",
        "--plan", str((PROTOCOL_DIR / "shards.json").resolve()),
        "--shard_jsons", *(str(path.resolve()) for path in shard_jsons),
        "--output", str(merged_txt.resolve()),
    ]
    run_command(merge_command, checkpoint_dir / "merge.stdout.log")
    merged_json = merged_txt.with_suffix(".json")
    payload = json.loads(merged_json.read_text(encoding="utf-8"))
    payload["physics"] = PHYSICS
    payload["fixed_protocol"] = str((PROTOCOL_DIR / "protocol.json").resolve())
    atomic_json(merged_json, payload)
    validate_evaluation(merged_json, checkpoint)
    log(f"{checkpoint_name}: seven shards merged and strictly validated")

    completed_previous = []
    if PROGRESS_CSV.is_file():
        with PROGRESS_CSV.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = row.get("checkpoint", "")
                if row.get("status") != "complete" or not name.startswith("model_"):
                    continue
                candidate = RUN_DIR / f"{name}.pt"
                candidate_merged, _, _ = result_paths(candidate)
                if candidate.is_file() and checkpoint_step(candidate) < step:
                    try:
                        evaluated = validate_evaluation(candidate_merged, candidate)
                    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                        continue
                    completed_previous.append((candidate, candidate_merged, evaluated))
    completed_previous.sort(key=lambda item: checkpoint_step(item[0]))
    report = EVALUATION_DIR / f"model_{step}_comprehensive.md"
    report_command = [
        str(PYTHON), str(SCRIPT_DIR / "build_comprehensive_checkpoint_report.py"),
        "--checkpoint", str(checkpoint.resolve()),
        "--evaluation", str(merged_json.resolve()),
        "--output", str(report.resolve()),
        "--progress_csv", str(PROGRESS_CSV.resolve()),
    ]
    if completed_previous:
        previous_checkpoint, previous_eval, _ = completed_previous[-1]
        best_checkpoint, best_eval, _ = max(
            completed_previous,
            key=lambda item: item[2]["results"][0]["macro_success_rate"],
        )
        report_command.extend(
            [
                "--previous_checkpoint", str(previous_checkpoint.resolve()),
                "--previous_evaluation", str(previous_eval.resolve()),
                "--best_checkpoint", str(best_checkpoint.resolve()),
                "--best_evaluation", str(best_eval.resolve()),
            ]
        )
    run_command(report_command, checkpoint_dir / "report.stdout.log")
    if not validate_comprehensive(checkpoint):
        raise ValueError(f"comprehensive output validation failed for {checkpoint_name}")
    upsert_progress(
        checkpoint_name,
        "complete",
        frame_zero_reset_sync_version=RESET_SYNC_VERSION,
        completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        error="",
    )
    log(f"{checkpoint_name}: comprehensive Chinese report/raw JSON complete")


def main() -> None:
    expected_run = Path(
        os.environ.get("ENGINEAI_FRAME0_EXPECTED_RUN_DIR", str(DEFAULT_RUN_DIR))
    ).resolve()
    if RUN_DIR != expected_run:
        raise RuntimeError(f"run scope mismatch: configured={RUN_DIR}, expected={expected_run}")
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(RUN_DIR)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = (EVALUATION_DIR / ".monitor.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another monitor instance already owns the run lock; exiting")
        return
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()
    atomic_json(
        EVALUATION_DIR / "monitor_state.json",
        {
            "pid": os.getpid(),
            "run": str(RUN_DIR),
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "seed": SEED,
            "physics": PHYSICS,
            "status": "starting",
        },
    )
    while True:
        try:
            plan = ensure_protocol()
            break
        except (FileNotFoundError, RuntimeError) as exc:
            waiting_for_model_500 = (
                isinstance(exc, FileNotFoundError)
                and Path(exc.filename or "").name == "model_500.pt"
            ) or "model_500.pt must be stable" in str(exc)
            if not waiting_for_model_500:
                raise
            atomic_json(
                EVALUATION_DIR / "monitor_state.json",
                {
                    "pid": os.getpid(), "run": str(RUN_DIR),
                    "checkpoint_interval": CHECKPOINT_INTERVAL, "seed": SEED,
                    "physics": PHYSICS, "status": "waiting_for_protocol_checkpoint",
                    "checkpoint": "model_500", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            log("waiting for stable model_500.pt before preparing the fixed protocol")
            time.sleep(POLL_SECONDS)
    retry_after: dict[int, float] = {}
    log("run-specific monitor active; evaluating positive checkpoints divisible by 500")
    while True:
        checkpoints = eligible_checkpoints()
        rows = progress_rows()
        for checkpoint in checkpoints:
            name = checkpoint.stem
            step = checkpoint_step(checkpoint)
            if validate_comprehensive(checkpoint):
                upsert_progress(name, "complete")
                continue
            if rows.get(name, {}).get("status") == "incomplete_invalid_robot_state":
                # This checkpoint was independently confirmed to violate the
                # fixed protocol.  Preserve its evidence, never call it
                # complete, and do not let it permanently block later models.
                continue
            if time.time() < retry_after.get(step, 0):
                break
            if same_checkpoint_process(checkpoint):
                upsert_progress(name, "external_process_detected")
                log(f"{name}: same-checkpoint evaluation already running; not starting a duplicate")
                break
            if not stable_checkpoint(checkpoint):
                upsert_progress(name, "waiting_for_stable_checkpoint")
                break
            upsert_progress(name, "evaluating", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            atomic_json(
                EVALUATION_DIR / "monitor_state.json",
                {
                    "pid": os.getpid(), "run": str(RUN_DIR), "checkpoint_interval": CHECKPOINT_INTERVAL,
                    "seed": SEED, "physics": PHYSICS, "status": "evaluating", "checkpoint": name,
                },
            )
            try:
                evaluate_checkpoint(checkpoint, plan)
            except Exception as exc:  # noqa: BLE001 - persist diagnostics and retry later.
                # Never reuse an older shard's invalid count as evidence for the
                # current attempt.  Only a freshly rewritten JSON can contribute.
                invalid = (
                    exc.invalid_robot_state
                    if isinstance(exc, EvaluationAttemptError)
                    and exc.invalid_robot_state is not None
                    else 0
                )
                prior_failures = int(rows.get(name, {}).get("validation_failure_count") or 0)
                failure_count = prior_failures + (1 if invalid > 0 else 0)
                if invalid > 0 and failure_count >= MAX_INVALID_RETRIES:
                    upsert_progress(
                        name,
                        "incomplete_invalid_robot_state",
                        invalid_robot_state=invalid,
                        validation_failure_count=failure_count,
                        error=repr(exc),
                    )
                    log(
                        f"{name}: invalid_robot_state={invalid} reproduced "
                        f"{failure_count} times; preserving incomplete result and continuing"
                    )
                    continue
                retry_after[step] = time.time() + RETRY_SECONDS
                upsert_progress(
                    name,
                    "failed_validation_or_execution",
                    invalid_robot_state=invalid,
                    validation_failure_count=failure_count,
                    error=repr(exc),
                )
                atomic_json(
                    EVALUATION_DIR / "monitor_state.json",
                    {
                        "pid": os.getpid(), "run": str(RUN_DIR), "checkpoint_interval": CHECKPOINT_INTERVAL,
                        "seed": SEED, "physics": PHYSICS, "status": "retry_wait", "checkpoint": name,
                        "error": repr(exc), "retry_after_epoch": retry_after[step],
                    },
                )
                log(f"{name}: evaluation/validation failed; retrying in {RETRY_SECONDS}s: {exc!r}")
                break
        atomic_json(
            EVALUATION_DIR / "monitor_state.json",
            {
                "pid": os.getpid(), "run": str(RUN_DIR), "checkpoint_interval": CHECKPOINT_INTERVAL,
                "seed": SEED, "physics": PHYSICS, "status": "watching",
                "eligible_checkpoints": [path.stem for path in checkpoints],
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
