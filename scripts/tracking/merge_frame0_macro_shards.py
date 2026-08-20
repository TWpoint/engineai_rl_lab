"""Prepare length-balanced evaluation shards and merge their frame-zero results."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict

import numpy as np
import yaml

LENGTH_BINS = (
    ("le250", -math.inf, 250),
    ("251_500", 250, 500),
    ("501_1000", 500, 1000),
    ("1001_2000", 1000, 2000),
    ("2001_5000", 2000, 5000),
    ("5001_10000", 5000, 10000),
    ("gt10000", 10000, math.inf),
)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"p{p}": float(np.quantile(values, p / 100.0)) for p in (10, 50, 90)}


def _source_name(path: str) -> str:
    parts = pathlib.Path(path).parts
    for marker in ("t800_datasets", "motions"):
        try:
            return parts[parts.index(marker) + 1]
        except (ValueError, IndexError):
            pass
    return "other"


def prepare(reference_json: pathlib.Path, output_dir: pathlib.Path) -> None:
    reference = json.loads(reference_json.read_text(encoding="utf-8"))
    motions = reference["results"][0]["per_motion"]
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    assigned = set()
    for index, (label, lower, upper) in enumerate(LENGTH_BINS):
        selected = [m for m in motions if lower < int(m["frames"]) <= upper]
        if not selected:
            continue
        paths = [m["file"] for m in selected]
        overlap = assigned.intersection(paths)
        if overlap:
            raise RuntimeError(f"Motions assigned more than once: {sorted(overlap)[:3]}")
        assigned.update(paths)
        manifest = output_dir / f"shard_{index:02d}_{label}.yaml"
        manifest.write_text(yaml.safe_dump({"files": paths}, sort_keys=False), encoding="utf-8")
        shards.append(
            {
                "label": label,
                "manifest": str(manifest.resolve()),
                "num_motions": len(paths),
                "num_envs": len(paths),
                "min_frames": min(int(m["frames"]) for m in selected),
                "max_frames": max(int(m["frames"]) for m in selected),
            }
        )
    if len(assigned) != len(motions):
        raise RuntimeError(f"Assigned {len(assigned)} of {len(motions)} reference motions")
    plan = {
        "reference_json": str(reference_json.resolve()),
        "num_motions": len(motions),
        "ordered_files": [m["file"] for m in motions],
        "shards": shards,
    }
    plan_path = output_dir / "shards.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(plan_path)
    for shard in shards:
        print(
            f"{shard['label']}: motions={shard['num_motions']}, frames=[{shard['min_frames']}, {shard['max_frames']}]"
        )


def _merge_checkpoint(checkpoint_results: list[dict], ordered_files: list[str]) -> dict:
    by_file = {}
    for result in checkpoint_results:
        for motion in result["per_motion"]:
            if motion["file"] in by_file:
                raise RuntimeError(f"Duplicate motion in shards: {motion['file']}")
            by_file[motion["file"]] = motion
    missing = set(ordered_files).difference(by_file)
    extra = set(by_file).difference(ordered_files)
    if missing or extra:
        raise RuntimeError(f"Shard mismatch: missing={len(missing)}, extra={len(extra)}")

    per_motion = []
    for motion_id, path in enumerate(ordered_files):
        motion = dict(by_file[path])
        motion["motion_id"] = motion_id
        motion["source"] = _source_name(path)
        per_motion.append(motion)

    attempts = np.asarray([m["attempts"] for m in per_motion], dtype=np.int64)
    successes = np.asarray([m["successes"] for m in per_motion], dtype=np.int64)
    failed = np.asarray([m["failed_attempts"] for m in per_motion], dtype=np.int64)
    rates = successes / attempts
    failure_names = checkpoint_results[0]["failure_termination_names"]
    success_names = checkpoint_results[0]["success_termination_names"]
    termination_names = list(per_motion[0]["termination_counts"])
    termination_rates = {
        name: float(sum(m["termination_counts"][name] for m in per_motion) / attempts.sum())
        for name in termination_names
    }

    error_names = list(per_motion[0]["tracking_errors"])
    error_summary = {}
    step_counts = np.asarray([m["mean_episode_length"] * m["attempts"] for m in per_motion], dtype=np.float64)
    for name in error_names:
        values = np.asarray([m["tracking_errors"][name] for m in per_motion], dtype=np.float64)
        error_summary[name] = {
            "macro_mean": float(values.mean()),
            "transition_weighted_mean": float(np.average(values, weights=step_counts)),
            **_quantiles(values),
        }

    source_ids: dict[str, list[int]] = defaultdict(list)
    for index, motion in enumerate(per_motion):
        source_ids[motion["source"]].append(index)
    source_summary = {}
    for source, ids in sorted(source_ids.items()):
        source_rates = rates[np.asarray(ids)]
        source_summary[source] = {
            "motions": len(ids),
            "macro_success_rate": float(source_rates.mean()),
            "zero_success_motion_fraction": float(np.mean(source_rates == 0.0)),
            "all_success_motion_fraction": float(np.mean(source_rates == 1.0)),
            "success_rate_quantiles": _quantiles(source_rates),
        }

    worst = sorted(
        per_motion,
        key=lambda item: (
            item["success_rate"],
            -item["failed_attempts"],
            item["mean_episode_length"],
        ),
    )[:50]
    return {
        "checkpoint": checkpoint_results[0]["checkpoint"],
        "motions": len(per_motion),
        "trials_per_motion": int(attempts[0]),
        "episodes": int(attempts.sum()),
        "macro_success_rate": float(rates.mean()),
        "micro_success_rate": float(successes.sum() / attempts.sum()),
        "success_rate_quantiles": _quantiles(rates),
        "zero_success_motions": int(np.sum(rates == 0.0)),
        "zero_success_motion_fraction": float(np.mean(rates == 0.0)),
        "all_success_motions": int(np.sum(rates == 1.0)),
        "all_success_motion_fraction": float(np.mean(rates == 1.0)),
        "partial_success_motions": int(np.sum((rates > 0.0) & (rates < 1.0))),
        "failure_rate": float(failed.sum() / attempts.sum()),
        "failure_termination_names": failure_names,
        "success_termination_names": success_names,
        "termination_rates": termination_rates,
        "macro_mean_episode_reward": float(np.mean([m["mean_episode_reward"] for m in per_motion])),
        "macro_mean_episode_length": float(np.mean([m["mean_episode_length"] for m in per_motion])),
        "macro_mean_step_reward": float(np.mean([m["mean_step_reward"] for m in per_motion])),
        "tracking_error_summary": error_summary,
        "source_summary": source_summary,
        "worst_motions": worst,
        "per_motion": per_motion,
    }


def _format_result(result: dict) -> list[str]:
    q = result["success_rate_quantiles"]
    lines = [
        f"Checkpoint: {result['checkpoint']}",
        f"  motions: {result['motions']:,}",
        f"  equal trials per motion: {result['trials_per_motion']}",
        f"  episodes: {result['episodes']:,}",
        f"  macro success rate: {result['macro_success_rate']:.6f}",
        f"  per-motion success P10/P50/P90: {q['p10']:.6f}/{q['p50']:.6f}/{q['p90']:.6f}",
        f"  zero-success motions: {result['zero_success_motions']:,} ({result['zero_success_motion_fraction']:.6f})",
        f"  partial-success motions: {result['partial_success_motions']:,}",
        f"  all-success motions: {result['all_success_motions']:,} ({result['all_success_motion_fraction']:.6f})",
        f"  episode failure rate (union): {result['failure_rate']:.6f}",
        f"  macro mean episode length: {result['macro_mean_episode_length']:.3f}",
        f"  macro mean episode reward: {result['macro_mean_episode_reward']:.6f}",
        f"  macro mean step reward: {result['macro_mean_step_reward']:.6f}",
        "  termination rates (may overlap):",
    ]
    for name, rate in result["termination_rates"].items():
        role = "success" if name in result["success_termination_names"] else "failure"
        lines.append(f"    {name} [{role}]: {rate:.6f}")
    lines.append("  tracking errors (macro mean / P50 / P90):")
    for name, summary in result["tracking_error_summary"].items():
        lines.append(f"    {name}: {summary['macro_mean']:.6f} / {summary['p50']:.6f} / {summary['p90']:.6f}")
    return lines


def merge(plan_path: pathlib.Path, shard_jsons: list[pathlib.Path], output: pathlib.Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shard_jsons]
    checkpoint_names = [[pathlib.Path(r["checkpoint"]).name for r in p["results"]] for p in payloads]
    if any(names != checkpoint_names[0] for names in checkpoint_names[1:]):
        raise RuntimeError(f"Checkpoint ordering differs across shards: {checkpoint_names}")
    merged_results = []
    for index in range(len(checkpoint_names[0])):
        merged_results.append(_merge_checkpoint([p["results"][index] for p in payloads], plan["ordered_files"]))
    first = payloads[0]
    merged = {
        "protocol": "frame_zero_equal_trials_per_motion_length_sharded",
        "task": first["task"],
        "environment_config_snapshot": first["environment_config_snapshot"],
        "agent_config_snapshot": first["agent_config_snapshot"],
        "source_motion_file": plan["reference_json"],
        "evaluation_manifest": str(plan_path.resolve()),
        "seed": first["seed"],
        "num_motions": len(plan["ordered_files"]),
        "num_envs": {s["label"]: s["num_envs"] for s in plan["shards"]},
        "trials_per_motion": first["trials_per_motion"],
        "results": merged_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = [
        "Frame-zero per-motion macro checkpoint evaluation (length-sharded)",
        f"Task: {first['task']}",
        f"Seed: {first['seed']}",
        f"Motions: {len(plan['ordered_files'])}",
        f"Equal trials per motion: {first['trials_per_motion']}",
        "Episode start: frame zero",
        "Aggregation: equal-weight per-motion macro average",
        "",
    ]
    for result in merged_results:
        report.extend(_format_result(result))
        report.append("")
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)
    print(output.with_suffix(".json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--reference_json", type=pathlib.Path, required=True)
    prepare_parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--plan", type=pathlib.Path, required=True)
    merge_parser.add_argument("--shard_jsons", type=pathlib.Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.reference_json, args.output_dir)
    else:
        merge(args.plan, args.shard_jsons, args.output)


if __name__ == "__main__":
    main()
