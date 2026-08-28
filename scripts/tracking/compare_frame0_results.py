"""Compare two validated frame-zero macro evaluation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BUCKETS = (
    ("≤250", 0, 250),
    ("251–500", 251, 500),
    ("501–1000", 501, 1000),
    ("1001–2000", 1001, 2000),
    ("2001–5000", 2001, 5000),
    ("5001–10000", 5001, 10000),
    (">10000", 10001, None),
)


def load(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload["results"]) != 1:
        raise ValueError(f"Expected one checkpoint result in {path}")
    result = payload["results"][0]
    motions = result["per_motion"]
    files = [item["file"] for item in motions]
    attempts = [item["attempts"] for item in motions]
    invalid = sum(item["termination_counts"].get("invalid_robot_state", 0) for item in motions)
    if not (
        payload.get("frame_zero_reset_sync_version") == 1
        and payload.get("physics") == "newton_mjwarp"
        and payload["seed"] == 20260818
        and payload["num_motions"] == result["motions"] == len(motions) == len(set(files)) == 8192
        and payload["trials_per_motion"] == 3
        and result["episodes"] == sum(attempts) == 24576
        and all(value == 3 for value in attempts)
        and invalid == 0
        and len(result["tracking_error_summary"]) == 10
    ):
        raise ValueError(f"Protocol validation failed for {path}")
    return payload, result


def bootstrap(delta: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    batch = 1000
    for start in range(0, draws, batch):
        end = min(start + batch, draws)
        indices = rng.integers(0, len(delta), size=(end - start, len(delta)))
        means[start:end] = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def termination_summary(result: dict, *, include_invalid: bool = False) -> dict[str, dict]:
    """Return exact episode counts and rates for every active termination reason."""

    excluded = set() if include_invalid else {"invalid_robot_state"}
    success_names = set(result.get("success_termination_names", ()))
    failure_names = set(result.get("failure_termination_names", ()))
    names = [name for name in result["termination_rates"] if name not in excluded]
    attempts = sum(item["attempts"] for item in result["per_motion"])
    summary = {}
    for name in names:
        count = sum(item["termination_counts"].get(name, 0) for item in result["per_motion"])
        role = "success" if name in success_names else "failure" if name in failure_names else "other"
        summary[name] = {"role": role, "count": int(count), "rate": float(count / attempts)}
    return summary


def grouped(result: dict, key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for item in result["per_motion"]:
        if key == "length":
            frames = item["frames"]
            name = next(label for label, low, high in BUCKETS if frames >= low and (high is None or frames <= high))
        else:
            name = item["source"]
        groups.setdefault(name, []).append(item)
    return {
        name: {
            "motions": len(items),
            "success_rate": float(np.mean([item["success_rate"] for item in items])),
            "body_pos_failure_rate": float(
                sum(item["termination_counts"].get("body_pos", 0) for item in items)
                / sum(item["attempts"] for item in items)
            ),
            "termination_rates": {
                termination: float(
                    sum(item["termination_counts"].get(termination, 0) for item in items)
                    / sum(item["attempts"] for item in items)
                )
                for termination in result["termination_rates"]
                if termination != "invalid_robot_state"
            },
        }
        for name, items in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    baseline_payload, baseline = load(args.baseline)
    target_payload, target = load(args.target)
    baseline_files = [item["file"] for item in baseline["per_motion"]]
    target_files = [item["file"] for item in target["per_motion"]]
    if baseline_files != target_files:
        raise ValueError("Per-motion file ordering differs")
    baseline_rates = np.asarray([item["success_rate"] for item in baseline["per_motion"]])
    target_rates = np.asarray([item["success_rate"] for item in target["per_motion"]])
    delta = target_rates - baseline_rates
    ci = bootstrap(delta, args.seed, args.bootstrap_draws)
    length_baseline, length_target = grouped(baseline, "length"), grouped(target, "length")
    source_baseline, source_target = grouped(baseline, "source"), grouped(target, "source")
    errors = {
        name: {
            "baseline": baseline["tracking_error_summary"][name]["macro_mean"],
            "target": target["tracking_error_summary"][name]["macro_mean"],
            "delta": target["tracking_error_summary"][name]["macro_mean"]
            - baseline["tracking_error_summary"][name]["macro_mean"],
        }
        for name in target["tracking_error_summary"]
    }
    baseline_terminations = termination_summary(baseline)
    target_terminations = termination_summary(target)
    termination_names = list(dict.fromkeys([*baseline_terminations, *target_terminations]))
    terminations = {
        name: {
            "role": target_terminations.get(name, baseline_terminations[name])["role"],
            "baseline": baseline_terminations.get(name, {"count": 0, "rate": 0.0}),
            "target": target_terminations.get(name, {"count": 0, "rate": 0.0}),
            "rate_delta": target_terminations.get(name, {"rate": 0.0})["rate"]
            - baseline_terminations.get(name, {"rate": 0.0})["rate"],
        }
        for name in termination_names
    }
    comparison = {
        "scope": "same_run_only",
        "run": args.target.parents[1].name,
        "protocol": {
            "motions": 8192,
            "episodes": 24576,
            "trials_per_motion": 3,
            "seed": args.seed,
            "start_frame": 0,
            "physics": "newton_mjwarp",
            "length_buckets": [item[0] for item in BUCKETS],
            "bootstrap_draws": args.bootstrap_draws,
        },
        "baseline": baseline,
        "target": target,
        "success_delta": float(delta.mean()),
        "success_delta_95_ci": ci,
        "improved_motions": int(np.sum(delta > 0)),
        "unchanged_motions": int(np.sum(delta == 0)),
        "regressed_motions": int(np.sum(delta < 0)),
        "length_buckets": {
            name: {"baseline": length_baseline[name], "target": length_target[name],
                   "success_delta": length_target[name]["success_rate"] - length_baseline[name]["success_rate"]}
            for name, _, _ in BUCKETS
        },
        "sources": {
            name: {"baseline": source_baseline[name], "target": source_target[name],
                   "success_delta": source_target[name]["success_rate"] - source_baseline[name]["success_rate"]}
            for name in sorted(target_payload["results"][0]["source_summary"])
        },
        "tracking_errors": errors,
        "termination_reasons": terminations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pct(value: float) -> str:
        return f"{value * 100:.4f}%"

    lines = [
        f"# {comparison['run']}：{Path(target['checkpoint']).stem} vs {Path(baseline['checkpoint']).stem}",
        "",
        "## 结论",
        "",
        f"目标 success 为 **{pct(target['macro_success_rate'])}**，基准为 **{pct(baseline['macro_success_rate'])}**；差值 **{delta.mean() * 100:+.4f} pp**，100,000 次逐 motion paired bootstrap 95% CI **[{ci[0] * 100:+.4f}, {ci[1] * 100:+.4f}] pp**。",
        "",
        "仅比较本 run 内 checkpoint；训练在线 termination 不与本报告的固定 frame-0 failure 直接比较。",
        "",
        "## 核心指标",
        "",
        "| 指标 | 基准 | 目标 |",
        "|---|---:|---:|",
        f"| Success | {pct(baseline['macro_success_rate'])} | {pct(target['macro_success_rate'])} |",
        f"| body_pos failure | {pct(baseline['termination_rates']['body_pos'])} | {pct(target['termination_rates']['body_pos'])} |",
        f"| invalid_robot_state | {baseline['termination_rates'].get('invalid_robot_state', 0):.0f} | {target['termination_rates'].get('invalid_robot_state', 0):.0f} |",
        f"| Zero / Partial / All success motions | {baseline['zero_success_motions']} / {baseline['partial_success_motions']} / {baseline['all_success_motions']} | {target['zero_success_motions']} / {target['partial_success_motions']} / {target['all_success_motions']} |",
        f"| 平均 episode 长度 | {baseline['macro_mean_episode_length']:.3f} | {target['macro_mean_episode_length']:.3f} |",
        "",
        f"逐 motion：改善 {comparison['improved_motions']}，不变 {comparison['unchanged_motions']}，回退 {comparison['regressed_motions']}。",
        "",
        "## 全部终止原因（排除 invalid_robot_state）",
        "",
        "终止项可能重叠；比例分母均为 24,576 episodes。",
        "",
        "| 终止原因 | 类型 | 基准 count / rate | 目标 count / rate | rate 差值 pp |",
        "|---|---|---:|---:|---:|",
    ]
    for name, item in terminations.items():
        lines.append(
            f"| {name} | {item['role']} | {item['baseline']['count']} / {pct(item['baseline']['rate'])} | "
            f"{item['target']['count']} / {pct(item['target']['rate'])} | {item['rate_delta'] * 100:+.4f} |"
        )
    lines += [
        "",
        "## 长度分桶",
        "",
        "| 桶 | motions | 基准 success | 目标 success | 差值 pp |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, _, _ in BUCKETS:
        item = comparison["length_buckets"][name]
        lines.append(f"| {name} | {item['target']['motions']} | {pct(item['baseline']['success_rate'])} | {pct(item['target']['success_rate'])} | {item['success_delta'] * 100:+.4f} |")
    lines += ["", "## 数据来源", "", "| 来源 | motions | 基准 success | 目标 success | 差值 pp |", "|---|---:|---:|---:|---:|"]
    for name, item in comparison["sources"].items():
        lines.append(f"| {name} | {item['target']['motions']} | {pct(item['baseline']['success_rate'])} | {pct(item['target']['success_rate'])} | {item['success_delta'] * 100:+.4f} |")
    lines += ["", "## Tracking errors（per-motion macro mean，越低越好）", "", "| 指标 | 基准 | 目标 | 差值 |", "|---|---:|---:|---:|"]
    for name, item in errors.items():
        lines.append(f"| {name} | {item['baseline']:.6f} | {item['target']:.6f} | {item['delta']:+.6f} |")
    lines += ["", "## 协议与校验", "", "8,192 个唯一 motion、每条 3 trials，共 24,576 episodes；seed 20260818；frame 0；`physics=newton_mjwarp`；七长度桶；两个 checkpoint 均使用目标 run 自带 `params/env.pkl` 与 `params/agent.pkl`。两侧 motions、episodes、attempts、唯一文件数均通过校验，`invalid_robot_state=0`。", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")

    table_path = args.output.with_name("checkpoint_progress.csv")
    termination_columns = [f"termination_{name}_rate" for name in termination_names]
    columns = [
        "checkpoint", "success_rate", "body_pos_failure_rate", "zero_success", "partial_success",
        "all_success", "mean_episode_length", "invalid_robot_state", "motions", "episodes", "seed",
        *termination_columns,
    ]
    rows: dict[str, dict[str, str]] = {}
    if table_path.is_file():
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = {row["checkpoint"]: row for row in reader}
            for field in reader.fieldnames or ():
                if field not in columns:
                    columns.append(field)
    for result in (baseline, target):
        checkpoint = Path(result["checkpoint"]).stem
        row = {
            "checkpoint": checkpoint,
            "success_rate": str(result["macro_success_rate"]),
            "body_pos_failure_rate": str(result["termination_rates"]["body_pos"]),
            "zero_success": str(result["zero_success_motions"]),
            "partial_success": str(result["partial_success_motions"]),
            "all_success": str(result["all_success_motions"]),
            "mean_episode_length": str(result["macro_mean_episode_length"]),
            "invalid_robot_state": str(result["termination_rates"].get("invalid_robot_state", 0)),
            "motions": str(result["motions"]),
            "episodes": str(result["episodes"]),
            "seed": str(args.seed),
        }
        for name, item in termination_summary(result).items():
            row[f"termination_{name}_rate"] = str(item["rate"])
        rows.setdefault(checkpoint, {}).update(row)

    def checkpoint_step(name: str) -> int:
        return int(name.rsplit("_", 1)[-1])

    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for checkpoint in sorted(rows, key=checkpoint_step):
            writer.writerow(rows[checkpoint])
    print(args.output)
    print(args.output.with_suffix(".json"))
    print(table_path)


if __name__ == "__main__":
    main()
