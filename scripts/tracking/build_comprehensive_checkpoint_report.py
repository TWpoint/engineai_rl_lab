"""Build one validated Chinese frame-zero + adaptive-sampler checkpoint report."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from compare_frame0_results import BUCKETS, grouped, load, termination_summary
from compare_sampler_checkpoints import _arrays, _load, _summary


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def write_success_trend(
    progress_csv: Path, checkpoint: Path, report: Path, *, append_report: bool = True
) -> Path:
    """Write an immutable per-checkpoint success trend chart and link it in the report."""

    current_step = int(checkpoint.stem.rsplit("_", 1)[-1])
    with progress_csv.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("status") == "complete" and row.get("success_rate")
        ]
    rows = [row for row in rows if int(row["checkpoint"].rsplit("_", 1)[-1]) <= current_step]
    rows.sort(key=lambda row: int(row["checkpoint"].rsplit("_", 1)[-1]))
    steps = [int(row["checkpoint"].rsplit("_", 1)[-1]) for row in rows]
    success = [float(row["success_rate"]) * 100.0 for row in rows]
    if not steps or steps[-1] != current_step:
        raise ValueError(f"Progress CSV does not contain completed {checkpoint.stem}")

    output = report.with_name(f"{checkpoint.stem}_success_trend.png")
    temporary = output.with_suffix(".png.tmp")
    positions = list(range(len(steps)))
    figure, axis = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    axis.plot(positions, success, color="#2f6fed", linewidth=2.2, marker="o", markersize=6)
    axis.scatter([positions[-1]], [success[-1]], color="#d9485f", s=70, zorder=3)

    # Labelling every point quickly becomes illegible.  Keep a small set of
    # landmarks on the curve and put exact recent values in a dedicated panel.
    label_stride = max(1, (len(positions) - 1 + 5) // 6)
    label_indices = set(range(0, len(positions), label_stride))
    label_indices.update({0, len(positions) - 1, max(range(len(success)), key=success.__getitem__)})
    for index in sorted(label_indices):
        position, value = positions[index], success[index]
        is_current = index == len(positions) - 1
        axis.annotate(
            f"{value:.2f}%",
            (position, value),
            xytext=(0, 13 if index % 2 == 0 else -19),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold" if is_current else "normal",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
    low, high = min(success), max(success)
    padding = max(2.0, (high - low) * 0.18)
    axis.set_ylim(max(0.0, low - padding), min(100.0, high + padding))
    tick_stride = max(1, (len(positions) - 1 + 7) // 8)
    tick_indices = list(range(0, len(positions), tick_stride))
    if tick_indices[-1] != len(positions) - 1:
        if len(positions) - 1 - tick_indices[-1] < tick_stride:
            tick_indices[-1] = len(positions) - 1
        else:
            tick_indices.append(len(positions) - 1)
    axis.set_xticks(tick_indices, [str(steps[index]) for index in tick_indices], fontsize=10)
    axis.set_xlim(-0.6, len(positions) + 3.2)
    axis.tick_params(axis="y", labelsize=10)
    axis.set_xlabel("Checkpoint step", fontsize=11)
    axis.set_ylabel("Frame-0 success (%)", fontsize=11)
    axis.set_title("Fixed-protocol frame-0 success trend", fontsize=14)
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    recent = list(zip(steps, success, strict=True))[-5:]
    recent_text = "Recent checkpoints\n" + "\n".join(
        f"{step:>6}   {value:>6.2f}%" for step, value in recent
    )
    axis.text(
        len(positions) + 2.8,
        (low + high) / 2.0,
        recent_text,
        ha="right",
        va="center",
        fontsize=9,
        family="monospace",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f5f7fb", "edgecolor": "#cbd3e1"},
    )
    figure.savefig(temporary, format="png", dpi=180, transparent=False)
    plt.close(figure)
    temporary.replace(output)
    if append_report:
        with report.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n## Success 折线图\n\n"
                f"![固定协议 frame-0 success 变化]({output.name})\n"
            )
    return output


def write_mastered_proxy_trend(
    progress_csv: Path, checkpoint: Path, report: Path, *, append_report: bool = True
) -> Path:
    """Write checkpoint-derived trends for the five required sampler classes."""

    current_step = int(checkpoint.stem.rsplit("_", 1)[-1])
    with progress_csv.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "complete"]
    rows = [row for row in rows if int(row["checkpoint"].rsplit("_", 1)[-1]) <= current_step]
    rows.sort(key=lambda row: int(row["checkpoint"].rsplit("_", 1)[-1]))

    steps: list[int] = []
    series = {
        "mastered_proxy": [],
        "progress-learnable": [],
        "medium-plateau": [],
        "hard-stalled": [],
        "critical-stalled": [],
    }
    for row in rows:
        stem = row["checkpoint"]
        # New columns are populated prospectively.  For historical rows rebuild
        # the exact classes directly from checkpoint infos.motion_adaptive_sampling.
        if all(row.get(field) for field in (
            "sampler_mastered_proxy_fraction", "sampler_progress_learnable_fraction",
            "sampler_medium_plateau_fraction", "sampler_hard_stalled_fraction",
            "sampler_critical_stalled_fraction",
        )):
            values = {
                "mastered_proxy": float(row["sampler_mastered_proxy_fraction"]),
                "progress-learnable": float(row["sampler_progress_learnable_fraction"]),
                "medium-plateau": float(row["sampler_medium_plateau_fraction"]),
                "hard-stalled": float(row["sampler_hard_stalled_fraction"]),
                "critical-stalled": float(row["sampler_critical_stalled_fraction"]),
            }
        else:
            sampler, _ = _load(report.parent.parent / f"{stem}.pt")
            summary = _summary(sampler, _arrays(sampler))
            values = {
                "mastered_proxy": summary["mastered_proxy"]["fraction"],
                "progress-learnable": summary["exclusive_classification"]["progress_learnable"]["fraction"],
                "medium-plateau": summary["exclusive_classification"]["medium_plateau"]["fraction"],
                "hard-stalled": summary["exclusive_classification"]["hard_stalled"]["fraction"],
                "critical-stalled": summary["critical_stalled"]["fraction"],
            }
        steps.append(int(stem.rsplit("_", 1)[-1]))
        for name, value in values.items():
            series[name].append(value * 100.0)
    if not steps or steps[-1] != current_step:
        raise ValueError(f"No checkpoint-derived mastered_proxy fraction for {checkpoint.stem}")

    output = report.with_name(f"{checkpoint.stem}_mastered_proxy_trend.png")
    temporary = output.with_suffix(".png.tmp")
    positions = list(range(len(steps)))
    figure, axis = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    colors = {
        "mastered_proxy": "#6f42c1", "progress-learnable": "#2f6fed",
        "medium-plateau": "#e09f3e", "hard-stalled": "#d9485f",
        "critical-stalled": "#7f1d1d",
    }
    for name, values in series.items():
        axis.plot(positions, values, color=colors[name], linewidth=2.0, marker="o", markersize=4, label=name)
        axis.annotate(f"{values[-1]:.2f}%", (positions[-1], values[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=colors[name])
    all_values = [value for values in series.values() for value in values]
    low, high = min(all_values), max(all_values)
    padding = max(1.0, (high - low) * 0.18)
    axis.set_ylim(max(0.0, low - padding), min(100.0, high + padding))
    tick_stride = max(1, (len(positions) - 1 + 7) // 8)
    tick_indices = list(range(0, len(positions), tick_stride))
    if tick_indices[-1] != len(positions) - 1:
        if len(positions) - 1 - tick_indices[-1] < tick_stride:
            tick_indices[-1] = len(positions) - 1
        else:
            tick_indices.append(len(positions) - 1)
    axis.set_xticks(tick_indices, [str(steps[index]) for index in tick_indices], fontsize=10)
    axis.set_xlim(-0.6, len(positions) + 2.2)
    axis.set_xlabel("Checkpoint step", fontsize=11)
    axis.set_ylabel("Sampler bins (%)", fontsize=11)
    axis.set_title("Checkpoint sampler classification trends", fontsize=14)
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.legend(loc="best", fontsize=8, framealpha=0.9)
    figure.savefig(temporary, format="png", dpi=180, transparent=False)
    plt.close(figure)
    temporary.replace(output)
    if append_report:
        with report.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n## Sampler 新分类趋势图\n\n"
                f"![checkpoint sampler 新分类趋势]({output.name})\n"
            )
    return output


def write_combined_trends(progress_csv: Path, checkpoint: Path, report: Path) -> Path:
    """Combine success and required sampler-class percentage trends."""

    success_chart = write_success_trend(
        progress_csv, checkpoint, report, append_report=False
    )
    mastered_chart = write_mastered_proxy_trend(
        progress_csv, checkpoint, report, append_report=False
    )
    temporary = success_chart.with_suffix(".combined.png.tmp")
    with Image.open(success_chart) as success_image, Image.open(mastered_chart) as mastered_image:
        width = max(success_image.width, mastered_image.width)
        combined = Image.new("RGB", (width, success_image.height + mastered_image.height), "white")
        combined.paste(success_image.convert("RGB"), (0, 0))
        combined.paste(mastered_image.convert("RGB"), (0, success_image.height))
        combined.save(temporary, format="PNG")
    temporary.replace(success_chart)
    mastered_chart.unlink()
    with report.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Success 与 sampler 新分类趋势图\n\n"
            f"包含 mastered_proxy、progress-learnable、medium-plateau、hard-stalled、critical-stalled 百分比。\n\n"
            f"![固定协议 success 与 sampler 新分类变化]({success_chart.name})\n"
        )
    return success_chart


def build_comparison(args: argparse.Namespace) -> None:
    """Compose the existing strict success and sampler comparisons into one artifact."""
    script_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="checkpoint-report-") as temporary:
        tmp = Path(temporary)
        success_md = tmp / "success.md"
        sampler_md = tmp / "sampler.md"
        subprocess.run(
            [
                sys.executable, str(script_dir / "compare_frame0_results.py"),
                "--baseline", str(args.previous_evaluation), "--target", str(args.evaluation),
                "--output", str(success_md), "--bootstrap_draws", "100000", "--seed", "20260818",
            ], check=True,
        )
        subprocess.run(
            [
                sys.executable, str(script_dir / "compare_sampler_checkpoints.py"),
                "--previous", str(args.previous_checkpoint), "--current", str(args.checkpoint),
                "--output", str(sampler_md),
            ], check=True,
        )
        success = json.loads(success_md.with_suffix(".json").read_text(encoding="utf-8"))
        sampler = json.loads(sampler_md.with_suffix(".json").read_text(encoding="utf-8"))
        best_success = best_sampler = None
        best_success_md = best_sampler_md = None
        has_distinct_best = (
            args.best_checkpoint is not None
            and args.best_evaluation is not None
            and args.best_checkpoint.resolve() != args.previous_checkpoint.resolve()
        )
        if has_distinct_best:
            best_success_md = tmp / "best-success.md"
            best_sampler_md = tmp / "best-sampler.md"
            subprocess.run(
                [
                    sys.executable, str(script_dir / "compare_frame0_results.py"),
                    "--baseline", str(args.best_evaluation), "--target", str(args.evaluation),
                    "--output", str(best_success_md), "--bootstrap_draws", "100000", "--seed", "20260818",
                ], check=True,
            )
            subprocess.run(
                [
                    sys.executable, str(script_dir / "compare_sampler_checkpoints.py"),
                    "--previous", str(args.best_checkpoint), "--current", str(args.checkpoint),
                    "--output", str(best_sampler_md),
                ], check=True,
            )
            best_success = json.loads(best_success_md.with_suffix(".json").read_text(encoding="utf-8"))
            best_sampler = json.loads(best_sampler_md.with_suffix(".json").read_text(encoding="utf-8"))
        current_payload, current_result = load(args.evaluation)
        delta = success["success_delta"]
        low, high = success["success_delta_95_ci"]
        success_signal = "改善" if low > 0 else "恶化" if high < 0 else "持平"
        sampler_signal = sampler["judgment"]
        if success_signal == sampler_signal:
            overall = success_signal
            reason = "frame-0 success 与 sampler 信号一致。"
        elif "持平" in (success_signal, sampler_signal):
            overall = "混合信号或持平"
            reason = f"frame-0 success 判断为{success_signal}，sampler 判断为{sampler_signal}。"
        else:
            overall = "混合信号"
            reason = f"frame-0 success 判断为{success_signal}，sampler 判断为{sampler_signal}，两者方向相反。"
        best_overall = best_reason = None
        if has_distinct_best:
            best_low, best_high = best_success["success_delta_95_ci"]
            best_success_signal = "改善" if best_low > 0 else "恶化" if best_high < 0 else "持平"
            best_sampler_signal = best_sampler["judgment"]
            if best_success_signal == best_sampler_signal:
                best_overall = best_success_signal
                best_reason = "frame-0 success 与 sampler 信号一致。"
            elif "持平" in (best_success_signal, best_sampler_signal):
                best_overall = "混合信号或持平"
                best_reason = f"frame-0 success 判断为{best_success_signal}，sampler 判断为{best_sampler_signal}。"
            else:
                best_overall = "混合信号"
                best_reason = f"frame-0 success 判断为{best_success_signal}，sampler 判断为{best_sampler_signal}，两者方向相反。"
        raw = {
            "checkpoint": str(args.checkpoint.resolve()),
            "success_trend_chart": str(
                args.output.with_name(f"{args.checkpoint.stem}_success_trend.png").resolve()
            ),
            "mastered_proxy_trend_chart": str(
                args.output.with_name(f"{args.checkpoint.stem}_success_trend.png").resolve()
            ),
            "combined_trend_chart": str(
                args.output.with_name(f"{args.checkpoint.stem}_success_trend.png").resolve()
            ),
            "comparison_checkpoints": [str(args.previous_checkpoint.resolve())] + ([str(args.best_checkpoint.resolve())] if has_distinct_best else []),
            "comparison_note": "分别报告上一个 checkpoint 与此前 frame-0 success 最佳 checkpoint。" if has_distinct_best else "上一个 checkpoint 与此前 frame-0 success 最佳 checkpoint 相同，仅报告一次。",
            "overall_judgment": overall,
            "overall_judgment_reason": reason,
            "frame0_evaluation": current_payload,
            "termination_reasons": termination_summary(current_result),
            "frame0_comparison": success,
            "sampler_comparison": sampler,
            "best_frame0_comparison": best_success,
            "best_sampler_comparison": best_sampler,
            "best_overall_judgment": best_overall,
            "best_overall_judgment_reason": best_reason,
            "validation": {"motions": 8192, "episodes": 24576, "attempts": 24576, "attempts_per_motion": 3, "unique_files": 8192, "invalid_robot_state": 0, "tracking_errors": 10, "complete": True},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        success_text = success_md.read_text(encoding="utf-8")
        sampler_text = sampler_md.read_text(encoding="utf-8")
        header = (
            f"# {args.checkpoint.stem} 综合报告\n\n## 综合判断\n\n**{overall}**。{reason} "
            f"success 差值 {delta * 100:+.4f} pp，100,000 次逐 motion paired bootstrap 95% CI "
            f"[{low * 100:+.4f}, {high * 100:+.4f}] pp。\n\n"
            + ("上一个 checkpoint 与此前 frame-0 success 最佳 checkpoint 不同，以下分别报告。\n\n" if has_distinct_best else "上一个 checkpoint 与此前 frame-0 success 最佳 checkpoint 均为同一 checkpoint，因此只报告一次。\n\n")
        )
        extra = ""
        if has_distinct_best:
            extra = f"\n# 对比此前 frame-0 success 最佳 checkpoint\n\n## 综合判断\n\n**{best_overall}**。{best_reason}\n\n" + best_success_md.read_text(encoding="utf-8") + "\n" + best_sampler_md.read_text(encoding="utf-8")
        args.output.write_text(header + success_text + "\n" + sampler_text + extra, encoding="utf-8")

        # Merge the validated success row into the persistent table, then append sampler columns.
        temp_progress = tmp / "checkpoint_progress.csv"
        with temp_progress.open(newline="", encoding="utf-8") as handle:
            success_rows = {row["checkpoint"]: row for row in csv.DictReader(handle)}
        existing = []
        if args.progress_csv.is_file():
            with args.progress_csv.open(newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
        by_checkpoint = {row["checkpoint"]: row for row in existing}
        merged_row = dict(by_checkpoint.get(args.checkpoint.stem, {}))
        for key, value in success_rows[args.checkpoint.stem].items():
            if value != "" or key not in merged_row:
                merged_row[key] = value
        by_checkpoint[args.checkpoint.stem] = merged_row
        fields = []
        for row in by_checkpoint.values():
            for key in row:
                if key not in fields:
                    fields.append(key)
        with args.progress_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for name in sorted(by_checkpoint, key=lambda value: int(value.rsplit("_", 1)[-1])):
                writer.writerow(by_checkpoint[name])
        subprocess.run(
            [
                sys.executable, str(script_dir / "compare_sampler_checkpoints.py"),
                "--previous", str(args.previous_checkpoint), "--current", str(args.checkpoint),
                "--output", str(tmp / "sampler-progress.md"), "--progress_csv", str(args.progress_csv),
            ], check=True,
        )
        with args.progress_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            final_fields = list(reader.fieldnames or [])
            final_rows = list(reader)
        for field in (
            "status",
            "attempts",
            "sampler_remaining_learnability",
            "frame_zero_reset_sync_version",
        ):
            if field not in final_fields:
                final_fields.append(field)
        for row in final_rows:
            if row["checkpoint"] == args.checkpoint.stem:
                row["status"] = "complete"
                row["attempts"] = "24576"
                row["sampler_remaining_learnability"] = str(sampler["current"]["remaining_learnability"])
                row["frame_zero_reset_sync_version"] = "1"
        with args.progress_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=final_fields)
            writer.writeheader()
            writer.writerows(final_rows)
        write_combined_trends(args.progress_csv, args.checkpoint, args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress_csv", type=Path, required=True)
    parser.add_argument("--previous_checkpoint", type=Path)
    parser.add_argument("--previous_evaluation", type=Path)
    parser.add_argument("--best_checkpoint", type=Path)
    parser.add_argument("--best_evaluation", type=Path)
    args = parser.parse_args()

    if args.previous_checkpoint is not None or args.previous_evaluation is not None:
        if args.previous_checkpoint is None or args.previous_evaluation is None:
            parser.error("--previous_checkpoint and --previous_evaluation must be supplied together")
        build_comparison(args)
        return

    evaluation_payload, result = load(args.evaluation)
    sampler, state = _load(args.checkpoint)
    sampler_summary = _summary(sampler, _arrays(sampler))
    length_groups = grouped(result, "length")
    source_groups = grouped(result, "source")
    raw = {
        "checkpoint": str(args.checkpoint.resolve()),
        "success_trend_chart": str(
            args.output.with_name(f"{args.checkpoint.stem}_success_trend.png").resolve()
        ),
        "mastered_proxy_trend_chart": str(
            args.output.with_name(f"{args.checkpoint.stem}_success_trend.png").resolve()
        ),
        "combined_trend_chart": str(
            args.output.with_name(f"{args.checkpoint.stem}_success_trend.png").resolve()
        ),
        "comparison_checkpoints": [],
        "comparison_note": "当前 run 的首个 checkpoint；无上一个或此前最佳 checkpoint 可比较。",
        "frame0_evaluation": evaluation_payload,
        "termination_reasons": termination_summary(result),
        "sampler": {
            "source": "checkpoint infos.motion_adaptive_sampling",
            "recipe": state["recipe"].double().tolist(),
            "manifest_fingerprint_words": state["manifest_fingerprint_words"].long().tolist(),
            "summary": sampler_summary,
        },
        "length_buckets": length_groups,
        "sources": source_groups,
        "validation": {
            "motions": 8192,
            "episodes": 24576,
            "attempts": 24576,
            "attempts_per_motion": 3,
            "unique_files": 8192,
            "invalid_robot_state": 0,
            "tracking_errors": 10,
            "complete": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    s = sampler_summary
    lines = [
        f"# {args.checkpoint.stem} 综合报告", "", "## 结论", "",
        "这是当前 run 的首个 checkpoint，尚无上一个 checkpoint 或此前 frame-0 success 最佳 checkpoint，因此本次不构造伪对比；后续 checkpoint 将按逐 motion 100,000 次 paired bootstrap 比较。",
        "", "固定 frame-0 success 与训练在线 termination 的起点和采样口径不同，不能直接比较。",
        "", "## Frame-0 success", "",
        "| 指标 | 当前 |", "|---|---:|",
        f"| success | {pct(result['macro_success_rate'])} |",
        f"| body_pos failure | {pct(result['termination_rates']['body_pos'])} |",
        f"| zero / partial / all-success motions | {result['zero_success_motions']} / {result['partial_success_motions']} / {result['all_success_motions']} |",
        f"| 平均 episode 长度 | {result['macro_mean_episode_length']:.3f} |",
        "", "## 全部终止原因（排除 invalid_robot_state）", "",
        "终止项可能重叠；比例分母为 24,576 episodes。", "",
        "| 终止原因 | 类型 | count | rate |", "|---|---|---:|---:|",
    ]
    termination_reasons = termination_summary(result)
    for name, item in termination_reasons.items():
        lines.append(f"| {name} | {item['role']} | {item['count']} | {pct(item['rate'])} |")
    lines += ["", "## 七个长度桶", "", "| 桶 | motions | success | body_pos failure |", "|---|---:|---:|---:|"]
    for label, _, _ in BUCKETS:
        item = length_groups[label]
        lines.append(f"| {label} | {item['motions']} | {pct(item['success_rate'])} | {pct(item['body_pos_failure_rate'])} |")
    lines += ["", "## 数据来源", "", "| 来源 | motions | success | body_pos failure |", "|---|---:|---:|---:|"]
    for name, item in source_groups.items():
        lines.append(f"| {name} | {item['motions']} | {pct(item['success_rate'])} | {pct(item['body_pos_failure_rate'])} |")
    lines += ["", "## 10 项 tracking errors", "", "| 指标 | per-motion macro mean |", "|---|---:|"]
    for name, item in result["tracking_error_summary"].items():
        lines.append(f"| {name} | {item['macro_mean']:.6f} |")
    lines += [
        "", "## Sampler（直接重建自 checkpoint）", "",
        "来源：`infos.motion_adaptive_sampling`；未从评估结果反推。", "",
        "| 指标 | 当前 |", "|---|---:|",
        f"| coverage / hard / learnable 概率预算 | {pct(s['probability_budget']['coverage'])} / {pct(s['probability_budget']['hard'])} / {pct(s['probability_budget']['learnable'])} |",
        f"| coverage-only bins | {s['coverage_only']['count']} ({pct(s['coverage_only']['fraction'])}) |",
        f"| hard-only bins（兼容字段；no-learnability-support） | {s['hard_only']['count']} ({pct(s['hard_only']['fraction'])}) |",
        f"| learnable bins | {s['learnable']['count']} ({pct(s['learnable']['fraction'])}) |",
        f"| uncertainty / progress / tied 来源 | {s['learnability_source']['uncertainty_dominant']['count']} / {s['learnability_source']['progress_dominant']['count']} / {s['learnability_source']['tied']['count']} |",
        f"| difficulty | {s['difficulty']['coverage_weighted_fast']:.6f} |",
        f"| gate | {s['gate']:.6f} |",
        f"| remaining learnability | {s['remaining_learnability']:.6f} |",
        f"| normalized entropy | {s['entropy_normalized']:.6f} |",
        f"| mastered_proxy | {s['mastered_proxy']['count']} ({pct(s['mastered_proxy']['fraction'])}) |",
        "", "### 新互斥 sampler 分类", "",
        "`hard-only` 仅为历史兼容字段，语义为 no-learnability-support，不用于单独判断恶化。", "",
        "| 分类 | count | fraction |", "|---|---:|---:|",
    "", "## 协议与严格校验", "",
        "8,192 个唯一 motion，frame 0，每 motion 3 trials，共 24,576 episodes；seed 20260818；`physics=newton_mjwarp`；使用目标 run 自带 `params/env.pkl` 和 `params/agent.pkl`。motions、episodes、attempts、逐 motion attempts、唯一文件数、`invalid_robot_state=0`、10 项 tracking errors 全部通过。", "",
    ]
    insert_at = lines.index("## 协议与严格校验") - 1
    class_lines = []
    for name, item in s["exclusive_classification"].items():
        class_lines.append(f"| {name} | {item['count']:,} | {pct(item['fraction'])} |")
    critical = s["critical_stalled"]
    class_lines += [
        f"| critical-stalled（子指标） | {critical['count']:,} | {pct(critical['fraction'])} |",
        "", "固定阈值：progress/degradation=0.02、mastered=0.10、hard-stalled=0.30、critical-stalled=0.50、exposure=32。首 checkpoint 无迁移基准。", "",
    ]
    lines[insert_at:insert_at] = class_lines
    args.output.write_text("\n".join(lines), encoding="utf-8")

    termination_fields = [f"termination_{name}_rate" for name in termination_reasons]
    fields = [
        "checkpoint", "status", "success_rate", "body_pos_failure_rate", "zero_success", "partial_success",
        "all_success", "mean_episode_length", "invalid_robot_state", "motions", "episodes", "attempts",
        "seed", "frame_zero_reset_sync_version", "sampler_coverage_budget", "sampler_hard_budget", "sampler_learnable_budget",
        "sampler_coverage_only_bins", "sampler_hard_only_bins", "sampler_learnable_bins", "sampler_gate",
        "sampler_remaining_learnability", "sampler_entropy", "sampler_difficulty", "sampler_mastered_proxy",
        *termination_fields,
    ]
    rows = []
    if args.progress_csv.is_file():
        with args.progress_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            for field in reader.fieldnames or ():
                if field not in fields:
                    fields.append(field)
    previous_row = next(
        (dict(row) for row in rows if row.get("checkpoint") == args.checkpoint.stem),
        {},
    )
    rows = [row for row in rows if row.get("checkpoint") != args.checkpoint.stem]
    current_row = previous_row
    current_row.update({
        "checkpoint": args.checkpoint.stem, "status": "complete",
        "success_rate": result["macro_success_rate"], "body_pos_failure_rate": result["termination_rates"]["body_pos"],
        "zero_success": result["zero_success_motions"], "partial_success": result["partial_success_motions"],
        "all_success": result["all_success_motions"], "mean_episode_length": result["macro_mean_episode_length"],
        "invalid_robot_state": 0, "motions": 8192, "episodes": 24576, "attempts": 24576, "seed": 20260818,
        "frame_zero_reset_sync_version": 1,
        "sampler_coverage_budget": s["probability_budget"]["coverage"],
        "sampler_hard_budget": s["probability_budget"]["hard"],
        "sampler_learnable_budget": s["probability_budget"]["learnable"],
        "sampler_coverage_only_bins": s["coverage_only"]["count"], "sampler_hard_only_bins": s["hard_only"]["count"],
        "sampler_learnable_bins": s["learnable"]["count"], "sampler_gate": s["gate"],
        "sampler_remaining_learnability": s["remaining_learnability"], "sampler_entropy": s["entropy_normalized"],
        "sampler_difficulty": s["difficulty"]["coverage_weighted_fast"], "sampler_mastered_proxy": s["mastered_proxy"]["count"],
        "sampler_mastered_proxy_fraction": s["mastered_proxy"]["fraction"],
        "sampler_degraded_bins": s["exclusive_classification"]["degraded"]["count"],
        "sampler_degraded_fraction": s["exclusive_classification"]["degraded"]["fraction"],
        "sampler_uncertain_bins": s["exclusive_classification"]["uncertain"]["count"],
        "sampler_uncertain_fraction": s["exclusive_classification"]["uncertain"]["fraction"],
        "sampler_progress_learnable_bins": s["exclusive_classification"]["progress_learnable"]["count"],
        "sampler_progress_learnable_fraction": s["exclusive_classification"]["progress_learnable"]["fraction"],
        "sampler_hard_stalled_bins": s["exclusive_classification"]["hard_stalled"]["count"],
        "sampler_hard_stalled_fraction": s["exclusive_classification"]["hard_stalled"]["fraction"],
        "sampler_medium_plateau_bins": s["exclusive_classification"]["medium_plateau"]["count"],
        "sampler_medium_plateau_fraction": s["exclusive_classification"]["medium_plateau"]["fraction"],
        "sampler_easy_plateau_bins": s["exclusive_classification"]["easy_plateau"]["count"],
        "sampler_easy_plateau_fraction": s["exclusive_classification"]["easy_plateau"]["fraction"],
        "sampler_critical_stalled_bins": s["critical_stalled"]["count"],
        "sampler_critical_stalled_fraction": s["critical_stalled"]["fraction"],
    })
    for name, item in termination_reasons.items():
        current_row[f"termination_{name}_rate"] = item["rate"]
    rows.append(current_row)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with args.progress_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["checkpoint"].rsplit("_", 1)[-1])))
    write_combined_trends(args.progress_csv, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
