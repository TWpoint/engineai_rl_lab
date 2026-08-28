"""Rebuild and compare AdaptiveSamplerV1 state stored in two checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from engineai_rl_lab.tasks.tracking.mdp.command_v1 import AdaptiveSamplerV1, AdaptiveSamplerV1Cfg


PROGRESS_THRESHOLD = 0.02
DEGRADATION_THRESHOLD = 0.02
MASTERED_THRESHOLD = 0.10
HARD_STALLED_THRESHOLD = 0.30
CRITICAL_STALLED_THRESHOLD = 0.50
EXPOSURE_THRESHOLD = 32.0
MATERIAL_FRACTION_DELTA = 0.001  # 0.10 percentage point


def _cfg(recipe: torch.Tensor) -> AdaptiveSamplerV1Cfg:
    values = recipe.cpu().double().tolist()
    if len(values) not in (11, 19):
        raise ValueError(f"Unsupported adaptive-sampler recipe length: {len(values)}")
    kwargs = dict(
        bin_size=int(values[0]),
        equal_motion_weighting=bool(values[1]),
        coverage_fraction=values[2],
        pre_failure_window=int(values[3]),
        tracking_error_scale=values[4],
        fast_half_life=values[5],
        slow_half_life=values[6],
        uncertainty_exposure=values[7],
        learnability_full_scale=values[8],
        max_learnable_fraction=values[9],
        probability_cap_ratio=values[10],
    )
    if len(values) == 19:
        kwargs.update(
            global_tracking_error_weight=values[11],
            relative_position_error_weight=values[12],
            relative_position_error_scale=values[13],
            relative_orientation_error_weight=values[14],
            relative_orientation_error_scale=values[15],
            local_position_error_weight=values[16],
            local_position_error_scale=values[17],
            deduplicate_failure_events=bool(values[18]),
        )
    return AdaptiveSamplerV1Cfg(**kwargs)


def _load(path: Path) -> tuple[AdaptiveSamplerV1, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("infos", {}).get("motion_adaptive_sampling")
    if not isinstance(state, dict):
        raise ValueError(f"No motion_adaptive_sampling state in {path}")
    required = {
        "version", "recipe", "total_exposure", "difficulty_fast", "difficulty_slow",
        "bin_size", "manifest_fingerprint_words", "motion_lengths", "body_names_utf8",
    }
    if required - state.keys():
        raise ValueError(f"Incomplete sampler state in {path}: {sorted(required - state.keys())}")
    sampler = AdaptiveSamplerV1(state["motion_lengths"], _cfg(state["recipe"]), device="cpu")
    if not sampler.load_state_dict(state):
        raise ValueError(f"Sampler state failed strict recipe/layout restoration: {path}")
    return sampler, state


def _arrays(sampler: AdaptiveSamplerV1) -> dict[str, torch.Tensor]:
    initialized = (sampler.difficulty_fast >= 0) & (sampler.difficulty_slow >= 0)
    difficulty = torch.where(initialized, sampler.difficulty_fast.double().clamp(0, 1), 1.0)
    slow = torch.where(initialized, sampler.difficulty_slow.double().clamp(0, 1), 1.0)
    uncertainty = (1.0 - sampler.total_exposure.double() / sampler.cfg.uncertainty_exposure).clamp(0, 1)
    progress = torch.where(
        initialized,
        ((slow - difficulty).clamp_min(0) / slow.clamp_min(1e-6)).clamp_max(1),
        0.0,
    )
    learnability = torch.maximum(uncertainty, progress)
    hard_support = difficulty > torch.finfo(torch.float64).tiny
    learnable_support = hard_support & (learnability > torch.finfo(torch.float64).tiny)
    coverage_only = ~hard_support
    hard_only = hard_support & ~learnable_support
    exposure = sampler.total_exposure.double()
    mastered = initialized & (exposure >= EXPOSURE_THRESHOLD) & (difficulty <= MASTERED_THRESHOLD) & (
        slow <= MASTERED_THRESHOLD
    )
    relative_progress = (slow - difficulty) / slow.clamp_min(1e-6)
    degraded = initialized & ((difficulty - slow) >= DEGRADATION_THRESHOLD)
    # ``mastered`` remains the independent mastered_proxy metric with its
    # documented definition.  The mutually exclusive classification applies
    # the requested priority, so a bin that is both degraded and below the
    # mastered thresholds belongs to degraded, not mastered_proxy.
    mastered_class = ~degraded & mastered
    uncertain = ~degraded & ~mastered & (exposure < EXPOSURE_THRESHOLD)
    progress_learnable = (
        ~degraded & ~mastered & ~uncertain & (relative_progress >= PROGRESS_THRESHOLD)
    )
    no_progress = relative_progress < PROGRESS_THRESHOLD
    hard_stalled = (
        ~degraded & ~mastered & ~uncertain & ~progress_learnable
        & no_progress & (difficulty > HARD_STALLED_THRESHOLD)
    )
    medium_plateau = (
        ~degraded & ~mastered & ~uncertain & ~progress_learnable & ~hard_stalled
        & no_progress & (difficulty > MASTERED_THRESHOLD)
        & (difficulty <= HARD_STALLED_THRESHOLD)
    )
    easy_plateau = ~(
        degraded | mastered | uncertain | progress_learnable | hard_stalled | medium_plateau
    )
    critical_stalled = (
        (exposure >= EXPOSURE_THRESHOLD) & no_progress
        & (difficulty > CRITICAL_STALLED_THRESHOLD)
    )
    exclusive = {
        "degraded": degraded,
        "mastered_proxy": mastered_class,
        "uncertain": uncertain,
        "progress_learnable": progress_learnable,
        "hard_stalled": hard_stalled,
        "medium_plateau": medium_plateau,
        "easy_plateau": easy_plateau,
    }
    if sum(int(mask.sum().item()) for mask in exclusive.values()) != difficulty.numel():
        raise RuntimeError("New sampler classifications are not exhaustive and exclusive")
    probabilities = sampler.distribution().double()
    entropy = float((-(probabilities * probabilities.clamp_min(1e-15).log()).sum() / math.log(len(probabilities))).item())
    return {
        "initialized": initialized,
        "difficulty": difficulty,
        "slow": slow,
        "uncertainty": uncertainty,
        "progress": progress,
        "learnability": learnability,
        "coverage_only": coverage_only,
        "hard_only": hard_only,
        "learnable": learnable_support,
        "mastered": mastered,
        "exclusive_classification": exclusive,
        "critical_stalled": critical_stalled,
        "probabilities": probabilities,
        "entropy": torch.tensor(entropy),
    }


def _count(mask: torch.Tensor, total: int) -> dict[str, float | int]:
    count = int(mask.sum().item())
    return {"count": count, "fraction": count / total}


def _summary(sampler: AdaptiveSamplerV1, arrays: dict[str, torch.Tensor]) -> dict:
    total = sampler.bin_count
    initialized = arrays["initialized"]
    fast_values = sampler.difficulty_fast[initialized].double()
    slow_values = sampler.difficulty_slow[initialized].double()
    learnable = arrays["learnable"]
    uncertainty = arrays["uncertainty"]
    progress = arrays["progress"]
    uncertainty_only = learnable & (uncertainty > progress)
    progress_only = learnable & (progress > uncertainty)
    tied = learnable & (progress == uncertainty)
    return {
        "bins": total,
        "probability_budget": {
            "coverage": sampler.last_coverage_mass,
            "hard": sampler.last_hard_mass,
            "learnable": sampler.last_learnable_mass,
        },
        "coverage_only": _count(arrays["coverage_only"], total),
        "hard_only": _count(arrays["hard_only"], total),
        "learnable": _count(learnable, total),
        "learnability_source": {
            "uncertainty_dominant": _count(uncertainty_only, total),
            "progress_dominant": _count(progress_only, total),
            "tied": _count(tied, total),
        },
        "initialized": _count(initialized, total),
        "difficulty": {
            "coverage_weighted_fast": sampler.last_difficulty_mean,
            "initialized_fast_mean": float(fast_values.mean().item()) if len(fast_values) else None,
            "initialized_slow_mean": float(slow_values.mean().item()) if len(slow_values) else None,
            "initialized_fast_p50": float(fast_values.median().item()) if len(fast_values) else None,
            "initialized_slow_p50": float(slow_values.median().item()) if len(slow_values) else None,
        },
        "gate": sampler.last_gate,
        "remaining_learnability": sampler.last_remaining_learnability,
        "entropy_normalized": float(arrays["entropy"].item()),
        "mastered_proxy": _count(arrays["mastered"], total),
        "exclusive_classification": {
            name: _count(mask, total)
            for name, mask in arrays["exclusive_classification"].items()
        },
        "critical_stalled": _count(arrays["critical_stalled"], total),
        "classification_thresholds": {
            "progress": PROGRESS_THRESHOLD,
            "degradation": DEGRADATION_THRESHOLD,
            "mastered": MASTERED_THRESHOLD,
            "hard_stalled": HARD_STALLED_THRESHOLD,
            "critical_stalled": CRITICAL_STALLED_THRESHOLD,
            "exposure": EXPOSURE_THRESHOLD,
        },
    }


def _transitions(previous: dict[str, torch.Tensor], current: dict[str, torch.Tensor]) -> dict:
    names = ("coverage_only", "hard_only", "learnable")
    matrix = {
        old: {new: int((previous[old] & current[new]).sum().item()) for new in names}
        for old in names
    }
    new_names = tuple(current["exclusive_classification"])
    new_matrix = {
        old: {
            new: int((previous["exclusive_classification"][old] & current["exclusive_classification"][new]).sum().item())
            for new in new_names
        }
        for old in new_names
    }
    fast_delta = current["difficulty"] - previous["difficulty"]
    slow_delta = current["slow"] - previous["slow"]
    comparable = previous["initialized"] & current["initialized"]
    entered_mastered = ~previous["mastered"] & current["mastered"]
    exited_mastered = previous["mastered"] & ~current["mastered"]
    degraded = comparable & ((fast_delta > 0.02) | (slow_delta > 0.02))
    degraded |= exited_mastered
    # Apply the documented priority literally so every bin belongs to exactly
    # one interpretation category. Entering mastered can coincide with a
    # >0.02 movement in one EMA, and stalled can overlap a degraded hard-only
    # bin unless the higher-priority masks are explicitly excluded.
    gradual = (
        (comparable & (fast_delta < -0.02) & (slow_delta < -0.02))
        | entered_mastered
    ) & ~degraded
    restarted = previous["hard_only"] & current["learnable"] & ~gradual & ~degraded
    stalled = previous["hard_only"] & current["hard_only"] & comparable & (
        fast_delta.abs() <= 0.02
    ) & (slow_delta.abs() <= 0.02) & (current["difficulty"] > 0.10)
    stalled &= ~degraded & ~gradual & ~restarted
    classified = degraded | gradual | restarted | stalled
    other = ~classified
    category_total = sum(
        int(mask.sum().item()) for mask in (gradual, restarted, stalled, degraded, other)
    )
    if category_total != current["difficulty"].numel():
        raise RuntimeError(
            "Difficulty interpretation categories are not exhaustive/exclusive: "
            f"{category_total} != {current['difficulty'].numel()}"
        )
    return {
        "category_matrix": matrix,
        "exclusive_classification_matrix": new_matrix,
        "exclusive_classification_order": list(new_names),
        "hard_to_learnable": matrix["hard_only"]["learnable"],
        "learnable_to_hard": matrix["learnable"]["hard_only"],
        "entered_mastered_proxy": int(entered_mastered.sum().item()),
        "exited_mastered_proxy": int(exited_mastered.sum().item()),
        "difficulty_change": {
            "fast_mean_delta_initialized_common": float(fast_delta[comparable].mean().item()),
            "slow_mean_delta_initialized_common": float(slow_delta[comparable].mean().item()),
        },
        "interpretation": {
            "逐渐掌握": int(gradual.sum().item()),
            "重新开始学习": int(restarted.sum().item()),
            "困难停滞": int(stalled.sum().item()),
            "发生退化": int(degraded.sum().item()),
            "其他": int(other.sum().item()),
            "rule": "阈值0.02；退化优先，其次双EMA下降/进入mastered、hard→learnable、hard持续且变化≤0.02。",
        },
    }


def _judgment(previous: dict, current: dict, transitions: dict) -> tuple[str, str]:
    fast_delta = (
        current["difficulty"]["coverage_weighted_fast"]
        - previous["difficulty"]["coverage_weighted_fast"]
    )
    mastered_delta = current["mastered_proxy"]["count"] - previous["mastered_proxy"]["count"]
    hard_delta = current["exclusive_classification"]["hard_stalled"]["fraction"] - previous["exclusive_classification"]["hard_stalled"]["fraction"]
    critical_delta = current["critical_stalled"]["fraction"] - previous["critical_stalled"]["fraction"]
    mastered_fraction_delta = current["mastered_proxy"]["fraction"] - previous["mastered_proxy"]["fraction"]
    reason = (
        f"coverage-weighted difficulty 变化 {fast_delta:+.4f}，"
        f"mastered_proxy 净变化 {mastered_delta:+d}，hard-stalled 变化 {hard_delta * 100:+.4f} pp，"
        f"critical-stalled 变化 {critical_delta * 100:+.4f} pp。"
    )
    stalled_up = hard_delta >= MATERIAL_FRACTION_DELTA or critical_delta >= MATERIAL_FRACTION_DELTA
    mastered_up = mastered_fraction_delta >= MATERIAL_FRACTION_DELTA
    if mastered_up and stalled_up:
        return "混合", reason + "mastered 与 hard/critical stalled 同时明显上升。"
    if mastered_up and not stalled_up and fast_delta <= 0:
        return "改善", reason + "mastered 上升，hard/critical stalled 未明显上升，difficulty 未上升。"
    if stalled_up and fast_delta > 0:
        return "恶化", reason + "hard/critical stalled 上升且 difficulty 上升。"
    if abs(mastered_fraction_delta) < MATERIAL_FRACTION_DELTA and abs(hard_delta) < MATERIAL_FRACTION_DELTA and abs(critical_delta) < MATERIAL_FRACTION_DELTA and abs(fast_delta) <= 0.001:
        return "持平", reason + "变化均较小。"
    return "混合", reason + "各 sampler 信号未形成一致的单向结论。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress_csv", type=Path)
    args = parser.parse_args()
    previous_sampler, previous_state = _load(args.previous)
    current_sampler, current_state = _load(args.current)
    for key in ("recipe", "bin_size", "manifest_fingerprint_words", "motion_lengths", "body_names_utf8"):
        if not torch.equal(previous_state[key], current_state[key]):
            raise ValueError(f"Sampler layout/recipe mismatch: {key}")
    previous_arrays, current_arrays = _arrays(previous_sampler), _arrays(current_sampler)
    previous_summary = _summary(previous_sampler, previous_arrays)
    current_summary = _summary(current_sampler, current_arrays)
    transitions = _transitions(previous_arrays, current_arrays)
    judgment, reason = _judgment(previous_summary, current_summary, transitions)
    payload = {
        "run": args.current.parent.name,
        "source": "checkpoint infos.motion_adaptive_sampling (direct reconstruction; not inferred from evaluation results)",
        "previous_checkpoint": str(args.previous.resolve()),
        "current_checkpoint": str(args.current.resolve()),
        "recipe": current_state["recipe"].double().tolist(),
        "manifest_fingerprint_words": current_state["manifest_fingerprint_words"].long().tolist(),
        "previous": previous_summary,
        "current": current_summary,
        "transitions": transitions,
        "judgment": judgment,
        "judgment_reason": reason,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pct(value: float) -> str:
        return f"{value * 100:.4f}%"

    lines = [
        f"# Sampler：{args.current.stem} vs {args.previous.stem}", "", "## 判断", "",
        f"**{judgment}**。{reason}", "",
        "数据来源：两个 checkpoint 的 `infos.motion_adaptive_sampling` 直接重建；未从评估结果反推。", "",
        "## 概览", "",
        "| 指标 | 上一 checkpoint | 当前 checkpoint |", "|---|---:|---:|",
    ]
    for key, label in (("coverage", "coverage 概率预算"), ("hard", "hard 概率预算"), ("learnable", "learnable 概率预算")):
        lines.append(f"| {label} | {pct(previous_summary['probability_budget'][key])} | {pct(current_summary['probability_budget'][key])} |")
    for key, label in (("coverage_only", "coverage-only bins"), ("hard_only", "hard-only bins（兼容字段；解释为 no-learnability-support）"), ("learnable", "learnable bins"), ("mastered_proxy", "mastered_proxy")):
        old, new = previous_summary[key], current_summary[key]
        lines.append(f"| {label} | {old['count']:,} ({pct(old['fraction'])}) | {new['count']:,} ({pct(new['fraction'])}) |")
    lines += [
        f"| gate | {previous_summary['gate']:.6f} | {current_summary['gate']:.6f} |",
        f"| remaining learnability | {previous_summary['remaining_learnability']:.6f} | {current_summary['remaining_learnability']:.6f} |",
        f"| normalized entropy | {previous_summary['entropy_normalized']:.6f} | {current_summary['entropy_normalized']:.6f} |",
        f"| coverage-weighted difficulty | {previous_summary['difficulty']['coverage_weighted_fast']:.6f} | {current_summary['difficulty']['coverage_weighted_fast']:.6f} |",
        "", "## Learnability 来源", "", "| 来源 | 上一 checkpoint | 当前 checkpoint |", "|---|---:|---:|",
    ]
    for key, label in (("uncertainty_dominant", "uncertainty 主导"), ("progress_dominant", "progress 主导"), ("tied", "两者相等")):
        old, new = previous_summary["learnability_source"][key], current_summary["learnability_source"][key]
        lines.append(f"| {label} | {old['count']:,} ({pct(old['fraction'])}) | {new['count']:,} ({pct(new['fraction'])}) |")
    lines += [
        "", "## 新互斥 sampler 分类", "",
        "判定优先级固定为 degraded → mastered_proxy → uncertain → progress-learnable → hard-stalled → medium-plateau → easy-plateau。",
        "`hard-only` 仅为历史兼容字段，语义为 no-learnability-support；不得用其总占比单独判断恶化。", "",
        "| 分类 | 上一 checkpoint | 当前 checkpoint |", "|---|---:|---:|",
    ]
    for key in previous_summary["exclusive_classification"]:
        old = previous_summary["exclusive_classification"][key]
        new = current_summary["exclusive_classification"][key]
        lines.append(f"| {key} | {old['count']:,} ({pct(old['fraction'])}) | {new['count']:,} ({pct(new['fraction'])}) |")
    old_critical, new_critical = previous_summary["critical_stalled"], current_summary["critical_stalled"]
    lines.append(f"| critical-stalled（hard-stalled 子指标） | {old_critical['count']:,} ({pct(old_critical['fraction'])}) | {new_critical['count']:,} ({pct(new_critical['fraction'])}) |")
    lines += ["", "## 逐 bin 迁移", "", "| From \\ To | coverage-only | hard-only | learnable |", "|---|---:|---:|---:|"]
    for old in ("coverage_only", "hard_only", "learnable"):
        row = transitions["category_matrix"][old]
        lines.append(f"| {old} | {row['coverage_only']:,} | {row['hard_only']:,} | {row['learnable']:,} |")
    new_names = transitions["exclusive_classification_order"]
    lines += ["", "## 新互斥分类完整迁移矩阵", "", "| From \\ To | " + " | ".join(new_names) + " |", "|---|" + "---:|" * len(new_names)]
    for old in new_names:
        row = transitions["exclusive_classification_matrix"][old]
        lines.append("| " + old + " | " + " | ".join(f"{row[new]:,}" for new in new_names) + " |")
    lines += [
        "", f"- hard→learnable：{transitions['hard_to_learnable']:,}",
        f"- learnable→hard：{transitions['learnable_to_hard']:,}",
        f"- 进入 mastered_proxy：{transitions['entered_mastered_proxy']:,}",
        f"- 退出 mastered_proxy：{transitions['exited_mastered_proxy']:,}",
        "", "## 训练状态解释", "",
    ]
    for name in ("逐渐掌握", "重新开始学习", "困难停滞", "发生退化", "其他"):
        lines.append(f"- {name}：{transitions['interpretation'][name]:,} bins")
    lines += ["", f"口径：{transitions['interpretation']['rule']}", "", "固定阈值：progress/degradation=0.02、mastered=0.10、hard-stalled=0.30、critical-stalled=0.50、exposure=32。", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")

    if args.progress_csv:
        rows = []
        if args.progress_csv.is_file():
            with args.progress_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        by_checkpoint = {row["checkpoint"]: row for row in rows}
        for checkpoint, summary in ((args.previous.stem, previous_summary), (args.current.stem, current_summary)):
            row = by_checkpoint.setdefault(checkpoint, {"checkpoint": checkpoint})
            row.update({
                "sampler_coverage_budget": summary["probability_budget"]["coverage"],
                "sampler_hard_budget": summary["probability_budget"]["hard"],
                "sampler_learnable_budget": summary["probability_budget"]["learnable"],
                "sampler_coverage_only_bins": summary["coverage_only"]["count"],
                "sampler_hard_only_bins": summary["hard_only"]["count"],
                "sampler_learnable_bins": summary["learnable"]["count"],
                "sampler_gate": summary["gate"], "sampler_entropy": summary["entropy_normalized"],
                "sampler_difficulty": summary["difficulty"]["coverage_weighted_fast"],
                "sampler_mastered_proxy": summary["mastered_proxy"]["count"],
                "sampler_mastered_proxy_fraction": summary["mastered_proxy"]["fraction"],
                "sampler_degraded_bins": summary["exclusive_classification"]["degraded"]["count"],
                "sampler_degraded_fraction": summary["exclusive_classification"]["degraded"]["fraction"],
                "sampler_uncertain_bins": summary["exclusive_classification"]["uncertain"]["count"],
                "sampler_uncertain_fraction": summary["exclusive_classification"]["uncertain"]["fraction"],
                "sampler_progress_learnable_bins": summary["exclusive_classification"]["progress_learnable"]["count"],
                "sampler_progress_learnable_fraction": summary["exclusive_classification"]["progress_learnable"]["fraction"],
                "sampler_hard_stalled_bins": summary["exclusive_classification"]["hard_stalled"]["count"],
                "sampler_hard_stalled_fraction": summary["exclusive_classification"]["hard_stalled"]["fraction"],
                "sampler_medium_plateau_bins": summary["exclusive_classification"]["medium_plateau"]["count"],
                "sampler_medium_plateau_fraction": summary["exclusive_classification"]["medium_plateau"]["fraction"],
                "sampler_easy_plateau_bins": summary["exclusive_classification"]["easy_plateau"]["count"],
                "sampler_easy_plateau_fraction": summary["exclusive_classification"]["easy_plateau"]["fraction"],
                "sampler_critical_stalled_bins": summary["critical_stalled"]["count"],
                "sampler_critical_stalled_fraction": summary["critical_stalled"]["fraction"],
            })
        fields = list(rows[0].keys()) if rows else ["checkpoint"]
        for row in by_checkpoint.values():
            for key in row:
                if key not in fields:
                    fields.append(key)
        with args.progress_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for checkpoint in sorted(by_checkpoint, key=lambda value: int(value.rsplit("_", 1)[-1])):
                writer.writerow(by_checkpoint[checkpoint])
    print(args.output)
    print(args.output.with_suffix(".json"))


if __name__ == "__main__":
    main()
