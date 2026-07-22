#!/usr/bin/env python3
"""Offline threshold, confidence-interval, and multi-VLM Joint analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from joint_grounding_analysis import (
    ROOT,
    REPORT_DIR,
    external_semantic_rows,
    sid_explain_candidate_rows,
    sid_fa_candidate_rows,
)
from stratified_statistics import (
    core_strata_by_id,
    design_metadata,
    mover_wilson_interval,
    scale_strata_by_id,
    stratum_counts,
    validate_expected_strata,
    xfer_stratum,
)

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
XFER_SELECTION_FILE = ROOT / "experiments/runs/external_semantic_eval_rows.jsonl"
MODEL_FILES = {
    "Qwen2.5-VL-3B": ROOT / "experiments/runs/sid_fa200_candidate_top3_qwen25vl_predictions.jsonl",
    "Qwen2.5-VL-7B": ROOT / "experiments/runs/sid_fa200_candidate_top3_ollama7b_predictions.jsonl",
    "MiniCPM-V 4.5": ROOT / "experiments/runs/sid_fa200_candidate_top3_minicpm45_predictions.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def proportion(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float:
    return sum(1 for row in rows if predicate(row)) / len(rows) if rows else 0.0


def benchmark_strata(
    dataset: str,
    rows: list[dict[str, Any]],
    core_map: dict[str, str],
    scale_map: dict[str, str],
    xfer_map: dict[str, str],
) -> list[str]:
    if dataset == "JSG-Core":
        strata = [core_map[str(row["img_id"])] for row in rows]
    elif dataset == "JSG-Scale":
        strata = [scale_map[str(row["img_id"])] for row in rows]
    elif dataset == "JSG-Xfer":
        strata = [xfer_map[str(row["img_id"])] for row in rows]
    else:
        raise ValueError(f"Unknown benchmark dataset: {dataset}")
    validate_expected_strata(dataset, strata)
    return strata


def artifact_ps(row: dict[str, Any]) -> bool:
    return bool(row.get("artifact_primary_or_secondary_correct", row.get("artifact_correct")))


def joint_predicate(threshold: float, *, ps: bool = False) -> Callable[[dict[str, Any]], bool]:
    return lambda row: bool(
        row.get("prediction_correct")
        and (artifact_ps(row) if ps else row.get("artifact_correct"))
        and row.get("target_correct")
        and float(row.get("bbox_iou") or 0.0) >= threshold
    )


def threshold_summary(
    dataset: str,
    rows: list[dict[str, Any]],
    strata: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(rows),
        "sampling_design": design_metadata(dataset, strata),
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        primary = joint_predicate(threshold)
        ps = joint_predicate(threshold, ps=True)
        primary_values = np.asarray([float(primary(row)) for row in rows], dtype=np.float64)
        ps_values = np.asarray([float(ps(row)) for row in rows], dtype=np.float64)
        result["thresholds"][str(threshold)] = {
            "joint_primary": float(primary_values.mean()),
            "joint_primary_ci95": mover_wilson_interval(primary_values, strata),
            "joint_primary_or_secondary": float(ps_values.mean()),
            "joint_primary_or_secondary_ci95": mover_wilson_interval(ps_values, strata),
            "interval_method": "stratum-specific Wilson intervals combined by MOVER",
        }
    return result


def multivlm_summary(path: Path, scale_map: dict[str, str]) -> dict[str, Any]:
    rows = read_jsonl(path)
    strata = [scale_map[str(row["img_id"])] for row in rows]
    result: dict[str, Any] = {
        "samples": len(rows),
        "sampling_design": {
            "estimand": "frozen JSG-Scale-200 benchmark-composition mean",
            "stratification": "mask-area band",
            "stratum_counts": stratum_counts(strata),
            "design_weights": {
                key: value / len(rows) for key, value in stratum_counts(strata).items()
            },
            "population_inference": False,
        },
        "parse_rate": proportion(rows, lambda row: bool((row.get("prediction") or {}).get("parsed", True))),
        "prediction_accuracy": proportion(rows, lambda row: bool(row.get("prediction_correct"))),
        "artifact_accuracy": proportion(rows, lambda row: bool(row.get("artifact_correct"))),
        "target_accuracy": proportion(rows, lambda row: bool(row.get("target_correct"))),
    }
    for threshold in (0.1, 0.3):
        result[f"iou_ge_{threshold}"] = proportion(rows, lambda row, t=threshold: float(row.get("bbox_iou") or 0.0) >= t)
        predicate = joint_predicate(threshold)
        values = np.asarray([float(predicate(row)) for row in rows], dtype=np.float64)
        result[f"joint_{threshold}"] = float(values.mean())
        result[f"joint_{threshold}_ci95"] = mover_wilson_interval(values, strata)
        result[f"joint_{threshold}_interval_method"] = (
            "stratum-specific Wilson intervals combined by MOVER"
        )
    return result


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def ci_text(values: list[float]) -> str:
    return f"[{pct(values[0])}, {pct(values[1])}]"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Joint Threshold and Multi-VLM Analysis", "", "## Threshold Sensitivity", ""]
    lines.extend([
        "| Setting | N | t | Joint Primary | 95% MOVER-Wilson CI | Joint P/S | 95% MOVER-Wilson CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for setting, data in report["threshold_sensitivity"].items():
        for threshold, row in data["thresholds"].items():
            lines.append(
                f"| {setting} | {data['samples']} | {threshold} | {pct(row['joint_primary'])} | "
                f"{ci_text(row['joint_primary_ci95'])} | {pct(row['joint_primary_or_secondary'])} | "
                f"{ci_text(row['joint_primary_or_secondary_ci95'])} |"
            )
    lines.extend(["", "## Fixed-Schema Multi-VLM Joint Evaluation", ""])
    lines.extend([
        "| Model | N | Parse | Pred | Artifact | Target | IoU@0.1 | Joint@0.1 | CI95 | IoU@0.3 | Joint@0.3 | CI95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model, row in report["multivlm_joint"].items():
        lines.append(
            f"| {model} | {row['samples']} | {pct(row['parse_rate'])} | {pct(row['prediction_accuracy'])} | "
            f"{pct(row['artifact_accuracy'])} | {pct(row['target_accuracy'])} | {pct(row['iou_ge_0.1'])} | "
            f"{pct(row['joint_0.1'])} | {ci_text(row['joint_0.1_ci95'])} | {pct(row['iou_ge_0.3'])} | "
            f"{pct(row['joint_0.3'])} | {ci_text(row['joint_0.3_ci95'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    core_map = core_strata_by_id()
    scale_map = scale_strata_by_id()
    xfer_map = {
        str(row["img_id"]): xfer_stratum(
            row.get("external_dataset"), float(row.get("bbox_iou") or 0.0)
        )
        for row in read_jsonl(XFER_SELECTION_FILE)
    }
    threshold_rows = {
        "JSG-Core": sid_explain_candidate_rows(),
        "JSG-Scale": sid_fa_candidate_rows(),
        "JSG-Xfer": external_semantic_rows(),
    }
    threshold_strata = {
        dataset: benchmark_strata(dataset, rows, core_map, scale_map, xfer_map)
        for dataset, rows in threshold_rows.items()
    }
    report = {
        "stage": "joint_threshold_multivlm_analysis",
        "configuration": {
            "binary_interval": "stratum-specific Wilson intervals combined by MOVER",
            "estimand": "frozen benchmark-design weighted mean",
        },
        "threshold_sensitivity": {
            dataset: threshold_summary(dataset, rows, threshold_strata[dataset])
            for dataset, rows in threshold_rows.items()
        },
        "multivlm_joint": {
            model: multivlm_summary(path, scale_map)
            for model, path in MODEL_FILES.items()
            if path.exists()
        },
        "scope": (
            "Threshold results reuse frozen per-sample outputs. Multi-VLM rows use the same frozen "
            "JSG-Scale-200 candidate IDs and schema; they diagnose model/prompt transfer rather than "
            "universal capability. JSG-Xfer intervals describe stability under its deterministic "
            "stratified stress-test composition, not source-population uncertainty."
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "joint_threshold_multivlm_analysis.json"
    md_path = REPORT_DIR / "joint_threshold_multivlm_analysis.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
