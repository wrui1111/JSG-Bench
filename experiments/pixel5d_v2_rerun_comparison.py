#!/usr/bin/env python3
"""Compare frozen legacy TD-EGFA results with the controlled pixel5d-v2 rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
REPORTS = ROOT / "experiments" / "reports"
THRESHOLDS = (0.1, 0.3, 0.5)
SEED = 20260721
BOOTSTRAP_REPLICATES = 20000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_map(path: Path, key: str = "img_id") -> dict[str, dict[str, Any]]:
    return {row[key]: row for row in read_jsonl(path)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def td_conditions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        prediction = row.get("prediction") or {}
        result[row["img_id"]] = {
            "parse": bool(prediction.get("parsed")),
            "prediction": bool(row.get("prediction_correct")),
            "artifact": bool(row.get("artifact_correct")),
            "target": bool(row.get("target_correct")),
            "iou": float(row.get("bbox_iou") or 0.0),
        }
    return result


def xfer_legacy_conditions() -> dict[str, dict[str, Any]]:
    traces = row_map(RUNS / "jsg_xfer_td_egfa_audit_trace_v1.jsonl", "sample_id")
    semantic = row_map(RUNS / "external_semantic_eval_rows.jsonl")
    result: dict[str, dict[str, Any]] = {}
    for sample_id, trace in traces.items():
        primary = trace["audit_profiles"]["primary_strict"]["conditions"]
        sem = semantic[sample_id]
        result[sample_id] = {
            "parse": bool(primary["parse"]),
            "prediction": bool(primary["prediction"]),
            "artifact": bool(sem["artifact_correct"]),
            "artifact_ps": bool(sem["artifact_primary_or_secondary_correct"]),
            "target": bool(sem["target_correct"]),
            "iou": float(trace["evaluated_region"]["iou"] or 0.0),
        }
    return result


def xfer_new_conditions() -> dict[str, dict[str, Any]]:
    outputs = row_map(RUNS / "jsg_xfer_outputs_pixel5d_v2.jsonl")
    predictions = row_map(RUNS / "jsg_xfer_predictions_pixel5d_v2.jsonl")
    semantic = row_map(RUNS / "jsg_xfer_semantic_rows_pixel5d_v2.jsonl")
    result: dict[str, dict[str, Any]] = {}
    for sample_id, sem in semantic.items():
        result[sample_id] = {
            "parse": isinstance(outputs[sample_id].get("vlm_result"), dict),
            "prediction": predictions[sample_id].get("prediction") == "tampered",
            "artifact": bool(sem["artifact_correct"]),
            "artifact_ps": bool(sem["artifact_primary_or_secondary_correct"]),
            "target": bool(sem["target_correct"]),
            "iou": float(sem["bbox_iou"] or 0.0),
        }
    return result


def xfer_uniform_legacy_conditions() -> dict[str, dict[str, Any]]:
    outputs = row_map(RUNS / "jsg_xfer_outputs_legacy6d_uniform_qwen25vl.jsonl")
    predictions = row_map(RUNS / "jsg_xfer_predictions_legacy6d_uniform_qwen25vl.jsonl")
    semantic = row_map(RUNS / "jsg_xfer_semantic_rows_legacy6d_uniform_qwen25vl.jsonl")
    result: dict[str, dict[str, Any]] = {}
    for sample_id, sem in semantic.items():
        result[sample_id] = {
            "parse": isinstance(outputs[sample_id].get("vlm_result"), dict),
            "prediction": predictions[sample_id].get("prediction") == "tampered",
            "artifact": bool(sem["artifact_correct"]),
            "artifact_ps": bool(sem["artifact_primary_or_secondary_correct"]),
            "target": bool(sem["target_correct"]),
            "iou": float(sem["bbox_iou"] or 0.0),
        }
    return result


def joint(row: dict[str, Any], threshold: float, artifact_key: str = "artifact") -> bool:
    return bool(
        row["parse"]
        and row["prediction"]
        and row[artifact_key]
        and row["target"]
        and row["iou"] >= threshold
    )


def summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    n = len(values)
    summary: dict[str, Any] = {
        "n": n,
        "parse": sum(row["parse"] for row in values),
        "prediction": sum(row["prediction"] for row in values),
        "artifact": sum(row["artifact"] for row in values),
        "target": sum(row["target"] for row in values),
        "mean_iou": sum(row["iou"] for row in values) / n,
    }
    if all("artifact_ps" in row for row in values):
        summary["artifact_ps"] = sum(row["artifact_ps"] for row in values)
    for threshold in THRESHOLDS:
        label = str(threshold)
        summary[f"spatial@{label}"] = sum(row["iou"] >= threshold for row in values)
        summary[f"joint@{label}"] = sum(joint(row, threshold) for row in values)
        if "artifact_ps" in summary:
            summary[f"joint_ps@{label}"] = sum(joint(row, threshold, "artifact_ps") for row in values)
    return summary


def paired_bootstrap_ci(delta: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = len(delta)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        estimates[index] = delta[rng.integers(0, n, n)].mean()
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def paired_comparison(
    old: dict[str, dict[str, Any]],
    new: dict[str, dict[str, Any]],
    include_ps: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if set(old) != set(new):
        raise ValueError("Paired comparison requires identical sample IDs")
    ids = sorted(old)
    metrics = ["parse", "prediction", "artifact", "target", "spatial@0.1", "joint@0.1", "iou"]
    if include_ps:
        metrics.insert(3, "artifact_ps")
        metrics.insert(-1, "joint_ps@0.1")
    result: dict[str, Any] = {}
    for metric in metrics:
        def value(row: dict[str, Any]) -> float:
            if metric == "iou":
                return float(row["iou"])
            if metric == "spatial@0.1":
                return float(row["iou"] >= 0.1)
            if metric == "joint@0.1":
                return float(joint(row, 0.1))
            if metric == "joint_ps@0.1":
                return float(joint(row, 0.1, "artifact_ps"))
            return float(row[metric])

        old_values = np.asarray([value(old[sample_id]) for sample_id in ids], dtype=np.float64)
        new_values = np.asarray([value(new[sample_id]) for sample_id in ids], dtype=np.float64)
        delta = new_values - old_values
        item: dict[str, Any] = {
            "old": float(old_values.mean()),
            "new": float(new_values.mean()),
            "delta": float(delta.mean()),
            "paired_bootstrap_95ci": paired_bootstrap_ci(delta, rng),
        }
        if metric != "iou":
            item["old_pass_new_fail"] = int(((old_values == 1) & (new_values == 0)).sum())
            item["old_fail_new_pass"] = int(((old_values == 0) & (new_values == 1)).sum())
        result[metric] = item
    return result


def union_top3(row: dict[str, Any]) -> list[int]:
    boxes = [item["bbox"] for item in row["lowlevel_candidates"][:3]]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def lowlevel_comparison(old_path: Path, new_path: Path) -> dict[str, Any]:
    old = row_map(old_path)
    new = row_map(new_path)
    if set(old) != set(new):
        raise ValueError("Low-level files have different IDs")
    ids = sorted(old)
    return {
        "n": len(ids),
        "top3_union_changed": sum(union_top3(old[sample_id]) != union_top3(new[sample_id]) for sample_id in ids),
        "top_feature_ranking_changed": sum(
            old[sample_id]["top_lowlevel_features"] != new[sample_id]["top_lowlevel_features"]
            for sample_id in ids
        ),
        "artifact_hints_changed": sum(
            old[sample_id]["lowlevel_artifact_types"] != new[sample_id]["lowlevel_artifact_types"]
            for sample_id in ids
        ),
        "rounded_score_changed": sum(
            old[sample_id]["lowlevel_score"] != new[sample_id]["lowlevel_score"]
            for sample_id in ids
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS / "pixel5d_v2_rerun_comparison.json",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    datasets = {
        "Core": (
            td_conditions(RUNS / "sid_explain300_tampered100_candidate_top3_revised_predictions.jsonl"),
            td_conditions(RUNS / "jsg_core_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl"),
            False,
        ),
        "Scale": (
            td_conditions(RUNS / "sid_fa600_candidate_top3_postreview_predictions.jsonl"),
            td_conditions(RUNS / "jsg_scale_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl"),
            False,
        ),
        "Xfer_frozen_mixed": (xfer_legacy_conditions(), xfer_new_conditions(), True),
        "Xfer_uniform_runtime_controlled": (
            xfer_uniform_legacy_conditions(),
            xfer_new_conditions(),
            True,
        ),
    }

    result: dict[str, Any] = {
        "analysis_version": "pixel5d-v2-controlled-comparison-v1",
        "bootstrap": {"seed": SEED, "replicates": BOOTSTRAP_REPLICATES, "method": "paired percentile"},
        "scope": {
            "feature_change": "remove pooled rgb_mean; retain five distinct equally weighted features",
            "compression_suffix_rule": "unchanged in this experiment",
            "core_scale_prompt_control": "legacy static prompt preserved byte-for-byte; only bbox, score, hints, and assets changed",
            "xfer_caution": "the frozen comparison is runtime-confounded; Xfer_uniform_runtime_controlled is the primary feature-effect comparison",
        },
        "lowlevel": {
            "Core": lowlevel_comparison(
                RUNS / "sid_explain300_tampered100_lowlevel_top3.jsonl",
                RUNS / "jsg_core_lowlevel_pixel5d_v2.jsonl",
            ),
            "Scale": lowlevel_comparison(
                RUNS / "sid_fa600_lowlevel_top3.jsonl",
                RUNS / "jsg_scale_lowlevel_pixel5d_v2.jsonl",
            ),
            "Xfer": lowlevel_comparison(
                RUNS / "jsg_xfer_td_egfa_lowlevel_v2.jsonl",
                RUNS / "jsg_xfer_lowlevel_pixel5d_v2.jsonl",
            ),
        },
        "datasets": {},
        "manuscript_integrity": {
            "main_en_tex_sha256": sha256(ROOT / "ieee-journal-latex-package" / "main_en.tex"),
            "main_en_pdf_sha256": sha256(ROOT / "ieee-journal-latex-package" / "main_en.pdf"),
        },
    }
    for name, (old, new, include_ps) in datasets.items():
        result["datasets"][name] = {
            "old": summarize(old),
            "pixel5d_v2": summarize(new),
            "paired": paired_comparison(old, new, include_ps, rng),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote controlled comparison to {args.output}")


if __name__ == "__main__":
    main()
