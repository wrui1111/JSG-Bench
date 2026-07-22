#!/usr/bin/env python3
"""Analyze the completed External-Semantic-100 evaluation.

This script reads completed human labels and frozen candidate predictions. It
does not run model inference and must not be used for prompt/rule tuning.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGETS = ["person", "animal", "object", "background", "text", "face", "other", "none", ""]
ARTIFACTS = [
    "boundary_seam",
    "texture_smoothness",
    "lighting_shadow",
    "geometry_structure",
    "resolution_noise",
    "semantic_implausibility",
    "compression_artifact",
    "other",
    "",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def flt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ious = [float(row.get("bbox_iou") or 0.0) for row in rows]
    return {
        "samples": n,
        "artifact_accuracy": sum(1 for row in rows if row.get("artifact_correct")) / n if n else None,
        "artifact_primary_accuracy": sum(1 for row in rows if row.get("artifact_correct")) / n if n else None,
        "artifact_primary_or_secondary_accuracy": sum(1 for row in rows if row.get("artifact_primary_or_secondary_correct")) / n if n else None,
        "target_accuracy": sum(1 for row in rows if row.get("target_correct")) / n if n else None,
        "mean_iou": mean(ious),
        "iou_0_1": sum(1 for v in ious if v >= 0.1) / n if n else None,
        "iou_0_3": sum(1 for v in ious if v >= 0.3) / n if n else None,
        "iou_0_5": sum(1 for v in ious if v >= 0.5) / n if n else None,
    }


def by_dimension(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return {name: metric(items) for name, items in sorted(grouped.items())}


def majority_baseline(rows: list[dict[str, Any]], gold_key: str) -> dict[str, Any]:
    counts = Counter(str(row.get(gold_key, "")) for row in rows)
    majority, support = counts.most_common(1)[0]
    n = len(rows)
    per_class = {label: (1.0 if label == majority else 0.0) for label in sorted(counts)}
    return {
        "majority_label": majority,
        "support": dict(sorted(counts.items())),
        "weighted_accuracy": support / n if n else None,
        "macro_accuracy": sum(per_class.values()) / len(per_class) if per_class else None,
    }


def macro_accuracy(rows: list[dict[str, Any]], gold_key: str, correct_key: str) -> dict[str, Any]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(gold_key, ""))].append(bool(row.get(correct_key)))
    per_class = {
        label: {
            "n": len(values),
            "accuracy": sum(values) / len(values) if values else None,
        }
        for label, values in sorted(grouped.items())
    }
    vals = [item["accuracy"] for item in per_class.values() if item["accuracy"] is not None]
    return {
        "macro_accuracy": sum(vals) / len(vals) if vals else None,
        "per_class": per_class,
    }


def iou_bucket(iou: float) -> str:
    if iou >= 0.5:
        return "iou_ge_0.5"
    if iou >= 0.3:
        return "iou_0.3_0.5"
    if iou >= 0.1:
        return "iou_0.1_0.3"
    return "iou_lt_0.1"


def select_cases(rows: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = {
        "bbox_and_semantic_success": lambda r: r["bbox_iou"] >= 0.3 and r["artifact_correct"] and r["target_correct"],
        "bbox_good_artifact_wrong": lambda r: r["bbox_iou"] >= 0.3 and not r["artifact_correct"],
        "bbox_good_target_wrong": lambda r: r["bbox_iou"] >= 0.3 and not r["target_correct"],
        "bbox_bad_semantic_partial": lambda r: r["bbox_iou"] < 0.1 and (r["artifact_correct"] or r["target_correct"]),
        "casia_failure": lambda r: r["external_dataset"] == "CASIA2" and r["bbox_iou"] < 0.1,
        "imd_failure": lambda r: r["external_dataset"] == "IMD2020" and r["bbox_iou"] < 0.1,
    }
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for case_type, predicate in buckets.items():
        candidates = [row for row in rows if row["img_id"] not in used and predicate(row)]
        candidates.sort(key=lambda r: (r["external_dataset"], -float(r["bbox_iou"]), r["img_id"]))
        if not candidates:
            continue
        row = candidates[0]
        used.add(row["img_id"])
        ann = annotations.get(row["img_id"], {})
        selected.append(
            {
                "case_type": case_type,
                "external_dataset": row["external_dataset"],
                "img_id": row["img_id"],
                "bbox_iou": row["bbox_iou"],
                "gold_artifact": row["gold_artifact"],
                "pred_artifact": row["pred_artifact"],
                "gold_target": row["gold_target"],
                "pred_target": row["pred_target"],
                "image_path": ann.get("image_path"),
                "mask_path": ann.get("mask_path"),
                "candidate_marked_path": ann.get("candidate_marked_path"),
                "candidate_crop_path": ann.get("candidate_crop_path"),
                "short_note": ann.get("short_note"),
            }
        )
    return selected


def table(title: str, rows: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Group | N | Artifact Primary Acc | Artifact Primary/Secondary Acc | Target Acc | mean IoU | IoU@0.1 | IoU@0.3 | IoU@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['samples']} | {pct(row['artifact_primary_accuracy'])} | "
            f"{pct(row['artifact_primary_or_secondary_accuracy'])} | "
            f"{pct(row['target_accuracy'])} | {flt(row['mean_iou'])} | {pct(row['iou_0_1'])} | "
            f"{pct(row['iou_0_3'])} | {pct(row['iou_0_5'])} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("external/annotations/External-Semantic-100_template.jsonl"))
    parser.add_argument("--rows", type=Path, default=Path("experiments/runs/external_semantic_eval_rows.jsonl"))
    parser.add_argument("--eval-json", type=Path, default=Path("experiments/reports/external_semantic_eval.json"))
    parser.add_argument("--sid-explain", type=Path, default=Path("experiments/reports/sid_explain300_tampered100_candidate_top3_revised_metrics.json"))
    parser.add_argument("--sid-fa", type=Path, default=Path("experiments/reports/sid_fa600_candidate_top3_postreview_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/reports/external_semantic_analysis.json"))
    parser.add_argument("--summary-md", type=Path, default=Path("experiments/reports/external_semantic_analysis.md"))
    parser.add_argument("--case-md", type=Path, default=Path("experiments/reports/external_semantic_case_index.md"))
    args = parser.parse_args()

    annotations = {row["img_id"]: row for row in read_jsonl(args.annotations)}
    rows = read_jsonl(args.rows)
    for row in rows:
        row["iou_bucket"] = iou_bucket(float(row.get("bbox_iou") or 0.0))

    overall = metric(rows)
    artifact_macro = macro_accuracy(rows, "gold_artifact", "artifact_correct")
    target_macro = macro_accuracy(rows, "gold_target", "target_correct")
    report = {
        "stage": "external_semantic_analysis",
        "annotations": str(args.annotations),
        "rows": str(args.rows),
        "samples": len(rows),
        "overall": overall,
        "sampling_note": "External-Semantic-100 is a stratified semantic sanity subset: 50 CASIA2 and 50 IMD2020 samples selected to cover low/mid/high candidate IoU buckets. Its semantic metrics are reportable, but its bbox IoU distribution should not be treated as a full-dataset estimate.",
        "artifact_primary_majority_baseline": majority_baseline(rows, "gold_artifact"),
        "target_majority_baseline": majority_baseline(rows, "gold_target"),
        "artifact_macro": artifact_macro,
        "target_macro": target_macro,
        "by_dataset": by_dimension(rows, "external_dataset"),
        "by_gold_artifact": by_dimension(rows, "gold_artifact"),
        "by_gold_target": by_dimension(rows, "gold_target"),
        "by_iou_bucket": by_dimension(rows, "iou_bucket"),
        "internal_comparison": {
            "SID-Explain-300 tampered100": read_json(args.sid_explain),
            "SID-FA-600 postreview": read_json(args.sid_fa),
            "External-Semantic-100": read_json(args.eval_json),
        },
        "cases": select_cases(rows, annotations),
    }
    write_json(args.output, report)

    lines = [
        "# External-Semantic-100 Analysis",
        "",
        "This report uses completed human labels only for evaluation. It does not tune prompts, thresholds, or candidate selection.",
        "",
        "Sampling note: External-Semantic-100 is a stratified semantic sanity subset: 50 CASIA2 and 50 IMD2020 samples selected to cover low/mid/high candidate IoU buckets. Its semantic metrics are reportable, but its bbox IoU distribution should not be treated as a full-dataset estimate.",
        "",
        "## Overall",
        "",
        "| N | Artifact Primary Acc | Artifact Primary/Secondary Acc | Artifact Primary Macro | Artifact Majority | Target Acc | Target Macro | Target Majority | mean IoU | IoU@0.1 | IoU@0.3 | IoU@0.5 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {overall['samples']} | {pct(overall['artifact_primary_accuracy'])} | "
            f"{pct(overall['artifact_primary_or_secondary_accuracy'])} | {pct(artifact_macro['macro_accuracy'])} | "
            f"{pct(report['artifact_primary_majority_baseline']['weighted_accuracy'])} | {pct(overall['target_accuracy'])} | "
            f"{pct(target_macro['macro_accuracy'])} | {pct(report['target_majority_baseline']['weighted_accuracy'])} | "
            f"{flt(overall['mean_iou'])} | {pct(overall['iou_0_1'])} | {pct(overall['iou_0_3'])} | {pct(overall['iou_0_5'])} |"
        ),
        "",
        f"- Artifact majority label: `{report['artifact_primary_majority_baseline']['majority_label']}`.",
        f"- Target majority label: `{report['target_majority_baseline']['majority_label']}`.",
        "",
        "Interpretation: primary-only external artifact typing is above the dominant-label majority baseline after the completed annotation revision; allowing secondary artifact labels further raises the hit rate. Target accuracy remains above the target majority baseline. This should still be written as a stratified external semantic boundary result, not as strong cross-dataset semantic generalization.",
        "",
        *table("By Dataset", report["by_dataset"]),
        "",
        *table("By Gold Artifact", report["by_gold_artifact"]),
        "",
        *table("By Gold Target", report["by_gold_target"]),
        "",
        *table("By IoU Bucket", report["by_iou_bucket"]),
        "",
        "## Internal Comparison",
        "",
        "| Dataset | N | Artifact Primary Acc | Artifact Primary/Secondary Acc | Target Acc | mean IoU | IoU@0.1 | IoU@0.3 | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        "| SID-Explain-300 tampered100 | 100 | 74.00 | - | 91.00 | - | 59.00 | - | easier internal explanation subset |",
        "| SID-FA-600 postreview | 600 | 79.50 | - | 58.00 | 0.1720 | 52.00 | 24.00 | artifact weighted acc below 84.33 majority baseline |",
        f"| External-Semantic-100 | {overall['samples']} | {pct(overall['artifact_primary_accuracy'])} | {pct(overall['artifact_primary_or_secondary_accuracy'])} | {pct(overall['target_accuracy'])} | {flt(overall['mean_iou'])} | {pct(overall['iou_0_1'])} | {pct(overall['iou_0_3'])} | stratified external semantic sanity subset |",
    ]
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    case_lines = [
        "# External Semantic Case Index",
        "",
        "Use these cases for qualitative external sanity-check panels. Each panel should show original image, mask overlay, candidate marked image, crop, and the evidence card.",
        "",
        "| Case | Dataset | img_id | IoU | Gold Artifact | Pred Artifact | Gold Target | Pred Target | Assets |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    for item in report["cases"]:
        assets = ", ".join(
            f"`{item[key]}`" for key in ["image_path", "mask_path", "candidate_marked_path", "candidate_crop_path"] if item.get(key)
        )
        case_lines.append(
            f"| {item['case_type']} | {item['external_dataset']} | {item['img_id']} | {item['bbox_iou']:.4f} | "
            f"{item['gold_artifact']} | {item['pred_artifact']} | {item['gold_target']} | {item['pred_target']} | {assets} |"
        )
    args.case_md.write_text("\n".join(case_lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.summary_md), "cases": str(args.case_md), "overall": overall}, ensure_ascii=False))


if __name__ == "__main__":
    main()
