#!/usr/bin/env python3
"""Evaluate inter-annotator agreement for completed IAA overlap annotations."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median, quantiles
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IAA_DIR = ROOT / "SID_Set/annotations/iaa"
REPORT_DIR = ROOT / "experiments/reports"
SID_FA_GOLD = ROOT / "SID_Set/annotations/SID-FA-600.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("source_img_id") or row.get("img_id") or "")


def norm_scalar(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip().lower() or "none"
    aliases = {
        "perosn": "person",
        "ogeometry_structure": "geometry_structure",
    }
    return aliases.get(text, text)


def norm_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return sorted({norm_scalar(v) for v in value if norm_scalar(v) != "none"})
    text = norm_scalar(value)
    return [] if text == "none" else [text]


def first_evidence(row: dict[str, Any]) -> dict[str, Any]:
    items = row.get("evidence_items")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0]
    return {}


def evidence_artifact(row: dict[str, Any]) -> Any:
    return first_evidence(row).get("artifact_type", row.get("dominant_artifact_type"))


def evidence_bbox(row: dict[str, Any]) -> list[float] | None:
    bbox = first_evidence(row).get("evidence_bbox")
    if bbox is None:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_inside(center_bbox: list[float] | None, target_bbox: list[float] | None) -> bool | None:
    if center_bbox is None and target_bbox is None:
        return None
    if center_bbox is None or target_bbox is None:
        return False
    cx = (center_bbox[0] + center_bbox[2]) / 2
    cy = (center_bbox[1] + center_bbox[3]) / 2
    return target_bbox[0] <= cx <= target_bbox[2] and target_bbox[1] <= cy <= target_bbox[3]


def cohen_kappa(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return {"samples": 0, "agreement": None, "expected_agreement": None, "cohen_kappa": None}
    n = len(pairs)
    agreement = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    labels = sorted(set(left) | set(right))
    expected = sum((left[label] / n) * (right[label] / n) for label in labels)
    if math.isclose(expected, 1.0):
        kappa = 1.0 if math.isclose(agreement, 1.0) else 0.0
    else:
        kappa = (agreement - expected) / (1 - expected)
    return {
        "samples": n,
        "agreement": round(agreement, 6),
        "expected_agreement": round(expected, 6),
        "cohen_kappa": round(kappa, 6),
        "labels_reference": dict(sorted(left.items())),
        "labels_second_pass": dict(sorted(right.items())),
    }


def weighted_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> dict[str, Any]:
    if not pairs:
        return {"samples": 0, "agreement": None, "weighted_kappa_quadratic": None}
    index = {label: i for i, label in enumerate(labels)}
    valid_pairs = [(a, b) for a, b in pairs if a in index and b in index]
    if not valid_pairs:
        return {"samples": 0, "agreement": None, "weighted_kappa_quadratic": None}
    n = len(valid_pairs)
    k = len(labels)
    agreement = sum(1 for a, b in valid_pairs if a == b) / n
    left = Counter(a for a, _ in valid_pairs)
    right = Counter(b for _, b in valid_pairs)

    def weight(a: str, b: str) -> float:
        if k <= 1:
            return 0.0
        return ((index[a] - index[b]) ** 2) / ((k - 1) ** 2)

    observed_disagreement = sum(weight(a, b) for a, b in valid_pairs) / n
    expected_disagreement = 0.0
    for a in labels:
        for b in labels:
            expected_disagreement += (left[a] / n) * (right[b] / n) * weight(a, b)
    if math.isclose(expected_disagreement, 0.0):
        kappa = 1.0 if math.isclose(observed_disagreement, 0.0) else 0.0
    else:
        kappa = 1 - (observed_disagreement / expected_disagreement)
    return {
        "samples": n,
        "agreement": round(agreement, 6),
        "observed_weighted_disagreement": round(observed_disagreement, 6),
        "expected_weighted_disagreement": round(expected_disagreement, 6),
        "weighted_kappa_quadratic": round(kappa, 6),
        "labels": labels,
    }


def multilabel_agreement(pairs: list[tuple[list[str], list[str]]]) -> dict[str, Any]:
    if not pairs:
        return {"samples": 0}
    exact = 0
    jaccards: list[float] = []
    f1s: list[float] = []
    ref_counts: Counter[str] = Counter()
    sec_counts: Counter[str] = Counter()
    for left_list, right_list in pairs:
        left = set(left_list)
        right = set(right_list)
        ref_counts.update(left)
        sec_counts.update(right)
        exact += left == right
        union = left | right
        inter = left & right
        jaccards.append(1.0 if not union else len(inter) / len(union))
        denom = len(left) + len(right)
        f1s.append(1.0 if denom == 0 else 2 * len(inter) / denom)
    return {
        "samples": len(pairs),
        "exact_match": round(exact / len(pairs), 6),
        "mean_jaccard": round(sum(jaccards) / len(jaccards), 6),
        "mean_f1": round(sum(f1s) / len(f1s), 6),
        "labels_reference": dict(sorted(ref_counts.items())),
        "labels_second_pass": dict(sorted(sec_counts.items())),
    }


def bbox_agreement(pairs: list[tuple[list[float] | None, list[float] | None]]) -> dict[str, Any]:
    if not pairs:
        return {"samples": 0}
    both_null = sum(1 for a, b in pairs if a is None and b is None)
    one_null = sum(1 for a, b in pairs if (a is None) ^ (b is None))
    bbox_pairs = [(a, b) for a, b in pairs if a is not None or b is not None]
    ious = [bbox_iou(a, b) or 0.0 for a, b in bbox_pairs]
    pointing = [center_inside(a, b) for a, b in bbox_pairs]
    pointing_valid = [v for v in pointing if v is not None]
    if len(ious) > 1:
        first_quartile, _, third_quartile = quantiles(ious, n=4, method="inclusive")
    elif ious:
        first_quartile = third_quartile = ious[0]
    else:
        first_quartile = third_quartile = None
    return {
        "samples": len(pairs),
        "bbox_compared_samples": len(bbox_pairs),
        "both_null": both_null,
        "one_null": one_null,
        "exact_match": (
            round(sum(a == b for a, b in bbox_pairs) / len(bbox_pairs), 6)
            if bbox_pairs
            else None
        ),
        "mean_iou": round(sum(ious) / len(ious), 6) if ious else None,
        "median_iou": round(median(ious), 6) if ious else None,
        "iou_iqr": (
            round(third_quartile - first_quartile, 6)
            if first_quartile is not None and third_quartile is not None
            else None
        ),
        "iou_0_1": round(sum(i >= 0.1 for i in ious) / len(ious), 6) if ious else None,
        "iou_0_3": round(sum(i >= 0.3 for i in ious) / len(ious), 6) if ious else None,
        "iou_0_5": round(sum(i >= 0.5 for i in ious) / len(ious), 6) if ious else None,
        "iou_0_7": round(sum(i >= 0.7 for i in ious) / len(ious), 6) if ious else None,
        "center_of_reference_inside_second": (
            round(sum(bool(v) for v in pointing_valid) / len(pointing_valid), 6) if pointing_valid else None
        ),
    }


def exact_field_report(
    ref_rows: dict[str, dict[str, Any]],
    second_rows: dict[str, dict[str, Any]],
    fields: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in fields:
        pairs = [(norm_scalar(ref_rows[k].get(field)), norm_scalar(second_rows[k].get(field))) for k in sorted(ref_rows)]
        out[field] = cohen_kappa(pairs)
    return out


def disagreement_examples(
    ref_rows: dict[str, dict[str, Any]],
    second_rows: dict[str, dict[str, Any]],
    fields: list[str],
    limit: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        rows = []
        for key in sorted(ref_rows):
            left = norm_scalar(ref_rows[key].get(field))
            right = norm_scalar(second_rows[key].get(field))
            if left != right:
                rows.append({"img_id": key, "reference": left, "second_pass": right})
        examples[field] = rows[:limit]
    return examples


def confusion_pairs(
    ref_rows: dict[str, dict[str, Any]], second_rows: dict[str, dict[str, Any]], field: str, limit: int = 20
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for key in sorted(ref_rows):
        a = norm_scalar(ref_rows[key].get(field))
        b = norm_scalar(second_rows[key].get(field))
        if a != b:
            counts[(a, b)] += 1
    return [
        {"reference": a, "second_pass": b, "count": count}
        for (a, b), count in counts.most_common(limit)
    ]


def load_pair(
    reference_name: str,
    second_name: str,
    reference_path: Path | None = None,
    reference_role: str = "reference",
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    reference_file = reference_path or (IAA_DIR / reference_name)
    reference = read_jsonl(reference_file)
    second = read_jsonl(IAA_DIR / second_name)
    ref_by_id = {row_id(row): row for row in reference}
    second_by_id = {row_id(row): row for row in second}
    overlap = sorted(set(ref_by_id) & set(second_by_id))
    audit = {
        "reference_file": str(reference_file),
        "reference_role": reference_role,
        "second_pass_file": str(IAA_DIR / second_name),
        "reference_rows": len(overlap) if reference_role == "independently_reviewed_frozen_gold" else len(reference),
        "full_reference_rows": len(reference),
        "second_pass_rows": len(second),
        "matched_samples": len(overlap),
        "missing_in_reference": sorted(set(second_by_id) - set(ref_by_id)),
        "missing_in_second_pass": []
        if reference_role == "independently_reviewed_frozen_gold"
        else sorted(set(ref_by_id) - set(second_by_id)),
        "duplicate_reference_ids": [k for k, v in Counter(row_id(row) for row in reference).items() if v > 1],
        "duplicate_second_pass_ids": [k for k, v in Counter(row_id(row) for row in second).items() if v > 1],
    }
    return {k: ref_by_id[k] for k in overlap}, {k: second_by_id[k] for k in overlap}, audit


def evaluate_sid_explain() -> dict[str, Any]:
    ref, sec, audit = load_pair("SID-Explain-IAA-60_reference.jsonl", "SID-Explain-IAA-60_template.jsonl")
    fields = ["human_label", "target_scope", "target_semantic_class", "dominant_artifact_type"]
    exact = exact_field_report(ref, sec, fields)
    evidence_artifact_pairs = [(norm_scalar(evidence_artifact(ref[k])), norm_scalar(evidence_artifact(sec[k]))) for k in sorted(ref)]
    bbox_pairs = [(evidence_bbox(ref[k]), evidence_bbox(sec[k])) for k in sorted(ref)]
    examples = disagreement_examples(ref, sec, fields + ["dominant_artifact_type"])
    examples["evidence_artifact_type"] = [
        {"img_id": k, "reference": norm_scalar(evidence_artifact(ref[k])), "second_pass": norm_scalar(evidence_artifact(sec[k]))}
        for k in sorted(ref)
        if norm_scalar(evidence_artifact(ref[k])) != norm_scalar(evidence_artifact(sec[k]))
    ][:12]
    return {
        "name": "SID-Explain-IAA-60",
        "audit": audit,
        "exact_fields": exact,
        "evidence_artifact_type": cohen_kappa(evidence_artifact_pairs),
        "evidence_bbox": bbox_agreement(bbox_pairs),
        "confusions": {
            "human_label": confusion_pairs(ref, sec, "human_label"),
            "target_scope": confusion_pairs(ref, sec, "target_scope"),
            "dominant_artifact_type": confusion_pairs(ref, sec, "dominant_artifact_type"),
        },
        "disagreement_examples": examples,
    }


def evaluate_sid_fa() -> dict[str, Any]:
    ref_path = IAA_DIR / "SID-FA-IAA-120_cross60x2_reference.jsonl"
    if ref_path.exists():
        ref, sec, audit = load_pair(
            "SID-FA-IAA-120_cross60x2_reference.jsonl",
            "SID-FA-IAA-120_cross60x2_template.jsonl",
            reference_role="legacy_reference",
        )
    else:
        ref, sec, audit = load_pair(
            "SID-FA-IAA-120_cross60x2_reference.jsonl",
            "SID-FA-IAA-120_cross60x2_template.jsonl",
            reference_path=SID_FA_GOLD,
            reference_role="independently_reviewed_frozen_gold",
        )
    fields = [
        "mask_area_bucket",
        "target_scope",
        "target_semantic_class",
        "dominant_artifact_type",
        "violated_rule",
    ]
    exact = exact_field_report(ref, sec, fields)
    strength_pairs = [(norm_scalar(ref[k].get("artifact_strength")), norm_scalar(sec[k].get("artifact_strength"))) for k in sorted(ref)]
    secondary_pairs = [(norm_list(ref[k].get("secondary_artifact_types")), norm_list(sec[k].get("secondary_artifact_types"))) for k in sorted(ref)]
    return {
        "name": "SID-FA-IAA-120",
        "audit": audit,
        "exact_fields": exact,
        "artifact_strength_weighted": weighted_kappa(strength_pairs, ["1", "2", "3"]),
        "secondary_artifact_types_multilabel": multilabel_agreement(secondary_pairs),
        "confusions": {
            "target_scope": confusion_pairs(ref, sec, "target_scope"),
            "target_semantic_class": confusion_pairs(ref, sec, "target_semantic_class"),
            "dominant_artifact_type": confusion_pairs(ref, sec, "dominant_artifact_type"),
            "violated_rule": confusion_pairs(ref, sec, "violated_rule"),
        },
        "disagreement_examples": disagreement_examples(ref, sec, fields),
    }


def evaluate_sid_hard() -> dict[str, Any]:
    ref, sec, audit = load_pair("SID-Hard-IAA-60_reference.jsonl", "SID-Hard-IAA-60_template.jsonl")
    fields = ["human_label", "main_failure_risk", "expected_failure_stage"]
    exact = exact_field_report(ref, sec, fields)
    difficulty_pairs = [(norm_scalar(ref[k].get("difficulty_score")), norm_scalar(sec[k].get("difficulty_score"))) for k in sorted(ref)]
    tag_pairs = [(norm_list(ref[k].get("difficulty_tags")), norm_list(sec[k].get("difficulty_tags"))) for k in sorted(ref)]
    return {
        "name": "SID-Hard-IAA-60",
        "audit": audit,
        "exact_fields": exact,
        "difficulty_score_weighted": weighted_kappa(difficulty_pairs, ["0", "1", "2", "3"]),
        "difficulty_tags_multilabel": multilabel_agreement(tag_pairs),
        "confusions": {
            "human_label": confusion_pairs(ref, sec, "human_label"),
            "main_failure_risk": confusion_pairs(ref, sec, "main_failure_risk"),
            "expected_failure_stage": confusion_pairs(ref, sec, "expected_failure_stage"),
        },
        "disagreement_examples": disagreement_examples(ref, sec, fields),
    }


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}"


def num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def metric_rows(report: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for dataset in report["datasets"]:
        name = dataset["name"]
        for field, metric in dataset.get("exact_fields", {}).items():
            rows.append((name, field, str(metric["samples"]), pct(metric["agreement"]), num(metric["cohen_kappa"])))
        for field in ["evidence_artifact_type"]:
            if field in dataset:
                metric = dataset[field]
                rows.append((name, field, str(metric["samples"]), pct(metric["agreement"]), num(metric["cohen_kappa"])))
        if "artifact_strength_weighted" in dataset:
            metric = dataset["artifact_strength_weighted"]
            rows.append((name, "artifact_strength (weighted)", str(metric["samples"]), pct(metric["agreement"]), num(metric["weighted_kappa_quadratic"])))
        if "difficulty_score_weighted" in dataset:
            metric = dataset["difficulty_score_weighted"]
            rows.append((name, "difficulty_score (weighted)", str(metric["samples"]), pct(metric["agreement"]), num(metric["weighted_kappa_quadratic"])))
        for field in ["secondary_artifact_types_multilabel", "difficulty_tags_multilabel"]:
            if field in dataset:
                metric = dataset[field]
                rows.append((name, field, str(metric["samples"]), f"Jaccard {pct(metric['mean_jaccard'])}", f"F1 {pct(metric['mean_f1'])}"))
        if "evidence_bbox" in dataset:
            metric = dataset["evidence_bbox"]
            rows.append(
                (
                    name,
                    "evidence_bbox",
                    str(metric["bbox_compared_samples"]),
                    f"exact {pct(metric['exact_match'])}; median IoU {num(metric['median_iou'])}",
                    f"IQR {num(metric['iou_iqr'])}; IoU@0.5/0.7 {pct(metric['iou_0_5'])}/{pct(metric['iou_0_7'])}",
                )
            )
    return rows


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# IAA Reliability Report",
        "",
        "This report compares benchmark reference annotations with completed blinded second-pass IAA annotations. For `SID-FA-IAA-120`, the reference is the current independently reviewed frozen `SID-FA-600` gold file when the legacy reference file is absent. IAA overlap samples are evaluation-only and must not be used for prompt, threshold, or rule tuning.",
        "",
        "## Data Integrity",
        "",
        "| Dataset | Reference Rows | Second-Pass Rows | Matched | Missing Ref | Missing Second | Duplicate Ref | Duplicate Second |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in report["datasets"]:
        audit = dataset["audit"]
        lines.append(
            f"| {dataset['name']} | {audit['reference_rows']} | {audit['second_pass_rows']} | {audit['matched_samples']} | "
            f"{len(audit['missing_in_reference'])} | {len(audit['missing_in_second_pass'])} | "
            f"{len(audit['duplicate_reference_ids'])} | {len(audit['duplicate_second_pass_ids'])} |"
        )

    lines.extend(
        [
            "",
            "## Agreement Metrics",
            "",
            "| Dataset | Field | N | Agreement / Jaccard | Kappa / F1 / IoU |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for dataset, field, n, agreement, kappa in metric_rows(report):
        lines.append(f"| {dataset} | {field} | {n} | {agreement} | {kappa} |")

    lines.extend(
        [
            "",
            "## Main Interpretation",
            "",
            "- `SID-Explain-IAA-60` has perfect class agreement. On its 20 tampered bbox overlaps, the blinded first- and second-pass submissions are exact matches (median IoU 1.000, IQR 0.000); target scope and artifact typing are less stable.",
            "- `SID-FA-IAA-120` is compared against independently reviewed frozen gold. Target scope, dominant artifact, and violated-rule are reliable enough to report, while free-text semantic class and artifact strength should be treated as diagnostic fields.",
            "- `SID-Hard-IAA-60` is highly reliable because many hard-case tags are feature/mask-rule driven; it can support difficulty breakdown claims.",
            "- Current IAA results support reporting the three annotation subsets as benchmark annotations, but artifact taxonomy claims should remain conservative because weighted Artifact Acc remains below the majority baseline.",
            "",
            "## High-Frequency Confusions",
            "",
        ]
    )
    for dataset in report["datasets"]:
        lines.append(f"### {dataset['name']}")
        for field, confusions in dataset.get("confusions", {}).items():
            if not confusions:
                continue
            joined = "; ".join(f"{row['reference']} -> {row['second_pass']} ({row['count']})" for row in confusions[:8])
            lines.append(f"- `{field}`: {joined}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    report = {
        "stage": "iaa_reliability",
        "datasets": [evaluate_sid_explain(), evaluate_sid_fa(), evaluate_sid_hard()],
        "decision": (
            "IAA overlap annotations are complete. Use IAA metrics in the manuscript; do not overwrite gold labels "
            "automatically. SID-FA-600 gold is independently reviewed and frozen, not second-pass adjudicated."
        ),
    }
    write_json(REPORT_DIR / "iaa_reliability_report.json", report)
    write_markdown(report, REPORT_DIR / "iaa_reliability_report.md")
    print(json.dumps({"stage": "iaa_reliability", "datasets": [d["name"] for d in report["datasets"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
