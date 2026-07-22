#!/usr/bin/env python3
"""Utilities for SID-Hard-300 and external CASIA2 explanation checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from training_free_pipeline import (
    bbox_iou,
    lowlevel_evidence_for_row,
    read_jsonl,
    valid_bbox,
    write_json,
    write_jsonl,
)


ALLOWED_HARD_STAGES = {
    "structured_observation",
    "artifact_typing",
    "evidence_aggregation",
    "grounding_verification",
    "none",
}

PREDICTION_ALIASES = {
    "real": "real",
    "authentic": "real",
    "genuine": "real",
    "full_synthetic": "full_synthetic",
    "fully_synthetic": "full_synthetic",
    "synthetic": "full_synthetic",
    "ai_generated": "full_synthetic",
    "ai-generated": "full_synthetic",
    "global_generation": "full_synthetic",
    "tampered": "tampered",
    "manipulated": "tampered",
    "local_tampering": "tampered",
    "none": "real",
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_prediction(value: Any) -> str:
    return PREDICTION_ALIASES.get(normalize_text(value), "unknown")


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def vlm_result(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("vlm_result"), dict):
        return row["vlm_result"]
    if isinstance(row.get("vlm_raw_text"), str):
        return extract_json_object(row["vlm_raw_text"]) or {}
    return {}


def mask_bbox(mask_path: Path) -> list[int] | None:
    with Image.open(mask_path) as img:
        arr = np.asarray(img.convert("L")) > 0
    if not arr.any():
        return None
    ys, xs = np.where(arr)
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def center_inside(pred: list[int] | None, gt: list[int] | None) -> bool:
    pred = valid_bbox(pred)
    gt = valid_bbox(gt)
    if not pred or not gt:
        return False
    cx = (pred[0] + pred[2]) / 2.0
    cy = (pred[1] + pred[3]) / 2.0
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def union_bbox(boxes: list[list[int]]) -> list[int] | None:
    valid = [box for box in (valid_bbox(box) for box in boxes) if box]
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def topk_union(low: dict[str, Any], top_k: int = 3) -> list[int] | None:
    boxes = []
    candidates = low.get("lowlevel_candidates")
    if isinstance(candidates, list):
        for item in candidates[:top_k]:
            if isinstance(item, dict):
                box = valid_bbox(item.get("bbox"))
                if box:
                    boxes.append(box)
    return union_bbox(boxes) or valid_bbox(low.get("lowlevel_candidate_bbox"))


def hard_gold_bbox(row: dict[str, Any]) -> list[int] | None:
    if row.get("human_label") != "tampered":
        return None
    auto = row.get("auto_features")
    if isinstance(auto, dict):
        box = valid_bbox(auto.get("mask_bbox"))
        if box:
            return box
    return valid_bbox(row.get("mask_bbox"))


def hard_qc(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.annotations)
    missing = Counter()
    invalid = defaultdict(list)
    dist = {
        "human_label": Counter(),
        "difficulty_score": Counter(),
        "difficulty_tags": Counter(),
        "expected_failure_stage": Counter(),
        "main_failure_risk": Counter(),
    }
    required = [
        "annotation_task",
        "annotation_status",
        "do_not_use_for_tuning",
        "img_id",
        "human_label",
        "image_path",
        "width",
        "height",
        "difficulty_tags",
        "difficulty_score",
        "expected_failure_stage",
    ]
    for row in rows:
        for key in required:
            if key not in row:
                missing[key] += 1
        if row.get("annotation_task") != "SID-Hard":
            invalid["annotation_task"].append(row.get("img_id"))
        if row.get("do_not_use_for_tuning") is not True:
            invalid["do_not_use_for_tuning"].append(row.get("img_id"))
        if row.get("human_label") not in {"real", "full_synthetic", "tampered"}:
            invalid["human_label"].append(row.get("img_id"))
        if row.get("expected_failure_stage") not in ALLOWED_HARD_STAGES:
            invalid["expected_failure_stage"].append(row.get("img_id"))
        if row.get("human_label") == "tampered" and not hard_gold_bbox(row):
            invalid["tampered_mask_bbox"].append(row.get("img_id"))
        dist["human_label"][row.get("human_label")] += 1
        dist["difficulty_score"][str(row.get("difficulty_score"))] += 1
        dist["expected_failure_stage"][str(row.get("expected_failure_stage"))] += 1
        dist["main_failure_risk"][str(row.get("main_failure_risk"))] += 1
        for tag in row.get("difficulty_tags") or []:
            dist["difficulty_tags"][tag] += 1

    report = {
        "stage": "sid_hard300_quality_control",
        "annotations": str(args.annotations),
        "samples": len(rows),
        "missing_required_fields": dict(missing),
        "invalid_value_counts": {key: len(value) for key, value in invalid.items()},
        "invalid_values": {key: value[:20] for key, value in invalid.items()},
        "distributions": {key: dict(counter.most_common()) for key, counter in dist.items()},
        "policy": "SID-Hard-300 is a frozen hard-case evaluation subset and must not be used for prompt or rule tuning.",
    }
    write_json(args.output, report)
    write_text(args.summary_output, hard_qc_summary(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def hard_qc_summary(report: dict[str, Any]) -> str:
    lines = [
        "# SID-Hard-300 Quality Control",
        "",
        report["policy"],
        "",
        f"- Samples: {report['samples']}",
        f"- Missing required fields: {sum(report['missing_required_fields'].values())}",
        f"- Invalid values: {sum(report['invalid_value_counts'].values())}",
        "",
    ]
    for key, values in report["distributions"].items():
        lines.extend([f"## {key}", "", "| Value | Count |", "|---|---:|"])
        for value, count in values.items():
            lines.append(f"| {value} | {count} |")
        lines.append("")
        if key == "expected_failure_stage":
            lines.extend(
                [
                    "Note: `expected_failure_stage`, `main_failure_risk`, `difficulty_tags`, and "
                    "`difficulty_score` were preannotated from visible image attributes before "
                    "frozen model-output evaluation. They are not derived from TD-EGFA predictions; "
                    "model results are used only as a post-hoc consistency check.",
                    "",
                ]
            )
    return "\n".join(lines)


def hard_annotation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        item = dict(row)
        item["class_dir"] = row.get("human_label")
        converted.append(item)
    return converted


def hard_lowlevel(args: argparse.Namespace) -> None:
    rows = hard_annotation_rows(read_jsonl(args.annotations))
    sid_root = args.sid_root.resolve()
    out = []
    for idx, row in enumerate(rows, start=1):
        out.append(lowlevel_evidence_for_row(row, sid_root))
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(rows)}")
    write_jsonl(args.output, out)
    print(f"Wrote {len(out)} low-level rows to {args.output}")


def hard_lowlevel_eval(args: argparse.Namespace) -> None:
    annotations = {row["img_id"]: row for row in read_jsonl(args.annotations)}
    lowlevel = {row["img_id"]: row for row in read_jsonl(args.lowlevel)}
    metrics = {
        "top1": [],
        "top3_union": [],
    }
    pointing = Counter()
    rows_out = []
    by_group = {
        "difficulty_score": defaultdict(lambda: {"top1": [], "top3_union": []}),
        "expected_failure_stage": defaultdict(lambda: {"top1": [], "top3_union": []}),
        "difficulty_tag": defaultdict(lambda: {"top1": [], "top3_union": []}),
    }
    for img_id, ann in annotations.items():
        if ann.get("human_label") != "tampered":
            continue
        gt = hard_gold_bbox(ann)
        low = lowlevel.get(img_id)
        if not gt or not low:
            continue
        preds = {
            "top1": valid_bbox(low.get("lowlevel_candidate_bbox")),
            "top3_union": topk_union(low, 3),
        }
        row_out = {"img_id": img_id, "gt_bbox": gt}
        for name, box in preds.items():
            value = bbox_iou(box, gt)
            metrics[name].append(value)
            pointing[name] += int(center_inside(box, gt))
            row_out[f"{name}_bbox"] = box
            row_out[f"{name}_iou"] = value
            by_group["difficulty_score"][str(ann.get("difficulty_score"))][name].append(value)
            by_group["expected_failure_stage"][str(ann.get("expected_failure_stage"))][name].append(value)
            for tag in ann.get("difficulty_tags") or ["none"]:
                by_group["difficulty_tag"][tag][name].append(value)
        rows_out.append(row_out)

    def summarize(values: list[float], point_hits: int | None = None) -> dict[str, Any]:
        arr = np.asarray(values, dtype=np.float32)
        return {
            "samples": int(len(values)),
            "mean_iou": float(arr.mean()) if len(arr) else 0.0,
            "iou_0_1": float((arr >= 0.1).mean()) if len(arr) else 0.0,
            "iou_0_3": float((arr >= 0.3).mean()) if len(arr) else 0.0,
            "iou_0_5": float((arr >= 0.5).mean()) if len(arr) else 0.0,
            "pointing_game": float(point_hits / len(values)) if point_hits is not None and values else None,
        }

    report = {
        "stage": "sid_hard300_lowlevel_evaluate",
        "annotations": str(args.annotations),
        "lowlevel": str(args.lowlevel),
        "metrics": {
            name: summarize(values, pointing[name]) for name, values in metrics.items()
        },
        "breakdown": {
            group: {
                key: {name: summarize(values) for name, values in methods.items()}
                for key, methods in groups.items()
            }
            for group, groups in by_group.items()
        },
    }
    write_json(args.output, report)
    if args.rows_output:
        write_jsonl(args.rows_output, rows_out)
    write_text(args.summary_output, lowlevel_summary(report, "SID-Hard-300 Low-Level Hard-Case Evaluation"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def hard_evaluate_free(args: argparse.Namespace) -> None:
    annotations = {row["img_id"]: row for row in read_jsonl(args.annotations)}
    outputs = read_jsonl(args.outputs)
    counts = Counter()
    by_label = defaultdict(Counter)
    by_score = defaultdict(Counter)
    by_stage = defaultdict(Counter)
    by_tag = defaultdict(Counter)
    ious = []
    pointing = []
    confusion = defaultdict(Counter)
    rows_out = []

    for out in outputs:
        img_id = out.get("img_id")
        ann = annotations.get(img_id)
        if not ann:
            continue
        result = vlm_result(out)
        pred_value = result.get("prediction") or result.get("class") or result.get("evidence_mode")
        pred = normalize_prediction(pred_value)
        gold = normalize_prediction(ann.get("human_label"))
        parsed = bool(result)
        ok = pred == gold
        score_key = str(ann.get("difficulty_score"))
        stage_key = str(ann.get("expected_failure_stage"))
        counts["samples"] += 1
        counts["parsed"] += int(parsed)
        counts["correct"] += int(ok)
        by_label[gold]["samples"] += 1
        by_label[gold]["correct"] += int(ok)
        by_score[score_key]["samples"] += 1
        by_score[score_key]["correct"] += int(ok)
        by_stage[stage_key]["samples"] += 1
        by_stage[stage_key]["correct"] += int(ok)
        for tag in ann.get("difficulty_tags") or ["none"]:
            by_tag[tag]["samples"] += 1
            by_tag[tag]["correct"] += int(ok)
        confusion[gold][pred] += 1

        gt = hard_gold_bbox(ann)
        pred_bbox = valid_bbox(result.get("evidence_bbox"))
        iou_value = None
        pointing_hit = None
        if gt is not None:
            iou_value = bbox_iou(pred_bbox, gt)
            pointing_hit = center_inside(pred_bbox, gt)
            ious.append(iou_value)
            pointing.append(pointing_hit)
            counts["bbox_samples"] += 1
            counts["bbox_iou_0_1"] += int(iou_value >= 0.1)
            counts["bbox_iou_0_3"] += int(iou_value >= 0.3)
            counts["bbox_iou_0_5"] += int(iou_value >= 0.5)
            counts["pointing_hit"] += int(pointing_hit)
        rows_out.append(
            {
                "img_id": img_id,
                "gold": gold,
                "prediction": pred,
                "parsed": parsed,
                "correct": ok,
                "bbox_iou": iou_value,
                "pointing_hit": pointing_hit,
                "difficulty_score": ann.get("difficulty_score"),
                "difficulty_tags": ann.get("difficulty_tags"),
                "expected_failure_stage": ann.get("expected_failure_stage"),
            }
        )

    total = counts["samples"]
    bbox_total = counts["bbox_samples"]

    def acc(counter: Counter) -> float:
        return counter["correct"] / counter["samples"] if counter["samples"] else 0.0

    report = {
        "stage": "sid_hard300_free_evaluate",
        "annotations": str(args.annotations),
        "outputs": str(args.outputs),
        "samples": total,
        "parse_rate": counts["parsed"] / total if total else 0.0,
        "prediction_accuracy": counts["correct"] / total if total else 0.0,
        "grounding": {
            "bbox_samples": bbox_total,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "iou_0_1": counts["bbox_iou_0_1"] / bbox_total if bbox_total else 0.0,
            "iou_0_3": counts["bbox_iou_0_3"] / bbox_total if bbox_total else 0.0,
            "iou_0_5": counts["bbox_iou_0_5"] / bbox_total if bbox_total else 0.0,
            "pointing_game": counts["pointing_hit"] / bbox_total if bbox_total else 0.0,
        },
        "by_label": {
            key: {"samples": value["samples"], "prediction_accuracy": acc(value)}
            for key, value in by_label.items()
        },
        "by_difficulty_score": {
            key: {"samples": value["samples"], "prediction_accuracy": acc(value)}
            for key, value in sorted(by_score.items())
        },
        "by_expected_failure_stage": {
            key: {"samples": value["samples"], "prediction_accuracy": acc(value)}
            for key, value in by_stage.items()
        },
        "by_difficulty_tag": {
            key: {"samples": value["samples"], "prediction_accuracy": acc(value)}
            for key, value in by_tag.items()
        },
        "confusion_matrix": {key: dict(value) for key, value in confusion.items()},
        "note": "SID-Hard lacks artifact/target attribution labels, so this report uses only classification and tampered grounding metrics.",
    }
    write_json(args.output, report)
    if args.rows_output:
        write_jsonl(args.rows_output, rows_out)
    write_text(args.summary_output, hard_free_summary(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def hard_free_summary(report: dict[str, Any]) -> str:
    g = report["grounding"]
    lines = [
        "# SID-Hard-300 Free Evidence Evaluation",
        "",
        report["note"],
        "",
        "| Samples | Parse | Pred Acc | BBox N | mean IoU | IoU@0.1 | IoU@0.3 | IoU@0.5 | Pointing |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {report['samples']} | {pct(report['parse_rate'])} | {pct(report['prediction_accuracy'])} | "
            f"{g['bbox_samples']} | {g['mean_iou']:.4f} | {pct(g['iou_0_1'])} | "
            f"{pct(g['iou_0_3'])} | {pct(g['iou_0_5'])} | {pct(g['pointing_game'])} |"
        ),
        "",
        "## By Label",
        "",
        "| Label | N | Pred Acc |",
        "|---|---:|---:|",
    ]
    for key, value in report["by_label"].items():
        lines.append(f"| {key} | {value['samples']} | {pct(value['prediction_accuracy'])} |")
    lines.extend(["", "## By Difficulty Score", "", "| Score | N | Pred Acc |", "|---|---:|---:|"])
    for key, value in report["by_difficulty_score"].items():
        lines.append(f"| {key} | {value['samples']} | {pct(value['prediction_accuracy'])} |")
    return "\n".join(lines) + "\n"


def lowlevel_summary(report: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| Method | N | mean IoU | IoU@0.1 | IoU@0.3 | IoU@0.5 | Pointing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["metrics"].items():
        lines.append(
            f"| {name} | {row['samples']} | {row['mean_iou']:.4f} | {pct(row['iou_0_1'])} | "
            f"{pct(row['iou_0_3'])} | {pct(row['iou_0_5'])} | {pct(row.get('pointing_game'))} |"
        )
    return "\n".join(lines) + "\n"


def casia_annotations(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.manifest)
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    if args.limit:
        rows = rows[: args.limit]
    out = []
    for row in rows:
        gt = mask_bbox(Path(row["mask_path"]))
        if not gt:
            continue
        with Image.open(row["image_path"]) as img:
            width, height = img.size
        out.append(
            {
                "annotation_task": "External-Mask",
                "annotation_status": "mask_only",
                "do_not_use_for_tuning": True,
                "img_id": row["img_id"],
                "human_label": "tampered",
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "width": width,
                "height": height,
                "mask_bbox": gt,
                "target_scope": "other",
                "dominant_artifact_type": "other",
                "short_note": "External mask-only sample; artifact and target labels are not manually annotated.",
            }
        )
    write_jsonl(args.output, out)
    report = {
        "stage": "external_mask_annotation_make",
        "manifest": str(args.manifest),
        "output": str(args.output),
        "samples": len(out),
        "note": "Artifact and target labels are placeholders and must not be reported as semantic accuracy.",
    }
    if args.summary_output:
        write_json(args.summary_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def consistency_audit(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.annotations)
    if args.limit:
        rows = rows[: args.limit]
    issues = []
    counters = Counter()
    seen = set()
    for row in rows:
        img_id = row.get("img_id")
        if img_id in seen:
            issues.append({"img_id": img_id, "issue": "duplicate_img_id"})
        seen.add(img_id)
        label = row.get("human_label")
        if row.get("annotation_task") == "SID-Hard":
            bbox = hard_gold_bbox(row)
        elif row.get("annotation_task") == "SID-Explain":
            items = row.get("evidence_items")
            item = items[0] if isinstance(items, list) and items else {}
            bbox = valid_bbox(item.get("evidence_bbox"))
        else:
            bbox = valid_bbox(row.get("mask_bbox"))
        if label == "tampered" and not bbox:
            issues.append({"img_id": img_id, "issue": "tampered_without_bbox"})
        if label != "tampered" and row.get("mask_path"):
            issues.append({"img_id": img_id, "issue": "non_tampered_with_mask_path"})
        tags = row.get("difficulty_tags") or []
        if row.get("annotation_task") == "SID-Hard":
            score = int(row.get("difficulty_score") or 0)
            if score == 0 and tags:
                issues.append({"img_id": img_id, "issue": "score_zero_with_tags"})
            if score > 0 and not tags:
                issues.append({"img_id": img_id, "issue": "positive_score_without_tags"})
            if row.get("expected_failure_stage") == "none" and score > 0:
                counters["none_stage_with_difficulty"] += 1
        counters["checked"] += 1

    report = {
        "stage": "annotation_consistency_audit",
        "annotations": str(args.annotations),
        "checked_samples": counters["checked"],
        "issue_count": len(issues),
        "issues_preview": issues[:50],
        "counters": dict(counters),
        "note": "This is an automated consistency audit, not inter-annotator agreement. True agreement requires a second independent annotator.",
    }
    write_json(args.output, report)
    lines = [
        "# Annotation Consistency Audit",
        "",
        report["note"],
        "",
        f"- Checked samples: {report['checked_samples']}",
        f"- Issue count: {report['issue_count']}",
        f"- none_stage_with_difficulty: {report['counters'].get('none_stage_with_difficulty', 0)}",
    ]
    write_text(args.summary_output, "\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def external_candidate_eval(args: argparse.Namespace) -> None:
    annotations = {row["img_id"]: row for row in read_jsonl(args.annotations)}
    requests = {row["img_id"]: row for row in read_jsonl(args.requests)} if args.requests else {}
    outputs = read_jsonl(args.outputs)
    counts = Counter()
    ious = []
    rows_out = []
    verdicts = Counter()
    for out in outputs:
        img_id = out.get("img_id")
        ann = annotations.get(img_id)
        if not ann:
            continue
        result = vlm_result(out)
        pred_value = result.get("prediction") or result.get("class") or result.get("evidence_mode")
        pred = normalize_prediction(pred_value)
        parsed = bool(result)
        req = requests.get(img_id, {})
        meta = req.get("metadata_for_evaluation_only") if isinstance(req, dict) else {}
        pred_bbox = valid_bbox(result.get("evidence_bbox"))
        candidate_bbox = valid_bbox(meta.get("candidate_bbox")) if isinstance(meta, dict) else None
        if pred == "tampered" and candidate_bbox:
            pred_bbox = candidate_bbox
        gt = valid_bbox(ann.get("mask_bbox"))
        iou_value = bbox_iou(pred_bbox, gt)
        point_hit = center_inside(pred_bbox, gt)
        counts["samples"] += 1
        counts["parsed"] += int(parsed)
        counts["tampered_pred"] += int(pred == "tampered")
        counts["iou_0_1"] += int(iou_value >= 0.1)
        counts["iou_0_3"] += int(iou_value >= 0.3)
        counts["iou_0_5"] += int(iou_value >= 0.5)
        counts["pointing_hit"] += int(point_hit)
        ious.append(iou_value)
        verdicts[normalize_text(result.get("candidate_verdict"))] += 1
        rows_out.append(
            {
                "img_id": img_id,
                "prediction": pred,
                "candidate_verdict": normalize_text(result.get("candidate_verdict")),
                "gt_bbox": gt,
                "pred_bbox": pred_bbox,
                "bbox_iou": iou_value,
                "pointing_hit": point_hit,
                "evidence_text": result.get("evidence_text"),
            }
        )

    total = counts["samples"]
    report = {
        "stage": "external_candidate_explanation_evaluate",
        "annotations": str(args.annotations),
        "outputs": str(args.outputs),
        "requests": str(args.requests) if args.requests else None,
        "samples": total,
        "parse_rate": counts["parsed"] / total if total else 0.0,
        "tampered_prediction_rate": counts["tampered_pred"] / total if total else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "iou_0_1": counts["iou_0_1"] / total if total else 0.0,
        "iou_0_3": counts["iou_0_3"] / total if total else 0.0,
        "iou_0_5": counts["iou_0_5"] / total if total else 0.0,
        "pointing_game": counts["pointing_hit"] / total if total else 0.0,
        "candidate_verdict_distribution": dict(verdicts),
        "note": "This external dataset has mask boxes but no manual artifact/target labels in this experiment; semantic attribution accuracy is intentionally not reported.",
    }
    write_json(args.output, report)
    if args.rows_output:
        write_jsonl(args.rows_output, rows_out)
    lines = [
        "# External Candidate Explanation Evaluation",
        "",
        report["note"],
        "",
        "| N | Parse | Tampered Rate | mean IoU | IoU@0.1 | IoU@0.3 | IoU@0.5 | Pointing |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {total} | {pct(report['parse_rate'])} | {pct(report['tampered_prediction_rate'])} | "
            f"{report['mean_iou']:.4f} | {pct(report['iou_0_1'])} | {pct(report['iou_0_3'])} | "
            f"{pct(report['iou_0_5'])} | {pct(report['pointing_game'])} |"
        ),
    ]
    write_text(args.summary_output, "\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    qc = subparsers.add_parser("hard-qc")
    qc.add_argument("--annotations", type=Path, required=True)
    qc.add_argument("--output", type=Path, required=True)
    qc.add_argument("--summary-output", type=Path, required=True)
    qc.set_defaults(func=hard_qc)

    low = subparsers.add_parser("hard-lowlevel")
    low.add_argument("--sid-root", type=Path, default=Path("SID_Set"))
    low.add_argument("--annotations", type=Path, required=True)
    low.add_argument("--output", type=Path, required=True)
    low.set_defaults(func=hard_lowlevel)

    lev = subparsers.add_parser("hard-evaluate-lowlevel")
    lev.add_argument("--annotations", type=Path, required=True)
    lev.add_argument("--lowlevel", type=Path, required=True)
    lev.add_argument("--output", type=Path, required=True)
    lev.add_argument("--summary-output", type=Path, required=True)
    lev.add_argument("--rows-output", type=Path)
    lev.set_defaults(func=hard_lowlevel_eval)

    hev = subparsers.add_parser("hard-evaluate-free")
    hev.add_argument("--annotations", type=Path, required=True)
    hev.add_argument("--outputs", type=Path, required=True)
    hev.add_argument("--output", type=Path, required=True)
    hev.add_argument("--summary-output", type=Path, required=True)
    hev.add_argument("--rows-output", type=Path)
    hev.set_defaults(func=hard_evaluate_free)

    casia = subparsers.add_parser("casia-annotations")
    casia.add_argument("--manifest", type=Path, required=True)
    casia.add_argument("--split")
    casia.add_argument("--limit", type=int, default=50)
    casia.add_argument("--output", type=Path, required=True)
    casia.add_argument("--summary-output", type=Path)
    casia.set_defaults(func=casia_annotations)

    audit = subparsers.add_parser("consistency-audit")
    audit.add_argument("--annotations", type=Path, required=True)
    audit.add_argument("--limit", type=int)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--summary-output", type=Path, required=True)
    audit.set_defaults(func=consistency_audit)

    extev = subparsers.add_parser("external-candidate-evaluate")
    extev.add_argument("--annotations", type=Path, required=True)
    extev.add_argument("--outputs", type=Path, required=True)
    extev.add_argument("--requests", type=Path)
    extev.add_argument("--output", type=Path, required=True)
    extev.add_argument("--summary-output", type=Path, required=True)
    extev.add_argument("--rows-output", type=Path)
    extev.set_defaults(func=external_candidate_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
