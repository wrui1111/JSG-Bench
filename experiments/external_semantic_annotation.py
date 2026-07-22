#!/usr/bin/env python3
"""Build and evaluate external semantic annotation templates.

This script does not create semantic labels. It prepares a frozen external
annotation subset and evaluates artifact/target metrics only after humans fill
the semantic fields.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TARGET_ALIASES = {
    "": "",
    "none": "none",
    "face": "face",
    "person": "person",
    "human": "person",
    "people": "person",
    "animal": "animal",
    "object": "object",
    "background": "background",
    "text": "text",
    "whole_image": "whole_image",
    "whole image": "whole_image",
    "other": "other",
}

ARTIFACT_ALIASES = {
    "": "",
    "none": "none",
    "boundary": "boundary_seam",
    "boundary_seam": "boundary_seam",
    "texture": "texture_smoothness",
    "texture_smoothness": "texture_smoothness",
    "lighting": "lighting_shadow",
    "lighting_shadow": "lighting_shadow",
    "geometry": "geometry_structure",
    "geometry_structure": "geometry_structure",
    "resolution": "resolution_noise",
    "resolution_noise": "resolution_noise",
    "semantic": "semantic_implausibility",
    "semantic_implausibility": "semantic_implausibility",
    "compression": "compression_artifact",
    "compression_artifact": "compression_artifact",
    "other": "other",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def valid_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def bbox_iou(a: Any, b: Any) -> float:
    box_a = valid_bbox(a)
    box_b = valid_bbox(b)
    if not box_a or not box_b:
        return 0.0
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


ALLOWED_TARGETS = ["person", "animal", "object", "background", "text", "face", "other"]
ALLOWED_ARTIFACTS = [
    "boundary_seam",
    "texture_smoothness",
    "lighting_shadow",
    "geometry_structure",
    "resolution_noise",
    "semantic_implausibility",
    "compression_artifact",
    "other",
]


def norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def norm_target(value: Any) -> str:
    return TARGET_ALIASES.get(norm(value), norm(value))


def norm_artifact(value: Any) -> str:
    return ARTIFACT_ALIASES.get(norm(value), norm(value))


def norm_artifact_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        artifact = norm_artifact(item)
        if artifact and artifact not in items:
            items.append(artifact)
    return items


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row["img_id"]: row for row in read_jsonl(path)}


def select_stratified(predictions: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Select a deterministic mix of low/mid/high bbox-overlap samples."""
    buckets: dict[str, list[dict[str, Any]]] = {"low": [], "mid": [], "high": []}
    for row in predictions:
        iou = float(row.get("bbox_iou") or 0.0)
        if iou >= 0.3:
            buckets["high"].append(row)
        elif iou >= 0.1:
            buckets["mid"].append(row)
        else:
            buckets["low"].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda item: str(item.get("img_id")))

    quota = {"low": n // 3, "mid": n // 3, "high": n - 2 * (n // 3)}
    selected: list[dict[str, Any]] = []
    for key in ["low", "mid", "high"]:
        selected.extend(buckets[key][: quota[key]])

    used = {row["img_id"] for row in selected}
    if len(selected) < n:
        leftovers = [row for key in ["low", "mid", "high"] for row in buckets[key] if row["img_id"] not in used]
        selected.extend(leftovers[: n - len(selected)])
    return selected[:n]


def make_template(args: argparse.Namespace) -> None:
    outputs = {
        "CASIA2": by_id(args.casia_outputs),
        "IMD2020": by_id(args.imd_outputs),
    }
    annotations = {
        "CASIA2": by_id(args.casia_annotations),
        "IMD2020": by_id(args.imd_annotations),
    }
    predictions = {
        "CASIA2": read_jsonl(args.casia_predictions),
        "IMD2020": read_jsonl(args.imd_predictions),
    }

    rows: list[dict[str, Any]] = []
    missing_assets = 0
    for dataset in ["CASIA2", "IMD2020"]:
        for pred in select_stratified(predictions[dataset], args.n_per_dataset):
            img_id = pred["img_id"]
            ann = annotations[dataset].get(img_id, {})
            out = outputs[dataset].get(img_id, {})
            image_paths = out.get("image_paths") if isinstance(out.get("image_paths"), list) else []
            marked = image_paths[0] if len(image_paths) >= 1 else ""
            crop = image_paths[1] if len(image_paths) >= 2 else ""
            asset_missing = bool((marked and not Path(marked).exists()) or (crop and not Path(crop).exists()))
            missing_assets += int(asset_missing)
            rows.append(
                {
                    "annotation_task": "External-Semantic-100",
                    "external_dataset": dataset,
                    "annotation_status": "pending_human",
                    "do_not_use_for_tuning": True,
                    "img_id": img_id,
                    "human_label": "tampered",
                    "image_path": ann.get("image_path") or out.get("image_path"),
                    "mask_path": ann.get("mask_path"),
                    "candidate_marked_path": marked,
                    "candidate_crop_path": crop,
                    "width": ann.get("width"),
                    "height": ann.get("height"),
                    "mask_bbox": ann.get("mask_bbox") or pred.get("gt_bbox"),
                    "candidate_bbox": pred.get("pred_bbox"),
                    "bbox_iou_mask_only": pred.get("bbox_iou"),
                    "pointing_mask_only": pred.get("pointing_hit"),
                    "target_scope": "",
                    "dominant_artifact_type": "",
                    "secondary_artifact_types": [],
                    "artifact_strength": "",
                    "violated_rule": "",
                    "short_note": "",
                    "allowed_target_scope": ALLOWED_TARGETS,
                    "allowed_artifact_types": ALLOWED_ARTIFACTS,
                    "annotation_guideline": (
                        "Fill target_scope and dominant_artifact_type from the image/mask/candidate evidence. "
                        "Do not inspect model artifact/target predictions while annotating."
                    ),
                    "asset_missing": asset_missing,
                }
            )

    rows.sort(key=lambda item: (item["external_dataset"], item["img_id"]))
    write_jsonl(args.output, rows)
    if args.csv_output:
        write_csv(args.csv_output, rows)

    dist = Counter(row["external_dataset"] for row in rows)
    report = {
        "stage": "external_semantic_template",
        "output": str(args.output),
        "csv_output": str(args.csv_output) if args.csv_output else None,
        "samples": len(rows),
        "by_dataset": dict(dist),
        "missing_candidate_assets": missing_assets,
        "status": "pending_human_annotation",
        "policy": "This template must not be used for prompt or threshold tuning. Semantic accuracy must not be reported until target/artifact labels are filled by humans.",
    }
    write_json(args.summary_output, report)
    write_summary(args.summary_md, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "external_dataset",
        "img_id",
        "annotation_status",
        "image_path",
        "mask_path",
        "candidate_marked_path",
        "candidate_crop_path",
        "mask_bbox",
        "candidate_bbox",
        "bbox_iou_mask_only",
        "target_scope",
        "dominant_artifact_type",
        "secondary_artifact_types",
        "artifact_strength",
        "violated_rule",
        "short_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), list) else row.get(key) for key in fields})


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# External Semantic Annotation Template",
        "",
        report["policy"],
        "",
        f"- Samples: {report['samples']}",
        f"- By dataset: {report['by_dataset']}",
        f"- Missing candidate assets: {report['missing_candidate_assets']}",
        f"- Status: {report['status']}",
        "",
        "Human fields to fill: `target_scope`, `dominant_artifact_type`, `secondary_artifact_types`, `artifact_strength`, `violated_rule`, `short_note`, and `annotation_status=completed`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    labels = read_jsonl(args.annotations)
    outputs: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    output_sources = [args.outputs] if args.outputs else [args.casia_outputs, args.imd_outputs]
    prediction_sources = [args.predictions] if args.predictions else [args.casia_predictions, args.imd_predictions]
    for path in output_sources:
        outputs.update(by_id(path))
    for path in prediction_sources:
        predictions.update(by_id(path))

    counts = Counter()
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    rows_out: list[dict[str, Any]] = []
    ious: list[float] = []
    unlabeled: list[str] = []

    for ann in labels:
        dataset = ann.get("external_dataset", "unknown")
        img_id = ann["img_id"]
        target = norm_target(ann.get("target_scope"))
        artifact = norm_artifact(ann.get("dominant_artifact_type"))
        secondary_artifacts = norm_artifact_list(ann.get("secondary_artifact_types"))
        acceptable_artifacts = [artifact] + [item for item in secondary_artifacts if item != artifact]
        completed = ann.get("annotation_status") == "completed" and target != "" and artifact != ""
        if not completed:
            unlabeled.append(img_id)
            continue

        out = outputs.get(img_id, {})
        result = out.get("vlm_result") if isinstance(out.get("vlm_result"), dict) else {}
        pred_target = norm_target(result.get("target_scope"))
        pred_artifact = norm_artifact(result.get("artifact_type"))
        pred = predictions.get(img_id, {})
        pred_bbox = valid_bbox(pred.get("pred_bbox") or result.get("evidence_bbox"))
        gt_bbox = valid_bbox(ann.get("mask_bbox"))
        iou = bbox_iou(pred_bbox, gt_bbox)
        ious.append(iou)

        artifact_ok = pred_artifact == artifact
        artifact_any_ok = pred_artifact in acceptable_artifacts
        target_ok = pred_target == target
        counts["samples"] += 1
        counts["artifact_correct"] += int(artifact_ok)
        counts["artifact_primary_or_secondary_correct"] += int(artifact_any_ok)
        counts["target_correct"] += int(target_ok)
        counts["iou_0_1"] += int(iou >= 0.1)
        counts["iou_0_3"] += int(iou >= 0.3)
        counts["iou_0_5"] += int(iou >= 0.5)
        for key, ok in [
            ("samples", True),
            ("artifact_correct", artifact_ok),
            ("artifact_primary_or_secondary_correct", artifact_any_ok),
            ("target_correct", target_ok),
            ("iou_0_1", iou >= 0.1),
            ("iou_0_3", iou >= 0.3),
            ("iou_0_5", iou >= 0.5),
        ]:
            by_dataset[dataset][key] += int(ok)

        rows_out.append(
            {
                "external_dataset": dataset,
                "img_id": img_id,
                "gold_artifact": artifact,
                "gold_secondary_artifacts": secondary_artifacts,
                "gold_acceptable_artifacts": acceptable_artifacts,
                "pred_artifact": pred_artifact,
                "artifact_correct": artifact_ok,
                "artifact_primary_or_secondary_correct": artifact_any_ok,
                "gold_target": target,
                "pred_target": pred_target,
                "target_correct": target_ok,
                "bbox_iou": iou,
            }
        )

    total = counts["samples"]
    status = "ready" if total else "pending_human_annotation"
    report = {
        "stage": "external_semantic_evaluate",
        "annotations": str(args.annotations),
        "outputs": [str(path) for path in output_sources],
        "predictions": [str(path) for path in prediction_sources],
        "status": status,
        "labeled_samples": total,
        "unlabeled_samples": len(unlabeled),
        "artifact_accuracy": counts["artifact_correct"] / total if total else None,
        "artifact_primary_accuracy": counts["artifact_correct"] / total if total else None,
        "artifact_primary_or_secondary_accuracy": counts["artifact_primary_or_secondary_correct"] / total if total else None,
        "target_accuracy": counts["target_correct"] / total if total else None,
        "mean_iou": sum(ious) / len(ious) if ious else None,
        "iou_0_1": counts["iou_0_1"] / total if total else None,
        "iou_0_3": counts["iou_0_3"] / total if total else None,
        "iou_0_5": counts["iou_0_5"] / total if total else None,
        "by_dataset": {
            key: {
                "samples": value["samples"],
                "artifact_accuracy": value["artifact_correct"] / value["samples"] if value["samples"] else None,
                "artifact_primary_accuracy": value["artifact_correct"] / value["samples"] if value["samples"] else None,
                "artifact_primary_or_secondary_accuracy": value["artifact_primary_or_secondary_correct"] / value["samples"] if value["samples"] else None,
                "target_accuracy": value["target_correct"] / value["samples"] if value["samples"] else None,
                "iou_0_1": value["iou_0_1"] / value["samples"] if value["samples"] else None,
                "iou_0_3": value["iou_0_3"] / value["samples"] if value["samples"] else None,
                "iou_0_5": value["iou_0_5"] / value["samples"] if value["samples"] else None,
            }
            for key, value in by_dataset.items()
        },
        "policy": "Semantic artifact/target metrics are reportable only when labeled_samples > 0 and labels were not used for tuning.",
    }
    write_json(args.output, report)
    if args.rows_output:
        write_jsonl(args.rows_output, rows_out)
    write_eval_summary(args.summary_md, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def write_eval_summary(path: Path, report: dict[str, Any]) -> None:
    mean_iou = "-" if report["mean_iou"] is None else f"{report['mean_iou']:.4f}"
    lines = [
        "# External Semantic Evaluation",
        "",
        report["policy"],
        "",
        f"- Status: {report['status']}",
        f"- Labeled samples: {report['labeled_samples']}",
        f"- Unlabeled samples: {report['unlabeled_samples']}",
        "",
        "| Scope | N | Artifact Primary Acc | Artifact Primary/Secondary Acc | Target Acc | mean IoU | IoU@0.1 | IoU@0.3 | IoU@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| all | {report['labeled_samples']} | {pct(report['artifact_primary_accuracy'])} | "
            f"{pct(report['artifact_primary_or_secondary_accuracy'])} | "
            f"{pct(report['target_accuracy'])} | {mean_iou} | "
            f"{pct(report['iou_0_1'])} | {pct(report['iou_0_3'])} | {pct(report['iou_0_5'])} |"
        ),
    ]
    for dataset, value in report["by_dataset"].items():
        lines.append(
            f"| {dataset} | {value['samples']} | {pct(value['artifact_primary_accuracy'])} | "
            f"{pct(value['artifact_primary_or_secondary_accuracy'])} | "
            f"{pct(value['target_accuracy'])} | - | {pct(value['iou_0_1'])} | "
            f"{pct(value['iou_0_3'])} | {pct(value['iou_0_5'])} |"
        )
    if report["status"] == "pending_human_annotation":
        lines.extend(
            [
                "",
                "No external semantic accuracy is reported yet because the template has not been human-completed.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-template")
    build.add_argument("--casia-annotations", type=Path, default=Path("experiments/runs/casia2_candidate996_annotations.jsonl"))
    build.add_argument("--casia-outputs", type=Path, default=Path("experiments/runs/casia2_candidate996_top3_qwen25vl_outputs.jsonl"))
    build.add_argument("--casia-predictions", type=Path, default=Path("experiments/runs/casia2_candidate996_top3_predictions.jsonl"))
    build.add_argument("--imd-annotations", type=Path, default=Path("experiments/runs/imd2020_candidate2010_annotations.jsonl"))
    build.add_argument("--imd-outputs", type=Path, default=Path("experiments/runs/imd2020_candidate2010_top3_qwen25vl_outputs.jsonl"))
    build.add_argument("--imd-predictions", type=Path, default=Path("experiments/runs/imd2020_candidate2010_top3_predictions.jsonl"))
    build.add_argument("--n-per-dataset", type=int, default=50)
    build.add_argument("--output", type=Path, default=Path("external/annotations/External-Semantic-100_template.jsonl"))
    build.add_argument("--csv-output", type=Path, default=Path("external/annotations/External-Semantic-100_template.csv"))
    build.add_argument("--summary-output", type=Path, default=Path("experiments/reports/external_semantic_template.json"))
    build.add_argument("--summary-md", type=Path, default=Path("experiments/reports/external_semantic_template.md"))
    build.set_defaults(func=make_template)

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--annotations", type=Path, default=Path("external/annotations/External-Semantic-100_template.jsonl"))
    eval_parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("experiments/runs/jsg_xfer_td_egfa_native_outputs_v2.jsonl"),
        help="Canonical JSG-Xfer VLM output JSONL.",
    )
    eval_parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("experiments/runs/jsg_xfer_td_egfa_predictions_v2.jsonl"),
        help="Canonical JSG-Xfer prediction JSONL.",
    )
    eval_parser.add_argument("--casia-outputs", type=Path, default=Path("experiments/runs/casia2_candidate996_top3_qwen25vl_outputs.jsonl"))
    eval_parser.add_argument("--casia-predictions", type=Path, default=Path("experiments/runs/casia2_candidate996_top3_predictions.jsonl"))
    eval_parser.add_argument("--imd-outputs", type=Path, default=Path("experiments/runs/imd2020_candidate2010_top3_qwen25vl_outputs.jsonl"))
    eval_parser.add_argument("--imd-predictions", type=Path, default=Path("experiments/runs/imd2020_candidate2010_top3_predictions.jsonl"))
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--summary-md", type=Path, required=True)
    eval_parser.add_argument("--rows-output", type=Path)
    eval_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
