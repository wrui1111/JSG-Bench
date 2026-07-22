#!/usr/bin/env python3
"""Evidence-grounded explanation experiment for SID annotation subsets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PREDICTION_ALIASES = {
    "real": "real",
    "authentic": "real",
    "genuine": "real",
    "full_synthetic": "full_synthetic",
    "fully_synthetic": "full_synthetic",
    "synthetic": "full_synthetic",
    "ai_generated": "full_synthetic",
    "ai-generated": "full_synthetic",
    "tampered": "tampered",
    "tampered_suspicious": "tampered",
    "manipulated": "tampered",
    "local_tampering": "tampered",
    "local_tampering_evidence": "tampered",
    "global_generation": "full_synthetic",
    "global_generation_evidence": "full_synthetic",
    "no_forensic_evidence": "real",
    "none": "real",
}

ARTIFACT_ALIASES = {
    None: "none",
    "": "none",
    "null": "none",
    "none": "none",
    "boundary_seam": "boundary_seam",
    "texture_smoothness": "texture_smoothness",
    "lighting_shadow": "lighting_shadow",
    "geometry_structure": "geometry_structure",
    "resolution_noise": "resolution_noise",
    "semantic_implausibility": "semantic_implausibility",
    "compression_artifact": "compression_artifact",
    "other": "other",
}

TARGET_ALIASES = {
    None: "none",
    "": "none",
    "null": "none",
    "none": "none",
    "face": "face",
    "person": "person",
    "human": "person",
    "body": "person",
    "person_body": "person",
    "whole_person": "person",
    "animal": "animal",
    "hand": "object",
    "object": "object",
    "background": "background",
    "text": "text",
    "whole_image": "whole_image",
    "whole image": "whole_image",
    "global": "whole_image",
    "other": "other",
}


PROMPT = """You are a forensic vision-language analyst.

Task: classify the image and provide a grounded evidence explanation.

Return valid JSON only:
{
  "prediction": "real | full_synthetic | tampered",
  "confidence": 0.0,
  "evidence_bbox": [x1, y1, x2, y2],
  "target_scope": "none | face | person | animal | object | background | text | whole_image | other",
  "artifact_type": "none | boundary_seam | texture_smoothness | lighting_shadow | geometry_structure | resolution_noise | semantic_implausibility | compression_artifact | other",
  "evidence_text": "one short sentence"
}

Definitions:
- real: a normal camera photograph with no clear AI-generation or local tampering evidence.
- full_synthetic: the whole image appears AI-generated; evidence may be global.
- tampered: one local region appears inserted, edited, replaced, or inconsistent with nearby context.

Evidence rules:
- evidence_bbox is the region that best supports your decision.
- For real, set evidence_bbox to null, target_scope to "none", artifact_type to "none".
- For full_synthetic, use target_scope "whole_image"; evidence_bbox may be null if the evidence is global.
- For tampered, evidence_bbox should roughly cover the suspicious local region.
- Do not use or mention any dataset label, filename, split, or mask.
- Output JSON only."""

PROMPT_V2 = """You are a forensic vision-language analyst.

Task: classify the image and provide a grounded evidence explanation.

Output exactly one JSON object with these keys:
- prediction
- confidence
- evidence_bbox
- target_scope
- artifact_type
- evidence_text

Allowed prediction values:
- real
- full_synthetic
- tampered

Allowed target_scope values:
- none
- face
- person
- animal
- object
- background
- text
- whole_image
- other

Allowed artifact_type values:
- none
- boundary_seam
- texture_smoothness
- lighting_shadow
- geometry_structure
- resolution_noise
- semantic_implausibility
- compression_artifact
- other

Decision rules:
1. If the image is a normal camera photograph with no clear AI-generation or local tampering evidence, set:
   prediction = real
   evidence_bbox = null
   target_scope = none
   artifact_type = none
2. If the whole image appears AI-generated, set:
   prediction = full_synthetic
   target_scope = whole_image
   evidence_bbox = null unless one specific region is the clearest evidence.
3. If one local region appears inserted, edited, replaced, or inconsistent with nearby context, set:
   prediction = tampered
   evidence_bbox = a rough box around that local suspicious region.

Strict output constraints:
- prediction must be exactly one value, not a list and not a string containing multiple choices.
- Do not output placeholder text such as "one short sentence".
- For real images, evidence_bbox must be null.
- Do not use or mention any dataset label, filename, split, or mask.
- Output JSON only, with no markdown fence."""

PROMPT_V3 = """You are a forensic vision-language analyst.

Task: find the strongest forensic evidence in the image. Do not directly solve a
three-class classification task. Focus on whether the visual evidence is absent,
global, or localized.

Output exactly one JSON object with these keys:
- evidence_mode
- confidence
- evidence_bbox
- target_scope
- artifact_type
- evidence_text

Allowed evidence_mode values:
- none
- global_generation
- local_tampering

Allowed target_scope values:
- none
- face
- person
- animal
- object
- background
- text
- whole_image
- other

Allowed artifact_type values:
- none
- boundary_seam
- texture_smoothness
- lighting_shadow
- geometry_structure
- resolution_noise
- semantic_implausibility
- compression_artifact
- other

Evidence rules:
1. Use evidence_mode = none when there is no clear forensic evidence. Then set:
   evidence_bbox = null
   target_scope = none
   artifact_type = none
2. Use evidence_mode = global_generation when the strongest evidence is distributed across the whole image, such as global AI-generation texture, unnatural semantics, or overall synthetic appearance. Then set:
   target_scope = whole_image
   evidence_bbox = null unless one region is clearly the strongest evidence.
3. Use evidence_mode = local_tampering only when one concrete local region is more suspicious than nearby context. Then set:
   evidence_bbox = a rough box around that local suspicious region.

Strict output constraints:
- evidence_mode must be exactly one value.
- Do not output a prediction key.
- Do not output placeholder text such as "one short sentence".
- Do not use or mention any dataset label, filename, split, or mask.
- Output JSON only, with no markdown fence."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_prediction(value: Any) -> str:
    return PREDICTION_ALIASES.get(normalize_text(value), "unknown")


def normalize_artifact(value: Any) -> str:
    return ARTIFACT_ALIASES.get(normalize_text(value), normalize_text(value) or "none")


def normalize_target(value: Any) -> str:
    return TARGET_ALIASES.get(normalize_text(value), normalize_text(value) or "none")


def valid_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_iou(a: list[int] | None, b: list[int] | None) -> float:
    if not a or not b:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def center_inside(pred: list[int] | None, gt: list[int] | None) -> bool:
    if not pred or not gt:
        return False
    cx = (pred[0] + pred[2]) / 2.0
    cy = (pred[1] + pred[3]) / 2.0
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def load_vlm_result(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("vlm_result"), dict):
        return row["vlm_result"]
    if isinstance(row.get("vlm_raw_text"), str):
        return extract_json_object(row["vlm_raw_text"]) or {}
    return {}


def gold_item(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("annotation_task") == "SID-FA":
        return {
            "prediction": normalize_prediction(row.get("human_label")),
            "artifact_type": normalize_artifact(row.get("dominant_artifact_type")),
            "target_scope": normalize_target(row.get("target_scope")),
            "evidence_bbox": valid_bbox(row.get("mask_bbox")),
            "evidence_text": row.get("short_note"),
        }
    if row.get("annotation_task") == "SID-Hard":
        auto = row.get("auto_features")
        auto_bbox = auto.get("mask_bbox") if isinstance(auto, dict) else None
        return {
            "prediction": normalize_prediction(row.get("human_label")),
            "artifact_type": "none",
            "target_scope": "none",
            "evidence_bbox": valid_bbox(row.get("mask_bbox")) or valid_bbox(auto_bbox),
            "evidence_text": row.get("human_note"),
        }
    items = row.get("evidence_items")
    item = items[0] if isinstance(items, list) and items else {}
    return {
        "prediction": normalize_prediction(row.get("human_label")),
        "artifact_type": normalize_artifact(row.get("dominant_artifact_type") or item.get("artifact_type")),
        "target_scope": normalize_target(row.get("target_scope")),
        "evidence_bbox": valid_bbox(item.get("evidence_bbox")),
        "evidence_text": item.get("evidence_text"),
    }


def prediction_item(row: dict[str, Any]) -> dict[str, Any]:
    result = load_vlm_result(row)
    prediction_value = result.get("prediction") or result.get("class") or result.get("evidence_mode")
    return {
        "prediction": normalize_prediction(prediction_value),
        "evidence_mode": normalize_text(result.get("evidence_mode")),
        "artifact_type": normalize_artifact(result.get("artifact_type")),
        "target_scope": normalize_target(result.get("target_scope")),
        "evidence_bbox": valid_bbox(result.get("evidence_bbox")),
        "evidence_text": result.get("evidence_text"),
        "confidence": result.get("confidence"),
        "parsed": bool(result),
    }


def make_requests(args: argparse.Namespace) -> None:
    sid_root = args.sid_root.resolve()
    annotations = read_jsonl(args.annotations)
    if args.prompt_version == "evidence_explanation_v3":
        prompt = PROMPT_V3
    elif args.prompt_version == "evidence_explanation_v2":
        prompt = PROMPT_V2
    else:
        prompt = PROMPT
    requests = []
    for row in annotations:
        requests.append(
            {
                "task_id": f"evidence_explanation_{row['img_id']}",
                "img_id": row["img_id"],
                "prompt_version": args.prompt_version,
                "image_path": str(sid_root / row["image_path"]),
                "metadata_for_evaluation_only": {
                    "human_label": row.get("human_label"),
                    "width": row.get("width"),
                    "height": row.get("height"),
                },
                "prompt": prompt,
            }
        )
    write_jsonl(args.output, requests)
    summary = {
        "stage": "evidence_explanation_make_requests",
        "annotations": str(args.annotations),
        "output": str(args.output),
        "total": len(requests),
    }
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    annotations = {row["img_id"]: row for row in read_jsonl(args.annotations)}
    outputs = read_jsonl(args.outputs)
    rows_out = []
    counts = Counter()
    by_class = {name: Counter() for name in ["real", "full_synthetic", "tampered"]}
    ious = []
    pointing_hits = []
    confusion = {name: Counter() for name in ["real", "full_synthetic", "tampered"]}

    for out in outputs:
        img_id = out.get("img_id")
        ann = annotations.get(img_id)
        if not ann:
            continue
        gold = gold_item(ann)
        pred = prediction_item(out)
        cls = gold["prediction"]
        counts["samples"] += 1
        counts["parsed"] += int(pred["parsed"])
        pred_ok = pred["prediction"] == gold["prediction"]
        artifact_ok = pred["artifact_type"] == gold["artifact_type"]
        target_ok = pred["target_scope"] == gold["target_scope"]
        counts["prediction_correct"] += int(pred_ok)
        counts["artifact_correct"] += int(artifact_ok)
        counts["target_correct"] += int(target_ok)
        by_class[cls]["samples"] += 1
        by_class[cls]["prediction_correct"] += int(pred_ok)
        by_class[cls]["artifact_correct"] += int(artifact_ok)
        by_class[cls]["target_correct"] += int(target_ok)
        confusion[cls][pred["prediction"]] += 1

        bbox_iou_value = None
        pointing_hit = None
        if gold["evidence_bbox"] is not None:
            bbox_iou_value = bbox_iou(pred["evidence_bbox"], gold["evidence_bbox"])
            pointing_hit = center_inside(pred["evidence_bbox"], gold["evidence_bbox"])
            ious.append(bbox_iou_value)
            pointing_hits.append(pointing_hit)
            counts["bbox_iou_0_1"] += int(bbox_iou_value >= 0.1)
            counts["bbox_iou_0_3"] += int(bbox_iou_value >= 0.3)
            counts["pointing_hit"] += int(pointing_hit)
            counts["bbox_samples"] += 1
            by_class[cls]["bbox_samples"] += 1
            by_class[cls]["bbox_iou_0_1"] += int(bbox_iou_value >= 0.1)
            by_class[cls]["bbox_iou_0_3"] += int(bbox_iou_value >= 0.3)
            by_class[cls]["pointing_hit"] += int(pointing_hit)
        elif pred["evidence_bbox"] is None:
            counts["null_bbox_correct"] += 1
            by_class[cls]["null_bbox_correct"] += 1
        counts["null_bbox_expected"] += int(gold["evidence_bbox"] is None)

        rows_out.append(
            {
                "img_id": img_id,
                "gold": gold,
                "prediction": pred,
                "prediction_correct": pred_ok,
                "artifact_correct": artifact_ok,
                "target_correct": target_ok,
                "bbox_iou": bbox_iou_value,
                "pointing_hit": pointing_hit,
            }
        )

    total = counts["samples"]
    bbox_total = counts["bbox_samples"]
    report = {
        "stage": "evidence_explanation_evaluate",
        "annotations": str(args.annotations),
        "outputs": str(args.outputs),
        "samples": total,
        "parse_rate": counts["parsed"] / total if total else 0.0,
        "prediction_accuracy": counts["prediction_correct"] / total if total else 0.0,
        "artifact_type_accuracy": counts["artifact_correct"] / total if total else 0.0,
        "target_scope_accuracy": counts["target_correct"] / total if total else 0.0,
        "null_bbox_accuracy_when_expected": counts["null_bbox_correct"] / counts["null_bbox_expected"]
        if counts["null_bbox_expected"]
        else None,
        "grounding": {
            "bbox_samples": bbox_total,
            "mean_iou": sum(ious) / bbox_total if bbox_total else 0.0,
            "iou_0_1": counts["bbox_iou_0_1"] / bbox_total if bbox_total else 0.0,
            "iou_0_3": counts["bbox_iou_0_3"] / bbox_total if bbox_total else 0.0,
            "pointing_game": counts["pointing_hit"] / bbox_total if bbox_total else 0.0,
        },
        "by_class": {
            cls: {
                "samples": c["samples"],
                "prediction_accuracy": c["prediction_correct"] / c["samples"] if c["samples"] else 0.0,
                "artifact_type_accuracy": c["artifact_correct"] / c["samples"] if c["samples"] else 0.0,
                "target_scope_accuracy": c["target_correct"] / c["samples"] if c["samples"] else 0.0,
                "bbox_iou_0_1": c["bbox_iou_0_1"] / c["bbox_samples"] if c["bbox_samples"] else None,
                "pointing_game": c["pointing_hit"] / c["bbox_samples"] if c["bbox_samples"] else None,
            }
            for cls, c in by_class.items()
        },
        "confusion_matrix": {cls: dict(counter) for cls, counter in confusion.items()},
    }
    write_json(args.output, report)
    if args.predictions_output:
        write_jsonl(args.predictions_output, rows_out)
    if args.summary_output:
        write_summary(args.summary_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def write_summary(path: Path, report: dict[str, Any]) -> None:
    g = report["grounding"]
    lines = [
        "# Evidence Explanation Evaluation",
        "",
        f"Outputs: `{report['outputs']}`",
        "",
        "| Samples | Parse | Prediction Acc | Artifact Acc | Target Acc | Null BBox Acc | BBox IoU@0.1 | Pointing |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {report['samples']} | {pct(report['parse_rate'])} | {pct(report['prediction_accuracy'])} | "
            f"{pct(report['artifact_type_accuracy'])} | {pct(report['target_scope_accuracy'])} | "
            f"{pct(report['null_bbox_accuracy_when_expected'])} | {pct(g['iou_0_1'])} | {pct(g['pointing_game'])} |"
        ),
        "",
        "## By Class",
        "",
        "| Class | N | Prediction Acc | Artifact Acc | Target Acc | BBox IoU@0.1 | Pointing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cls, row in report["by_class"].items():
        lines.append(
            f"| {cls} | {row['samples']} | {pct(row['prediction_accuracy'])} | "
            f"{pct(row['artifact_type_accuracy'])} | {pct(row['target_scope_accuracy'])} | "
            f"{pct(row['bbox_iou_0_1'])} | {pct(row['pointing_game'])} |"
        )
    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "| GT | pred real | pred full_synthetic | pred tampered | pred unknown |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cls, row in report["confusion_matrix"].items():
        lines.append(
            f"| {cls} | {row.get('real', 0)} | {row.get('full_synthetic', 0)} | "
            f"{row.get('tampered', 0)} | {row.get('unknown', 0)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    make_parser = subparsers.add_parser("make-requests")
    make_parser.add_argument("--sid-root", type=Path, default=Path("SID_Set"))
    make_parser.add_argument("--annotations", type=Path, required=True)
    make_parser.add_argument(
        "--prompt-version",
        choices=["evidence_explanation_v1", "evidence_explanation_v2", "evidence_explanation_v3"],
        default="evidence_explanation_v1",
    )
    make_parser.add_argument("--output", type=Path, required=True)
    make_parser.add_argument("--summary-output", type=Path)
    make_parser.set_defaults(func=make_requests)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--annotations", type=Path, required=True)
    eval_parser.add_argument("--outputs", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--summary-output", type=Path)
    eval_parser.add_argument("--predictions-output", type=Path)
    eval_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
