#!/usr/bin/env python3
"""Compute joint grounding metrics from frozen evaluation outputs.

This script does not run model inference. It reads existing per-sample
prediction rows and reports whether classification, semantic explanation, and
coarse evidence bbox succeed on the same sample.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "experiments" / "reports"


def read_jsonl(rel: str) -> list[dict[str, Any]]:
    path = ROOT / rel
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def flt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def acceptable_artifacts(row: dict[str, Any]) -> set[str]:
    values = {str(row.get("dominant_artifact_type", "")).strip()}
    for item in row.get("secondary_artifact_types") or []:
        values.add(str(item).strip())
    values.discard("")
    return values


def summarize_joint(rows: list[dict[str, Any]], thresholds: tuple[float, ...] = (0.1, 0.3, 0.5)) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {
        "samples": n,
        "prediction_accuracy": mean([1.0 if row.get("prediction_correct") else 0.0 for row in rows]),
        "artifact_primary_accuracy": mean([1.0 if row.get("artifact_correct") else 0.0 for row in rows]),
        "artifact_primary_or_secondary_accuracy": mean(
            [1.0 if row.get("artifact_primary_or_secondary_correct", row.get("artifact_correct")) else 0.0 for row in rows]
        ),
        "target_accuracy": mean([1.0 if row.get("target_correct") else 0.0 for row in rows]),
        "mean_iou": mean([float(row.get("bbox_iou") or 0.0) for row in rows]),
    }
    for threshold in thresholds:
        suffix = str(threshold).replace(".", "_")
        out[f"bbox_iou_ge_{suffix}"] = sum(1 for row in rows if float(row.get("bbox_iou") or 0.0) >= threshold) / n if n else None
        out[f"joint_primary_iou_ge_{suffix}"] = (
            sum(
                1
                for row in rows
                if row.get("prediction_correct")
                and row.get("artifact_correct")
                and row.get("target_correct")
                and float(row.get("bbox_iou") or 0.0) >= threshold
            )
            / n
            if n
            else None
        )
        out[f"joint_primary_or_secondary_iou_ge_{suffix}"] = (
            sum(
                1
                for row in rows
                if row.get("prediction_correct")
                and row.get("artifact_primary_or_secondary_correct", row.get("artifact_correct"))
                and row.get("target_correct")
                and float(row.get("bbox_iou") or 0.0) >= threshold
            )
            / n
            if n
            else None
        )
    return out


def by_dimension(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return {name: summarize_joint(items) for name, items in sorted(groups.items())}


def bbox_correct(row: dict[str, Any], threshold: float = 0.1) -> bool:
    return float(row.get("bbox_iou") or 0.0) >= threshold


def compact_case(row: dict[str, Any], reason: str) -> dict[str, Any]:
    gold = row.get("gold", {})
    pred = row.get("prediction", {})
    return {
        "img_id": row.get("img_id"),
        "dataset": row.get("dataset"),
        "reason": reason,
        "gold_prediction": gold.get("prediction"),
        "pred_prediction": pred.get("prediction"),
        "gold_artifact": row.get("gold_artifact", gold.get("artifact_type")),
        "pred_artifact": row.get("pred_artifact", pred.get("artifact_type")),
        "artifact_correct": bool(row.get("artifact_correct")),
        "gold_target": row.get("gold_target", gold.get("target_scope")),
        "pred_target": row.get("pred_target", pred.get("target_scope")),
        "target_correct": bool(row.get("target_correct")),
        "bbox_iou": float(row.get("bbox_iou") or 0.0),
        "gold_bbox": gold.get("evidence_bbox"),
        "pred_bbox": pred.get("evidence_bbox"),
        "gold_note": gold.get("evidence_text"),
        "pred_note": pred.get("evidence_text"),
    }


def motivating_examples(sid_explain: list[dict[str, Any]], sid_fa: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for rows, dataset in [(sid_explain, "SID-Explain-300"), (sid_fa, "SID-FA-600")]:
        semantic_ok_spatial_fail = [
            row
            for row in rows
            if row.get("prediction_correct")
            and row.get("artifact_correct")
            and row.get("target_correct")
            and not bbox_correct(row)
        ]
        if semantic_ok_spatial_fail:
            examples.append(
                compact_case(
                    semantic_ok_spatial_fail[0],
                    f"{dataset}: prediction, artifact and target are correct, but bbox IoU<0.1 so joint success fails.",
                )
            )

        spatial_ok_semantic_fail = [
            row
            for row in rows
            if row.get("prediction_correct")
            and bbox_correct(row)
            and (not row.get("artifact_correct") or not row.get("target_correct"))
        ]
        if spatial_ok_semantic_fail:
            examples.append(
                compact_case(
                    spatial_ok_semantic_fail[0],
                    f"{dataset}: prediction and bbox are acceptable, but artifact or target is wrong so joint success fails.",
                )
            )
    return examples


def sid_explain_candidate_rows() -> list[dict[str, Any]]:
    rows = read_jsonl("experiments/runs/jsg_core_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl")
    annotations = {
        row["img_id"]: row for row in read_jsonl("experiments/runs/sid_explain300_tampered100_annotations_revised.jsonl")
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        ann = annotations.get(row["img_id"], {})
        pred_artifact = row.get("prediction", {}).get("artifact_type")
        acceptable = acceptable_artifacts(ann)
        out.append(
            {
                **row,
                "dataset": "SID-Explain-300 tampered100",
                "gold_artifact": row.get("gold", {}).get("artifact_type"),
                "gold_target": row.get("gold", {}).get("target_scope"),
                "gold_secondary_artifacts": ann.get("secondary_artifact_types") or [],
                "pred_artifact": pred_artifact,
                "pred_target": row.get("prediction", {}).get("target_scope"),
                "artifact_primary_or_secondary_correct": pred_artifact in acceptable if acceptable else row.get("artifact_correct"),
            }
        )
    return out


def sid_fa_candidate_rows() -> list[dict[str, Any]]:
    rows = read_jsonl("experiments/runs/jsg_scale_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl")
    annotations = {row["img_id"]: row for row in read_jsonl("experiments/runs/sid_fa600_annotations_postreview_eval.jsonl")}
    out: list[dict[str, Any]] = []
    for row in rows:
        ann = annotations.get(row["img_id"], {})
        pred_artifact = row.get("prediction", {}).get("artifact_type")
        acceptable = acceptable_artifacts(ann)
        out.append(
            {
                **row,
                "dataset": "SID-FA-600",
                "mask_area_bucket": ann.get("mask_area_bucket"),
                "gold_artifact": row.get("gold", {}).get("artifact_type"),
                "gold_target": row.get("gold", {}).get("target_scope"),
                "gold_secondary_artifacts": ann.get("secondary_artifact_types") or [],
                "pred_artifact": pred_artifact,
                "pred_target": row.get("prediction", {}).get("target_scope"),
                "artifact_primary_or_secondary_correct": pred_artifact in acceptable if acceptable else row.get("artifact_correct"),
            }
        )
    return out


def external_semantic_rows() -> list[dict[str, Any]]:
    rows = read_jsonl("experiments/runs/jsg_xfer_semantic_rows_pixel5d_v2.jsonl")
    predictions = {
        row["img_id"]: row
        for row in read_jsonl("experiments/runs/jsg_xfer_predictions_pixel5d_v2.jsonl")
    }

    out: list[dict[str, Any]] = []
    for row in rows:
        pred = predictions.get(row["img_id"], {})
        prediction_correct = pred.get("prediction") == "tampered"
        out.append(
            {
                **row,
                "dataset": "External-Semantic-100",
                "prediction_correct": prediction_correct,
                "artifact_primary_or_secondary_correct": row.get("artifact_primary_or_secondary_correct"),
                "target_correct": row.get("target_correct"),
                "artifact_correct": row.get("artifact_correct"),
                "pred_class": pred.get("prediction"),
            }
        )
    return out


def sid_explain_free_all300() -> dict[str, Any]:
    rows = read_jsonl("experiments/runs/sid_explain300_evidence_explanation_v2_revised_predictions.jsonl")
    all_count = len(rows)
    class_appropriate = 0
    for row in rows:
        if not (row.get("prediction_correct") and row.get("artifact_correct") and row.get("target_correct")):
            continue
        gold_class = row.get("gold", {}).get("prediction")
        if gold_class == "tampered":
            class_appropriate += int(float(row.get("bbox_iou") or 0.0) >= 0.1)
        else:
            class_appropriate += int(row.get("prediction", {}).get("evidence_bbox") is None)
    tampered = [row for row in rows if row.get("gold", {}).get("prediction") == "tampered"]
    return {
        "samples": all_count,
        "class_appropriate_joint_success": class_appropriate / all_count if all_count else None,
        "tampered_joint_primary_iou_ge_0_1": summarize_joint(tampered)["joint_primary_iou_ge_0_1"],
        "note": (
            "For real/full_synthetic rows, class-appropriate joint success requires correct prediction, "
            "artifact, target, and null bbox. For tampered rows it requires IoU>=0.1."
        ),
    }


def sid_explain_free_tampered_breakdown() -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl("experiments/runs/sid_explain300_evidence_explanation_v2_revised_predictions.jsonl")
        if row.get("gold", {}).get("prediction") == "tampered"
    ]
    by_prediction: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"samples": 0, "target_correct": 0, "artifact_correct": 0, "bbox_iou_ge_0_1": 0}
    )
    for row in rows:
        pred_label = str(row.get("prediction", {}).get("prediction") or "unknown")
        bucket = by_prediction[pred_label]
        bucket["samples"] += 1
        bucket["target_correct"] += int(bool(row.get("target_correct")))
        bucket["artifact_correct"] += int(bool(row.get("artifact_correct")))
        bucket["bbox_iou_ge_0_1"] += int(float(row.get("bbox_iou") or 0.0) >= 0.1)

    for bucket in by_prediction.values():
        n = bucket["samples"]
        bucket["target_accuracy"] = bucket["target_correct"] / n if n else None
        bucket["artifact_accuracy"] = bucket["artifact_correct"] / n if n else None
        bucket["bbox_iou_ge_0_1_rate"] = bucket["bbox_iou_ge_0_1"] / n if n else None

    return {
        "samples": len(rows),
        "by_predicted_label": dict(sorted(by_prediction.items())),
        "overall_target_correct": sum(int(bool(row.get("target_correct"))) for row in rows),
        "overall_artifact_correct": sum(int(bool(row.get("artifact_correct"))) for row in rows),
        "note": (
            "Target Acc is field-level and is not conditioned on prediction correctness; many wrong-label "
            "free-prompt outputs still coincidentally predict object-like targets."
        ),
    }


def report_rows_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Dataset / Method | N | Pred Acc | Artifact Primary | Artifact Primary/Secondary | Target Acc | mean IoU | IoU@0.1 | Joint@0.1 Primary | Joint@0.1 Primary/Secondary | Joint@0.3 Primary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["sid_explain_candidate_top3", "sid_fa_candidate_top3", "external_semantic_100"]:
        row = report[key]["overall"]
        label = report[key]["label"]
        artifact_primary = pct(row["artifact_primary_accuracy"])
        if key == "sid_fa_candidate_top3":
            artifact_primary += "*"
        lines.append(
            f"| {label} | {row['samples']} | {pct(row['prediction_accuracy'])} | "
            f"{artifact_primary} | {pct(row['artifact_primary_or_secondary_accuracy'])} | "
            f"{pct(row['target_accuracy'])} | {flt(row['mean_iou'])} | {pct(row['bbox_iou_ge_0_1'])} | "
            f"{pct(row['joint_primary_iou_ge_0_1'])} | {pct(row['joint_primary_or_secondary_iou_ge_0_1'])} | "
            f"{pct(row['joint_primary_iou_ge_0_3'])} |"
        )
    return lines


def main() -> None:
    sid_explain = sid_explain_candidate_rows()
    sid_fa = sid_fa_candidate_rows()
    external = external_semantic_rows()

    report = {
        "stage": "joint_grounding_analysis",
        "definition": (
            "Joint success requires prediction_correct AND artifact_correct AND target_correct AND "
            "bbox IoU >= threshold. The primary/secondary variant counts a prediction as artifact-correct "
            "when it matches either the dominant artifact or one of the human-labeled secondary artifacts."
        ),
        "bbox_gt_sources": {
            "SID-Explain-300 tampered-only candidate top-3": (
                "human evidence_bbox; current tampered evidence boxes are mask-aligned with mask_bbox "
                "(mean IoU 1.0000 in mask/evidence audit)"
            ),
            "SID-FA-600 candidate top-3": "mask-derived bbox from SID-Set tampered masks",
            "External-Semantic-100 candidate top-3": "mask-derived bbox from CASIA2/IMD2020 masks",
        },
        "thresholds": [0.1, 0.3, 0.5],
        "sid_explain_free_all300": sid_explain_free_all300(),
        "sid_explain_free_tampered_breakdown": sid_explain_free_tampered_breakdown(),
        "sid_explain_candidate_top3": {
            "label": "SID-Explain-300 tampered-only candidate top-3",
            "overall": summarize_joint(sid_explain),
        },
        "sid_fa_candidate_top3": {
            "label": "SID-FA-600 candidate top-3",
            "overall": summarize_joint(sid_fa),
            "by_mask_area_bucket": by_dimension(sid_fa, "mask_area_bucket"),
        },
        "external_semantic_100": {
            "label": "External-Semantic-100 candidate top-3",
            "overall": summarize_joint(external),
            "by_dataset": by_dimension(external, "external_dataset"),
        },
        "motivating_examples": motivating_examples(sid_explain, sid_fa),
        "remaining_6000_note": (
            "remaining_6000 is retained as a negative three-class boundary experiment. It lacks human artifact "
            "and target labels, so the four-condition joint explanation metric is not applicable there."
        ),
    }

    json_path = REPORT_DIR / "joint_grounding_analysis.json"
    md_path = REPORT_DIR / "joint_grounding_analysis.md"
    write_json(json_path, report)

    lines = [
        "# Joint Grounding Analysis",
        "",
        "This report uses frozen per-sample outputs only. It does not run new VLM inference.",
        "",
        "Definition: `joint success = prediction correct ∧ artifact correct ∧ target correct ∧ bbox IoU >= t`. The broader variant counts a predicted artifact as correct if it matches either the dominant artifact or a labeled secondary artifact.",
        "",
        "Main Joint Success is computed only for rows with local evidence regions: tampered rows in SID-derived subsets and the external tampered semantic subset. `SID-Explain-300` real/full_synthetic rows are reported with separate prediction/null-bbox diagnostics and are not in the main joint denominator.",
        "",
        "BBox GT sources: `SID-Explain-300` uses human evidence bbox that is currently mask-aligned; `SID-FA-600` uses SID-Set mask-derived bbox; `External-Semantic-100` uses CASIA2/IMD2020 mask-derived bbox. The rows are therefore reported as region-level grounding checks, but not as pixel-level edit localization or an absolute cross-dataset difficulty ranking. External-Semantic-100 is stratified by candidate IoU, so its bbox IoU distribution is not a full external-dataset estimate.",
        "",
        "## Main Joint Metrics",
        "",
        *report_rows_table(report),
        "",
        "*`SID-FA-600` artifact majority baseline is 84.33 because `geometry_structure` accounts for 506/600 independently reviewed gold labels. The weighted artifact score should be read with macro/per-class diagnostics.",
        "",
        "Interpretation: separate artifact, target, and bbox scores substantially overstate explanation reliability when considered alone. Joint success is the stricter evidence-grounded metric and should be the main benchmark-facing number.",
        "",
        "## SID-Explain Free-Prompt Diagnostic",
        "",
        f"- Free all-300 class-appropriate joint success: {pct(report['sid_explain_free_all300']['class_appropriate_joint_success'])}.",
        f"- Free tampered-only joint@0.1 primary: {pct(report['sid_explain_free_all300']['tampered_joint_primary_iou_ge_0_1'])}.",
        f"- Note: {report['sid_explain_free_all300']['note']}",
        f"- Tampered100 target-field note: {report['sid_explain_free_tampered_breakdown']['note']}",
        "",
        "| Free tampered predicted label | N | Target hits | Target Acc | Artifact hits | Artifact Acc | IoU@0.1 hits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pred_label, row in report["sid_explain_free_tampered_breakdown"]["by_predicted_label"].items():
        lines.append(
            f"| {pred_label} | {row['samples']} | {row['target_correct']} | {pct(row['target_accuracy'])} | "
            f"{row['artifact_correct']} | {pct(row['artifact_accuracy'])} | {row['bbox_iou_ge_0_1']} |"
        )
    lines.extend(
        [
            "",
            "## SID-FA Joint@0.1 by Mask Area",
            "",
            "| Bucket | N | IoU@0.1 | Joint@0.1 Primary | Joint@0.1 Primary/Secondary |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bucket, row in report["sid_fa_candidate_top3"]["by_mask_area_bucket"].items():
        lines.append(
            f"| {bucket} | {row['samples']} | {pct(row['bbox_iou_ge_0_1'])} | "
            f"{pct(row['joint_primary_iou_ge_0_1'])} | {pct(row['joint_primary_or_secondary_iou_ge_0_1'])} |"
        )
    lines.extend(
        [
            "",
            "## External-Semantic-100 by Dataset",
            "",
            "| Dataset | N | Pred Acc | Joint@0.1 Primary | Joint@0.1 Primary/Secondary | Joint@0.3 Primary |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, row in report["external_semantic_100"]["by_dataset"].items():
        lines.append(
            f"| {dataset} | {row['samples']} | {pct(row['prediction_accuracy'])} | "
            f"{pct(row['joint_primary_iou_ge_0_1'])} | {pct(row['joint_primary_or_secondary_iou_ge_0_1'])} | "
            f"{pct(row['joint_primary_iou_ge_0_3'])} |"
        )
    lines.extend(
        [
            "",
            "## Motivating Joint-Failure Examples",
            "",
            "| Dataset | Image | Failure Type | Artifact | Target | BBox IoU | Why Joint Fails |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for row in report["motivating_examples"]:
        lines.append(
            f"| {row['dataset']} | {row['img_id']} | {row['reason'].split(': ', 1)[1]} | "
            f"{row['gold_artifact']} -> {row['pred_artifact']} | {row['gold_target']} -> {row['pred_target']} | "
            f"{row['bbox_iou']:.4f} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Applicability Note",
            "",
            report["remaining_6000_note"],
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
