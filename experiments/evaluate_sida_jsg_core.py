#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from stratified_statistics import (
    core_strata_by_id,
    design_metadata,
    mover_wilson_interval,
    stratified_percentile_interval_list,
    validate_expected_strata,
)


ROOT = Path(__file__).resolve().parents[1]
SIDA_DIR = ROOT / "remote_sida/jsg_core_native100"
NATIVE_OUTPUTS = SIDA_DIR / "native_outputs.jsonl"
PRIVATE_MAP = ROOT / "remote_sida/jsg_core_private/sample_map_DO_NOT_UPLOAD.jsonl"
GOLD_FILE = ROOT / "experiments/runs/sid_explain300_tampered100_annotations_revised.jsonl"
TD_FILE = ROOT / "experiments/runs/jsg_core_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl"
ADAPTED_FILE = ROOT / "experiments/runs/sida_jsg_core_adapted_predictions.jsonl"
REPORT_JSON = ROOT / "experiments/reports/sida_jsg_core_native_eval.json"
REPORT_MD = ROOT / "experiments/reports/sida_jsg_core_native_eval.md"
COMPARISON_CSV = ROOT / "experiments/reports/sida_td_jsg_core_system_comparison.csv"
OVERLAP_CSV = ROOT / "experiments/reports/sida_td_jsg_core_passing_overlap.csv"
FAILURE_CSV = ROOT / "experiments/reports/sida_td_jsg_core_failure_signatures.csv"
DIFFERENCE_CSV = ROOT / "experiments/reports/sida_td_jsg_core_paired_differences.csv"
CONFUSION_CSV = ROOT / "experiments/reports/sida_td_jsg_core_confusions.csv"

CONDITIONS = ["parse", "prediction", "artifact", "target", "spatial"]
TAG_TO_ARTIFACT = {
    "edges": "boundary_seam",
    "edge": "boundary_seam",
    "boundary": "boundary_seam",
    "seam": "boundary_seam",
    "lighting": "lighting_shadow",
    "light": "lighting_shadow",
    "shadows": "lighting_shadow",
    "shadow": "lighting_shadow",
    "resolution": "resolution_noise",
    "noise": "resolution_noise",
    "texture": "texture_smoothness",
    "smoothness": "texture_smoothness",
    "geometry": "geometry_structure",
    "structure": "geometry_structure",
    "semantic": "semantic_implausibility",
    "compression": "compression_artifact",
    "jpeg": "compression_artifact",
}
TARGET_KEYWORDS = {
    "face": ["face", "facial", "head"],
    "person": [
        "person", "people", "human", "man", "woman", "boy", "girl", "child", "skier",
        "body", "arm", "leg", "hand", "foot", "knee", "ankle", "torso", "shoulder",
    ],
    "animal": ["animal", "dog", "cat", "bird", "horse", "cow", "sheep", "bear"],
    "text": ["text", "letter", "word", "logo", "caption", "sign", "writing"],
    "background": ["background", "sky", "ground", "wall", "road", "landscape", "scene"],
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def native_prediction(row: dict) -> str | None:
    value = row.get("native_class_text_parse")
    if value in {"real", "full_synthetic", "tampered"}:
        return value
    text = " ".join(str(row.get("native_text_normalized") or "").lower().split())
    if "image is tampered" in text or "classified as tampered" in text or "[cls] is tampered" in text:
        return "tampered"
    if "fully synthetic" in text or "classified as synthetic" in text:
        return "full_synthetic"
    if "image is real" in text or "classified as real" in text or "authentic image" in text:
        return "real"
    return None


def section(text: str, start: str, end: str | None = None) -> str:
    lower = text.lower()
    start_index = lower.find(start.lower())
    if start_index < 0:
        return ""
    start_index += len(start)
    if end is None:
        return text[start_index:]
    end_index = lower.find(end.lower(), start_index)
    return text[start_index:] if end_index < 0 else text[start_index:end_index]


def target_from_text(text: str) -> tuple[str | None, str]:
    content = section(text, "Tampered Content:", "Visual Inconsistencies:").strip()
    normalized = " ".join(content.lower().split())
    if not normalized:
        return None, content
    for target in ["face", "person", "animal", "text", "background"]:
        if any(re.search(rf"\b{re.escape(keyword)}\b", normalized) for keyword in TARGET_KEYWORDS[target]):
            return target, content
    boilerplate = {"types of objects or parts", "object or part", "unknown", "none"}
    if normalized.strip(" .:{}<>") in boilerplate:
        return None, content
    return "object", content


def artifacts_from_text(text: str) -> tuple[str | None, list[str], list[dict]]:
    visual = section(text, "Visual Inconsistencies:")
    evidence = []
    artifacts = []
    for match in re.finditer(r"<([^>]+)>([^<]*)", visual):
        tag = " ".join(match.group(1).lower().split())
        content = " ".join(match.group(2).split())
        mapped = TAG_TO_ARTIFACT.get(tag)
        if mapped is None:
            for keyword, artifact in TAG_TO_ARTIFACT.items():
                if re.search(rf"\b{re.escape(keyword)}\b", tag):
                    mapped = artifact
                    break
        if mapped and content:
            evidence.append({"tag": tag, "content": content, "artifact": mapped})
            if mapped not in artifacts:
                artifacts.append(mapped)
            normalized_content = content.lower()
            for keyword, content_artifact in TAG_TO_ARTIFACT.items():
                if re.search(rf"\b{re.escape(keyword)}\b", normalized_content):
                    evidence.append(
                        {
                            "tag": tag,
                            "content": content,
                            "artifact": content_artifact,
                            "source": f"phrase:{keyword}",
                        }
                    )
                    if content_artifact not in artifacts:
                        artifacts.append(content_artifact)
    if not artifacts:
        normalized = " ".join(visual.lower().split())
        positions = []
        for keyword, artifact in TAG_TO_ARTIFACT.items():
            match = re.search(rf"\b{re.escape(keyword)}\b", normalized)
            if match:
                positions.append((match.start(), artifact, keyword))
        for _, artifact, keyword in sorted(positions):
            if artifact not in artifacts:
                artifacts.append(artifact)
                evidence.append({"tag": keyword, "content": "lexical fallback", "artifact": artifact})
    return (artifacts[0] if artifacts else None), artifacts, evidence


def resolve_mask_paths(row: dict) -> list[Path]:
    paths = []
    for raw_path in row.get("mask_files") or []:
        normalized = str(raw_path).replace("\\", "/")
        path = SIDA_DIR / normalized
        if path.exists():
            paths.append(path)
    return paths


def bbox_from_masks(paths: list[Path]) -> tuple[list[int] | None, int]:
    union = None
    for path in paths:
        mask = np.asarray(Image.open(path).convert("L")) > 0
        union = mask if union is None else np.logical_or(union, mask)
    if union is None or not union.any():
        return None, 0
    ys, xs = np.where(union)
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)], int(union.sum())


def bbox_iou(first: list[int] | None, second: list[int] | None) -> float:
    if first is None or second is None:
        return 0.0
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def gold_record(row: dict) -> dict:
    items = row.get("evidence_items") or []
    item = items[0] if items else {}
    return {
        "prediction": row.get("human_label"),
        "artifact_type": row.get("dominant_artifact_type") or item.get("artifact_type"),
        "target_scope": row.get("target_scope"),
        "evidence_bbox": item.get("evidence_bbox"),
    }


def condition_record(parse: bool, prediction: bool, artifact: bool, target: bool, spatial: bool) -> dict:
    values = {
        "parse": bool(parse),
        "prediction": bool(prediction),
        "artifact": bool(artifact),
        "target": bool(target),
        "spatial": bool(spatial),
    }
    failed = [condition for condition in CONDITIONS if not values[condition]]
    return {"conditions": values, "failed_conditions": failed, "joint": not failed}


def summarize(rows: list[dict], strata: list[str], prefix: str = "") -> dict:
    total = len(rows)
    keys = ["parse", "prediction", "artifact", "target", "spatial", "joint"]
    output = {"samples": total}
    for key in keys:
        if key == "joint":
            successes = sum(bool(row[f"{prefix}joint"]) for row in rows)
        else:
            successes = sum(bool(row[f"{prefix}conditions"][key]) for row in rows)
        output[key] = successes / total
        output[f"{key}_count"] = successes
        values = np.asarray(
            [
                float(row[f"{prefix}joint"])
                if key == "joint"
                else float(row[f"{prefix}conditions"][key])
                for row in rows
            ],
            dtype=np.float64,
        )
        output[f"{key}_mover_wilson_95"] = mover_wilson_interval(values, strata)
    return output


def bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    strata: list[str],
    samples: int = 20000,
) -> list[float]:
    return stratified_percentile_interval_list(
        first - second,
        strata,
        seed=20260715,
        reps=samples,
    )


def mcnemar_exact(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    smaller = min(first_only, second_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) * (0.5 ** discordant)
    return min(1.0, 2 * tail)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    native = {row["sample_id"]: row for row in read_jsonl(NATIVE_OUTPUTS)}
    private = {row["sample_id"]: row for row in read_jsonl(PRIVATE_MAP)}
    gold = {row["img_id"]: gold_record(row) for row in read_jsonl(GOLD_FILE)}
    td = {row["img_id"]: row for row in read_jsonl(TD_FILE)}
    core_strata = core_strata_by_id(GOLD_FILE)
    if not (len(native) == len(private) == len(gold) == len(td) == 100):
        raise ValueError("Expected 100 unique records in every input")

    adapted = []
    for sample_id in sorted(native):
        native_row = native[sample_id]
        img_id = private[sample_id]["img_id"]
        gold_row = gold[img_id]
        text = str(native_row.get("native_text_normalized") or "")
        prediction = native_prediction(native_row)
        target, target_text = target_from_text(text)
        artifact, artifact_set, artifact_evidence = artifacts_from_text(text)
        mask_paths = resolve_mask_paths(native_row)
        evidence_bbox, mask_pixels = bbox_from_masks(mask_paths)
        parse_valid = native_row.get("runtime_status") == "ok" and bool(text) and prediction is not None
        iou = bbox_iou(evidence_bbox, gold_row["evidence_bbox"])
        strict = condition_record(
            parse_valid,
            prediction == gold_row["prediction"],
            artifact == gold_row["artifact_type"],
            target == gold_row["target_scope"],
            iou >= 0.1,
        )
        artifact_set_correct = gold_row["artifact_type"] in artifact_set
        set_variant = condition_record(
            parse_valid,
            prediction == gold_row["prediction"],
            artifact_set_correct,
            target == gold_row["target_scope"],
            iou >= 0.1,
        )
        adapted.append(
            {
                "sample_id": sample_id,
                "img_id": img_id,
                "system_id": "SIDA-7B-description-FP16",
                "native_text": text,
                "native_mask_paths": [str(path.relative_to(SIDA_DIR)) for path in mask_paths],
                "prediction": prediction,
                "artifact_type": artifact,
                "artifact_set": artifact_set,
                "artifact_evidence": artifact_evidence,
                "target_scope": target,
                "target_text": target_text,
                "evidence_bbox": evidence_bbox,
                "bbox_source": "sida_native_mask_tight_bbox",
                "mask_positive_pixels": mask_pixels,
                "parse_valid": parse_valid,
                "bbox_iou": iou,
                "conditions": strict["conditions"],
                "failed_conditions": strict["failed_conditions"],
                "joint": strict["joint"],
                "artifact_set_conditions": set_variant["conditions"],
                "artifact_set_failed_conditions": set_variant["failed_conditions"],
                "artifact_set_joint": set_variant["joint"],
                "gold": gold_row,
                "sampling_stratum": core_strata[img_id],
            }
        )
    write_jsonl(ADAPTED_FILE, adapted)

    td_rows = []
    for row in adapted:
        td_row = td[row["img_id"]]
        prediction = td_row.get("prediction") or {}
        parse_valid = bool(prediction.get("parsed"))
        conditions = condition_record(
            parse_valid,
            bool(td_row.get("prediction_correct")),
            bool(td_row.get("artifact_correct")),
            bool(td_row.get("target_correct")),
            float(td_row.get("bbox_iou") or 0.0) >= 0.1,
        )
        td_rows.append(
            {
                "img_id": row["img_id"],
                "conditions": conditions["conditions"],
                "failed_conditions": conditions["failed_conditions"],
                "joint": conditions["joint"],
                "bbox_iou": float(td_row.get("bbox_iou") or 0.0),
                "sampling_stratum": core_strata[row["img_id"]],
            }
        )

    strata = [row["sampling_stratum"] for row in adapted]
    validate_expected_strata("JSG-Core", strata)
    sida_summary = summarize(adapted, strata)
    sida_set_summary = summarize(adapted, strata, prefix="artifact_set_")
    td_summary = summarize(td_rows, strata)
    for summary in [sida_summary, td_summary]:
        summary["marginal_mean"] = float(np.mean([summary[k] for k in ["prediction", "artifact", "target", "spatial"]]))
        summary["marginal_product"] = float(np.prod([summary[k] for k in ["prediction", "artifact", "target", "spatial"]]))

    for summary, rows, condition_prefix in [
        (sida_summary, adapted, ""),
        (td_summary, td_rows, ""),
    ]:
        shared_passes = sum(
            row[f"{condition_prefix}conditions"]["parse"]
            and row[f"{condition_prefix}conditions"]["prediction"]
            and row[f"{condition_prefix}conditions"]["target"]
            and row[f"{condition_prefix}conditions"]["spatial"]
            for row in rows
        )
        summary["shared_prediction_target_spatial_joint"] = shared_passes / len(rows)
        summary["shared_prediction_target_spatial_joint_count"] = shared_passes

    sida_pass = {row["img_id"] for row in adapted if row["joint"]}
    td_pass = {row["img_id"] for row in td_rows if row["joint"]}
    both = sida_pass & td_pass
    sida_only = sida_pass - td_pass
    td_only = td_pass - sida_pass
    neither = set(gold) - (sida_pass | td_pass)
    union = sida_pass | td_pass
    overlap = {
        "both_pass": len(both),
        "sida_only": len(sida_only),
        "td_egfa_only": len(td_only),
        "neither": len(neither),
        "jaccard": len(both) / len(union) if union else 0.0,
        "overlap_coefficient": len(both) / min(len(sida_pass), len(td_pass)) if sida_pass and td_pass else 0.0,
        "oracle_union_rate": len(union) / 100,
        "mcnemar_exact_p": mcnemar_exact(len(sida_only), len(td_only)),
    }
    sida_joint = np.asarray([row["joint"] for row in adapted], dtype=float)
    td_joint = np.asarray([row["joint"] for row in td_rows], dtype=float)
    overlap["sida_minus_td_joint"] = float((sida_joint - td_joint).mean())
    overlap["sida_minus_td_joint_stratified_bootstrap_95"] = bootstrap_difference(
        sida_joint,
        td_joint,
        strata,
    )

    sida_set_pass = {row["img_id"] for row in adapted if row["artifact_set_joint"]}
    set_both = sida_set_pass & td_pass
    set_sida_only = sida_set_pass - td_pass
    set_td_only = td_pass - sida_set_pass
    set_neither = set(gold) - (sida_set_pass | td_pass)
    set_union = sida_set_pass | td_pass
    artifact_set_overlap = {
        "both_pass": len(set_both),
        "sida_only": len(set_sida_only),
        "td_egfa_only": len(set_td_only),
        "neither": len(set_neither),
        "jaccard": len(set_both) / len(set_union) if set_union else 0.0,
        "oracle_union_rate": len(set_union) / 100,
        "mcnemar_exact_p": mcnemar_exact(len(set_sida_only), len(set_td_only)),
    }

    metric_rows = []
    metric_names = ["parse", "prediction", "artifact", "target", "spatial", "marginal_mean", "marginal_product", "joint"]
    for metric in metric_names:
        values = {"TD-EGFA": td_summary[metric], "SIDA": sida_summary[metric]}
        ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
        ranks = {system: rank for rank, (system, _) in enumerate(ordered, start=1)}
        for system in ["TD-EGFA", "SIDA"]:
            metric_rows.append({"metric": metric, "system": system, "value": values[system], "rank": ranks[system]})
    write_csv(COMPARISON_CSV, ["metric", "system", "value", "rank"], metric_rows)

    paired_rows = []
    for condition in ["parse", "prediction", "artifact", "target", "spatial"]:
        sida_values = np.asarray([row["conditions"][condition] for row in adapted], dtype=float)
        td_values = np.asarray([row["conditions"][condition] for row in td_rows], dtype=float)
        sida_only_condition = int(np.sum((sida_values == 1) & (td_values == 0)))
        td_only_condition = int(np.sum((sida_values == 0) & (td_values == 1)))
        interval = bootstrap_difference(sida_values, td_values, strata)
        paired_rows.append(
            {
                "condition": condition,
                "sida_rate": float(sida_values.mean()),
                "td_egfa_rate": float(td_values.mean()),
                "sida_minus_td": float((sida_values - td_values).mean()),
                "bootstrap_95_low": interval[0],
                "bootstrap_95_high": interval[1],
                "sida_only_pass": sida_only_condition,
                "td_egfa_only_pass": td_only_condition,
                "mcnemar_exact_p": mcnemar_exact(sida_only_condition, td_only_condition),
                "interval_method": "paired stratified percentile bootstrap over mask-area bands",
            }
        )
    strict_interval = bootstrap_difference(sida_joint, td_joint, strata)
    paired_rows.append(
        {
            "condition": "joint_strict",
            "sida_rate": float(sida_joint.mean()),
            "td_egfa_rate": float(td_joint.mean()),
            "sida_minus_td": float((sida_joint - td_joint).mean()),
            "bootstrap_95_low": strict_interval[0],
            "bootstrap_95_high": strict_interval[1],
            "sida_only_pass": len(sida_only),
            "td_egfa_only_pass": len(td_only),
            "mcnemar_exact_p": mcnemar_exact(len(sida_only), len(td_only)),
            "interval_method": "paired stratified percentile bootstrap over mask-area bands",
        }
    )
    write_csv(
        DIFFERENCE_CSV,
        [
            "condition", "sida_rate", "td_egfa_rate", "sida_minus_td",
            "bootstrap_95_low", "bootstrap_95_high", "sida_only_pass",
            "td_egfa_only_pass", "mcnemar_exact_p",
            "interval_method",
        ],
        paired_rows,
    )

    confusion_rows = []
    adapted_by_id = {row["img_id"]: row for row in adapted}
    for img_id in sorted(gold):
        sida_row = adapted_by_id[img_id]
        td_prediction = td[img_id].get("prediction") or {}
        for field, gold_key, sida_key, td_key in [
            ("artifact", "artifact_type", "artifact_type", "artifact_type"),
            ("target", "target_scope", "target_scope", "target_scope"),
        ]:
            confusion_rows.extend(
                [
                    {
                        "system": "SIDA",
                        "field": field,
                        "gold": gold[img_id][gold_key],
                        "predicted": sida_row[sida_key] or "missing",
                    },
                    {
                        "system": "TD-EGFA",
                        "field": field,
                        "gold": gold[img_id][gold_key],
                        "predicted": td_prediction.get(td_key) or "missing",
                    },
                ]
            )
    confusion_counts = Counter(
        (row["system"], row["field"], row["gold"], row["predicted"])
        for row in confusion_rows
    )
    write_csv(
        CONFUSION_CSV,
        ["system", "field", "gold", "predicted", "count"],
        [
            {"system": key[0], "field": key[1], "gold": key[2], "predicted": key[3], "count": count}
            for key, count in sorted(confusion_counts.items())
        ],
    )

    write_csv(
        OVERLAP_CSV,
        ["td_egfa", "sida", "count"],
        [
            {"td_egfa": "pass", "sida": "pass", "count": len(both)},
            {"td_egfa": "pass", "sida": "fail", "count": len(td_only)},
            {"td_egfa": "fail", "sida": "pass", "count": len(sida_only)},
            {"td_egfa": "fail", "sida": "fail", "count": len(neither)},
        ],
    )

    failure_rows = []
    for system, rows in [("TD-EGFA", td_rows), ("SIDA", adapted)]:
        counts = Counter("PASS" if not row["failed_conditions"] else "+".join(row["failed_conditions"]) for row in rows)
        for signature, count in counts.most_common():
            failure_rows.append({"system": system, "failure_signature": signature, "count": count, "percent": count / len(rows)})
    set_counts = Counter(
        "PASS" if not row["artifact_set_failed_conditions"] else "+".join(row["artifact_set_failed_conditions"])
        for row in adapted
    )
    for signature, count in set_counts.most_common():
        failure_rows.append(
            {
                "system": "SIDA artifact-set sensitivity",
                "failure_signature": signature,
                "count": count,
                "percent": count / len(adapted),
            }
        )
    write_csv(FAILURE_CSV, ["system", "failure_signature", "count", "percent"], failure_rows)

    report = {
        "evaluation": "SIDA native evaluation on JSG-Core",
        "adapter_contract": "remote_sida/SIDA_TO_JSG_ADAPTER_CONTRACT.md",
        "sampling_design": design_metadata("JSG-Core", strata),
        "interval_methods": {
            "binary_rates": "stratum-specific Wilson intervals combined by MOVER",
            "paired_differences": "paired stratified percentile bootstrap over mask-area bands",
        },
        "sida": sida_summary,
        "sida_artifact_set_sensitivity": sida_set_summary,
        "td_egfa": td_summary,
        "passing_set_overlap": overlap,
        "artifact_set_passing_overlap": artifact_set_overlap,
        "sida_primary_artifact_distribution": Counter(row["artifact_type"] or "missing" for row in adapted),
        "sida_target_distribution": Counter(row["target_scope"] or "missing" for row in adapted),
        "sida_mask_count": sum(bool(row["native_mask_paths"]) for row in adapted),
        "outputs": {
            "adapted_predictions": str(ADAPTED_FILE.relative_to(ROOT)),
            "system_comparison": str(COMPARISON_CSV.relative_to(ROOT)),
            "passing_overlap": str(OVERLAP_CSV.relative_to(ROOT)),
            "failure_signatures": str(FAILURE_CSV.relative_to(ROOT)),
            "paired_differences": str(DIFFERENCE_CSV.relative_to(ROOT)),
            "confusions": str(CONFUSION_CSV.relative_to(ROOT)),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def percentage(value: float) -> str:
        return f"{100 * value:.2f}"

    lines = [
        "# SIDA Native Evaluation on JSG-Core",
        "",
        "The strict headline uses the first recognized native inconsistency tag as the primary artifact. The artifact-set variant is a diagnostic sensitivity result.",
        "",
        "## System Comparison",
        "",
        "| System | Parse | Prediction | Artifact | Target | IoU@0.1 | Marginal mean | Marginal product | Joint@0.1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| TD-EGFA | {percentage(td_summary['parse'])} | {percentage(td_summary['prediction'])} | {percentage(td_summary['artifact'])} | {percentage(td_summary['target'])} | {percentage(td_summary['spatial'])} | {percentage(td_summary['marginal_mean'])} | {percentage(td_summary['marginal_product'])} | {percentage(td_summary['joint'])} |",
        f"| SIDA | {percentage(sida_summary['parse'])} | {percentage(sida_summary['prediction'])} | {percentage(sida_summary['artifact'])} | {percentage(sida_summary['target'])} | {percentage(sida_summary['spatial'])} | {percentage(sida_summary['marginal_mean'])} | {percentage(sida_summary['marginal_product'])} | {percentage(sida_summary['joint'])} |",
        "",
        f"Shared prediction+target+spatial Joint is {percentage(td_summary['shared_prediction_target_spatial_joint'])}% for TD-EGFA and {percentage(sida_summary['shared_prediction_target_spatial_joint'])}% for SIDA.",
        "",
        "## Artifact-Set Sensitivity",
        "",
        f"SIDA artifact-set accuracy is {percentage(sida_set_summary['artifact'])}%, and artifact-set Joint@0.1 is {percentage(sida_set_summary['joint'])}%.",
        f"Artifact-set passing overlap with TD-EGFA: both={artifact_set_overlap['both_pass']}, SIDA-only={artifact_set_overlap['sida_only']}, TD-EGFA-only={artifact_set_overlap['td_egfa_only']}, neither={artifact_set_overlap['neither']}, Jaccard={artifact_set_overlap['jaccard']:.4f}.",
        "",
        "## Passing-Set Overlap",
        "",
        "| TD-EGFA | SIDA | N |",
        "|---|---|---:|",
        f"| pass | pass | {len(both)} |",
        f"| pass | fail | {len(td_only)} |",
        f"| fail | pass | {len(sida_only)} |",
        f"| fail | fail | {len(neither)} |",
        "",
        f"- Jaccard overlap: {overlap['jaccard']:.4f}",
        f"- Overlap coefficient: {overlap['overlap_coefficient']:.4f}",
        f"- Oracle union pass rate: {percentage(overlap['oracle_union_rate'])}%",
        f"- SIDA minus TD-EGFA Joint difference: {percentage(overlap['sida_minus_td_joint'])} points",
        f"- Paired stratified-bootstrap 95% CI: [{percentage(overlap['sida_minus_td_joint_stratified_bootstrap_95'][0])}, {percentage(overlap['sida_minus_td_joint_stratified_bootstrap_95'][1])}] points",
        f"- Exact McNemar p-value: {overlap['mcnemar_exact_p']:.6f}",
        "",
        "## Failure Signatures",
        "",
        "| System | Signature | N | Rate |",
        "|---|---|---:|---:|",
    ]
    for row in failure_rows:
        lines.append(f"| {row['system']} | {row['failure_signature']} | {row['count']} | {percentage(row['percent'])} |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
