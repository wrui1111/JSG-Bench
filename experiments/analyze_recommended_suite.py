#!/usr/bin/env python3
"""Analyze the reviewer-recommended routing and provenance experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from candidate_evidence_explanation import (
    bbox_iou,
    extract_json_object,
    gold_item,
    normalize_prediction,
    normalize_text,
    read_jsonl,
    valid_bbox,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "experiments" / "reports" / "recommended_suite"
CLASS_NAMES = ["real", "full_synthetic", "tampered"]
MASK_AREA_BUCKETS = ["tiny", "small", "medium", "large"]
BOOTSTRAP_SEED = 20260729
BOOTSTRAP_REPLICATES = 10000


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int | None | bool]:
    if total <= 0:
        return {
            "successes": successes,
            "total": total,
            "estimate": None,
            "low": None,
            "high": None,
            "applicable": False,
        }
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": estimate,
        "low": max(0.0, center - radius),
        "high": min(1.0, center + radius),
        "applicable": True,
    }


def mover_weighted(
    intervals: list[dict[str, float | int | None | bool]], weights: list[float]
) -> dict[str, float | str]:
    if not intervals or len(intervals) != len(weights):
        raise ValueError("MOVER requires one applicable interval per fixed weight")
    if any(interval.get("estimate") is None for interval in intervals):
        raise ValueError("MOVER received a non-applicable interval")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("MOVER weights must sum to one")
    estimate = sum(weight * float(interval["estimate"]) for weight, interval in zip(weights, intervals))
    lower_radius = math.sqrt(
        sum(
            weight * weight * (float(interval["estimate"]) - float(interval["low"])) ** 2
            for weight, interval in zip(weights, intervals)
        )
    )
    upper_radius = math.sqrt(
        sum(
            weight * weight * (float(interval["high"]) - float(interval["estimate"])) ** 2
            for weight, interval in zip(weights, intervals)
        )
    )
    return {
        "estimate": estimate,
        "low": max(0.0, estimate - lower_radius),
        "high": min(1.0, estimate + upper_radius),
        "method": "Wilson-MOVER",
    }


def unique_row_map(path: Path, id_key: str = "img_id") -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        if id_key not in row:
            raise ValueError(f"Missing {id_key} in {path}")
        row_id = str(row[id_key])
        if row_id in mapping:
            raise ValueError(f"Duplicate {id_key}={row_id} in {path}")
        mapping[row_id] = row
    return mapping


def mask_area_bucket(row: dict[str, Any]) -> str:
    return str(
        (row.get("auto_features") or {}).get("mask_area_bucket")
        or row.get("mask_area_bucket")
        or "not_applicable"
    )


def assert_exact_ids(expected: set[str], observed: set[str], label: str) -> None:
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label} does not match the fixed denominator: "
            f"missing={missing[:5]} ({len(missing)} total), extra={extra[:5]} ({len(extra)} total)"
        )


def fixed_weight_mean(
    rows: list[dict[str, Any]],
    stratum: Callable[[dict[str, Any]], str],
    value: Callable[[dict[str, Any]], float],
    expected_strata: list[str],
) -> float:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    if set(groups) != set(expected_strata) or any(not groups[name] for name in expected_strata):
        raise ValueError(
            f"Observed strata {sorted(groups)} do not match fixed strata {sorted(expected_strata)}"
        )
    return float(np.mean([np.mean([value(row) for row in groups[name]]) for name in expected_strata]))


def derived_seed(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def exact_two_sided_binomial(b: int, c: int) -> dict[str, Any]:
    """Two-sided exact McNemar tail with stable log-space evaluation."""
    discordant = b + c
    if discordant == 0:
        return {
            "baseline_only_success": b,
            "variant_only_success": c,
            "discordant": 0,
            "p_value": 1.0,
            "log_p_value": 0.0,
            "underflow_bounded": False,
        }
    cutoff = min(b, c)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
        for k in range(cutoff + 1)
    ]
    maximum = max(log_terms)
    log_tail = maximum + math.log(sum(math.exp(value - maximum) for value in log_terms))
    log_p = min(0.0, math.log(2.0) + log_tail)
    min_positive = float(np.nextafter(0.0, 1.0))
    underflow = log_p < math.log(min_positive)
    p_value = min_positive if underflow else math.exp(log_p)
    return {
        "baseline_only_success": b,
        "variant_only_success": c,
        "discordant": discordant,
        "p_value": p_value,
        "log_p_value": log_p,
        "underflow_bounded": underflow,
    }


def percentile_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "low": float(np.quantile(array, 0.025)),
        "high": float(np.quantile(array, 0.975)),
    }


def macro_f1(rows: list[dict[str, Any]], classes: list[str] | None = None) -> float:
    classes = classes or [name for name in CLASS_NAMES if any(row["gold_class"] == name for row in rows)]
    scores = []
    for class_name in classes:
        tp = sum(row["gold_class"] == class_name and row["predicted_class"] == class_name for row in rows)
        fp = sum(row["gold_class"] != class_name and row["predicted_class"] == class_name for row in rows)
        fn = sum(row["gold_class"] == class_name and row["predicted_class"] != class_name for row in rows)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def stratified_bootstrap_rows(
    rows: list[dict[str, Any]],
    stratum: Callable[[dict[str, Any]], str],
    statistic: Callable[[list[dict[str, Any]]], float],
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    rng = np.random.default_rng(seed)
    values = []
    group_values = list(groups.values())
    for _ in range(replicates):
        sample = []
        for group in group_values:
            indices = rng.integers(0, len(group), size=len(group))
            sample.extend(group[index] for index in indices)
        values.append(statistic(sample))
    return values


def _result_from_output(output: dict[str, Any]) -> dict[str, Any]:
    result = output.get("vlm_result")
    if isinstance(result, dict):
        return result
    raw = output.get("vlm_raw_text")
    parsed = extract_json_object(raw) if isinstance(raw, str) else None
    return parsed if isinstance(parsed, dict) else {}


def assert_output_request_link(
    img_id: str,
    request: dict[str, Any],
    output: dict[str, Any],
    label: str,
    require_request_hash: bool = False,
) -> bool:
    if str(request.get("img_id")) != img_id or str(output.get("img_id")) != img_id:
        raise ValueError(f"{label}: img_id linkage failure for {img_id}")
    expected_hash = canonical_hash(request)
    recorded_hash = output.get("request_sha256")
    if recorded_hash is not None and recorded_hash != expected_hash:
        raise ValueError(f"{label}: request_sha256 mismatch for {img_id}")
    if require_request_hash and recorded_hash is None:
        raise ValueError(f"{label}: missing required request_sha256 for {img_id}")
    for field in ["image_path", "image_paths"]:
        if field in request and field in output and request.get(field) != output.get(field):
            raise ValueError(f"{label}: {field} mismatch for {img_id}")
    return recorded_hash is not None


def assert_candidate_prediction_link(
    img_id: str,
    request: dict[str, Any],
    output: dict[str, Any],
    prediction_row: dict[str, Any],
    label: str,
    require_request_hash: bool = False,
) -> bool:
    hash_present = assert_output_request_link(
        img_id, request, output, label, require_request_hash=require_request_hash
    )
    if str(prediction_row.get("img_id")) != img_id:
        raise ValueError(f"{label}: prediction img_id mismatch for {img_id}")
    result = _result_from_output(output)
    prediction = prediction_row.get("prediction") or {}
    parsed = bool(prediction.get("parsed"))
    expected_parsed = bool(result) and not output.get("error")
    if parsed != expected_parsed:
        raise ValueError(f"{label}: parsed status disagrees with supplied output for {img_id}")
    if not parsed:
        return hash_present
    raw_class = normalize_prediction(result.get("prediction"))
    evaluated_class = normalize_prediction(prediction.get("prediction"))
    if raw_class != evaluated_class:
        raise ValueError(f"{label}: prediction class disagrees with supplied output for {img_id}")
    for field in ["artifact_type", "target_scope", "candidate_verdict"]:
        if normalize_text(result.get(field)) != normalize_text(prediction.get(field)):
            raise ValueError(f"{label}: {field} disagrees with supplied output for {img_id}")
    candidate_bbox = valid_bbox(
        (request.get("metadata_for_evaluation_only") or {}).get("candidate_bbox")
    )
    expected_bbox = valid_bbox(result.get("evidence_bbox"))
    if evaluated_class == "tampered" and candidate_bbox is not None:
        expected_bbox = candidate_bbox
    if valid_bbox(prediction.get("evidence_bbox")) != expected_bbox:
        raise ValueError(f"{label}: evaluated bbox disagrees with output/provenance rule for {img_id}")
    return hash_present


def _prediction_row_map(path: Path) -> dict[str, dict[str, Any]]:
    return unique_row_map(path)


def routing_analysis(args: argparse.Namespace) -> None:
    annotations = unique_row_map(args.annotations)
    requests = unique_row_map(args.requests)
    outputs = unique_row_map(args.outputs)
    predictions = _prediction_row_map(args.predictions)
    lowlevel = unique_row_map(args.lowlevel)
    ids = list(requests)
    expected_ids = set(annotations)
    for label, mapping in [
        ("requests", requests),
        ("outputs", outputs),
        ("predictions", predictions),
        ("low-level evidence", lowlevel),
    ]:
        assert_exact_ids(expected_ids, set(mapping), label)
    class_counts = Counter(str(gold_item(row)["prediction"]) for row in annotations.values())
    if args.design == "sid_explain_300":
        if len(expected_ids) != 300 or class_counts != Counter({name: 100 for name in CLASS_NAMES}):
            raise ValueError(
                f"sid_explain_300 requires 300 unique IDs and 100 per class; observed {dict(class_counts)}"
            )
        tampered_bands = Counter(
            mask_area_bucket(row)
            for row in annotations.values()
            if str(gold_item(row)["prediction"]) == "tampered"
        )
        if tampered_bands != Counter({name: 25 for name in MASK_AREA_BUCKETS}):
            raise ValueError(f"Unexpected tampered mask-area strata: {dict(tampered_bands)}")
    elif args.design == "negative_2000":
        if len(expected_ids) != 2000 or class_counts != Counter({"real": 1000, "full_synthetic": 1000}):
            raise ValueError(
                f"negative_2000 requires 2,000 unique IDs and 1,000 per class; observed {dict(class_counts)}"
            )
    else:
        raise ValueError(f"Unsupported routing design: {args.design}")

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    candidate_verdicts: dict[str, Counter[str]] = defaultdict(Counter)
    for img_id in ids:
        annotation = annotations[img_id]
        request = requests[img_id]
        output = outputs[img_id]
        prediction_row = predictions[img_id]
        assert_candidate_prediction_link(
            img_id,
            request,
            output,
            prediction_row,
            "routing audit",
            require_request_hash=True,
        )
        prediction = prediction_row.get("prediction") or {}
        gold = gold_item(annotation)
        parsed = bool(prediction.get("parsed"))
        predicted_class = normalize_prediction(prediction.get("prediction")) if parsed else "invalid/unknown"
        if predicted_class not in CLASS_NAMES:
            predicted_class = "invalid/unknown"
        gold_class = str(gold["prediction"])
        candidate_verdict = normalize_text(prediction.get("candidate_verdict")) or "missing"
        candidate_verdicts[gold_class][candidate_verdict] += 1
        evaluated_bbox = valid_bbox(prediction.get("evidence_bbox"))
        raw_result = _result_from_output(output)
        raw_bbox = valid_bbox(raw_result.get("evidence_bbox"))
        meta = request.get("metadata_for_evaluation_only") or {}
        experimental_provenance = request.get("experimental_provenance") or {}
        candidate_bbox = valid_bbox(meta.get("candidate_bbox"))
        correct = bool(parsed and predicted_class == gold_class)
        row = {
            "img_id": img_id,
            "gold_class": gold_class,
            "predicted_class": predicted_class,
            "parsed": parsed,
            "correct": correct,
            "false_tampered": bool(parsed and predicted_class == "tampered" and gold_class != "tampered"),
            "non_tampered_prediction": bool(parsed and predicted_class in {"real", "full_synthetic"}),
            "candidate_verdict": candidate_verdict,
            "null_bbox": evaluated_bbox is None,
            "artifact_correct": bool(parsed and prediction_row.get("artifact_correct")),
            "target_correct": bool(parsed and prediction_row.get("target_correct")),
            "bbox_iou": float(prediction_row.get("bbox_iou") or 0.0),
            "mask_area_bucket": mask_area_bucket(annotation),
        }
        row["tampered_joint_0_1"] = bool(
            gold_class == "tampered"
            and parsed
            and correct
            and row["artifact_correct"]
            and row["target_correct"]
            and evaluated_bbox is not None
            and row["bbox_iou"] >= 0.1
        )
        row["negative_schema_pass"] = bool(
            gold_class != "tampered"
            and parsed
            and correct
            and row["artifact_correct"]
            and row["target_correct"]
            and evaluated_bbox is None
        )
        rows.append(row)

        failures = []
        if not parsed:
            failures.append("parse_failure")
        elif predicted_class == "invalid/unknown":
            failures.append("out_of_schema_prediction")
        if not correct:
            failures.append(f"class_{gold_class}_to_{predicted_class}")
        if row["false_tampered"]:
            failures.append("spurious_tampered")
        if predicted_class == "tampered" and candidate_verdict != "accepted":
            failures.append("tampered_with_nonaccepted_candidate_verdict")
        if gold_class != "tampered" and evaluated_bbox is not None:
            failures.append("negative_non_null_bbox")
        if gold_class != "tampered" and not row["artifact_correct"]:
            failures.append("wrong_negative_artifact_field")
        if gold_class != "tampered" and not row["target_correct"]:
            failures.append("wrong_negative_target_field")
        if gold_class == "tampered":
            if not row["artifact_correct"]:
                failures.append("tampered_artifact_failure")
            if not row["target_correct"]:
                failures.append("tampered_target_failure")
            if evaluated_bbox is None or row["bbox_iou"] < 0.1:
                failures.append("tampered_spatial_failure")

        image_hashes = []
        for image_path in request.get("image_paths") or []:
            path = Path(str(image_path))
            image_hashes.append({"path": str(path), "sha256": sha256(path) if path.is_file() else None})
        bbox_source = "null"
        if evaluated_bbox is not None:
            bbox_source = "inherited_candidate" if predicted_class == "tampered" and candidate_bbox == evaluated_bbox else "raw"
        traces.append(
            {
                "trace_version": "recommended-routing-trace-v1.0",
                "sample_id": img_id,
                "denominator_inclusion": (
                    "fixed_class_balanced_sid_explain300"
                    if args.design == "sid_explain_300"
                    else "fixed_equal_weight_negative2000"
                ),
                "gold_class": gold_class,
                "class_stratum": gold_class,
                "inference_provenance_id": output.get("inference_provenance_id"),
                "prompt_version": request.get("prompt_version"),
                "request_sha256": canonical_hash(request),
                "raw_output_sha256": canonical_hash(output),
                "prompt_sha256": hashlib.sha256(str(request.get("prompt") or "").encode("utf-8")).hexdigest(),
                "input_view_mode": meta.get("image_mode") or experimental_provenance.get("view_mode"),
                "image_assets": image_hashes,
                "candidate": {
                    "mode": meta.get("candidate_mode"),
                    "top3_boxes": [item.get("bbox") for item in lowlevel[img_id].get("lowlevel_candidates", [])[:3]],
                    "union_bbox": candidate_bbox,
                    "lowlevel_row_sha256": canonical_hash(lowlevel[img_id]),
                },
                "parser": {"parsed": parsed, "error": output.get("error")},
                "raw_fields": {
                    "prediction": raw_result.get("prediction"),
                    "artifact_type": raw_result.get("artifact_type"),
                    "target_scope": raw_result.get("target_scope"),
                    "candidate_verdict": raw_result.get("candidate_verdict"),
                    "evidence_bbox": raw_bbox,
                },
                "normalized_fields": {
                    "prediction": predicted_class,
                    "artifact_type": prediction.get("artifact_type"),
                    "target_scope": prediction.get("target_scope"),
                    "candidate_verdict": candidate_verdict,
                    "evidence_bbox": evaluated_bbox,
                },
                "evaluated_region": {
                    "bbox": evaluated_bbox,
                    "bbox_source": bbox_source,
                    "inheritance_triggered": bbox_source == "inherited_candidate",
                    "iou": row["bbox_iou"] if gold_class == "tampered" else None,
                },
                "applicability": {
                    "classification": True,
                    "artifact_target_spatial_joint": gold_class == "tampered",
                    "negative_schema_diagnostic": gold_class != "tampered",
                },
                "classification_correct": correct,
                "tampered_joint_0_1": row["tampered_joint_0_1"] if gold_class == "tampered" else None,
                "negative_schema_pass": row["negative_schema_pass"] if gold_class != "tampered" else None,
                "failure_signatures": failures,
            }
        )

    by_class = {class_name: [row for row in rows if row["gold_class"] == class_name] for class_name in CLASS_NAMES}
    class_metrics: dict[str, Any] = {}
    confusion: dict[str, dict[str, int]] = {}
    for class_name, class_rows in by_class.items():
        confusion[class_name] = dict(Counter(row["predicted_class"] for row in class_rows))
        class_metrics[class_name] = {
            "n": len(class_rows),
            "parse_rate": wilson(sum(row["parsed"] for row in class_rows), len(class_rows)),
            "recall": wilson(sum(row["correct"] for row in class_rows), len(class_rows)),
            "false_tampered_emission": (
                wilson(sum(row["false_tampered"] for row in class_rows), len(class_rows))
                if class_name != "tampered" and class_rows
                else None
            ),
            "candidate_verdict_counts": dict(candidate_verdicts[class_name]),
            "candidate_acceptance": wilson(candidate_verdicts[class_name]["accepted"], len(class_rows)),
            "null_bbox_rate": wilson(sum(row["null_bbox"] for row in class_rows), len(class_rows)),
            "schema_consistent_report": wilson(sum(row["negative_schema_pass"] for row in class_rows), len(class_rows))
            if class_name != "tampered"
            else None,
        }
    present_classes = [class_name for class_name in CLASS_NAMES if by_class[class_name]]
    recall_intervals = [class_metrics[class_name]["recall"] for class_name in present_classes]
    negative_specificity_intervals = []
    negative_ftr_intervals = []
    for class_name in ["real", "full_synthetic"]:
        class_rows = by_class[class_name]
        negative_specificity_intervals.append(
            wilson(sum(row["non_tampered_prediction"] for row in class_rows), len(class_rows))
        )
        negative_ftr_intervals.append(class_metrics[class_name]["false_tampered_emission"])
    accuracy = sum(row["correct"] for row in rows) / len(rows)
    f1 = macro_f1(rows, present_classes)
    f1_bootstrap = stratified_bootstrap_rows(
        rows,
        lambda row: row["gold_class"],
        lambda sample: macro_f1(sample, present_classes),
    )
    parse_intervals = [class_metrics[class_name]["parse_rate"] for class_name in present_classes]
    tampered_joint_interval = None
    tampered_joint_by_band = None
    if by_class["tampered"]:
        tampered_joint_by_band = {
            bucket: wilson(
                sum(row["tampered_joint_0_1"] for row in by_class["tampered"] if row["mask_area_bucket"] == bucket),
                sum(row["mask_area_bucket"] == bucket for row in by_class["tampered"]),
            )
            for bucket in MASK_AREA_BUCKETS
        }
        tampered_joint_interval = mover_weighted(
            [tampered_joint_by_band[bucket] for bucket in MASK_AREA_BUCKETS],
            [0.25] * len(MASK_AREA_BUCKETS),
        )
    report = {
        "stage": (
            "same_runtime_three_class_routing_audit"
            if args.design == "sid_explain_300"
            else "same_runtime_large_negative_class_audit"
        ),
        "design": args.design,
        "estimand": args.estimand
        or (
            "fixed 1:1:1 class-balanced SID-Explain-300 mixture"
            if by_class["tampered"]
            else "fixed 1:1 balanced real/full-synthetic negative-class mixture"
        ),
        "all_300_joint_formed": False,
        "sample_count": len(rows),
        "class_counts": {class_name: len(class_rows) for class_name, class_rows in by_class.items()},
        "overall_parse_rate": mover_weighted(
            parse_intervals, [1 / len(present_classes)] * len(present_classes)
        ),
        "accuracy": accuracy,
        "balanced_accuracy": mover_weighted(
            recall_intervals, [1 / len(present_classes)] * len(present_classes)
        ),
        "accuracy_equals_balanced_accuracy_because_equal_present_class_sizes": len(
            {len(by_class[class_name]) for class_name in present_classes}
        )
        == 1,
        "macro_f1": {
            "estimate": f1,
            **percentile_interval(f1_bootstrap),
            "method": f"class-stratified percentile bootstrap, {BOOTSTRAP_REPLICATES} replicates",
        },
        "non_tampered_specificity": mover_weighted(negative_specificity_intervals, [0.5, 0.5]),
        "equal_weight_negative_false_tampered_emission": mover_weighted(negative_ftr_intervals, [0.5, 0.5]),
        "class_metrics": class_metrics,
        "confusion_matrix_counts": confusion,
        "tampered_strict_primary_joint_0_1": tampered_joint_interval,
        "tampered_strict_primary_joint_0_1_by_mask_area": tampered_joint_by_band,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
        "linkage_verification": {
            "request_hash_verified_rows": len(rows),
            "prediction_output_rederived_rows": len(rows),
        },
        "inputs": {
            "annotations": str(args.annotations.relative_to(ROOT)),
            "requests": str(args.requests.relative_to(ROOT)),
            "outputs": str(args.outputs.relative_to(ROOT)),
            "predictions": str(args.predictions.relative_to(ROOT)),
            "lowlevel": str(args.lowlevel.relative_to(ROOT)),
            "hashes": {
                "annotations": sha256(args.annotations),
                "requests": sha256(args.requests),
                "outputs": sha256(args.outputs),
                "predictions": sha256(args.predictions),
                "lowlevel": sha256(args.lowlevel),
            },
        },
    }
    write_json(args.output, report)
    write_jsonl(args.trace_output, traces)
    is_three_class = bool(by_class["tampered"])
    lines = [
        "# Same-Runtime Three-Class Routing Audit"
        if is_three_class
        else "# Same-Runtime Negative-Class Routing Audit",
        "",
        (
            "Fixed 1:1:1 SID-Explain-300 mixture; this report does not form an all-300 Joint score."
            if is_three_class
            else "Fixed 1:1 real/full-synthetic mixture; Joint is not applicable to this negative-class audit."
        ),
        "",
        "| Metric | Estimate | 95% CI |",
        "|---|---:|---:|",
    ]
    for label, interval in [
        ("Parse rate", report["overall_parse_rate"]),
        ("Balanced accuracy", report["balanced_accuracy"]),
        ("Non-tampered specificity", report["non_tampered_specificity"]),
        ("Negative false-tampered emission", report["equal_weight_negative_false_tampered_emission"]),
        ("Macro-F1", report["macro_f1"]),
    ]:
        lines.append(f"| {label} | {interval['estimate'] * 100:.2f}% | [{interval['low'] * 100:.2f}, {interval['high'] * 100:.2f}] |")
    lines.extend(["", "| GT | real | full-synthetic | tampered | invalid/unknown |", "|---|---:|---:|---:|---:|"])
    for class_name in present_classes:
        counts = confusion[class_name]
        lines.append(
            f"| {class_name} | {counts.get('real', 0)} | {counts.get('full_synthetic', 0)} | "
            f"{counts.get('tampered', 0)} | {counts.get('invalid/unknown', 0)} |"
        )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_variant(value: str) -> tuple[str, Path, Path, Path]:
    parts = value.split("|")
    if len(parts) != 4 or not all(parts):
        raise argparse.ArgumentTypeError(
            "Variant must be NAME|PREDICTIONS_JSONL|REQUESTS_JSONL|OUTPUTS_JSONL"
        )
    return parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3])


def variant_row(
    prediction_row: dict[str, Any],
    annotation: dict[str, Any],
    request: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    prediction = prediction_row.get("prediction") or {}
    parsed = bool(prediction.get("parsed"))
    predicted = normalize_prediction(prediction.get("prediction")) if parsed else "invalid/unknown"
    if predicted not in CLASS_NAMES:
        predicted = "invalid/unknown"
    gold = str(gold_item(annotation)["prediction"])
    bbox = valid_bbox(prediction.get("evidence_bbox"))
    iou = float(prediction_row.get("bbox_iou") or 0.0)
    artifact = bool(parsed and prediction_row.get("artifact_correct"))
    target = bool(parsed and prediction_row.get("target_correct"))
    candidate_bbox = valid_bbox(
        (request.get("metadata_for_evaluation_only") or {}).get("candidate_bbox")
    )
    experimental_provenance = request.get("experimental_provenance") or {}
    bbox_source = "null"
    if bbox is not None:
        bbox_source = (
            "inherited_candidate"
            if predicted == "tampered" and bbox == candidate_bbox
            else "model_emitted"
        )
    failures = []
    if not parsed:
        failures.append("parse_failure")
    elif predicted == "invalid/unknown":
        failures.append("out_of_schema_prediction")
    if not (parsed and predicted == gold):
        failures.append(f"class_{gold}_to_{predicted}")
    if parsed and predicted == "tampered" and gold != "tampered":
        failures.append("spurious_tampered")
    if gold == "tampered":
        if not artifact:
            failures.append("tampered_artifact_failure")
        if not target:
            failures.append("tampered_target_failure")
        if not (parsed and bbox is not None and iou >= 0.1):
            failures.append("tampered_spatial_failure")
    return {
        "img_id": str(prediction_row["img_id"]),
        "gold_class": gold,
        "predicted_class": predicted,
        "parsed": parsed,
        "correct": bool(parsed and predicted == gold),
        "false_tampered": bool(parsed and predicted == "tampered" and gold != "tampered"),
        "tampered_recall": bool(gold == "tampered" and parsed and predicted == "tampered"),
        "artifact": artifact,
        "target": target,
        "spatial": bool(parsed and bbox is not None and iou >= 0.1),
        "joint": bool(
            gold == "tampered"
            and parsed
            and predicted == "tampered"
            and artifact
            and target
            and bbox is not None
            and iou >= 0.1
        ),
        "mask_area_bucket": mask_area_bucket(annotation),
        "artifact_label": prediction.get("artifact_type"),
        "target_label": prediction.get("target_scope"),
        "candidate_verdict": prediction.get("candidate_verdict"),
        "candidate_bbox": candidate_bbox,
        "evaluated_bbox": bbox,
        "bbox_source": bbox_source,
        "bbox_iou": iou,
        "view_mode": (
            (request.get("metadata_for_evaluation_only") or {}).get("image_mode")
            or experimental_provenance.get("view_mode")
        ),
        "crop_donor_img_id": experimental_provenance.get("crop_donor_img_id"),
        "proposal_donor_img_id": experimental_provenance.get("proposal_donor_img_id"),
        "request_sha256": canonical_hash(request),
        "prompt_sha256": hashlib.sha256(str(request.get("prompt") or "").encode("utf-8")).hexdigest(),
        "output_sha256": canonical_hash(output),
        "prediction_row_sha256": canonical_hash(prediction_row),
        "inference_provenance_id": output.get("inference_provenance_id"),
        "output_error": output.get("error"),
        "failure_signatures": failures,
    }


def endpoint_value(row: dict[str, Any], endpoint: str) -> float:
    if endpoint == "balanced_accuracy":
        return float(row["correct"])
    if endpoint == "negative_ftr":
        return float(row["false_tampered"])
    if endpoint == "tampered_recall":
        return float(row["tampered_recall"])
    if endpoint == "tampered_joint":
        return float(row["joint"])
    if endpoint == "tampered_spatial":
        return float(row["spatial"])
    raise KeyError(endpoint)


def endpoint_subset(rows: list[dict[str, Any]], endpoint: str) -> list[dict[str, Any]]:
    if endpoint == "negative_ftr":
        return [row for row in rows if row["gold_class"] != "tampered"]
    if endpoint in {"tampered_recall", "tampered_joint", "tampered_spatial"}:
        return [row for row in rows if row["gold_class"] == "tampered"]
    return rows


def endpoint_strata(endpoint: str) -> list[str]:
    if endpoint == "balanced_accuracy":
        return CLASS_NAMES
    if endpoint == "negative_ftr":
        return ["real", "full_synthetic"]
    if endpoint in {"tampered_recall", "tampered_joint", "tampered_spatial"}:
        return MASK_AREA_BUCKETS
    raise KeyError(endpoint)


def endpoint_stratum(row: dict[str, Any], endpoint: str) -> str:
    if endpoint in {"balanced_accuracy", "negative_ftr"}:
        return row["gold_class"]
    return row["mask_area_bucket"]


def weighted_endpoint_mean(rows: list[dict[str, Any]], endpoint: str) -> float:
    subset = endpoint_subset(rows, endpoint)
    return fixed_weight_mean(
        subset,
        lambda row: endpoint_stratum(row, endpoint),
        lambda row: endpoint_value(row, endpoint),
        endpoint_strata(endpoint),
    )


def paired_bootstrap_delta(
    baseline: list[dict[str, Any]], variant: list[dict[str, Any]], endpoint: str, seed: int
) -> dict[str, float]:
    base_map = {row["img_id"]: row for row in endpoint_subset(baseline, endpoint)}
    var_map = {row["img_id"]: row for row in endpoint_subset(variant, endpoint)}
    assert_exact_ids(set(base_map), set(var_map), f"paired {endpoint} variant")
    ids = sorted(base_map)
    groups: dict[str, list[str]] = defaultdict(list)
    for img_id in ids:
        row = base_map[img_id]
        key = endpoint_stratum(row, endpoint)
        groups[key].append(img_id)
    expected_strata = endpoint_strata(endpoint)
    if set(groups) != set(expected_strata):
        raise ValueError(f"Unexpected strata for {endpoint}: {sorted(groups)}")
    observed = float(
        np.mean(
            [
                np.mean(
                    [
                        endpoint_value(var_map[img_id], endpoint)
                        - endpoint_value(base_map[img_id], endpoint)
                        for img_id in groups[stratum]
                    ]
                )
                for stratum in expected_strata
            ]
        )
    )
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        stratum_means = []
        for stratum in expected_strata:
            group_ids = groups[stratum]
            indices = rng.integers(0, len(group_ids), size=len(group_ids))
            differences = []
            for index in indices:
                img_id = group_ids[index]
                differences.append(
                    endpoint_value(var_map[img_id], endpoint)
                    - endpoint_value(base_map[img_id], endpoint)
                )
            stratum_means.append(float(np.mean(differences)))
        draws.append(float(np.mean(stratum_means)))
    return {
        "estimate": observed,
        **percentile_interval(draws),
        "seed": seed,
        "replicates": BOOTSTRAP_REPLICATES,
        "fixed_equal_weight_strata": expected_strata,
    }


def exact_mcnemar(baseline: list[dict[str, Any]], variant: list[dict[str, Any]], endpoint: str) -> dict[str, Any]:
    base_map = {row["img_id"]: row for row in endpoint_subset(baseline, endpoint)}
    var_map = {row["img_id"]: row for row in endpoint_subset(variant, endpoint)}
    assert_exact_ids(set(base_map), set(var_map), f"McNemar {endpoint} variant")
    ids = sorted(base_map)
    b = sum(endpoint_value(base_map[img_id], endpoint) == 1 and endpoint_value(var_map[img_id], endpoint) == 0 for img_id in ids)
    c = sum(endpoint_value(base_map[img_id], endpoint) == 0 and endpoint_value(var_map[img_id], endpoint) == 1 for img_id in ids)
    return exact_two_sided_binomial(b, c)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = (count - rank) * p_values[name]
        running = max(running, candidate)
        adjusted[name] = min(1.0, running)
    return adjusted


def variants_analysis(args: argparse.Namespace) -> None:
    annotations = unique_row_map(args.annotations)
    expected_ids = set(annotations)
    class_counts = Counter(str(gold_item(row)["prediction"]) for row in annotations.values())
    if args.design == "sid_explain_300_views":
        if len(expected_ids) != 300 or class_counts != Counter({name: 100 for name in CLASS_NAMES}):
            raise ValueError(f"View design requires 100/100/100 classes; observed {dict(class_counts)}")
        primary_endpoint = "negative_ftr"
        stage = "sid_explain_300_view_decoupling"
        expected_band_size = 25
    elif args.design == "jsg_core_proposal":
        if len(expected_ids) != 100 or class_counts != Counter({"tampered": 100}):
            raise ValueError(f"Proposal design requires 100 tampered reports; observed {dict(class_counts)}")
        primary_endpoint = "tampered_joint"
        stage = "jsg_core_proposal_package_specificity"
        expected_band_size = 25
    elif args.design == "sid_fa600_prompt_sensitivity":
        if len(expected_ids) != 600 or class_counts != Counter({"tampered": 600}):
            raise ValueError(f"Prompt sensitivity requires 600 tampered reports; observed {dict(class_counts)}")
        primary_endpoint = None
        stage = "post_hoc_prompt_specification_sensitivity"
        expected_band_size = 150
    elif args.design == "sid_fa200_model_runtime":
        if len(expected_ids) != 200 or class_counts != Counter({"tampered": 200}):
            raise ValueError(f"Model-runtime scope check requires 200 tampered reports; observed {dict(class_counts)}")
        primary_endpoint = None
        stage = "sid_fa200_model_runtime_scope_check"
        expected_band_size = 50
    else:
        raise ValueError(f"Unsupported variants design: {args.design}")
    tampered_bands = Counter(
        mask_area_bucket(row)
        for row in annotations.values()
        if str(gold_item(row)["prediction"]) == "tampered"
    )
    if tampered_bands != Counter({name: expected_band_size for name in MASK_AREA_BUCKETS}):
        raise ValueError(f"Unexpected tampered mask-area strata: {dict(tampered_bands)}")
    variants: dict[str, list[dict[str, Any]]] = {}
    paths: dict[str, dict[str, Path]] = {}
    input_diagnostics: dict[str, dict[str, int]] = {}
    if len({name for name, *_ in args.variant}) != len(args.variant):
        raise ValueError("Variant names must be unique")
    for name, prediction_path, request_path, output_path in args.variant:
        prediction_path = prediction_path.resolve()
        request_path = request_path.resolve()
        output_path = output_path.resolve()
        prediction_map = unique_row_map(prediction_path)
        request_map = unique_row_map(request_path)
        output_map = unique_row_map(output_path)
        for label, mapping in [
            (f"{name} predictions", prediction_map),
            (f"{name} requests", request_map),
        ]:
            assert_exact_ids(expected_ids, set(mapping), label)
        missing_outputs = expected_ids - set(output_map)
        if missing_outputs:
            raise ValueError(
                f"{name} outputs omit fixed-denominator IDs: {sorted(missing_outputs)[:5]} "
                f"({len(missing_outputs)} total)"
            )
        extra_outputs = set(output_map) - expected_ids
        linked_hash_rows = 0
        for img_id in expected_ids:
            linked_hash_rows += int(
                assert_candidate_prediction_link(
                    img_id,
                    request_map[img_id],
                    output_map[img_id],
                    prediction_map[img_id],
                    f"variant {name}",
                )
            )
        input_diagnostics[name] = {
            "fixed_denominator_rows": len(expected_ids),
            "prediction_rows": len(prediction_map),
            "request_rows": len(request_map),
            "output_rows": len(output_map),
            "unused_extra_output_rows": len(extra_outputs),
            "request_hash_verified_rows": linked_hash_rows,
            "request_hash_unavailable_rows": len(expected_ids) - linked_hash_rows,
        }
        paths[name] = {
            "predictions": prediction_path,
            "requests": request_path,
            "outputs": output_path,
        }
        variants[name] = [
            variant_row(
                prediction_map[img_id], annotations[img_id], request_map[img_id], output_map[img_id]
            )
            for img_id in annotations
        ]
    if args.baseline not in variants:
        raise ValueError(f"Baseline {args.baseline} is not among variants")
    baseline = variants[args.baseline]

    summaries: dict[str, Any] = {}
    for name, rows in variants.items():
        by_class = {class_name: [row for row in rows if row["gold_class"] == class_name] for class_name in CLASS_NAMES}
        negative = [row for row in rows if row["gold_class"] != "tampered"]
        tampered = by_class["tampered"]
        parse_strata = CLASS_NAMES if args.design == "sid_explain_300_views" else MASK_AREA_BUCKETS
        parse_key = (
            (lambda row: row["gold_class"])
            if args.design == "sid_explain_300_views"
            else (lambda row: row["mask_area_bucket"])
        )
        summaries[name] = {
            "n": len(rows),
            "class_counts": {key: len(value) for key, value in by_class.items()},
            "parse_rate": fixed_weight_mean(
                rows, parse_key, lambda row: float(row["parsed"]), parse_strata
            ),
            "balanced_accuracy": (
                weighted_endpoint_mean(rows, "balanced_accuracy")
                if args.design == "sid_explain_300_views"
                else None
            ),
            "negative_false_tampered_emission": (
                weighted_endpoint_mean(rows, "negative_ftr") if negative else None
            ),
            "tampered_recall": weighted_endpoint_mean(rows, "tampered_recall") if tampered else None,
            "tampered_artifact_pass": fixed_weight_mean(
                tampered,
                lambda row: row["mask_area_bucket"],
                lambda row: float(row["artifact"]),
                MASK_AREA_BUCKETS,
            ) if tampered else None,
            "tampered_target_pass": fixed_weight_mean(
                tampered,
                lambda row: row["mask_area_bucket"],
                lambda row: float(row["target"]),
                MASK_AREA_BUCKETS,
            ) if tampered else None,
            "tampered_spatial_pass_0_1": weighted_endpoint_mean(rows, "tampered_spatial") if tampered else None,
            "tampered_joint_0_1": weighted_endpoint_mean(rows, "tampered_joint") if tampered else None,
            "tampered_predicted_artifact_counts": dict(Counter(str(row["artifact_label"]) for row in tampered)),
            "tampered_predicted_target_counts": dict(Counter(str(row["target_label"]) for row in tampered)),
            "tampered_by_mask_area_bucket": {
                bucket: {
                    "n": len(bucket_rows),
                    "artifact_pass": float(np.mean([row["artifact"] for row in bucket_rows])),
                    "target_pass": float(np.mean([row["target"] for row in bucket_rows])),
                    "spatial_pass_0_1": float(np.mean([row["spatial"] for row in bucket_rows])),
                    "joint_0_1": float(np.mean([row["joint"] for row in bucket_rows])),
                }
                for bucket, bucket_rows in sorted(
                    {
                        value: [row for row in tampered if row["mask_area_bucket"] == value]
                        for value in {row["mask_area_bucket"] for row in tampered}
                    }.items()
                )
            },
            "tampered_joint_passing_ids": sorted(row["img_id"] for row in tampered if row["joint"]),
        }

    comparisons: dict[str, Any] = {}
    for name, rows in variants.items():
        if name == args.baseline:
            continue
        common_ids = set(row["img_id"] for row in baseline)
        assert_exact_ids(common_ids, {row["img_id"] for row in rows}, f"variant {name}")
        endpoints = {}
        endpoint_names = (
            ["balanced_accuracy", "negative_ftr", "tampered_recall", "tampered_spatial", "tampered_joint"]
            if args.design == "sid_explain_300_views"
            else ["tampered_recall", "tampered_spatial", "tampered_joint"]
        )
        for endpoint in endpoint_names:
            subset = endpoint_subset(baseline, endpoint)
            if not subset:
                continue
            seed = derived_seed(args.design, args.baseline, name, endpoint)
            endpoints[endpoint] = {
                "paired_delta_variant_minus_baseline": paired_bootstrap_delta(
                    baseline, rows, endpoint, seed
                ),
                "exact_mcnemar": exact_mcnemar(baseline, rows, endpoint),
                "inference_status": (
                    "primary_family"
                    if endpoint == primary_endpoint and name in args.primary_comparison
                    else (
                        "post_hoc_exploratory_unadjusted"
                        if args.design == "sid_fa600_prompt_sensitivity"
                        else "exploratory_unadjusted"
                    )
                ),
            }
        baseline_pass = set(summaries[args.baseline]["tampered_joint_passing_ids"])
        variant_pass = set(summaries[name]["tampered_joint_passing_ids"])
        union = baseline_pass | variant_pass
        comparisons[name] = {
            "common_sample_count": len(common_ids),
            "endpoints": endpoints,
            "tampered_joint_passing_set": {
                "baseline_count": len(baseline_pass),
                "variant_count": len(variant_pass),
                "intersection": len(baseline_pass & variant_pass),
                "union": len(union),
                "jaccard": len(baseline_pass & variant_pass) / len(union) if union else 1.0,
            },
        }

    if primary_endpoint is not None:
        if not args.primary_comparison:
            raise ValueError("At least one --primary-comparison is required")
        if len(set(args.primary_comparison)) != len(args.primary_comparison):
            raise ValueError("Primary comparisons must be unique")
        unknown_primary = sorted(set(args.primary_comparison) - set(comparisons))
        if unknown_primary:
            raise ValueError(f"Unknown primary comparisons: {unknown_primary}")
        primary_names = list(args.primary_comparison)
        raw_p = {
            name: comparisons[name]["endpoints"][primary_endpoint]["exact_mcnemar"]["p_value"]
            for name in primary_names
        }
        adjusted = holm_adjust(raw_p)
        for name, value in adjusted.items():
            comparisons[name]["endpoints"][primary_endpoint]["holm_adjusted_p_value"] = value
            comparisons[name]["endpoints"][primary_endpoint]["holm_family_size"] = len(raw_p)
    else:
        if args.primary_comparison:
            raise ValueError(f"{args.design} is diagnostic and accepts no primary comparisons")
        primary_names = []
        raw_p = {}

    report = {
        "stage": stage,
        "design": args.design,
        "baseline": args.baseline,
        "variant_summaries": summaries,
        "paired_comparisons": comparisons,
        "primary_endpoint": primary_endpoint,
        "primary_comparisons": primary_names,
        "multiplicity": (
            {
                "method": "Holm FWER 0.05 across the named primary comparisons",
                "family_size": len(raw_p),
                "direction": "lower is better" if primary_endpoint == "negative_ftr" else "higher is better",
                "confidence_intervals": "pointwise and unadjusted",
                "secondary_endpoint_p_values": "exploratory and unadjusted",
            }
            if primary_endpoint is not None
            else {
                "method": "none; diagnostic analysis with no primary hypothesis family",
                "family_size": 0,
                "confidence_intervals": "pointwise and exploratory",
                "p_values": "exploratory and unadjusted",
            }
        ),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_derivation": "SHA-256(design, baseline, variant, endpoint)",
            "frozen_derangement_conditioned_on_one_permutation": args.design == "sid_explain_300_views",
        },
        "input_hashes": {
            name: {
                kind: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for kind, path in artifact_paths.items()
            }
            for name, artifact_paths in paths.items()
        },
        "input_diagnostics": input_diagnostics,
    }
    write_json(args.output, report)
    baseline_ids = sorted(row["img_id"] for row in baseline)
    variant_maps = {name: {row["img_id"]: row for row in rows} for name, rows in variants.items()}
    matched_outcomes = []
    for img_id in baseline_ids:
        base_row = variant_maps[args.baseline][img_id]
        matched_outcomes.append(
            {
                "sample_id": img_id,
                "gold_class": base_row["gold_class"],
                "mask_area_bucket": base_row["mask_area_bucket"],
                "conditions": {
                    name: {
                        "available": True,
                        "predicted_class": mapping[img_id]["predicted_class"],
                        "parsed": mapping[img_id]["parsed"],
                        "classification_correct": mapping[img_id]["correct"],
                        "false_tampered": mapping[img_id]["false_tampered"],
                        "artifact_pass": mapping[img_id]["artifact"],
                        "target_pass": mapping[img_id]["target"],
                        "spatial_pass_0_1": mapping[img_id]["spatial"],
                        "strict_joint_0_1": mapping[img_id]["joint"],
                        "candidate_verdict": mapping[img_id]["candidate_verdict"],
                        "artifact_label": mapping[img_id]["artifact_label"],
                        "target_label": mapping[img_id]["target_label"],
                        "candidate_bbox": mapping[img_id]["candidate_bbox"],
                        "evaluated_bbox": mapping[img_id]["evaluated_bbox"],
                        "bbox_source": mapping[img_id]["bbox_source"],
                        "bbox_iou": mapping[img_id]["bbox_iou"],
                        "view_mode": mapping[img_id]["view_mode"],
                        "crop_donor_img_id": mapping[img_id]["crop_donor_img_id"],
                        "proposal_donor_img_id": mapping[img_id]["proposal_donor_img_id"],
                        "request_sha256": mapping[img_id]["request_sha256"],
                        "prompt_sha256": mapping[img_id]["prompt_sha256"],
                        "output_sha256": mapping[img_id]["output_sha256"],
                        "prediction_row_sha256": mapping[img_id]["prediction_row_sha256"],
                        "inference_provenance_id": mapping[img_id]["inference_provenance_id"],
                        "output_error": mapping[img_id]["output_error"],
                        "failure_signatures": mapping[img_id]["failure_signatures"],
                    }
                    for name, mapping in variant_maps.items()
                },
            }
        )
    write_jsonl(args.trace_output, matched_outcomes)
    lines = [
        "# Proposal Provenance and View Decoupling",
        "",
        "| Condition | N | Parse | Balanced acc. | Negative false-tampered | Tampered recall | Artifact | Target | Spatial@0.1 | Joint@0.1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        fmt = lambda value: "--" if value is None else f"{value * 100:.2f}"
        lines.append(
            f"| {name} | {summary['n']} | {fmt(summary['parse_rate'])} | {fmt(summary['balanced_accuracy'])} | "
            f"{fmt(summary['negative_false_tampered_emission'])} | {fmt(summary['tampered_recall'])} | "
            f"{fmt(summary['tampered_artifact_pass'])} | {fmt(summary['tampered_target_pass'])} | "
            f"{fmt(summary['tampered_spatial_pass_0_1'])} | {fmt(summary['tampered_joint_0_1'])} |"
        )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def bounded_bbox(value: Any, width: int, height: int) -> tuple[list[int] | None, str]:
    bbox = valid_bbox(value)
    if bbox is None:
        return None, "missing_or_malformed"
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return None, "out_of_bounds"
    return bbox, "valid"


def bbox_refine_analysis(args: argparse.Namespace) -> None:
    annotations = unique_row_map(args.annotations)
    requests = unique_row_map(args.requests)
    outputs = unique_row_map(args.outputs)
    expected_ids = set(annotations)
    assert_exact_ids(expected_ids, set(requests), "bbox-refinement requests")
    assert_exact_ids(expected_ids, set(outputs), "bbox-refinement outputs")
    class_counts = Counter(str(gold_item(row)["prediction"]) for row in annotations.values())
    if len(expected_ids) != 100 or class_counts != Counter({"tampered": 100}):
        raise ValueError(f"BBox refinement requires 100 tampered reports; observed {dict(class_counts)}")
    band_counts = Counter(
        mask_area_bucket(row)
        for row in annotations.values()
    )
    if band_counts != Counter({name: 25 for name in MASK_AREA_BUCKETS}):
        raise ValueError(f"Unexpected bbox-refinement mask-area strata: {dict(band_counts)}")

    rows = []
    for img_id in annotations:
        annotation = annotations[img_id]
        request = requests[img_id]
        output = outputs[img_id]
        assert_output_request_link(
            img_id,
            request,
            output,
            "bbox refinement",
            require_request_hash=True,
        )
        meta = request.get("metadata_for_evaluation_only") or {}
        width = int(meta.get("width") or annotation.get("width") or 0)
        height = int(meta.get("height") or annotation.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Missing image dimensions for {img_id}")
        gold_bbox, gold_status = bounded_bbox(gold_item(annotation).get("evidence_bbox"), width, height)
        candidate_bbox, candidate_status = bounded_bbox(meta.get("candidate_bbox"), width, height)
        if gold_status != "valid" or candidate_status != "valid":
            raise ValueError(
                f"Invalid fixed gold/candidate bbox for {img_id}: gold={gold_status}, candidate={candidate_status}"
            )
        result = _result_from_output(output)
        parsed = bool(result) and not output.get("error")
        prediction = normalize_prediction(result.get("prediction")) if parsed else "invalid/unknown"
        if prediction not in CLASS_NAMES:
            prediction = "invalid/unknown"
        raw_bbox = result.get("evidence_bbox") if parsed else None
        refined_bbox, refined_status = bounded_bbox(raw_bbox, width, height)
        shared_gate = bool(parsed and prediction == "tampered")
        candidate_raw_iou = bbox_iou(candidate_bbox, gold_bbox)
        refined_raw_iou = bbox_iou(refined_bbox, gold_bbox)
        candidate_gated_iou = candidate_raw_iou if shared_gate else 0.0
        refined_gated_iou = refined_raw_iou if shared_gate and refined_status == "valid" else 0.0
        if not shared_gate:
            adjustment_category = "shared_gate_failed"
        elif refined_status == "out_of_bounds":
            adjustment_category = "out_of_bounds"
        elif refined_status != "valid":
            adjustment_category = "missing_or_malformed"
        elif refined_bbox == candidate_bbox:
            adjustment_category = "copied_exactly"
        else:
            adjustment_category = "valid_changed"
        failures = []
        if not parsed:
            failures.append("parse_failure")
        elif prediction != "tampered":
            failures.append(f"non_tampered_prediction_{prediction}")
        if refined_status != "valid":
            failures.append(f"refined_bbox_{refined_status}")
        rows.append(
            {
                "img_id": img_id,
                "gold_class": "tampered",
                "mask_area_bucket": mask_area_bucket(annotation),
                "width": width,
                "height": height,
                "parsed": parsed,
                "prediction": prediction,
                "shared_gate_parse_and_tampered": shared_gate,
                "candidate_bbox": candidate_bbox,
                "raw_model_bbox": raw_bbox,
                "refined_bbox": refined_bbox,
                "refined_bbox_status": refined_status,
                "adjustment_category": adjustment_category,
                "candidate_raw_iou": candidate_raw_iou,
                "refined_raw_iou": refined_raw_iou,
                "candidate_gated_iou": candidate_gated_iou,
                "refined_gated_iou": refined_gated_iou,
                "request_sha256": canonical_hash(request),
                "prompt_sha256": hashlib.sha256(
                    str(request.get("prompt") or "").encode("utf-8")
                ).hexdigest(),
                "output_sha256": canonical_hash(output),
                "inference_provenance_id": output.get("inference_provenance_id"),
                "output_error": output.get("error"),
                "failure_signatures": failures,
            }
        )

    def weighted_rate(value: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        intervals = []
        for bucket in MASK_AREA_BUCKETS:
            bucket_rows = [row for row in rows if row["mask_area_bucket"] == bucket]
            intervals.append(wilson(sum(value(row) for row in bucket_rows), len(bucket_rows)))
        result = mover_weighted(intervals, [0.25] * len(MASK_AREA_BUCKETS))
        result["by_stratum"] = dict(zip(MASK_AREA_BUCKETS, intervals))
        return result

    def weighted_metric(sample: list[dict[str, Any]], field: str) -> float:
        return fixed_weight_mean(
            sample,
            lambda row: row["mask_area_bucket"],
            lambda row: float(row[field]),
            MASK_AREA_BUCKETS,
        )

    def delta_stat(sample: list[dict[str, Any]], key: str) -> float:
        if key == "mean_iou":
            return weighted_metric(sample, "refined_gated_iou") - weighted_metric(
                sample, "candidate_gated_iou"
            )
        threshold = float(key.split("@")[1])
        return fixed_weight_mean(
            sample,
            lambda row: row["mask_area_bucket"],
            lambda row: float(row["refined_gated_iou"] >= threshold)
            - float(row["candidate_gated_iou"] >= threshold),
            MASK_AREA_BUCKETS,
        )

    contrasts: dict[str, Any] = {}
    for key in ["mean_iou", "iou@0.1", "iou@0.3", "iou@0.5"]:
        seed = derived_seed("bbox_refinement", key)
        rng = np.random.default_rng(seed)
        groups = {
            bucket: [row for row in rows if row["mask_area_bucket"] == bucket]
            for bucket in MASK_AREA_BUCKETS
        }
        draws = []
        for _ in range(BOOTSTRAP_REPLICATES):
            sample = []
            for bucket in MASK_AREA_BUCKETS:
                group = groups[bucket]
                indices = rng.integers(0, len(group), size=len(group))
                sample.extend(group[index] for index in indices)
            draws.append(delta_stat(sample, key))
        contrasts[key] = {
            "estimate": delta_stat(rows, key),
            **percentile_interval(draws),
            "seed": seed,
            "replicates": BOOTSTRAP_REPLICATES,
            "fixed_equal_weight_strata": MASK_AREA_BUCKETS,
        }

    threshold_tests: dict[str, Any] = {}
    for threshold in [0.1, 0.3, 0.5]:
        candidate_success = [row["candidate_gated_iou"] >= threshold for row in rows]
        refined_success = [row["refined_gated_iou"] >= threshold for row in rows]
        b = sum(base and not variant for base, variant in zip(candidate_success, refined_success))
        c = sum(not base and variant for base, variant in zip(candidate_success, refined_success))
        threshold_tests[str(threshold)] = exact_two_sided_binomial(b, c)
    adjusted = holm_adjust(
        {threshold: result["p_value"] for threshold, result in threshold_tests.items()}
    )
    for threshold, value in adjusted.items():
        threshold_tests[threshold]["holm_adjusted_p_value"] = value
        threshold_tests[threshold]["holm_family_size"] = len(threshold_tests)

    category_counts = Counter(row["adjustment_category"] for row in rows)
    candidate_summary = {
        "mean_iou": weighted_metric(rows, "candidate_gated_iou"),
        **{
            f"iou_{threshold}": fixed_weight_mean(
                rows,
                lambda row: row["mask_area_bucket"],
                lambda row, threshold=threshold: float(row["candidate_gated_iou"] >= threshold),
                MASK_AREA_BUCKETS,
            )
            for threshold in [0.1, 0.3, 0.5]
        },
    }
    refined_summary = {
        "mean_iou": weighted_metric(rows, "refined_gated_iou"),
        **{
            f"iou_{threshold}": fixed_weight_mean(
                rows,
                lambda row: row["mask_area_bucket"],
                lambda row, threshold=threshold: float(row["refined_gated_iou"] >= threshold),
                MASK_AREA_BUCKETS,
            )
            for threshold in [0.1, 0.3, 0.5]
        },
    }
    report = {
        "stage": "proposal_conditioned_model_emitted_bbox_adjustment",
        "estimand": (
            "full-denominator fixed four-band mixture with the shared gate "
            "parse-valid AND predicted-tampered applied to both candidate and model-emitted boxes"
        ),
        "interpretation_scope": (
            "The model still observes the supplied marked region and crop; this is a proposal-conditioned "
            "adjustment diagnostic, not independent VLM localization. No refinement Joint is formed."
        ),
        "samples": len(rows),
        "mask_area_counts": dict(band_counts),
        "parse_rate": weighted_rate(lambda row: row["parsed"]),
        "tampered_prediction_rate": weighted_rate(lambda row: row["prediction"] == "tampered"),
        "shared_gate_rate": weighted_rate(lambda row: row["shared_gate_parse_and_tampered"]),
        "adjustment_category_counts": dict(category_counts),
        "valid_changed_rate_full_denominator": weighted_rate(
            lambda row: row["adjustment_category"] == "valid_changed"
        ),
        "candidate_shared_gate": candidate_summary,
        "refined_shared_gate": refined_summary,
        "raw_geometry_descriptive": {
            "candidate_mean_iou": fixed_weight_mean(
                rows,
                lambda row: row["mask_area_bucket"],
                lambda row: row["candidate_raw_iou"],
                MASK_AREA_BUCKETS,
            ),
            "valid_refined_bbox_count": sum(row["refined_bbox_status"] == "valid" for row in rows),
            "valid_refined_mean_iou_full_denominator": fixed_weight_mean(
                rows,
                lambda row: row["mask_area_bucket"],
                lambda row: row["refined_raw_iou"] if row["refined_bbox_status"] == "valid" else 0.0,
                MASK_AREA_BUCKETS,
            ),
        },
        "refined_better_count": sum(
            row["refined_gated_iou"] > row["candidate_gated_iou"] for row in rows
        ),
        "refined_worse_count": sum(
            row["refined_gated_iou"] < row["candidate_gated_iou"] for row in rows
        ),
        "refined_equal_count": sum(
            row["refined_gated_iou"] == row["candidate_gated_iou"] for row in rows
        ),
        "paired_refined_minus_candidate": contrasts,
        "threshold_exact_mcnemar": threshold_tests,
        "multiplicity": (
            "Holm FWER 0.05 across Spatial@0.1, @0.3, and @0.5 exact McNemar tests; "
            "bootstrap confidence intervals remain pointwise and unadjusted"
        ),
        "linkage_verification": {
            "request_hash_verified_rows": len(rows),
            "output_request_asset_fields_checked": True,
        },
        "inputs": {
            "annotations": str(args.annotations.relative_to(ROOT)),
            "requests": str(args.requests.relative_to(ROOT)),
            "outputs": str(args.outputs.relative_to(ROOT)),
            "hashes": {
                "annotations": sha256(args.annotations),
                "requests": sha256(args.requests),
                "outputs": sha256(args.outputs),
            },
        },
    }
    write_json(args.output, report)
    write_jsonl(args.trace_output, rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def negative_interface_analysis(args: argparse.Namespace) -> None:
    annotations = unique_row_map(args.annotations)
    candidate = _prediction_row_map(args.candidate_predictions)
    candidate_requests = unique_row_map(args.candidate_requests)
    candidate_outputs = unique_row_map(args.candidate_outputs)
    dedicated = unique_row_map(args.dedicated_predictions)
    expected_ids = set(annotations)
    assert_exact_ids(expected_ids, set(candidate), "candidate-conditioned predictions")
    assert_exact_ids(expected_ids, set(candidate_requests), "candidate-conditioned requests")
    assert_exact_ids(expected_ids, set(candidate_outputs), "candidate-conditioned outputs")
    assert_exact_ids(expected_ids, set(dedicated), "dedicated predictions")
    gold_counts = Counter(str(row.get("human_label")) for row in annotations.values())
    if len(expected_ids) != 2000 or gold_counts != Counter({"real": 1000, "full_synthetic": 1000}):
        raise ValueError(
            f"Negative-interface design requires 1,000 real and 1,000 full-synthetic IDs; "
            f"observed {dict(gold_counts)}"
        )
    rows = []
    for img_id in annotations:
        assert_candidate_prediction_link(
            img_id,
            candidate_requests[img_id],
            candidate_outputs[img_id],
            candidate[img_id],
            "negative-interface candidate arm",
            require_request_hash=True,
        )
        gold_class = str(annotations[img_id]["human_label"])
        candidate_prediction = candidate[img_id].get("prediction") or {}
        candidate_parsed = bool(candidate_prediction.get("parsed"))
        candidate_class = (
            normalize_prediction(candidate_prediction.get("prediction"))
            if candidate_parsed
            else "invalid/unknown"
        )
        if candidate_class not in CLASS_NAMES:
            candidate_class = "invalid/unknown"
        dedicated_row = dedicated[img_id]
        if normalize_prediction(dedicated_row.get("true_class")) != gold_class:
            raise ValueError(f"Dedicated true_class disagrees with annotation for {img_id}")
        dedicated_parsed = bool(dedicated_row.get("parsed"))
        dedicated_class = normalize_prediction(dedicated_row.get("pred_class")) if dedicated_parsed else "invalid/unknown"
        if dedicated_class not in CLASS_NAMES:
            dedicated_class = "invalid/unknown"
        if dedicated_class == "tampered":
            raise ValueError(f"Dedicated binary output unexpectedly emitted tampered for {img_id}")
        candidate_correct = bool(candidate_parsed and candidate_class == gold_class)
        dedicated_correct = bool(dedicated_parsed and dedicated_class == gold_class)
        failures = []
        if not candidate_parsed:
            failures.append("candidate_parse_failure")
        elif candidate_class == "invalid/unknown":
            failures.append("candidate_out_of_schema")
        if not dedicated_parsed:
            failures.append("dedicated_parse_failure")
        elif dedicated_class == "invalid/unknown":
            failures.append("dedicated_out_of_schema")
        if candidate_parsed and candidate_class == "tampered":
            failures.append("candidate_spurious_tampered")
        rows.append(
            {
                "img_id": img_id,
                "gold_class": gold_class,
                "candidate_class": candidate_class,
                "dedicated_class": dedicated_class,
                "candidate_parsed": candidate_parsed,
                "dedicated_parsed": dedicated_parsed,
                "candidate_correct": candidate_correct,
                "dedicated_correct": dedicated_correct,
                "candidate_false_tampered": bool(candidate_parsed and candidate_class == "tampered"),
                "candidate_prediction_row_sha256": canonical_hash(candidate[img_id]),
                "dedicated_prediction_row_sha256": canonical_hash(dedicated_row),
                "failure_signatures": failures,
            }
        )

    def paired_delta(sample: list[dict[str, Any]]) -> float:
        return fixed_weight_mean(
            sample,
            lambda row: row["gold_class"],
            lambda row: float(row["candidate_correct"]) - float(row["dedicated_correct"]),
            ["real", "full_synthetic"],
        )

    observed_delta = paired_delta(rows)
    bootstrap_seed = derived_seed("negative_interface", "accuracy_delta")
    rng = np.random.default_rng(bootstrap_seed)
    groups = {
        class_name: [row for row in rows if row["gold_class"] == class_name]
        for class_name in ["real", "full_synthetic"]
    }
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = []
        for class_name in ["real", "full_synthetic"]:
            group = groups[class_name]
            indices = rng.integers(0, len(group), size=len(group))
            sample.extend(group[index] for index in indices)
        draws.append(paired_delta(sample))
    discordant_dedicated_only = sum(row["dedicated_correct"] and not row["candidate_correct"] for row in rows)
    discordant_candidate_only = sum(row["candidate_correct"] and not row["dedicated_correct"] for row in rows)
    mcnemar = exact_two_sided_binomial(
        discordant_dedicated_only, discordant_candidate_only
    )
    mcnemar["dedicated_only_correct"] = mcnemar.pop("baseline_only_success")
    mcnemar["candidate_only_correct"] = mcnemar.pop("variant_only_success")

    class_metrics = {}
    for class_name in ["real", "full_synthetic"]:
        class_rows = [row for row in rows if row["gold_class"] == class_name]
        class_metrics[class_name] = {
            "n": len(class_rows),
            "candidate_parse_rate": wilson(
                sum(row["candidate_parsed"] for row in class_rows), len(class_rows)
            ),
            "dedicated_parse_rate": wilson(
                sum(row["dedicated_parsed"] for row in class_rows), len(class_rows)
            ),
            "candidate_recall": wilson(sum(row["candidate_correct"] for row in class_rows), len(class_rows)),
            "dedicated_recall": wilson(sum(row["dedicated_correct"] for row in class_rows), len(class_rows)),
            "candidate_false_tampered_emission": wilson(
                sum(row["candidate_false_tampered"] for row in class_rows), len(class_rows)
            ),
        }
    candidate_parse = mover_weighted(
        [class_metrics[name]["candidate_parse_rate"] for name in ["real", "full_synthetic"]],
        [0.5, 0.5],
    )
    dedicated_parse = mover_weighted(
        [class_metrics[name]["dedicated_parse_rate"] for name in ["real", "full_synthetic"]],
        [0.5, 0.5],
    )
    candidate_accuracy = mover_weighted(
        [class_metrics[name]["candidate_recall"] for name in ["real", "full_synthetic"]],
        [0.5, 0.5],
    )
    dedicated_accuracy = mover_weighted(
        [class_metrics[name]["dedicated_recall"] for name in ["real", "full_synthetic"]],
        [0.5, 0.5],
    )
    candidate_ftr = mover_weighted(
        [
            class_metrics[name]["candidate_false_tampered_emission"]
            for name in ["real", "full_synthetic"]
        ],
        [0.5, 0.5],
    )
    confusion_candidate = {
        class_name: dict(Counter(row["candidate_class"] for row in groups[class_name]))
        for class_name in groups
    }
    confusion_dedicated = {
        class_name: dict(Counter(row["dedicated_class"] for row in groups[class_name]))
        for class_name in groups
    }
    report = {
        "stage": "paired_negative_interface_comparison",
        "estimand": "fixed 1:1 real/full-synthetic mixture on identical 2,000 images",
        "samples": len(rows),
        "class_counts": dict(gold_counts),
        "candidate_conditioned_parse_rate": candidate_parse,
        "dedicated_binary_parse_rate": dedicated_parse,
        "candidate_conditioned_accuracy": candidate_accuracy,
        "dedicated_binary_accuracy": dedicated_accuracy,
        "paired_accuracy_delta_candidate_minus_dedicated": {
            "estimate": observed_delta,
            **percentile_interval(draws),
            "seed": bootstrap_seed,
            "replicates": BOOTSTRAP_REPLICATES,
            "fixed_equal_weight_strata": ["real", "full_synthetic"],
        },
        "exact_mcnemar": mcnemar,
        "multiplicity": "single primary accuracy test; no multiplicity adjustment",
        "candidate_false_tampered_emission": candidate_ftr,
        "dedicated_false_tampered_emission": None,
        "dedicated_false_tampered_reason": "N/A: the dedicated output label space is binary",
        "class_metrics": class_metrics,
        "candidate_confusion_matrix_counts": confusion_candidate,
        "dedicated_confusion_matrix_counts": confusion_dedicated,
        "interpretation": (
            "The interfaces differ in prompt, visible evidence, and output label space; the paired "
            "contrast is interface-specific and is not a causal estimate of candidate effects."
        ),
        "bootstrap": {
            "seed": bootstrap_seed,
            "replicates": BOOTSTRAP_REPLICATES,
            "fixed_equal_weight_strata": ["real", "full_synthetic"],
        },
        "linkage_verification": {
            "candidate_request_hash_verified_rows": len(rows),
            "candidate_prediction_output_rederived_rows": len(rows),
            "dedicated_annotation_label_verified_rows": len(rows),
        },
        "inputs": {
            "annotations": str(args.annotations.relative_to(ROOT)),
            "candidate_predictions": str(args.candidate_predictions.relative_to(ROOT)),
            "candidate_requests": str(args.candidate_requests.relative_to(ROOT)),
            "candidate_outputs": str(args.candidate_outputs.relative_to(ROOT)),
            "dedicated_predictions": str(args.dedicated_predictions.relative_to(ROOT)),
            "hashes": {
                "annotations": sha256(args.annotations),
                "candidate_predictions": sha256(args.candidate_predictions),
                "candidate_requests": sha256(args.candidate_requests),
                "candidate_outputs": sha256(args.candidate_outputs),
                "dedicated_predictions": sha256(args.dedicated_predictions),
            },
        },
    }
    write_json(args.output, report)
    write_jsonl(args.trace_output, rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def replay_analysis(args: argparse.Namespace) -> None:
    original = unique_row_map(args.original_outputs)
    rerun = unique_row_map(args.rerun_outputs)
    if args.expected_common is not None:
        if len(original) != args.expected_common:
            raise ValueError(
                f"Expected {args.expected_common} original rows, found {len(original)}"
            )
        missing = set(original) - set(rerun)
        if missing:
            raise ValueError(f"Replay omits original IDs: {sorted(missing)[:5]} ({len(missing)} total)")
        common_ids = sorted(original)
    else:
        assert_exact_ids(set(original), set(rerun), "replay outputs")
        common_ids = sorted(original)
    comparisons = []
    for img_id in common_ids:
        source = original[img_id]
        replay = rerun[img_id]
        comparisons.append(
            {
                "img_id": img_id,
                "raw_text_exact": source.get("vlm_raw_text") == replay.get("vlm_raw_text"),
                "parsed_json_exact": source.get("vlm_result") == replay.get("vlm_result"),
                "source_error": source.get("error"),
                "replay_error": replay.get("error"),
            }
        )
    raw_exact = sum(row["raw_text_exact"] for row in comparisons)
    json_exact = sum(row["parsed_json_exact"] for row in comparisons)
    error_free = sum(not row["source_error"] and not row["replay_error"] for row in comparisons)
    verdict = (
        "REPRODUCIBLE"
        if raw_exact == len(comparisons)
        and json_exact == len(comparisons)
        and error_free == len(comparisons)
        else "NOT_REPRODUCIBLE"
    )
    report = {
        "stage": "deterministic_replay_verification",
        "classification": "deterministic greedy inference under matched checkpoint, code, data, and environment",
        "verdict": verdict,
        "common_rows": len(comparisons),
        "raw_text_exact_rows": raw_exact,
        "parsed_json_exact_rows": json_exact,
        "error_free_rows": error_free,
        "mismatch_ids": [
            row["img_id"]
            for row in comparisons
            if not row["raw_text_exact"] or not row["parsed_json_exact"]
        ],
        "timing_compared": False,
        "inputs": {
            "original_outputs": str(args.original_outputs.relative_to(ROOT)),
            "rerun_outputs": str(args.rerun_outputs.relative_to(ROOT)),
            "original_sha256": sha256(args.original_outputs),
            "rerun_sha256": sha256(args.rerun_outputs),
        },
        "per_sample": comparisons,
    }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_sample"}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    routing = subparsers.add_parser("routing")
    routing.add_argument(
        "--design", choices=["sid_explain_300", "negative_2000"], required=True
    )
    routing.add_argument("--annotations", type=Path, required=True)
    routing.add_argument("--requests", type=Path, required=True)
    routing.add_argument("--outputs", type=Path, required=True)
    routing.add_argument("--predictions", type=Path, required=True)
    routing.add_argument("--lowlevel", type=Path, required=True)
    routing.add_argument("--estimand")
    routing.add_argument("--output", type=Path, default=DEFAULT_REPORT_DIR / "same_runtime_routing_audit.json")
    routing.add_argument("--trace-output", type=Path, default=ROOT / "experiments" / "runs" / "recommended_suite" / "same_runtime_routing_trace.jsonl")
    routing.add_argument("--summary-output", type=Path, default=DEFAULT_REPORT_DIR / "same_runtime_routing_audit.md")
    routing.set_defaults(func=routing_analysis)

    variants = subparsers.add_parser("variants")
    variants.add_argument(
        "--design",
        choices=[
            "sid_explain_300_views",
            "jsg_core_proposal",
            "sid_fa600_prompt_sensitivity",
            "sid_fa200_model_runtime",
        ],
        required=True,
    )
    variants.add_argument("--annotations", type=Path, required=True)
    variants.add_argument("--variant", type=parse_variant, action="append", required=True)
    variants.add_argument("--baseline", required=True)
    variants.add_argument("--primary-comparison", action="append", default=[])
    variants.add_argument("--output", type=Path, default=DEFAULT_REPORT_DIR / "proposal_view_decoupling.json")
    variants.add_argument("--summary-output", type=Path, default=DEFAULT_REPORT_DIR / "proposal_view_decoupling.md")
    variants.add_argument(
        "--trace-output",
        type=Path,
        default=ROOT / "experiments" / "runs" / "recommended_suite" / "proposal_view_matched_outcomes.jsonl",
    )
    variants.set_defaults(func=variants_analysis)

    bbox = subparsers.add_parser("bbox-refine")
    bbox.add_argument("--annotations", type=Path, required=True)
    bbox.add_argument("--requests", type=Path, required=True)
    bbox.add_argument("--outputs", type=Path, required=True)
    bbox.add_argument("--output", type=Path, default=DEFAULT_REPORT_DIR / "bbox_refinement_diagnostic.json")
    bbox.add_argument(
        "--trace-output",
        type=Path,
        default=ROOT / "experiments" / "runs" / "recommended_suite" / "bbox_refinement_trace.jsonl",
    )
    bbox.set_defaults(func=bbox_refine_analysis)

    interface = subparsers.add_parser("negative-interface")
    interface.add_argument("--annotations", type=Path, required=True)
    interface.add_argument("--candidate-predictions", type=Path, required=True)
    interface.add_argument("--candidate-requests", type=Path, required=True)
    interface.add_argument("--candidate-outputs", type=Path, required=True)
    interface.add_argument("--dedicated-predictions", type=Path, required=True)
    interface.add_argument(
        "--output", type=Path, default=DEFAULT_REPORT_DIR / "paired_negative_interface_comparison.json"
    )
    interface.add_argument(
        "--trace-output",
        type=Path,
        default=ROOT / "experiments" / "runs" / "recommended_suite" / "paired_negative_interface_trace.jsonl",
    )
    interface.set_defaults(func=negative_interface_analysis)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--original-outputs", type=Path, required=True)
    replay.add_argument("--rerun-outputs", type=Path, required=True)
    replay.add_argument("--expected-common", type=int)
    replay.add_argument(
        "--output", type=Path, default=DEFAULT_REPORT_DIR / "core_exact_replay_verification.json"
    )
    replay.set_defaults(func=replay_analysis)

    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if hasattr(args, "variant"):
        args.variant = [
            (name, prediction.resolve(), request.resolve(), output.resolve())
            for name, prediction, request, output in args.variant
        ]
    args.func(args)


if __name__ == "__main__":
    main()
