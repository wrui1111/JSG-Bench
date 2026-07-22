#!/usr/bin/env python3
"""Export the versioned JSG-Bench contract and per-report audit traces.

The exporter joins existing frozen requests, raw native outputs, and normalized
evaluation records by image ID. It performs no model inference and never uses
gold fields to normalize system output.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
REPORTS = ROOT / "experiments" / "reports"
CONTRACT_VERSION = "jsg-bench-audit-v1.0"
RUN_MANIFEST_VERSION = "jsg-run-manifest-v1.0"
IOU_THRESHOLD = 0.1

TD_SPECS = {
    "JSG-Core": {
        "system_id": "TD-EGFA/Qwen2.5-VL-3B-Instruct",
        "input_condition": "known-tampered/proposal-grounded/B_top3",
        "predictions": RUNS / "sid_explain300_tampered100_candidate_top3_revised_predictions.jsonl",
        "requests": RUNS / "sid_explain300_tampered100_candidate_top3_requests.jsonl",
        "raw_outputs": RUNS / "sid_explain300_tampered100_candidate_top3_qwen25vl_outputs.jsonl",
        "output": RUNS / "jsg_core_td_egfa_audit_trace_v1.jsonl",
        "target_scope_labels": [
            "none",
            "face",
            "object",
            "background",
            "text",
            "whole_image",
            "other",
        ],
    },
    "JSG-Scale": {
        "system_id": "TD-EGFA/Qwen2.5-VL-3B-Instruct",
        "input_condition": "known-tampered/proposal-grounded/B_top3",
        "predictions": RUNS / "sid_fa600_candidate_top3_postreview_predictions.jsonl",
        "requests": RUNS / "sid_fa600_candidate_top3_requests.jsonl",
        "raw_outputs": RUNS / "sid_fa600_candidate_top3_qwen25vl_outputs.jsonl",
        "output": RUNS / "jsg_scale_td_egfa_audit_trace_v1.jsonl",
        "target_scope_labels": [
            "none",
            "face",
            "person",
            "animal",
            "object",
            "background",
            "text",
            "whole_image",
            "other",
        ],
    },
}

SIDA_SOURCE = RUNS / "sida_jsg_core_adapted_predictions.jsonl"
SIDA_OUTPUT = RUNS / "jsg_core_sida_audit_trace_v1.jsonl"

XFER_SEMANTIC = RUNS / "external_semantic_eval_rows.jsonl"
XFER_LOWLEVEL = RUNS / "jsg_xfer_td_egfa_lowlevel_v2.jsonl"
XFER_PREDICTIONS = [
    RUNS / "jsg_xfer_td_egfa_predictions_v2.jsonl",
]
XFER_REQUESTS = [
    RUNS / "jsg_xfer_td_egfa_requests_v2.jsonl",
]
XFER_RAW_OUTPUTS = [
    RUNS / "jsg_xfer_td_egfa_native_outputs_v2.jsonl",
]
XFER_OUTPUT = RUNS / "jsg_xfer_td_egfa_audit_trace_v1.jsonl"


def read_jsonl_indexed(path: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[int, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    index: dict[str, tuple[int, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            img_id = str(row["img_id"])
            if img_id in index:
                raise ValueError(f"Duplicate img_id {img_id} in {path}")
            rows.append(row)
            index[img_id] = (line_number, row)
    return rows, index


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference(path: Path, line_number: int) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "line": line_number}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def top3_union(row: dict[str, Any]) -> list[int] | None:
    boxes = [
        item.get("bbox")
        for item in (row.get("lowlevel_candidates") or [])[:3]
        if isinstance(item, dict)
        and isinstance(item.get("bbox"), list)
        and len(item["bbox"]) == 4
    ]
    if not boxes:
        candidate = row.get("lowlevel_candidate_bbox")
        return candidate if isinstance(candidate, list) and len(candidate) == 4 else None
    return [
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    ]


def merged_index(paths: list[Path]) -> dict[str, tuple[Path, int, dict[str, Any]]]:
    index: dict[str, tuple[Path, int, dict[str, Any]]] = {}
    for path in paths:
        _, local_index = read_jsonl_indexed(path)
        for img_id, (line_number, row) in local_index.items():
            if img_id in index:
                raise ValueError(f"Duplicate img_id {img_id} across merged sources")
            index[img_id] = (path, line_number, row)
    return index


def profile(conditions: dict[str, bool], condition_order: list[str]) -> dict[str, Any]:
    passes = {name: bool(conditions[name]) for name in condition_order}
    failed = [name for name in condition_order if not passes[name]]
    return {
        "condition_order": condition_order,
        "pass_vector": [passes[name] for name in condition_order],
        "conditions": passes,
        "joint": not failed,
        "failed_conditions": failed,
    }


def td_trace_rows(dataset: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    prediction_rows, predictions = read_jsonl_indexed(spec["predictions"])
    _, requests = read_jsonl_indexed(spec["requests"])
    _, raw_outputs = read_jsonl_indexed(spec["raw_outputs"])
    if set(predictions) != set(requests) or set(predictions) != set(raw_outputs):
        raise ValueError(f"Unaligned TD-EGFA sources for {dataset}")

    traces: list[dict[str, Any]] = []
    for prediction_row in prediction_rows:
        img_id = str(prediction_row["img_id"])
        prediction_line, frozen = predictions[img_id]
        request_line, request = requests[img_id]
        raw_line, _ = raw_outputs[img_id]
        normalized = frozen.get("prediction") or {}
        parse_valid = bool(normalized.get("parsed", True))
        output_bbox = normalized.get("evidence_bbox")
        gold = frozen.get("gold") or {}
        recomputed_semantic = {
            "prediction_correct": normalized.get("prediction") == gold.get("prediction"),
            "artifact_correct": normalized.get("artifact_type") == gold.get("artifact_type"),
            "target_correct": normalized.get("target_scope") == gold.get("target_scope"),
        }
        for key, value in recomputed_semantic.items():
            if bool(frozen.get(key)) != bool(parse_valid and value):
                raise ValueError(f"Stored/recomputed {key} mismatch for {dataset}/{img_id}")
        if output_bbox is not None:
            candidate_bbox = (request.get("metadata_for_evaluation_only") or {}).get("candidate_bbox")
            if output_bbox != candidate_bbox:
                raise ValueError(f"Output/proposal bbox mismatch for {dataset}/{img_id}")
            recomputed_iou = bbox_iou(output_bbox, gold["evidence_bbox"])
            if not math.isclose(recomputed_iou, float(frozen.get("bbox_iou") or 0.0), abs_tol=1e-12):
                raise ValueError(f"Stored/recomputed IoU mismatch for {dataset}/{img_id}")
        spatial = bool(parse_valid and output_bbox is not None and float(frozen.get("bbox_iou") or 0.0) >= IOU_THRESHOLD)
        conditions = {
            "parse": parse_valid,
            "prediction": bool(parse_valid and frozen.get("prediction_correct")),
            "artifact": bool(parse_valid and frozen.get("artifact_correct")),
            "target": bool(parse_valid and frozen.get("target_correct")),
            "spatial": spatial,
        }
        primary = profile(conditions, ["parse", "prediction", "artifact", "target", "spatial"])
        shared = profile(conditions, ["parse", "prediction", "target", "spatial"])
        request_metadata = request.get("metadata_for_evaluation_only") or {}
        traces.append(
            {
                "contract_version": CONTRACT_VERSION,
                "dataset": dataset,
                "sample_id": img_id,
                "system_id": spec["system_id"],
                "input_condition": spec["input_condition"],
                "parser_or_adapter_version": "td-egfa-json-normalizer-v1",
                "prompt_version": request.get("prompt_version"),
                "source_references": {
                    "request": reference(spec["requests"], request_line),
                    "raw_output": reference(spec["raw_outputs"], raw_line),
                    "normalized_evaluation": reference(spec["predictions"], prediction_line),
                },
                "normalized_fields": {
                    "prediction": normalized.get("prediction"),
                    "artifact_type": normalized.get("artifact_type"),
                    "target_scope": normalized.get("target_scope"),
                    "evidence_bbox": output_bbox,
                },
                "applicability": {
                    "prediction": True,
                    "artifact": True,
                    "target": True,
                    "spatial": True,
                    "gold_class": "tampered",
                },
                "evaluated_region": {
                    "bbox": output_bbox,
                    "provenance": "inherited_B_top3" if output_bbox is not None else "null_after_parse_or_rejection",
                    "candidate_bbox": request_metadata.get("candidate_bbox"),
                    "candidate_mode": request_metadata.get("candidate_mode"),
                    "iou_threshold": IOU_THRESHOLD,
                    "iou": float(frozen.get("bbox_iou") or 0.0),
                },
                "audit_profiles": {
                    "primary_strict": primary,
                    "shared_fields": shared,
                    "set_compatible": primary,
                },
            }
        )
    return traces


def sida_trace_rows() -> list[dict[str, Any]]:
    source_rows, source_index = read_jsonl_indexed(SIDA_SOURCE)
    traces: list[dict[str, Any]] = []
    for source_row in source_rows:
        img_id = str(source_row["img_id"])
        source_line, row = source_index[img_id]
        strict_conditions = {name: bool(value) for name, value in (row.get("conditions") or {}).items()}
        set_conditions = {
            name: bool(value) for name, value in (row.get("artifact_set_conditions") or {}).items()
        }
        for required in ("parse", "prediction", "artifact", "target", "spatial"):
            if required not in strict_conditions or required not in set_conditions:
                raise ValueError(f"Missing SIDA condition {required} for {img_id}")
        traces.append(
            {
                "contract_version": CONTRACT_VERSION,
                "dataset": "JSG-Core",
                "sample_id": img_id,
                "native_sample_id": row.get("sample_id"),
                "system_id": row.get("system_id") or "SIDA-7B-description-FP16",
                "input_condition": "known-tampered/native-original-image",
                "parser_or_adapter_version": "sida-native-posthoc-adapter-v1",
                "prompt_version": "sida-verbatim-native-instruction-v1",
                "source_references": {
                    "native_and_normalized_output": reference(SIDA_SOURCE, source_line),
                },
                "normalized_fields": {
                    "prediction": row.get("prediction"),
                    "artifact_type": row.get("artifact_type"),
                    "artifact_set": row.get("artifact_set") or [],
                    "target_scope": row.get("target_scope"),
                    "evidence_bbox": row.get("evidence_bbox"),
                },
                "applicability": {
                    "prediction": True,
                    "artifact": True,
                    "target": True,
                    "spatial": True,
                    "gold_class": "tampered",
                },
                "evaluated_region": {
                    "bbox": row.get("evidence_bbox"),
                    "provenance": row.get("bbox_source"),
                    "native_mask_paths": row.get("native_mask_paths") or [],
                    "iou_threshold": IOU_THRESHOLD,
                    "iou": float(row.get("bbox_iou") or 0.0),
                },
                "audit_profiles": {
                    "primary_strict": profile(
                        strict_conditions,
                        ["parse", "prediction", "artifact", "target", "spatial"],
                    ),
                    "shared_fields": profile(
                        strict_conditions,
                        ["parse", "prediction", "target", "spatial"],
                    ),
                    "set_compatible": profile(
                        set_conditions,
                        ["parse", "prediction", "artifact", "target", "spatial"],
                    ),
                },
            }
        )
    return traces


def xfer_trace_rows() -> list[dict[str, Any]]:
    semantic_rows, semantic_index = read_jsonl_indexed(XFER_SEMANTIC)
    lowlevels = merged_index([XFER_LOWLEVEL])
    predictions = merged_index(XFER_PREDICTIONS)
    requests = merged_index(XFER_REQUESTS)
    raw_outputs = merged_index(XFER_RAW_OUTPUTS)
    expected_ids = set(semantic_index)
    for source_name, source in (
        ("lowlevel", lowlevels),
        ("predictions", predictions),
        ("requests", requests),
        ("raw outputs", raw_outputs),
    ):
        if set(source) != expected_ids:
            raise ValueError(f"Unaligned JSG-Xfer {source_name} IDs")
    traces: list[dict[str, Any]] = []
    for semantic_row in semantic_rows:
        img_id = str(semantic_row["img_id"])
        if img_id not in predictions or img_id not in requests or img_id not in raw_outputs:
            raise ValueError(f"Missing JSG-Xfer source for {img_id}")
        semantic_line, semantic = semantic_index[img_id]
        lowlevel_path, lowlevel_line, lowlevel = lowlevels[img_id]
        prediction_path, prediction_line, prediction = predictions[img_id]
        request_path, request_line, request = requests[img_id]
        raw_path, raw_line, raw = raw_outputs[img_id]
        normalized_raw = raw.get("vlm_result") or {}
        parse_valid = bool(normalized_raw)
        output_bbox = prediction.get("pred_bbox")
        request_metadata = request.get("metadata_for_evaluation_only") or {}
        candidate_bbox = request_metadata.get("candidate_bbox")
        if top3_union(lowlevel) != candidate_bbox:
            raise ValueError(f"Low-level/request candidate bbox mismatch for JSG-Xfer/{img_id}")
        if output_bbox is not None and output_bbox != candidate_bbox:
            raise ValueError(f"Prediction/request candidate bbox mismatch for JSG-Xfer/{img_id}")
        if parse_valid:
            for raw_key, semantic_key in (("artifact_type", "pred_artifact"), ("target_scope", "pred_target")):
                if normalized_raw.get(raw_key) != semantic.get(semantic_key):
                    raise ValueError(f"Raw/semantic {raw_key} mismatch for JSG-Xfer/{img_id}")
        if output_bbox is not None:
            recomputed_iou = bbox_iou(output_bbox, prediction["gt_bbox"])
            if not math.isclose(recomputed_iou, float(semantic.get("bbox_iou") or 0.0), abs_tol=1e-12):
                raise ValueError(f"Stored/recomputed IoU mismatch for JSG-Xfer/{img_id}")
        conditions = {
            "parse": parse_valid,
            "prediction": bool(parse_valid and prediction.get("prediction") == "tampered"),
            "artifact": bool(parse_valid and semantic.get("artifact_correct")),
            "target": bool(parse_valid and semantic.get("target_correct")),
            "spatial": bool(
                parse_valid
                and output_bbox is not None
                and float(semantic.get("bbox_iou") or 0.0) >= IOU_THRESHOLD
            ),
        }
        set_conditions = {
            **conditions,
            "artifact": bool(parse_valid and semantic.get("artifact_primary_or_secondary_correct")),
        }
        primary = profile(conditions, ["parse", "prediction", "artifact", "target", "spatial"])
        shared = profile(conditions, ["parse", "prediction", "target", "spatial"])
        set_compatible = profile(
            set_conditions,
            ["parse", "prediction", "artifact", "target", "spatial"],
        )
        traces.append(
            {
                "contract_version": CONTRACT_VERSION,
                "dataset": "JSG-Xfer",
                "external_dataset": semantic.get("external_dataset"),
                "sample_id": img_id,
                "system_id": "TD-EGFA/Qwen2.5-VL-3B-Instruct",
                "input_condition": "known-tampered/proposal-grounded/B_top3/stratified-external",
                "parser_or_adapter_version": "td-egfa-external-semantic-normalizer-v1",
                "prompt_version": request.get("prompt_version"),
                "source_references": {
                    "lowlevel_candidate": reference(lowlevel_path, lowlevel_line),
                    "request": reference(request_path, request_line),
                    "raw_output": reference(raw_path, raw_line),
                    "candidate_prediction": reference(prediction_path, prediction_line),
                    "semantic_evaluation": reference(XFER_SEMANTIC, semantic_line),
                },
                "normalized_fields": {
                    "prediction": prediction.get("prediction"),
                    "artifact_type": semantic.get("pred_artifact"),
                    "target_scope": semantic.get("pred_target"),
                    "evidence_bbox": output_bbox,
                },
                "applicability": {
                    "prediction": True,
                    "artifact": True,
                    "target": True,
                    "spatial": True,
                    "gold_class": "tampered",
                },
                "evaluated_region": {
                    "bbox": output_bbox,
                    "provenance": "inherited_B_top3" if output_bbox is not None else "null_after_parse_or_rejection",
                    "candidate_bbox": request_metadata.get("candidate_bbox"),
                    "candidate_mode": request_metadata.get("candidate_mode"),
                    "candidate_geometry_version": request.get("candidate_geometry_version"),
                    "reference_source": "mask_derived_bbox",
                    "iou_threshold": IOU_THRESHOLD,
                    "iou": float(semantic.get("bbox_iou") or 0.0),
                },
                "audit_profiles": {
                    "primary_strict": primary,
                    "shared_fields": shared,
                    "set_compatible": set_compatible,
                },
            }
        )
    return traces


def summarize_trace(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = ("primary_strict", "shared_fields", "set_compatible")
    primary_j_auc = sum(
        float(row["evaluated_region"]["iou"])
        * float(
            all(
                value
                for name, value in row["audit_profiles"]["primary_strict"]["conditions"].items()
                if name != "spatial"
            )
        )
        for row in rows
    ) / len(rows)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "reports": len(rows),
        "primary_j_auc": primary_j_auc,
        "joint_passes": {
            name: sum(bool(row["audit_profiles"][name]["joint"]) for row in rows)
            for name in profiles
        },
    }


def main() -> None:
    contract = {
        "contract_id": "JSG-Bench",
        "version": CONTRACT_VERSION,
        "scope": "sample-level audit of structured image-forensic reports",
        "schema": ["prediction", "artifact_type", "target_scope", "evidence_bbox"],
        "tampered_applicability": ["prediction", "artifact", "target", "spatial"],
        "primary_condition_order": ["parse", "prediction", "artifact", "target", "spatial"],
        "spatial_rule": {"metric": "bbox_iou", "threshold": IOU_THRESHOLD, "comparison": ">="},
        "missing_policy": "A malformed report or missing applicable field fails its condition.",
        "denominator_policy": "Every declared evaluation sample remains in the Joint denominator.",
        "normalization_policy": "System output is normalized without evaluation-GT access.",
        "trace_required_fields": [
            "sample_id",
            "system_id",
            "input_condition",
            "parser_or_adapter_version",
            "source_references",
            "normalized_fields",
            "applicability",
            "evaluated_region",
            "audit_profiles",
        ],
        "profiles": {
            "primary_strict": ["parse", "prediction", "artifact", "target", "spatial"],
            "shared_fields": ["parse", "prediction", "target", "spatial"],
            "set_compatible": ["parse", "prediction", "artifact_set_hit", "target", "spatial"],
        },
    }
    contract_path = REPORTS / "jsg_audit_contract_v1.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "manifest_version": RUN_MANIFEST_VERSION,
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256(contract_path),
            "version": CONTRACT_VERSION,
        },
        "source_files": {},
        "traces": {},
    }
    for dataset, spec in TD_SPECS.items():
        rows = td_trace_rows(dataset, spec)
        write_jsonl(spec["output"], rows)
        manifest["traces"][f"{dataset}/TD-EGFA"] = summarize_trace(spec["output"], rows)
        for source_key in ("predictions", "requests", "raw_outputs"):
            source = spec[source_key]
            manifest["source_files"][str(source.relative_to(ROOT))] = sha256(source)

    xfer_rows = xfer_trace_rows()
    write_jsonl(XFER_OUTPUT, xfer_rows)
    manifest["traces"]["JSG-Xfer/TD-EGFA"] = summarize_trace(XFER_OUTPUT, xfer_rows)
    for source in [
        XFER_SEMANTIC,
        XFER_LOWLEVEL,
        *XFER_PREDICTIONS,
        *XFER_REQUESTS,
        *XFER_RAW_OUTPUTS,
    ]:
        manifest["source_files"][str(source.relative_to(ROOT))] = sha256(source)

    sida_rows = sida_trace_rows()
    write_jsonl(SIDA_OUTPUT, sida_rows)
    manifest["traces"]["JSG-Core/SIDA"] = summarize_trace(SIDA_OUTPUT, sida_rows)
    manifest["source_files"][str(SIDA_SOURCE.relative_to(ROOT))] = sha256(SIDA_SOURCE)

    manifest_path = REPORTS / "jsg_audit_trace_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contract_version": CONTRACT_VERSION, "traces": manifest["traces"]}))


if __name__ == "__main__":
    main()
