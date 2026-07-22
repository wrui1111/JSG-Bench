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
CORE_SCALE_RUN_CONFIG_ID = "td-egfa-qwen25vl-core-scale-v1"
XFER_PROVENANCE_ID = "jsg-xfer-v2-mixed"

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
XFER_REPAIR_REPORT = REPORTS / "jsg_xfer_bbox_remainder_fix.md"

QWEN_RUNNER = ROOT / "experiments" / "qwen25vl_sid_infer.py"
QWEN_CHECKPOINT = ROOT / "models" / "Qwen2.5-VL-3B-Instruct"
QWEN_CHECKPOINT_FILES = {
    "model-00001-of-00002.safetensors": "41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8",
    "model-00002-of-00002.safetensors": "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4",
    "model.safetensors.index.json": "c7dd78a4c6bea60b51332f1baf37b8f8124ecab2c35395a29a29825bf2619768",
    "config.json": "7ed3eed5be6924cc800e8a5e53fc405c1aab1aaf36bad65c33403b36c56827f5",
    "preprocessor_config.json": "f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0",
    "generation_config.json": "533f191cc257b7de37a4fccd0a7a1706d75e1aa660f93efaa54e5a2a9f9aace9",
}

EXPECTED_JOINT_PASSES = {
    "JSG-Core/TD-EGFA": {"primary_strict": 34, "shared_fields": 57, "set_compatible": 34},
    "JSG-Scale/TD-EGFA": {"primary_strict": 152, "shared_fields": 194, "set_compatible": 152},
    "JSG-Xfer/TD-EGFA": {"primary_strict": 26, "shared_fields": 40, "set_compatible": 32},
    "JSG-Core/SIDA": {"primary_strict": 0, "shared_fields": 32, "set_compatible": 13},
}


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


def core_scale_run_configuration() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for dataset, spec in TD_SPECS.items():
        request_rows, requests = read_jsonl_indexed(spec["requests"])
        output_rows, outputs = read_jsonl_indexed(spec["raw_outputs"])
        if set(requests) != set(outputs):
            raise ValueError(f"Unaligned declared run sources for {dataset}")
        if {row.get("prompt_version") for row in request_rows} != {"candidate_evidence_v1"}:
            raise ValueError(f"Unexpected prompt version in {dataset}")
        if {
            (row.get("metadata_for_evaluation_only") or {}).get("candidate_mode")
            for row in request_rows
        } != {"top3_union"}:
            raise ValueError(f"Unexpected candidate mode in {dataset}")
        if {len(row.get("image_paths") or []) for row in request_rows} != {2}:
            raise ValueError(f"Unexpected input-view count in {dataset}")
        target_scope_block = "Allowed target_scope values:\n" + "\n".join(
            f"- {label}" for label in spec["target_scope_labels"]
        )
        if any(target_scope_block not in str(row.get("prompt")) for row in request_rows):
            raise ValueError(f"Unexpected target-scope label space in {dataset}")
        evidence[dataset] = {
            "request_rows": len(request_rows),
            "native_output_rows": len(output_rows),
            "unique_aligned_sample_ids": len(requests),
            "retained_native_output_rows_per_report": 1,
        }

    checkpoint_files: dict[str, Any] = {}
    for name, expected_digest in QWEN_CHECKPOINT_FILES.items():
        path = QWEN_CHECKPOINT / name
        observed_digest = sha256(path)
        if observed_digest != expected_digest:
            raise ValueError(f"Frozen checkpoint hash changed for {path}")
        checkpoint_files[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": observed_digest,
        }

    return {
        "status": "declared_legacy_run_configuration",
        "applies_to": ["JSG-Core/TD-EGFA", "JSG-Scale/TD-EGFA"],
        "checkpoint": {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "local_path": str(QWEN_CHECKPOINT.relative_to(ROOT)),
            "post_training_quantization": False,
            "modelscope_weight_revision": "37ed0f575d34359b119fc3ac1e73c825f3308d29",
            "hash_capture": "post_hoc_local_snapshot",
            "files": checkpoint_files,
        },
        "prompt": {
            "version": "candidate_evidence_v1",
            "candidate_mode": "top3_union",
            "schema_and_complete_prompt": "stored_verbatim_in_hashed_requests",
            "label_spaces": {
                "prediction": ["real", "full_synthetic", "tampered"],
                "candidate_verdict": ["accepted", "rejected", "uncertain"],
                "target_scope_by_dataset": {
                    dataset: spec["target_scope_labels"] for dataset, spec in TD_SPECS.items()
                },
                "artifact_type": [
                    "none",
                    "boundary_seam",
                    "texture_smoothness",
                    "lighting_shadow",
                    "geometry_structure",
                    "resolution_noise",
                    "semantic_implausibility",
                    "compression_artifact",
                    "other",
                ],
            },
            "candidate_bbox_inheritance": (
                "A valid supplied candidate bbox becomes the audited evidence bbox when the "
                "normalized prediction is tampered."
            ),
        },
        "input": {
            "views": ["marked_full_image", "candidate_crop"],
            "images_per_request": 2,
            "max_pixels": 602112,
        },
        "generation": {
            "declared_generation_attempts_per_report": 1,
            "strategy": "greedy",
            "do_sample": False,
            "max_new_tokens": 512,
        },
        "runtime": {
            "backend": "transformers-mps",
            "device": "mps",
            "dtype": "float16",
            "attention_implementation": "sdpa",
            "framework_versions": {
                "python": "3.9.6",
                "torch": "2.8.0",
                "transformers": "4.57.6",
                "qwen_vl_utils": "0.0.14",
                "pillow": "11.3.0",
            },
            "version_capture": "post_hoc_from_preserved_local_environment",
        },
        "runner": {
            "path": str(QWEN_RUNNER.relative_to(ROOT)),
            "sha256": sha256(QWEN_RUNNER),
        },
        "artifact_evidence": evidence,
        "binding_limit": (
            "The legacy raw-output rows do not embed the command line or runtime metadata. "
            "Backend, dtype, framework, attention, checkpoint, quantization, and generation "
            "settings are a reconstructed run-level declaration based on the preserved local "
            "runner and environment plus the author run record. Request/output hashes bind one "
            "retained native-output row per sample but do not independently exclude upstream "
            "best-of-n output selection."
        ),
    }


def xfer_inference_provenance_summary() -> dict[str, Any]:
    rows, index = read_jsonl_indexed(XFER_RAW_OUTPUTS[0])
    with_provenance = [row for row in rows if isinstance(row.get("inference_provenance"), dict)]
    without_provenance = [row for row in rows if not isinstance(row.get("inference_provenance"), dict)]
    model_only = [row for row in without_provenance if row.get("model") is not None]
    without_model = [row for row in without_provenance if row.get("model") is None]
    backend_counts: dict[str, int] = {}
    for row in with_provenance:
        backend = str(row["inference_provenance"].get("backend"))
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
    observed = {
        "reports": len(rows),
        "with_structured_per_report_provenance": len(with_provenance),
        "without_structured_per_report_provenance": len(without_provenance),
        "retained_with_model_tag_only": len(model_only),
        "retained_without_model_or_runtime_tags": len(without_model),
        "backend_counts_for_regenerated_reports": backend_counts,
    }
    expected = {
        "reports": 100,
        "with_structured_per_report_provenance": 5,
        "without_structured_per_report_provenance": 95,
        "retained_with_model_tag_only": 47,
        "retained_without_model_or_runtime_tags": 48,
        "backend_counts_for_regenerated_reports": {"ollama": 3, "transformers-mps": 2},
    }
    if observed != expected:
        raise ValueError(f"JSG-Xfer inference provenance changed: {observed}")
    expected_regenerated_ids = {
        "Tp_S_NNN_S_N_pla20070_pla20070_01970",
        "Tp_S_NNN_S_N_pla20077_pla20077_02360",
        "Tp_S_NNN_S_N_sec20049_sec20049_01639",
        "1a5x44__c8ufqah_0",
        "1a9tss__c8vejd9_0",
    }
    if {str(row["img_id"]) for row in with_provenance} != expected_regenerated_ids:
        raise ValueError("JSG-Xfer regenerated report IDs changed")
    return {
        "status": "mixed_regenerated_and_legacy_outputs",
        **observed,
        "retained_outputs_without_backend_version_tags": 95,
        "raw_output_path": str(XFER_RAW_OUTPUTS[0].relative_to(ROOT)),
        "regenerated_report_records": [
            {
                "sample_id": str(row["img_id"]),
                "source_reference": reference(
                    XFER_RAW_OUTPUTS[0], index[str(row["img_id"])][0]
                ),
                "inference_provenance": row["inference_provenance"],
            }
            for row in with_provenance
        ],
        "repair_report": {
            "path": str(XFER_REPAIR_REPORT.relative_to(ROOT)),
            "sha256": sha256(XFER_REPAIR_REPORT),
        },
        "interpretation_limit": (
            "Source-specific contrasts are descriptive because runtime identity is not "
            "recoverable for the 95 retained outputs."
        ),
    }


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


def validate_manifest_counts(manifest: dict[str, Any]) -> None:
    observed = {key: value["joint_passes"] for key, value in manifest["traces"].items()}
    if observed != EXPECTED_JOINT_PASSES:
        raise ValueError(f"Trace pass counts changed: {observed}")


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
        "run_configurations": {
            CORE_SCALE_RUN_CONFIG_ID: core_scale_run_configuration(),
        },
        "inference_provenance": {
            XFER_PROVENANCE_ID: xfer_inference_provenance_summary(),
        },
        "source_files": {},
        "traces": {},
    }
    for dataset, spec in TD_SPECS.items():
        rows = td_trace_rows(dataset, spec)
        write_jsonl(spec["output"], rows)
        manifest["traces"][f"{dataset}/TD-EGFA"] = summarize_trace(spec["output"], rows)
        manifest["traces"][f"{dataset}/TD-EGFA"]["run_configuration"] = CORE_SCALE_RUN_CONFIG_ID
        for source_key in ("predictions", "requests", "raw_outputs"):
            source = spec[source_key]
            manifest["source_files"][str(source.relative_to(ROOT))] = sha256(source)

    xfer_rows = xfer_trace_rows()
    write_jsonl(XFER_OUTPUT, xfer_rows)
    manifest["traces"]["JSG-Xfer/TD-EGFA"] = summarize_trace(XFER_OUTPUT, xfer_rows)
    manifest["traces"]["JSG-Xfer/TD-EGFA"]["inference_provenance"] = XFER_PROVENANCE_ID
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

    validate_manifest_counts(manifest)

    manifest_path = REPORTS / "jsg_audit_trace_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contract_version": CONTRACT_VERSION, "traces": manifest["traces"]}))


if __name__ == "__main__":
    main()
