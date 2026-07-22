#!/usr/bin/env python3
"""Build canonical JSG audit traces for the controlled pixel5d-v2 results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stratified_statistics import xfer_stratum


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
CONTRACT_VERSION = "jsg-bench-audit-v1.0"
CONDITION_ORDER = ["parse", "prediction", "artifact", "target", "spatial"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_map(path: Path, key: str = "img_id") -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in read_jsonl(path)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def audit_profile(parse: bool, prediction: bool, artifact: bool, target: bool, iou: float, bbox: Any) -> dict[str, Any]:
    if not parse:
        prediction = artifact = target = False
    conditions = {
        "parse": parse,
        "prediction": prediction,
        "artifact": artifact,
        "target": target,
        "spatial": bool(parse and bbox is not None and iou >= 0.1),
    }
    return {
        "condition_order": CONDITION_ORDER,
        "pass_vector": [conditions[name] for name in CONDITION_ORDER],
        "conditions": conditions,
        "joint": all(conditions.values()),
        "failed_conditions": [name for name in CONDITION_ORDER if not conditions[name]],
    }


def td_rows(dataset: str, predictions_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction_row in read_jsonl(predictions_path):
        native = prediction_row.get("prediction") or {}
        sample_id = str(prediction_row["img_id"])
        parse = bool(native.get("parsed"))
        bbox = native.get("evidence_bbox") if parse else None
        iou = float(prediction_row.get("bbox_iou") or 0.0)
        profile = audit_profile(
            parse,
            bool(prediction_row.get("prediction_correct")),
            bool(prediction_row.get("artifact_correct")),
            bool(prediction_row.get("target_correct")),
            iou,
            bbox,
        )
        rows.append(
            {
                "contract_version": CONTRACT_VERSION,
                "dataset": dataset,
                "sample_id": sample_id,
                "system_id": "TD-EGFA/Qwen2.5-VL-3B-Instruct/pixel5d-v2",
                "input_condition": "known-tampered/proposal-grounded/B_top3/pixel5d-v2",
                "parser_or_adapter_version": "candidate-evidence-v1/pixel5d-v2",
                "normalized_fields": {
                    "prediction": native.get("prediction") if parse else None,
                    "artifact_type": native.get("artifact_type") if parse else None,
                    "target_scope": native.get("target_scope") if parse else None,
                    "evidence_bbox": bbox,
                },
                "evaluated_region": {
                    "bbox": bbox,
                    "provenance": "inherited_B_top3" if bbox is not None else "null",
                    "iou_threshold": 0.1,
                    "iou": iou,
                },
                "audit_profiles": {"primary_strict": profile},
            }
        )
    return rows


def xfer_rows() -> list[dict[str, Any]]:
    outputs = row_map(RUNS / "jsg_xfer_outputs_pixel5d_v2.jsonl")
    predictions = row_map(RUNS / "jsg_xfer_predictions_pixel5d_v2.jsonl")
    semantic = row_map(RUNS / "jsg_xfer_semantic_rows_pixel5d_v2.jsonl")
    selection = row_map(RUNS / "external_semantic_eval_rows.jsonl")
    rows: list[dict[str, Any]] = []
    for sample_id in semantic:
        output = outputs[sample_id]
        prediction = predictions[sample_id]
        sem = semantic[sample_id]
        selection_row = selection[sample_id]
        native = output.get("vlm_result") or {}
        parse = isinstance(output.get("vlm_result"), dict)
        bbox = prediction.get("pred_bbox") if parse else None
        iou = float(sem.get("bbox_iou") or 0.0)
        profile = audit_profile(
            parse,
            prediction.get("prediction") == "tampered",
            bool(sem.get("artifact_correct")),
            bool(sem.get("target_correct")),
            iou,
            bbox,
        )
        rows.append(
            {
                "contract_version": CONTRACT_VERSION,
                "dataset": "JSG-Xfer",
                "external_dataset": sem.get("external_dataset"),
                "sample_id": sample_id,
                "sampling_stratum": xfer_stratum(
                    selection_row.get("external_dataset"),
                    float(selection_row.get("bbox_iou") or 0.0),
                ),
                "selection_iou_legacy": float(selection_row.get("bbox_iou") or 0.0),
                "system_id": "TD-EGFA/Qwen2.5-VL-3B-Instruct/pixel5d-v2",
                "input_condition": "known-tampered/proposal-grounded/B_top3/pixel5d-v2/unified-runtime",
                "parser_or_adapter_version": "candidate-evidence-v1/pixel5d-v2",
                "normalized_fields": {
                    "prediction": prediction.get("prediction") if parse else None,
                    "artifact_type": sem.get("pred_artifact") if parse else None,
                    "target_scope": sem.get("pred_target") if parse else None,
                    "evidence_bbox": bbox,
                },
                "evaluated_region": {
                    "bbox": bbox,
                    "provenance": "inherited_B_top3" if bbox is not None else "null",
                    "iou_threshold": 0.1,
                    "iou": iou,
                },
                "audit_profiles": {"primary_strict": profile},
            }
        )
    return rows


def main() -> None:
    outputs = {
        RUNS / "jsg_core_td_egfa_audit_trace_pixel5d_v2.jsonl": td_rows(
            "JSG-Core", RUNS / "jsg_core_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl"
        ),
        RUNS / "jsg_scale_td_egfa_audit_trace_pixel5d_v2.jsonl": td_rows(
            "JSG-Scale", RUNS / "jsg_scale_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl"
        ),
        RUNS / "jsg_xfer_td_egfa_audit_trace_pixel5d_v2.jsonl": xfer_rows(),
    }
    for path, rows in outputs.items():
        write_jsonl(path, rows)
        print(f"Wrote {len(rows)} rows to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
