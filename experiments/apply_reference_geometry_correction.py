#!/usr/bin/env python3
"""Map SID-Set spatial references from the mask pixel grid to image coordinates.

Defect: the frozen SID-Set references were the tight ``mask > 0`` bbox measured in
the mask's own pixel grid. For masks whose size differs from the image (8 Core and
37 Scale reports), that bbox is not in image coordinates, while every evaluated box
(B_top3, free boxes, SIDA masks) is. The fix scales the mask-grid bbox by
(W_image / W_mask, H_image / H_mask) with outward rounding. Masks with matching
size are unchanged. JSG-Xfer (CASIA2/IMD2020) masks all match their images.

This script is idempotent. It:
  1. patches annotation files (adds ``reference_geometry`` with the frozen value);
  2. patches evaluate-format prediction rows (``gold.evidence_bbox``, ``bbox_iou``,
     ``pointing_hit``) after asserting the stored values equal the frozen ones;
  3. writes a manifest under experiments/reports/reference_geometry_correction.json.
Frozen copies are archived in experiments/runs/frozen_reference_v1/.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
VERSION = "image-aligned-v2"

ANNOTATION_FILES = [
    ROOT / "SID_Set/annotations/SID-Explain-300.jsonl",
    ROOT / "SID_Set/annotations/SID-FA-600.jsonl",
    RUNS / "sid_explain300_tampered100_annotations.jsonl",
    RUNS / "sid_explain300_tampered100_annotations_revised.jsonl",
    RUNS / "recommended_suite/sid_explain300_tampered100_annotations.jsonl",
    RUNS / "sid_fa600_annotations_postreview_eval.jsonl",
    RUNS / "sid_fa200_annotations.jsonl",
]

# Evaluate-format prediction files ("gold" + "prediction" + "bbox_iou").
PREDICTION_FILES = [
    "jsg_core_predictions_pixel5d_v2.jsonl",
    "jsg_core_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl",
    "jsg_scale_predictions_pixel5d_v2.jsonl",
    "jsg_scale_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl",
    "recommended_suite/jsg_core_shuffled_proposal_legacy_predictions.jsonl",
    "recommended_suite/sid_explain300_top3_crop_only_legacy_predictions.jsonl",
    "recommended_suite/sid_explain300_top3_marked_crop_legacy_predictions.jsonl",
    "recommended_suite/sid_explain300_top3_marked_only_legacy_predictions.jsonl",
    "recommended_suite/sid_explain300_top3_shuffled_crop_legacy_predictions.jsonl",
    "sid_explain300_evidence_explanation_v2_minicpm45_predictions.jsonl",
    "sid_explain300_evidence_explanation_v2_ollama7b_predictions.jsonl",
    "sid_explain300_evidence_explanation_v2_qwen25vl_recheck_predictions.jsonl",
    "sid_explain300_evidence_explanation_v2_revised_predictions.jsonl",
    "sid_explain300_tampered100_candidate_top1_predictions.jsonl",
    "sid_explain300_tampered100_candidate_top3_croponly_predictions.jsonl",
    "sid_explain300_tampered100_candidate_top3_markedonly_predictions.jsonl",
    "sid_explain300_tampered100_candidate_top3_revised_predictions.jsonl",
    "sid_fa200_candidate_top3_minicpm45_predictions.jsonl",
    "sid_fa200_candidate_top3_ollama7b_predictions.jsonl",
    "sid_fa200_candidate_top3_qwen25vl_predictions.jsonl",
    "sid_fa600_artifact_v2_top3_postreview_predictions.jsonl",
    "sid_fa600_candidate_top1_postreview_predictions.jsonl",
    "sid_fa600_candidate_top3_postreview_predictions.jsonl",
    "sid_fa600_free_v2_postreview_predictions.jsonl",
    "sid_fa600_target_v2_top3_postreview_predictions.jsonl",
]

# Files that store the frozen box but are not used by the manuscript; listed in the
# manifest and intentionally left untouched (historical oracle requests show what
# the model actually saw).
NOT_REGENERATED = [
    "sid_explain300_tampered100_gtbbox_oracle_*", "sid_fa600_gtbbox_oracle_*",
    "sid_explain300_tampered100_bbox_refine_predictions.jsonl",
    "sid_explain300_tampered100_lowlevel_candidate_oracle_rows.jsonl",
    "sid_explain300_tampered100_topk_sensitivity_rows.jsonl", "sid_fa600_lowlevel_eval_rows.jsonl",
    "sid_fa600_topk_sensitivity_rows.jsonl", "tifs_*_rows.jsonl", "sid_fa50_*", "sid_fa600_candidate_top3_limit60_*",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bbox_iou(a: list[int] | None, b: list[int] | None) -> float:
    """Identical to candidate_evidence_explanation.bbox_iou."""
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def center_inside(pred: list[int] | None, gt: list[int] | None) -> bool:
    if not pred or not gt:
        return False
    cx, cy = (pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def reference_table() -> dict[str, dict[str, Any]]:
    """Corrected boxes for every tampered SID-Set image referenced by the cohorts."""
    table: dict[str, dict[str, Any]] = {}
    seen = set()
    for path in ANNOTATION_FILES:
        for row in read_jsonl(path):
            img_id = row["img_id"]
            if img_id in seen or not row.get("mask_path"):
                continue
            seen.add(img_id)
            with Image.open(ROOT / "SID_Set" / row["image_path"]) as im:
                W, H = im.size
            with Image.open(ROOT / "SID_Set" / row["mask_path"]) as mk:
                arr = np.asarray(mk.convert("L")) > 0
            h, w = arr.shape
            if not arr.any():
                continue
            ys, xs = np.where(arr)
            native = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            if (w, h) == (W, H):
                continue
            sx, sy = W / w, H / h
            corrected = [int(math.floor(native[0] * sx)), int(math.floor(native[1] * sy)),
                         min(W, int(math.ceil(native[2] * sx))), min(H, int(math.ceil(native[3] * sy)))]
            table[img_id] = {"frozen": native, "corrected": corrected,
                             "mask_size": [w, h], "image_size": [W, H]}
    return table


def provenance(entry: dict[str, Any]) -> dict[str, Any]:
    return {"version": VERSION, "frozen_mask_grid_bbox": entry["frozen"],
            "mask_size": entry["mask_size"], "image_size": entry["image_size"],
            "rule": "scale mask-grid bbox by image/mask size, floor min and ceil max"}


def patch_annotations(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for path in ANNOTATION_FILES:
        rows = read_jsonl(path)
        changed = 0
        for row in rows:
            entry = table.get(row["img_id"])
            if entry is None:
                continue
            if (row.get("reference_geometry") or {}).get("version") == VERSION:
                continue
            frozen, corrected = entry["frozen"], entry["corrected"]
            touched = False
            if "mask_bbox" in row and row["mask_bbox"] is not None:
                assert row["mask_bbox"] == frozen, (path, row["img_id"])
                row["mask_bbox"] = corrected
                touched = True
            auto = row.get("auto_features")
            if isinstance(auto, dict) and auto.get("mask_bbox") is not None:
                assert auto["mask_bbox"] == frozen, (path, row["img_id"])
                auto["mask_bbox"] = corrected
                touched = True
            for item in row.get("evidence_items") or []:
                if isinstance(item, dict) and item.get("evidence_bbox") is not None:
                    assert item["evidence_bbox"] == frozen, (path, row["img_id"])
                    item["evidence_bbox"] = corrected
                    touched = True
            if touched:
                row["reference_geometry"] = provenance(entry)
                changed += 1
        if changed:
            write_jsonl(path, rows)
        report[str(path.relative_to(ROOT))] = {"rows_patched": changed, "sha256": sha256(path)}
    return report


def patch_predictions(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for name in PREDICTION_FILES:
        path = RUNS / name
        rows = read_jsonl(path)
        changed = 0
        for row in rows:
            entry = table.get(row.get("img_id"))
            gold = row.get("gold")
            if entry is None or not isinstance(gold, dict):
                continue
            if gold.get("evidence_bbox") == entry["corrected"]:
                continue
            assert gold.get("evidence_bbox") == entry["frozen"], (name, row["img_id"])
            pred_box = (row.get("prediction") or {}).get("evidence_bbox")
            stored = row.get("bbox_iou")
            if stored is not None:
                assert math.isclose(float(stored), bbox_iou(pred_box, entry["frozen"]), abs_tol=1e-12), (
                    name, row["img_id"], stored)
            gold["evidence_bbox"] = entry["corrected"]
            if stored is not None:
                row["bbox_iou"] = bbox_iou(pred_box, entry["corrected"])
            if row.get("pointing_hit") is not None:
                row["pointing_hit"] = center_inside(pred_box, entry["corrected"])
            row["reference_geometry_version"] = VERSION
            changed += 1
        if changed:
            write_jsonl(path, rows)
        report[name] = {"rows_patched": changed, "sha256": sha256(path)}
    return report


def main() -> None:
    table = reference_table()
    manifest = {
        "version": VERSION,
        "affected_images": {k: table[k] for k in sorted(table)},
        "annotation_files": patch_annotations(table),
        "prediction_files": patch_predictions(table),
        "not_regenerated_legacy_files": NOT_REGENERATED,
        "frozen_archive": "experiments/runs/frozen_reference_v1/",
    }
    out = ROOT / "experiments/reports/reference_geometry_correction.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "affected_images"}, indent=2))
    print("affected images:", len(table))


if __name__ == "__main__":
    main()
