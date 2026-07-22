#!/usr/bin/env python3
"""Post-process SID predictions with top-k low-level candidate boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CLASS_NAMES = ["real", "full_synthetic", "tampered"]
EPS = 1e-6


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


def bbox_area(box: list[int] | None) -> float:
    box = valid_bbox(box)
    if not box:
        return 0.0
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def bbox_center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


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


def mask_bbox(mask_path: Path) -> list[int] | None:
    with Image.open(mask_path) as img:
        arr = np.asarray(img.convert("L")) > 0
    if not arr.any():
        return None
    ys, xs = np.where(arr)
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


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


def clamp_bbox_to_image(box: list[int], width: int, height: int) -> list[int] | None:
    clipped = [
        max(0, min(width, int(box[0]))),
        max(0, min(height, int(box[1]))),
        max(0, min(width, int(box[2]))),
        max(0, min(height, int(box[3]))),
    ]
    return valid_bbox(clipped)


def padded_bbox(box: list[int], width: int, height: int, scale: float = 1.5) -> list[int] | None:
    cx, cy = bbox_center(box)
    half_w = (box[2] - box[0]) * scale / 2.0
    half_h = (box[3] - box[1]) * scale / 2.0
    return clamp_bbox_to_image(
        [
            int(round(cx - half_w)),
            int(round(cy - half_h)),
            int(round(cx + half_w)),
            int(round(cy + half_h)),
        ],
        width,
        height,
    )


def gradient_array(
    sid_root: Path,
    image_path: str,
    cache: dict[str, tuple[np.ndarray, int, int]],
) -> tuple[np.ndarray, int, int]:
    if image_path in cache:
        return cache[image_path]
    with Image.open(sid_root / image_path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    grad = np.sqrt(gx * gx + gy * gy)
    height, width = grad.shape[:2]
    cache[image_path] = (grad, width, height)
    return cache[image_path]


def gradient_variance_ratio(
    box: list[int],
    manifest_row: dict[str, Any],
    sid_root: Path,
    image_cache: dict[str, tuple[np.ndarray, int, int]],
) -> float:
    image_path = manifest_row.get("image_path")
    if not isinstance(image_path, str):
        return 0.0
    grad, width, height = gradient_array(sid_root, image_path, image_cache)
    box = clamp_bbox_to_image(box, width, height)
    if not box:
        return 0.0
    pad = padded_bbox(box, width, height, scale=1.5)
    if not pad:
        return 0.0

    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = pad
    inside = grad[y1:y2, x1:x2]
    padded = grad[py1:py2, px1:px2]
    if inside.size == 0 or padded.size == 0:
        return 0.0

    context_mask = np.ones(padded.shape, dtype=bool)
    ix1 = max(0, x1 - px1)
    iy1 = max(0, y1 - py1)
    ix2 = min(padded.shape[1], x2 - px1)
    iy2 = min(padded.shape[0], y2 - py1)
    context_mask[iy1:iy2, ix1:ix2] = False
    context = padded[context_mask]
    if context.size == 0:
        return 0.0
    return float(np.var(inside) / (np.var(context) + EPS))


def topk_candidate_boxes(lowlevel_row: dict[str, Any] | None, top_k: int) -> list[list[int]]:
    if not lowlevel_row:
        return []
    candidates = lowlevel_row.get("lowlevel_candidates")
    if not isinstance(candidates, list):
        return []
    boxes = []
    for candidate in candidates[:top_k]:
        if not isinstance(candidate, dict):
            continue
        box = valid_bbox(candidate.get("bbox"))
        if box:
            boxes.append(box)
    return boxes


def topk_candidate_items(lowlevel_row: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
    if not lowlevel_row:
        return []
    candidates = lowlevel_row.get("lowlevel_candidates")
    if not isinstance(candidates, list):
        return []
    items = []
    for idx, candidate in enumerate(candidates[:top_k]):
        if not isinstance(candidate, dict):
            continue
        box = valid_bbox(candidate.get("bbox"))
        if not box:
            continue
        items.append(
            {
                "rank": idx + 1,
                "bbox": box,
                "score": candidate.get("score"),
            }
        )
    return items


def select_heterogeneity_filtered_bbox(
    pred: dict[str, Any],
    lowlevel_row: dict[str, Any] | None,
    manifest_row: dict[str, Any],
    sid_root: Path,
    image_cache: dict[str, tuple[np.ndarray, int, int]],
    top_k: int,
    center_distance_threshold: float,
    max_union_old_area_ratio: float,
    max_union_image_area_ratio: float,
    min_keep_candidates: int,
) -> tuple[list[int] | None, dict[str, Any]]:
    old_box = valid_bbox(pred.get("pred_tampered_bbox"))
    items = topk_candidate_items(lowlevel_row, top_k)
    if not items:
        return old_box, {
            "note": "no_valid_topk_candidates",
            "old_bbox": old_box,
            "new_bbox": old_box,
            "candidate_count": 0,
        }

    for item in items:
        item["heterogeneity_score"] = gradient_variance_ratio(
            item["bbox"],
            manifest_row,
            sid_root,
            image_cache,
        )
    ranked = sorted(items, key=lambda item: item["heterogeneity_score"], reverse=True)
    anchor = ranked[0]
    ax, ay = bbox_center(anchor["bbox"])
    try:
        short_side = max(1.0, min(float(manifest_row.get("width") or 0), float(manifest_row.get("height") or 0)))
        image_area = max(1.0, float(manifest_row.get("width") or 0) * float(manifest_row.get("height") or 0))
    except (TypeError, ValueError):
        short_side = 1.0
        image_area = 1.0

    kept = []
    for item in ranked:
        cx, cy = bbox_center(item["bbox"])
        center_distance = float(np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2) / short_side)
        item["center_distance_to_anchor"] = center_distance
        if item is anchor or center_distance <= center_distance_threshold:
            kept.append(item)

    if len(kept) < min_keep_candidates:
        selected = anchor["bbox"]
        note = "fallback_anchor_not_enough_nearby_candidates"
    else:
        selected = union_bbox([item["bbox"] for item in kept]) or anchor["bbox"]
        note = "heterogeneity_sorted_center_filtered_union"

    if selected and bbox_area(selected) / image_area > max_union_image_area_ratio:
        selected = anchor["bbox"]
        note = "fallback_anchor_union_exceeds_image_area"

    if (
        selected
        and old_box
        and max_union_old_area_ratio > 0
        and bbox_area(selected) / max(bbox_area(old_box), 1.0) > max_union_old_area_ratio
    ):
        selected = anchor["bbox"]
        note = "fallback_anchor_union_exceeds_old_area"

    return selected, {
        "note": note,
        "old_bbox": old_box,
        "new_bbox": selected,
        "candidate_count": len(items),
        "kept_count": len(kept),
        "center_distance_threshold": center_distance_threshold,
        "max_union_old_area_ratio": max_union_old_area_ratio,
        "max_union_image_area_ratio": max_union_image_area_ratio,
        "min_keep_candidates": min_keep_candidates,
        "ranked_candidates": ranked,
    }


def select_bbox(
    mode: str,
    pred: dict[str, Any],
    lowlevel_row: dict[str, Any] | None,
    gt_bbox: list[int] | None,
    top_k: int,
    manifest_row: dict[str, Any],
    sid_root: Path,
    image_cache: dict[str, tuple[np.ndarray, int, int]],
    center_distance_threshold: float,
    max_union_old_area_ratio: float,
    max_union_image_area_ratio: float,
    min_keep_candidates: int,
) -> tuple[list[int] | None, dict[str, Any]]:
    old_box = valid_bbox(pred.get("pred_tampered_bbox"))
    candidate_boxes = topk_candidate_boxes(lowlevel_row, top_k)
    selected = old_box
    note = "keep_original"

    if mode == "keep":
        selected = old_box
        note = "keep_original"
    elif mode == "topk_first":
        selected = candidate_boxes[0] if candidate_boxes else old_box
        note = "topk_first"
    elif mode == "topk_union":
        selected = union_bbox(candidate_boxes) or old_box
        note = "topk_union"
    elif mode == "topk_area_guard":
        union = union_bbox(candidate_boxes)
        try:
            image_area = max(
                1.0,
                float(manifest_row.get("width") or 0) * float(manifest_row.get("height") or 0),
            )
        except (TypeError, ValueError):
            image_area = 1.0
        if union and bbox_area(union) / image_area <= max_union_image_area_ratio:
            selected = union
            note = "topk_union_with_image_area_guard"
        else:
            selected = candidate_boxes[0] if candidate_boxes else old_box
            note = "fallback_top1_union_exceeds_image_area"
    elif mode == "topk_heterogeneity_filter":
        selected, details = select_heterogeneity_filtered_bbox(
            pred,
            lowlevel_row,
            manifest_row,
            sid_root,
            image_cache,
            top_k,
            center_distance_threshold,
            max_union_old_area_ratio,
            max_union_image_area_ratio,
            min_keep_candidates,
        )
        details.update({"mode": mode, "top_k": top_k})
        return selected, details
    elif mode == "topk_oracle":
        if candidate_boxes and gt_bbox:
            selected = max(candidate_boxes, key=lambda box: bbox_iou(box, gt_bbox))
            note = "topk_oracle_uses_gt_mask_for_diagnostic_only"
        else:
            selected = old_box
            note = "topk_oracle_fallback_original"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return selected, {
        "mode": mode,
        "note": note,
        "top_k": top_k,
        "old_bbox": old_box,
        "new_bbox": selected,
        "candidate_count": len(candidate_boxes),
        "max_union_image_area_ratio": max_union_image_area_ratio,
    }


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    mask_by_id: dict[str, list[int] | None],
) -> dict[str, Any]:
    total = 0
    correct = 0
    by_class_total = {name: 0 for name in CLASS_NAMES}
    by_class_correct = {name: 0 for name in CLASS_NAMES}
    confusion = {name: {pred: 0 for pred in CLASS_NAMES + ["unknown"]} for name in CLASS_NAMES}
    ious = []

    for pred in predictions:
        row = manifest_by_id.get(pred["img_id"])
        if not row:
            continue
        true_class = row["class_dir"]
        pred_class = pred.get("pred_class") if pred.get("pred_class") in CLASS_NAMES else "unknown"
        total += 1
        by_class_total[true_class] += 1
        confusion[true_class][pred_class] += 1
        if pred_class == true_class:
            correct += 1
            by_class_correct[true_class] += 1

        if true_class == "tampered" and row.get("mask_path"):
            ious.append(bbox_iou(valid_bbox(pred.get("pred_tampered_bbox")), mask_by_id.get(pred["img_id"])))

    per_class = {
        name: by_class_correct[name] / by_class_total[name] if by_class_total[name] else 0.0
        for name in CLASS_NAMES
    }
    loc = {
        "samples": len(ious),
        "mean_bbox_iou": float(np.mean(ious)) if ious else 0.0,
        "hit_rate_iou_0_1": float(np.mean(np.asarray(ious) >= 0.1)) if ious else 0.0,
        "hit_rate_iou_0_3": float(np.mean(np.asarray(ious) >= 0.3)) if ious else 0.0,
        "hit_rate_iou_0_5": float(np.mean(np.asarray(ious) >= 0.5)) if ious else 0.0,
    }
    macro_acc = sum(per_class.values()) / len(CLASS_NAMES)
    return {
        "samples": total,
        "classification_accuracy": correct / total if total else 0.0,
        "macro_accuracy": macro_acc,
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion,
        "tampered_localization": loc,
    }


def candidate_upper_bounds(
    prediction_ids: set[str],
    manifest_by_id: dict[str, dict[str, Any]],
    lowlevel_by_id: dict[str, dict[str, Any]],
    mask_by_id: dict[str, list[int] | None],
    top_k: int,
) -> dict[str, Any]:
    modes = {
        "current_lowlevel_candidate": [],
        "topk_first": [],
        "topk_oracle_single": [],
        "topk_union": [],
    }
    best_rank_counts = [0 for _ in range(top_k)]
    missing_topk = 0

    for img_id in sorted(prediction_ids):
        row = manifest_by_id.get(img_id)
        if not row or row.get("class_dir") != "tampered" or not row.get("mask_path"):
            continue
        gt = mask_by_id.get(img_id)
        low = lowlevel_by_id.get(img_id, {})
        boxes = topk_candidate_boxes(low, top_k)
        if len(boxes) < top_k:
            missing_topk += 1
        box_ious = [bbox_iou(box, gt) for box in boxes]
        if box_ious:
            best_rank_counts[int(np.argmax(box_ious))] += 1
        modes["current_lowlevel_candidate"].append(bbox_iou(valid_bbox(low.get("lowlevel_candidate_bbox")), gt))
        modes["topk_first"].append(bbox_iou(boxes[0] if boxes else None, gt))
        modes["topk_oracle_single"].append(max(box_ious + [0.0]))
        modes["topk_union"].append(bbox_iou(union_bbox(boxes), gt))

    summary = {}
    for name, values in modes.items():
        arr = np.asarray(values, dtype=np.float32)
        summary[name] = {
            "samples": int(len(arr)),
            "mean_bbox_iou": float(arr.mean()) if len(arr) else 0.0,
            "hit_rate_iou_0_1": float((arr >= 0.1).mean()) if len(arr) else 0.0,
            "hit_rate_iou_0_3": float((arr >= 0.3).mean()) if len(arr) else 0.0,
            "hit_rate_iou_0_5": float((arr >= 0.5).mean()) if len(arr) else 0.0,
        }
    return {
        "top_k": top_k,
        "missing_topk": missing_topk,
        "best_rank_counts": best_rank_counts,
        "metrics": summary,
        "note": "These upper bounds ignore classification and evaluate candidate boxes on all GT tampered samples.",
    }


def run(args: argparse.Namespace) -> None:
    sid_root = args.sid_root.resolve()
    manifest_by_id = {row["img_id"]: row for row in read_jsonl(args.manifest)}
    lowlevel_by_id = {row["img_id"]: row for row in read_jsonl(args.lowlevel)}
    predictions = read_jsonl(args.predictions)
    prediction_ids = {row["img_id"] for row in predictions}
    mask_by_id = {}
    for img_id, row in manifest_by_id.items():
        if img_id in prediction_ids and row.get("mask_path"):
            mask_by_id[img_id] = mask_bbox(sid_root / row["mask_path"])

    out_rows = []
    changed_bbox_count = 0
    image_cache: dict[str, tuple[np.ndarray, int, int]] = {}
    for pred in predictions:
        row = manifest_by_id.get(pred["img_id"], {})
        gt = mask_by_id.get(pred["img_id"])
        out = dict(pred)
        if pred.get("pred_class") == "tampered":
            new_bbox, info = select_bbox(
                args.mode,
                pred,
                lowlevel_by_id.get(pred["img_id"]),
                gt,
                args.top_k,
                row,
                sid_root,
                image_cache,
                args.center_distance_threshold,
                args.max_union_old_area_ratio,
                args.max_union_image_area_ratio,
                args.min_keep_candidates,
            )
            old_bbox = valid_bbox(pred.get("pred_tampered_bbox"))
            out["pred_tampered_bbox"] = new_bbox
            changed_bbox_count += int(new_bbox != old_bbox)
            evidence_card = out.get("evidence_card") if isinstance(out.get("evidence_card"), dict) else {}
            evidence_card = dict(evidence_card)
            evidence_card["topk_bbox_postprocess"] = {
                **info,
                "true_class_for_evaluation_only": row.get("class_dir"),
            }
            out["evidence_card"] = evidence_card
        else:
            out["pred_tampered_bbox"] = None
        out_rows.append(out)

    metrics = evaluate_predictions(out_rows, manifest_by_id, mask_by_id)
    report = {
        "mode": args.mode,
        "top_k": args.top_k,
        "prediction_file": str(args.predictions),
        "lowlevel_file": str(args.lowlevel),
        "changed_bbox_count": changed_bbox_count,
        "metrics": metrics,
        "candidate_upper_bounds": candidate_upper_bounds(
            prediction_ids,
            manifest_by_id,
            lowlevel_by_id,
            mask_by_id,
            args.top_k,
        ),
    }

    write_jsonl(args.output, out_rows)
    if args.metrics_output:
        write_json(args.metrics_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--lowlevel", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "keep",
            "topk_first",
            "topk_union",
            "topk_area_guard",
            "topk_heterogeneity_filter",
            "topk_oracle",
        ],
        required=True,
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--center-distance-threshold", type=float, default=0.25)
    parser.add_argument("--max-union-old-area-ratio", type=float, default=2.0)
    parser.add_argument("--max-union-image-area-ratio", type=float, default=0.4)
    parser.add_argument("--min-keep-candidates", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
