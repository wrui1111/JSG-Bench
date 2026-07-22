#!/usr/bin/env python3
"""DCT/SRM low-level candidate experiment for SID-Set.

This script is intentionally independent from Qwen/VLM inference. It tests
whether adding block-level frequency and noise-residual signals improves the
pure low-level localization upper bound on the dev_300 tampered subset.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CLASS_NAMES = ["real", "full_synthetic", "tampered"]
EPS = 1e-6
GRID = 16
BASELINE_FEATURE_VERSION = "pixel5d-v2"


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
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def select_rows(rows: list[dict[str, Any]], per_class: int) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["class_dir"]].append(row)
    selected: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        selected.extend(by_class[class_name][:per_class])
    return selected


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


def image_array(path: Path, max_side: int = 512) -> tuple[np.ndarray, float, float]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        scale = min(max_side / max(orig_w, orig_h), 1.0)
        if scale < 1.0:
            img = img.resize((round(orig_w * scale), round(orig_h * scale)))
        arr = np.asarray(img, dtype=np.float32)
    scale_x = orig_w / arr.shape[1]
    scale_y = orig_h / arr.shape[0]
    return arr, scale_x, scale_y


def gray_edges(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    grad = np.sqrt(gx * gx + gy * gy)
    lap = np.zeros_like(gray)
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lap[1:-1, 1:-1] = (
            gray[2:, 1:-1]
            + gray[:-2, 1:-1]
            + gray[1:-1, 2:]
            + gray[1:-1, :-2]
            - 4 * gray[1:-1, 1:-1]
        )
    return grad, lap


def srm_like_residual(gray: np.ndarray) -> np.ndarray:
    """Small high-pass bank approximating SRM residual statistics."""
    residuals: list[np.ndarray] = []

    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        gray[2:, 1:-1]
        + gray[:-2, 1:-1]
        + gray[1:-1, 2:]
        + gray[1:-1, :-2]
        - 4 * gray[1:-1, 1:-1]
    )
    residuals.append(lap)

    h2 = np.zeros_like(gray)
    h2[:, 1:-1] = gray[:, 2:] - 2 * gray[:, 1:-1] + gray[:, :-2]
    residuals.append(h2)

    v2 = np.zeros_like(gray)
    v2[1:-1, :] = gray[2:, :] - 2 * gray[1:-1, :] + gray[:-2, :]
    residuals.append(v2)

    d1 = np.zeros_like(gray)
    d1[1:-1, 1:-1] = gray[2:, 2:] - 2 * gray[1:-1, 1:-1] + gray[:-2, :-2]
    residuals.append(d1)

    d2 = np.zeros_like(gray)
    d2[1:-1, 1:-1] = gray[2:, :-2] - 2 * gray[1:-1, 1:-1] + gray[:-2, 2:]
    residuals.append(d2)

    return np.mean(np.abs(np.stack(residuals, axis=0)), axis=0)


_DCT_CACHE: dict[int, np.ndarray] = {}


def dct_basis(n: int) -> np.ndarray:
    if n not in _DCT_CACHE:
        k = np.arange(n, dtype=np.float64)[:, None]
        x = np.arange(n, dtype=np.float64)[None, :]
        basis = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
        basis[0, :] *= np.sqrt(1.0 / n)
        if n > 1:
            basis[1:, :] *= np.sqrt(2.0 / n)
        _DCT_CACHE[n] = basis
    return _DCT_CACHE[n]


def high_frequency_dct_ratio(patch_gray: np.ndarray) -> float:
    if patch_gray.size == 0:
        return 0.0
    patch = np.asarray(patch_gray, dtype=np.float64)
    if patch.ndim != 2 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return 0.0
    patch = np.nan_to_num(patch, nan=0.0, posinf=255.0, neginf=0.0)
    patch = patch - float(patch.mean())
    h, w = patch.shape
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coeff = dct_basis(h) @ patch @ dct_basis(w).T
    coeff = np.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
    energy = coeff * coeff
    total = float(energy.sum()) + EPS
    yy = np.arange(h, dtype=np.float32)[:, None] / max(h - 1, 1)
    xx = np.arange(w, dtype=np.float32)[None, :] / max(w - 1, 1)
    high_mask = (yy + xx) >= 0.85
    high_mask[0, 0] = False
    return float(energy[high_mask].sum() / total)


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True) + EPS
    return np.abs((values - median) / (1.4826 * mad))


def local_contrast(scores: np.ndarray, indices: list[tuple[int, int]]) -> np.ndarray:
    grid_scores = np.zeros((GRID, GRID), dtype=np.float32)
    valid = np.zeros((GRID, GRID), dtype=bool)
    for score, (gy, gx) in zip(scores, indices):
        grid_scores[gy, gx] = float(score)
        valid[gy, gx] = True

    contrasts: list[float] = []
    for gy, gx in indices:
        y1 = max(0, gy - 1)
        y2 = min(GRID, gy + 2)
        x1 = max(0, gx - 1)
        x2 = min(GRID, gx + 2)
        window = grid_scores[y1:y2, x1:x2][valid[y1:y2, x1:x2]]
        if window.size <= 1:
            contrasts.append(0.0)
            continue
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med))) + EPS
        contrasts.append(abs(float(grid_scores[gy, gx]) - med) / (1.4826 * mad))
    return np.asarray(contrasts, dtype=np.float32)


def bbox_from_blocks(
    block_indices: list[tuple[int, int]],
    block_h: int,
    block_w: int,
    height: int,
    width: int,
    scale_x: float,
    scale_y: float,
) -> list[int] | None:
    if not block_indices:
        return None
    ys = [idx[0] for idx in block_indices]
    xs = [idx[1] for idx in block_indices]
    x1 = min(xs) * block_w
    y1 = min(ys) * block_h
    x2 = min((max(xs) + 1) * block_w, width)
    y2 = min((max(ys) + 1) * block_h, height)
    return [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    ]


def candidate_bboxes_from_scores(
    block_scores: np.ndarray,
    indices: list[tuple[int, int]],
    block_h: int,
    block_w: int,
    height: int,
    width: int,
    scale_x: float,
    scale_y: float,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    if len(block_scores) == 0:
        return []
    order = np.argsort(block_scores)[::-1]
    candidates: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    all_indices = set(indices)
    for score_idx in order:
        center = indices[int(score_idx)]
        if center in used:
            continue
        cy, cx = center
        local_blocks = [
            (y, x)
            for y in range(max(0, cy - 1), min(GRID, cy + 2))
            for x in range(max(0, cx - 1), min(GRID, cx + 2))
            if (y, x) in all_indices
        ]
        for block in local_blocks:
            used.add(block)
        bbox = bbox_from_blocks(local_blocks, block_h, block_w, height, width, scale_x, scale_y)
        if bbox:
            candidates.append(
                {
                    "bbox": bbox,
                    "score": round(float(block_scores[int(score_idx)]), 4),
                }
            )
        if len(candidates) >= top_k:
            break
    return candidates


def feature_matrix_for_row(row: dict[str, Any], sid_root: Path) -> tuple[list[str], np.ndarray, list[tuple[int, int]], dict[str, Any]]:
    arr, scale_x, scale_y = image_array(sid_root / row["image_path"])
    height, width = arr.shape[:2]
    gray = arr.mean(axis=2)
    grad, lap = gray_edges(gray)
    srm = srm_like_residual(gray)

    block_h = max(height // GRID, 1)
    block_w = max(width // GRID, 1)
    features: list[list[float]] = []
    indices: list[tuple[int, int]] = []

    for gy in range(GRID):
        for gx in range(GRID):
            y1 = gy * block_h
            x1 = gx * block_w
            y2 = height if gy == GRID - 1 else min((gy + 1) * block_h, height)
            x2 = width if gx == GRID - 1 else min((gx + 1) * block_w, width)
            patch = arr[y1:y2, x1:x2]
            patch_gray = gray[y1:y2, x1:x2]
            patch_grad = grad[y1:y2, x1:x2]
            patch_lap = lap[y1:y2, x1:x2]
            patch_srm = srm[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            features.append(
                [
                    float(patch.std()),
                    float(patch_gray.mean()),
                    float(patch_gray.std()),
                    float(patch_grad.mean()),
                    float(patch_lap.var()),
                    high_frequency_dct_ratio(patch_gray),
                    float(patch_srm.mean()),
                    float(patch_srm.std()),
                    float(patch_srm.var()),
                    float(np.percentile(patch_srm, 95) - np.percentile(patch_srm, 50)),
                ]
            )
            indices.append((gy, gx))

    feature_names = [
        "rgb_std",
        "gray_mean",
        "gray_std",
        "edge_mean",
        "lap_var",
        "dct_hf_ratio",
        "srm_abs_mean",
        "srm_abs_std",
        "srm_abs_var",
        "srm_tail_delta",
    ]
    meta = {
        "height": height,
        "width": width,
        "block_h": block_h,
        "block_w": block_w,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    return feature_names, np.asarray(features, dtype=np.float32), indices, meta


def variant_scores(feature_names: list[str], matrix: np.ndarray) -> dict[str, tuple[np.ndarray, list[str]]]:
    z = robust_z(matrix)
    idx = {name: i for i, name in enumerate(feature_names)}
    original_names = ["rgb_std", "gray_mean", "gray_std", "edge_mean", "lap_var"]
    original = z[:, [idx[name] for name in original_names]].mean(axis=1)
    dct_srm = z[:, [idx[name] for name in ["dct_hf_ratio", "srm_abs_mean", "srm_abs_std", "srm_abs_var", "srm_tail_delta"]]].mean(axis=1)
    contrast_original = local_contrast(original, [(i // GRID, i % GRID) for i in range(len(original))])
    contrast_dct_srm = local_contrast(dct_srm, [(i // GRID, i % GRID) for i in range(len(dct_srm))])

    edge_lap_dct_srm = (
        0.15 * z[:, idx["edge_mean"]]
        + 0.15 * z[:, idx["lap_var"]]
        + 0.25 * z[:, idx["dct_hf_ratio"]]
        + 0.25 * z[:, idx["srm_abs_mean"]]
        + 0.10 * z[:, idx["srm_abs_var"]]
        + 0.10 * z[:, idx["srm_tail_delta"]]
    )

    return {
        "original_reimpl": (
            original,
            original_names,
        ),
        "dct_srm_only": (
            dct_srm,
            ["dct_hf_ratio", "srm_abs_mean", "srm_abs_std", "srm_abs_var", "srm_tail_delta"],
        ),
        "augmented_equal": (
            z.mean(axis=1),
            feature_names,
        ),
        "augmented_weighted": (
            0.45 * original + 0.55 * dct_srm,
            ["original_features", "dct_srm_features"],
        ),
        "forensic_weighted": (
            edge_lap_dct_srm,
            ["edge_mean", "lap_var", "dct_hf_ratio", "srm_abs_mean", "srm_abs_var", "srm_tail_delta"],
        ),
        "forensic_context": (
            0.75 * edge_lap_dct_srm + 0.25 * local_contrast(edge_lap_dct_srm, [(i // GRID, i % GRID) for i in range(len(edge_lap_dct_srm))]),
            ["forensic_weighted", "local_score_contrast"],
        ),
        "dct_srm_context": (
            0.75 * dct_srm + 0.25 * contrast_dct_srm,
            ["dct_srm_features", "local_score_contrast"],
        ),
        "original_context": (
            0.75 * original + 0.25 * contrast_original,
            ["original_features", "local_score_contrast"],
        ),
    }


def artifact_types_for_features(features: list[str], image_path: str) -> list[str]:
    artifacts: list[str] = []
    feature_set = set(features)
    if {"edge_mean", "lap_var", "dct_hf_ratio"} & feature_set:
        artifacts.append("boundary_seam")
    if {"rgb_std", "gray_std", "srm_abs_std", "srm_abs_var"} & feature_set:
        artifacts.append("texture_smoothness")
    if {"srm_abs_mean", "srm_tail_delta"} & feature_set:
        artifacts.append("resolution_noise")
    if "gray_mean" in feature_set:
        artifacts.append("lighting_shadow")
    if image_path.lower().endswith((".jpg", ".jpeg")):
        artifacts.append("compression_artifact")
    return artifacts[:3] or ["resolution_noise"]


def lowlevel_row_for_variant(row: dict[str, Any], sid_root: Path, variant: str) -> dict[str, Any]:
    feature_names, matrix, indices, meta = feature_matrix_for_row(row, sid_root)
    scores_by_variant = variant_scores(feature_names, matrix)
    if variant not in scores_by_variant:
        raise ValueError(f"Unknown variant: {variant}")
    block_scores, top_features = scores_by_variant[variant]

    score_threshold = max(float(np.percentile(block_scores, 95)), 2.0)
    picked = [indices[i] for i, score in enumerate(block_scores) if score >= score_threshold]
    if len(picked) > 8:
        top_indices = np.argsort(block_scores)[-8:]
        picked = [indices[i] for i in top_indices]

    candidate_bbox = bbox_from_blocks(
        picked,
        int(meta["block_h"]),
        int(meta["block_w"]),
        int(meta["height"]),
        int(meta["width"]),
        float(meta["scale_x"]),
        float(meta["scale_y"]),
    )
    candidates = candidate_bboxes_from_scores(
        block_scores,
        indices,
        int(meta["block_h"]),
        int(meta["block_w"]),
        int(meta["height"]),
        int(meta["width"]),
        float(meta["scale_x"]),
        float(meta["scale_y"]),
    )
    artifacts = artifact_types_for_features(top_features, row["image_path"])
    for candidate in candidates:
        candidate["artifact_types"] = artifacts

    return {
        "img_id": row["img_id"],
        "image_path": row["image_path"],
        "class_dir_for_evaluation_only": row["class_dir"],
        "lowlevel_candidate_bbox": candidate_bbox,
        "lowlevel_candidates": candidates,
        "lowlevel_artifact_types": artifacts,
        "lowlevel_score": round(float(block_scores.max()), 4) if len(block_scores) else 0.0,
        "top_lowlevel_features": top_features,
        "lowlevel_variant": variant,
        "baseline_feature_version": BASELINE_FEATURE_VERSION,
        "method_note": "DCT/SRM experiment uses only image pixels. Masks are used only for oracle evaluation.",
    }


def topk_boxes(row: dict[str, Any], top_k: int = 3) -> list[list[int]]:
    candidates = row.get("lowlevel_candidates")
    if not isinstance(candidates, list):
        return []
    boxes: list[list[int]] = []
    for candidate in candidates[:top_k]:
        if isinstance(candidate, dict):
            box = valid_bbox(candidate.get("bbox"))
            if box:
                boxes.append(box)
    return boxes


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "samples": int(len(arr)),
        "mean_bbox_iou": float(arr.mean()) if len(arr) else 0.0,
        "hit_rate_iou_0_1": float((arr >= 0.1).mean()) if len(arr) else 0.0,
        "hit_rate_iou_0_3": float((arr >= 0.3).mean()) if len(arr) else 0.0,
        "hit_rate_iou_0_5": float((arr >= 0.5).mean()) if len(arr) else 0.0,
    }


def evaluate_lowlevel_rows(
    rows: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    sid_root: Path,
    top_k: int = 3,
) -> dict[str, Any]:
    modes = {
        "current_lowlevel_candidate": [],
        "topk_first": [],
        "topk_oracle_single": [],
        "topk_union": [],
    }
    best_rank_counts = [0 for _ in range(top_k)]
    missing_topk = 0
    sample_ids: list[str] = []

    for pred in rows:
        row = manifest_by_id.get(pred["img_id"])
        if not row or row.get("class_dir") != "tampered" or not row.get("mask_path"):
            continue
        gt = mask_bbox(sid_root / row["mask_path"])
        boxes = topk_boxes(pred, top_k)
        if len(boxes) < top_k:
            missing_topk += 1
        box_ious = [bbox_iou(box, gt) for box in boxes]
        if box_ious:
            best_rank_counts[int(np.argmax(box_ious))] += 1

        modes["current_lowlevel_candidate"].append(bbox_iou(valid_bbox(pred.get("lowlevel_candidate_bbox")), gt))
        modes["topk_first"].append(bbox_iou(boxes[0] if boxes else None, gt))
        modes["topk_oracle_single"].append(max(box_ious + [0.0]))
        modes["topk_union"].append(bbox_iou(union_bbox(boxes), gt))
        sample_ids.append(pred["img_id"])

    return {
        "tampered_samples": len(sample_ids),
        "top_k": top_k,
        "missing_topk": missing_topk,
        "best_rank_counts": best_rank_counts,
        "metrics": {name: summarize(values) for name, values in modes.items()},
    }


def write_summary_markdown(path: Path, report: dict[str, Any]) -> None:
    old_union = report["baseline_old_top3"]["metrics"]["topk_union"]["hit_rate_iou_0_1"]
    old_current = report["baseline_old_top3"]["metrics"]["current_lowlevel_candidate"]["hit_rate_iou_0_1"]
    best = report["best_variant"]
    best_metrics = report["variants"][best]["metrics"]
    best_union = best_metrics["topk_union"]["hit_rate_iou_0_1"]
    best_current = best_metrics["current_lowlevel_candidate"]["hit_rate_iou_0_1"]

    lines = [
        "# DCT/SRM Low-Level Stage 1 Summary",
        "",
        "## 结论",
        "",
    ]
    if report["decision"]["connect_to_main_pipeline"]:
        lines.append("- DCT/SRM 通过强门槛，建议接入主流程继续评估。")
    elif report["decision"]["beats_single_bbox_only"]:
        lines.append("- DCT/SRM 只超过旧单框基线，但没有超过现有 top-3 union 上界，不建议直接接入主流程。该诊断基于 dev_300/global 候选比较，不是 SID-FA tiny 区域专项替代方案验证。")
    else:
        lines.append("- DCT/SRM 未超过旧低层基线，不建议接入主流程。该诊断基于 dev_300/global 候选比较，不是 SID-FA tiny 区域专项替代方案验证。")

    lines.extend(
        [
            "",
            "## 关键指标",
            "",
            f"- 旧单框 IoU@0.1: {old_current:.3f}",
            f"- 旧 top-3 union oracle IoU@0.1: {old_union:.3f}",
            f"- 最佳变体: `{best}`",
            f"- 最佳变体单框 IoU@0.1: {best_current:.3f}",
            f"- 最佳变体 top-3 union oracle IoU@0.1: {best_union:.3f}",
            f"- 强门槛: top-3 union oracle IoU@0.1 >= {report['decision']['strong_gate_threshold']:.3f}",
            "",
            "## 各变体结果",
            "",
            "| variant | current@0.1 | union@0.1 | union@0.3 | union@0.5 | mean_union_iou |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant, variant_report in report["variants"].items():
        current = variant_report["metrics"]["current_lowlevel_candidate"]
        union = variant_report["metrics"]["topk_union"]
        lines.append(
            f"| `{variant}` | {current['hit_rate_iou_0_1']:.3f} | "
            f"{union['hit_rate_iou_0_1']:.3f} | {union['hit_rate_iou_0_3']:.3f} | "
            f"{union['hit_rate_iou_0_5']:.3f} | {union['mean_bbox_iou']:.4f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    sid_root = args.sid_root.resolve()
    manifest_rows = read_jsonl(args.manifest)
    selected = select_rows(manifest_rows, args.per_class)
    manifest_by_id = {row["img_id"]: row for row in manifest_rows}
    variants = [
        "original_reimpl",
        "original_context",
        "dct_srm_only",
        "dct_srm_context",
        "augmented_equal",
        "augmented_weighted",
        "forensic_weighted",
        "forensic_context",
    ]

    outputs: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    for idx, row in enumerate(selected, start=1):
        feature_names, matrix, indices, meta = feature_matrix_for_row(row, sid_root)
        scores_by_variant = variant_scores(feature_names, matrix)
        for variant in variants:
            block_scores, top_features = scores_by_variant[variant]
            score_threshold = max(float(np.percentile(block_scores, 95)), 2.0)
            picked = [indices[i] for i, score in enumerate(block_scores) if score >= score_threshold]
            if len(picked) > 8:
                top_indices = np.argsort(block_scores)[-8:]
                picked = [indices[i] for i in top_indices]
            candidate_bbox = bbox_from_blocks(
                picked,
                int(meta["block_h"]),
                int(meta["block_w"]),
                int(meta["height"]),
                int(meta["width"]),
                float(meta["scale_x"]),
                float(meta["scale_y"]),
            )
            candidates = candidate_bboxes_from_scores(
                block_scores,
                indices,
                int(meta["block_h"]),
                int(meta["block_w"]),
                int(meta["height"]),
                int(meta["width"]),
                float(meta["scale_x"]),
                float(meta["scale_y"]),
            )
            artifacts = artifact_types_for_features(top_features, row["image_path"])
            for candidate in candidates:
                candidate["artifact_types"] = artifacts
            outputs[variant].append(
                {
                    "img_id": row["img_id"],
                    "image_path": row["image_path"],
                    "class_dir_for_evaluation_only": row["class_dir"],
                    "lowlevel_candidate_bbox": candidate_bbox,
                    "lowlevel_candidates": candidates,
                    "lowlevel_artifact_types": artifacts,
                    "lowlevel_score": round(float(block_scores.max()), 4) if len(block_scores) else 0.0,
                    "top_lowlevel_features": top_features,
                    "lowlevel_variant": variant,
                    "method_note": "DCT/SRM experiment uses only image pixels. Masks are used only for oracle evaluation.",
                }
            )
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(selected)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    variant_reports: dict[str, Any] = {}
    for variant, rows in outputs.items():
        out_path = args.output_dir / f"dev_300_dct_srm_lowlevel_{variant}.jsonl"
        write_jsonl(out_path, rows)
        variant_reports[variant] = evaluate_lowlevel_rows(rows, manifest_by_id, sid_root, top_k=args.top_k)
        variant_reports[variant]["output_file"] = str(out_path)

    baseline_rows = read_jsonl(args.old_lowlevel)
    baseline_report = evaluate_lowlevel_rows(baseline_rows, manifest_by_id, sid_root, top_k=args.top_k)
    baseline_report["output_file"] = str(args.old_lowlevel)

    best_variant = max(
        variant_reports,
        key=lambda name: (
            variant_reports[name]["metrics"]["topk_union"]["hit_rate_iou_0_1"],
            variant_reports[name]["metrics"]["topk_union"]["hit_rate_iou_0_3"],
            variant_reports[name]["metrics"]["topk_union"]["mean_bbox_iou"],
        ),
    )
    best_rows = outputs[best_variant]
    best_path = args.output_dir / "dev_300_dct_srm_lowlevel_best.jsonl"
    write_jsonl(best_path, best_rows)

    old_current = baseline_report["metrics"]["current_lowlevel_candidate"]["hit_rate_iou_0_1"]
    old_union = baseline_report["metrics"]["topk_union"]["hit_rate_iou_0_1"]
    best_current = variant_reports[best_variant]["metrics"]["current_lowlevel_candidate"]["hit_rate_iou_0_1"]
    best_union = variant_reports[best_variant]["metrics"]["topk_union"]["hit_rate_iou_0_1"]
    strong_gate_threshold = old_union + args.min_improvement
    single_gate_threshold = old_current + args.min_improvement
    decision = {
        "min_improvement": args.min_improvement,
        "single_gate_threshold": single_gate_threshold,
        "strong_gate_threshold": strong_gate_threshold,
        "beats_single_bbox_only": best_current >= single_gate_threshold or best_union >= single_gate_threshold,
        "connect_to_main_pipeline": best_union >= strong_gate_threshold,
        "reason": (
            "connect only if DCT/SRM top-3 union oracle beats old top-3 union by the configured margin"
        ),
    }

    report = {
        "stage": "dct_srm_lowlevel_oracle",
        "sid_root": str(args.sid_root),
        "manifest": str(args.manifest),
        "samples": len(selected),
        "per_class": args.per_class,
        "baseline_old_top3": baseline_report,
        "variants": variant_reports,
        "best_variant": best_variant,
        "best_output_file": str(best_path),
        "decision": decision,
    }
    metrics_path = args.report_dir / "dev_300_dct_srm_oracle_metrics.json"
    summary_path = args.report_dir / "dct_srm_lowlevel_stage1_summary.md"
    write_json(metrics_path, report)
    write_summary_markdown(summary_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid-root", type=Path, default=Path("SID_Set"))
    parser.add_argument("--manifest", type=Path, default=Path("SID_Set/splits/dev.jsonl"))
    parser.add_argument("--old-lowlevel", type=Path, default=Path("experiments/runs/dev_300_lowlevel_top3_v2.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--report-dir", type=Path, default=Path("experiments/reports"))
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-improvement", type=float, default=0.04)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
