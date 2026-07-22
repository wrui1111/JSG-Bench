#!/usr/bin/env python3
"""
Prepare SID-Set for the proposed experiment.

This script does not copy, move, delete, or modify original images. It only
creates split manifests and annotation templates that reference existing files.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CLASS_NAMES = ["real", "full_synthetic", "tampered"]
LABEL_TO_CLASS = {0: "real", 1: "full_synthetic", 2: "tampered"}
ARTIFACT_TYPES = [
    "boundary_seam",
    "texture_smoothness",
    "lighting_shadow",
    "geometry_structure",
    "resolution_noise",
    "semantic_implausibility",
    "compression_artifact",
    "other",
]
TARGET_SCOPES = ["face", "object", "background", "text", "whole_image", "other"]
EXPECTED_FAILURE_STAGES = [
    "structured_observation",
    "artifact_typing",
    "evidence_aggregation",
    "grounding_verification",
    "none",
]


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


def attach_abs_path(row: dict[str, Any], sid_root: Path) -> dict[str, Any]:
    image_path = sid_root / row["image_path"]
    mask_path = sid_root / row["mask_path"] if row.get("mask_path") else None
    out = dict(row)
    out["_abs_image_path"] = str(image_path)
    out["_abs_mask_path"] = str(mask_path) if mask_path else None
    return out


def strip_private(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def mask_features(row: dict[str, Any], sid_root: Path) -> dict[str, Any]:
    if not row.get("mask_path"):
        return {
            "mask_area_ratio": None,
            "mask_bbox": None,
            "mask_area_bucket": None,
        }

    mask_path = sid_root / row["mask_path"]
    with Image.open(mask_path) as img:
        arr = np.asarray(img.convert("L")) > 0

    if not arr.any():
        return {
            "mask_area_ratio": 0.0,
            "mask_bbox": None,
            "mask_area_bucket": "empty",
        }

    ys, xs = np.where(arr)
    ratio = float(arr.mean())
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]

    if ratio < 0.01:
        bucket = "tiny"
    elif ratio < 0.05:
        bucket = "small"
    elif ratio < 0.20:
        bucket = "medium"
    else:
        bucket = "large"

    return {
        "mask_area_ratio": round(ratio, 6),
        "mask_bbox": bbox,
        "mask_area_bucket": bucket,
    }


def image_quality_features(row: dict[str, Any], sid_root: Path) -> dict[str, Any]:
    image_path = sid_root / row["image_path"]
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    pixels = max(width * height, 1)
    file_bytes = image_path.stat().st_size if image_path.exists() else 0
    bytes_per_pixel = file_bytes / pixels

    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            gray.thumbnail((256, 256))
            arr = np.asarray(gray, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - defensive data audit path
        return {
            "file_bytes": file_bytes,
            "bytes_per_pixel": round(bytes_per_pixel, 6),
            "quality_error": str(exc),
            "difficulty_tags_auto": ["read_error"],
            "difficulty_score_auto": 10,
        }

    contrast = float(arr.std()) if arr.size else 0.0
    if arr.shape[0] >= 3 and arr.shape[1] >= 3:
        gx = np.diff(arr, axis=1)
        gy = np.diff(arr, axis=0)
        edge_density = float(
            (np.mean(np.abs(gx) > 25.0) + np.mean(np.abs(gy) > 25.0)) / 2.0
        )
        lap = (
            arr[2:, 1:-1]
            + arr[:-2, 1:-1]
            + arr[1:-1, 2:]
            + arr[1:-1, :-2]
            - 4 * arr[1:-1, 1:-1]
        )
        laplacian_var = float(lap.var()) if lap.size else 0.0
    else:
        edge_density = 0.0
        laplacian_var = 0.0

    tags: list[str] = []
    if min(width, height) <= 512:
        tags.append("low_resolution")
    if width and height and max(width / height, height / width) >= 1.65:
        tags.append("extreme_aspect_ratio")
    if row["image_path"].lower().endswith((".jpg", ".jpeg")) and bytes_per_pixel < 0.18:
        tags.append("compression_artifact_candidate")
    if contrast < 35.0:
        tags.append("low_contrast")
    if laplacian_var < 120.0:
        tags.append("blur_or_smooth_candidate")
    if edge_density > 0.22:
        tags.append("complex_background_candidate")

    return {
        "file_bytes": file_bytes,
        "bytes_per_pixel": round(bytes_per_pixel, 6),
        "contrast_std": round(contrast, 4),
        "edge_density": round(edge_density, 6),
        "laplacian_var": round(laplacian_var, 4),
        "difficulty_tags_auto": tags,
        "difficulty_score_auto": len(tags),
    }


def expected_failure_stage(tags: list[str]) -> str:
    tag_set = set(tags)
    if {"very_small_region", "small_region", "low_resolution"} & tag_set:
        return "grounding_verification"
    if {"compression_artifact_candidate", "blur_or_smooth_candidate", "low_contrast"} & tag_set:
        return "artifact_typing"
    if "complex_background_candidate" in tag_set:
        return "evidence_aggregation"
    return "none"


def split_manifest(
    manifest: list[dict[str, Any]],
    rng: random.Random,
    dev_per_class: int,
    val_per_class: int,
) -> dict[str, list[dict[str, Any]]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        class_name = row.get("class_dir") or LABEL_TO_CLASS.get(row.get("label"))
        if class_name not in CLASS_NAMES:
            raise ValueError(f"Unknown class in manifest row: {row}")
        by_class[class_name].append(row)

    splits = {"dev": [], "val": [], "test": []}
    for class_name in CLASS_NAMES:
        rows = list(by_class[class_name])
        rng.shuffle(rows)
        needed = dev_per_class + val_per_class
        if len(rows) <= needed:
            raise ValueError(
                f"Class {class_name} has {len(rows)} rows, fewer than required {needed + 1}."
            )
        splits["dev"].extend({**r, "split": "dev"} for r in rows[:dev_per_class])
        splits["val"].extend(
            {**r, "split": "val"} for r in rows[dev_per_class:needed]
        )
        splits["test"].extend({**r, "split": "test"} for r in rows[needed:])

    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def count_by_class(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for row in rows:
        counts[row["class_dir"]] += 1
    return counts


def select_random(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    used_ids: set[str],
) -> list[dict[str, Any]]:
    pool = [r for r in rows if r["img_id"] not in used_ids]
    if len(pool) < n:
        raise ValueError(f"Not enough rows to select {n}; available={len(pool)}")
    picked = rng.sample(pool, n)
    used_ids.update(r["img_id"] for r in picked)
    return picked


def select_tampered_balanced(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    used_ids: set[str],
    features_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    bucket_order = ["tiny", "small", "medium", "large"]
    groups: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in bucket_order}
    for row in rows:
        if row["img_id"] in used_ids:
            continue
        bucket = features_by_id[row["img_id"]]["mask_area_bucket"]
        if bucket in groups:
            groups[bucket].append(row)

    selected: list[dict[str, Any]] = []
    base = n // len(bucket_order)
    remainder = n % len(bucket_order)
    targets = {
        bucket: base + (1 if i < remainder else 0)
        for i, bucket in enumerate(bucket_order)
    }

    for bucket in bucket_order:
        rng.shuffle(groups[bucket])
        take = min(targets[bucket], len(groups[bucket]))
        selected.extend(groups[bucket][:take])

    if len(selected) < n:
        selected_ids = {r["img_id"] for r in selected}
        rest = [
            r
            for bucket in bucket_order
            for r in groups[bucket]
            if r["img_id"] not in selected_ids
        ]
        rng.shuffle(rest)
        selected.extend(rest[: n - len(selected)])

    if len(selected) != n:
        raise ValueError(f"Could only select {len(selected)} tampered rows, need {n}.")

    used_ids.update(r["img_id"] for r in selected)
    return selected


def make_explain_row(
    row: dict[str, Any],
    sid_root: Path,
    mask_features_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    features = mask_features_by_id.get(row["img_id"], {})
    first_bbox = features.get("mask_bbox") if row["class_dir"] == "tampered" else None
    return {
        "annotation_task": "SID-Explain",
        "annotation_status": "unlabeled",
        "do_not_use_for_tuning": True,
        "source_split": "frozen_annotation_pool",
        "img_id": row["img_id"],
        "human_label": row["class_dir"],
        "image_path": row["image_path"],
        "mask_path": row.get("mask_path"),
        "width": row.get("width"),
        "height": row.get("height"),
        "auto_features": features,
        "evidence_items": [
            {
                "evidence_bbox": first_bbox,
                "artifact_type": None,
                "evidence_text": None,
                "confidence": None,
            }
        ],
        "summary_reason": None,
        "allowed_artifact_types": ARTIFACT_TYPES,
        "annotation_note": "人工填写 artifact_type、evidence_text、confidence、summary_reason；tampered 可参考 mask_bbox，但不要把 mask 作为模型输入。",
    }


def make_fa_row(
    row: dict[str, Any],
    mask_features_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    features = mask_features_by_id[row["img_id"]]
    return {
        "annotation_task": "SID-FA",
        "annotation_status": "unlabeled",
        "do_not_use_for_tuning": True,
        "source_split": "frozen_annotation_pool",
        "img_id": row["img_id"],
        "human_label": "tampered",
        "image_path": row["image_path"],
        "mask_path": row.get("mask_path"),
        "width": row.get("width"),
        "height": row.get("height"),
        "mask_bbox": features["mask_bbox"],
        "mask_area_ratio": features["mask_area_ratio"],
        "mask_area_bucket": features["mask_area_bucket"],
        "target_scope": None,
        "target_semantic_class": None,
        "dominant_artifact_type": None,
        "secondary_artifact_types": [],
        "artifact_strength": None,
        "violated_rule": None,
        "short_note": None,
        "allowed_target_scopes": TARGET_SCOPES,
        "allowed_artifact_types": ARTIFACT_TYPES,
    }


def make_hard_row(
    row: dict[str, Any],
    auto_features: dict[str, Any],
    mask_features_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tags = list(auto_features.get("difficulty_tags_auto") or [])
    mask_auto = mask_features_by_id.get(row["img_id"], {})
    if row["class_dir"] == "tampered":
        ratio = mask_auto.get("mask_area_ratio")
        if ratio is not None and ratio < 0.01:
            tags.append("very_small_region")
        elif ratio is not None and ratio < 0.03:
            tags.append("small_region")

    tags = sorted(set(tags))
    score = len(tags)
    return {
        "annotation_task": "SID-Hard",
        "annotation_status": "unlabeled",
        "do_not_use_for_tuning": True,
        "source_split": "frozen_annotation_pool",
        "img_id": row["img_id"],
        "human_label": row["class_dir"],
        "image_path": row["image_path"],
        "mask_path": row.get("mask_path"),
        "width": row.get("width"),
        "height": row.get("height"),
        "auto_features": {**auto_features, **mask_auto},
        "difficulty_tags_auto": tags,
        "difficulty_score_auto": score,
        "difficulty_tags": None,
        "difficulty_score": None,
        "main_failure_risk": None,
        "expected_failure_stage_auto": expected_failure_stage(tags),
        "expected_failure_stage": None,
        "human_note": None,
        "allowed_expected_failure_stages": EXPECTED_FAILURE_STAGES,
    }


def select_hard(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    used_ids: set[str],
    quality_features_by_id: dict[str, dict[str, Any]],
    mask_features_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    scored: list[tuple[int, float, dict[str, Any]]] = []
    for row in rows:
        if row["img_id"] in used_ids:
            continue
        features = quality_features_by_id[row["img_id"]]
        score = int(features.get("difficulty_score_auto") or 0)
        mask_auto = mask_features_by_id.get(row["img_id"], {})
        ratio = mask_auto.get("mask_area_ratio")
        if row["class_dir"] == "tampered" and ratio is not None:
            if ratio < 0.01:
                score += 3
            elif ratio < 0.03:
                score += 2
        jitter = rng.random()
        scored.append((score, jitter, row))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picked = [row for _, _, row in scored[:n]]
    if len(picked) != n:
        raise ValueError(f"Could only select {len(picked)} hard rows, need {n}.")
    used_ids.update(r["img_id"] for r in picked)
    return picked


def build_hard_pool(
    rows: list[dict[str, Any]],
    class_name: str,
    scan_limit: int,
    rng: random.Random,
    used_ids: set[str],
    mask_features_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    available = [row for row in rows if row["img_id"] not in used_ids]
    if len(available) <= scan_limit:
        return available

    if class_name == "tampered":
        sorted_by_area = sorted(
            available,
            key=lambda row: mask_features_by_id[row["img_id"]].get("mask_area_ratio") or 1.0,
        )
        small_region_quota = scan_limit // 2
        selected = sorted_by_area[:small_region_quota]
        selected_ids = {row["img_id"] for row in selected}
        rest = [row for row in available if row["img_id"] not in selected_ids]
        selected.extend(rng.sample(rest, scan_limit - len(selected)))
        rng.shuffle(selected)
        return selected

    return rng.sample(available, scan_limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid-root", type=Path, default=Path("SID_Set"))
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--dev-per-class", type=int, default=1000)
    parser.add_argument("--val-per-class", type=int, default=2000)
    parser.add_argument("--explain-per-class", type=int, default=100)
    parser.add_argument("--fa-tampered", type=int, default=600)
    parser.add_argument("--hard-per-class", type=int, default=100)
    parser.add_argument(
        "--hard-scan-per-class",
        type=int,
        default=1500,
        help="Number of per-class test candidates to scan before selecting SID-Hard.",
    )
    args = parser.parse_args()

    sid_root = args.sid_root.resolve()
    manifest_path = sid_root / "validation" / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    rng = random.Random(args.seed)
    manifest = read_jsonl(manifest_path)
    splits = split_manifest(
        manifest,
        rng,
        dev_per_class=args.dev_per_class,
        val_per_class=args.val_per_class,
    )

    split_dir = sid_root / "splits"
    for split_name, rows in splits.items():
        write_jsonl(split_dir / f"{split_name}.jsonl", [strip_private(r) for r in rows])

    test_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in splits["test"]:
        test_by_class[row["class_dir"]].append(row)

    print("Computing mask features for tampered test rows...")
    mask_features_by_id = {
        row["img_id"]: mask_features(row, sid_root)
        for row in test_by_class["tampered"]
    }

    print("Selecting SID-Explain and SID-FA candidates...")
    used_ids: set[str] = set()
    explain_rows: list[dict[str, Any]] = []
    explain_rows.extend(
        select_random(
            test_by_class["real"],
            args.explain_per_class,
            rng,
            used_ids,
        )
    )
    explain_rows.extend(
        select_random(
            test_by_class["full_synthetic"],
            args.explain_per_class,
            rng,
            used_ids,
        )
    )
    explain_rows.extend(
        select_tampered_balanced(
            test_by_class["tampered"],
            args.explain_per_class,
            rng,
            used_ids,
            mask_features_by_id,
        )
    )
    rng.shuffle(explain_rows)

    fa_rows = select_tampered_balanced(
        test_by_class["tampered"],
        args.fa_tampered,
        rng,
        used_ids,
        mask_features_by_id,
    )
    rng.shuffle(fa_rows)

    print("Building hard-set candidate pools...")
    hard_pool_by_class = {
        class_name: build_hard_pool(
            test_by_class[class_name],
            class_name,
            args.hard_scan_per_class,
            rng,
            used_ids,
            mask_features_by_id,
        )
        for class_name in CLASS_NAMES
    }
    hard_candidate_rows = [
        row for class_name in CLASS_NAMES for row in hard_pool_by_class[class_name]
    ]
    print(f"Computing image quality features for {len(hard_candidate_rows)} hard-set candidates...")
    quality_features_by_id = {
        row["img_id"]: image_quality_features(row, sid_root) for row in hard_candidate_rows
    }

    print("Selecting SID-Hard candidates...")
    hard_rows: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        hard_rows.extend(
            select_hard(
                hard_pool_by_class[class_name],
                args.hard_per_class,
                rng,
                used_ids,
                quality_features_by_id,
                mask_features_by_id,
            )
        )
    rng.shuffle(hard_rows)

    annotations_dir = sid_root / "annotations"
    explain_template = [
        make_explain_row(row, sid_root, mask_features_by_id) for row in explain_rows
    ]
    fa_template = [make_fa_row(row, mask_features_by_id) for row in fa_rows]
    hard_template = [
        make_hard_row(row, quality_features_by_id[row["img_id"]], mask_features_by_id)
        for row in hard_rows
    ]

    write_jsonl(annotations_dir / "SID-Explain-300.jsonl", explain_template)
    write_jsonl(annotations_dir / "SID-FA-600.jsonl", fa_template)
    write_jsonl(annotations_dir / "SID-Hard-300.jsonl", hard_template)

    annotated_ids = {r["img_id"] for r in explain_rows + fa_rows + hard_rows}
    test_main = [r for r in splits["test"] if r["img_id"] not in annotated_ids]
    write_jsonl(split_dir / "test_main_without_annotation_subsets.jsonl", test_main)

    mask_bucket_counts: dict[str, int] = defaultdict(int)
    for row in fa_rows:
        bucket = mask_features_by_id[row["img_id"]]["mask_area_bucket"]
        mask_bucket_counts[bucket] += 1

    summary = {
        "sid_root": str(sid_root),
        "seed": args.seed,
        "note": "Original SID-Set images are unchanged. Split and annotation files only reference original image paths.",
        "splits": {
            name: {"total": len(rows), "by_class": count_by_class(rows)}
            for name, rows in splits.items()
        },
        "annotation_subsets": {
            "SID-Explain-300": {
                "path": str(annotations_dir / "SID-Explain-300.jsonl"),
                "total": len(explain_template),
                "by_class": count_by_class(explain_rows),
            },
            "SID-FA-600": {
                "path": str(annotations_dir / "SID-FA-600.jsonl"),
                "total": len(fa_template),
                "by_class": count_by_class(fa_rows),
                "tampered_mask_area_buckets": dict(sorted(mask_bucket_counts.items())),
            },
            "SID-Hard-300": {
                "path": str(annotations_dir / "SID-Hard-300.jsonl"),
                "total": len(hard_template),
                "by_class": count_by_class(hard_rows),
            },
        },
        "clean_final_test": {
            "path": str(split_dir / "test_main_without_annotation_subsets.jsonl"),
            "total": len(test_main),
            "by_class": count_by_class(test_main),
        },
    }
    write_json(Path("experiments/reports/preparation_summary.json"), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
