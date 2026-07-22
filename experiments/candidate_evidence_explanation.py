#!/usr/bin/env python3
"""Candidate-constrained evidence explanation for annotated SID subsets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PREDICTION_ALIASES = {
    "real": "real",
    "authentic": "real",
    "genuine": "real",
    "full_synthetic": "full_synthetic",
    "fully_synthetic": "full_synthetic",
    "synthetic": "full_synthetic",
    "ai_generated": "full_synthetic",
    "ai-generated": "full_synthetic",
    "tampered": "tampered",
    "manipulated": "tampered",
    "local_tampering": "tampered",
    "global_generation": "full_synthetic",
    "none": "real",
}

ARTIFACT_ALIASES = {
    None: "none",
    "": "none",
    "null": "none",
    "none": "none",
    "boundary_seam": "boundary_seam",
    "texture_smoothness": "texture_smoothness",
    "lighting_shadow": "lighting_shadow",
    "geometry_structure": "geometry_structure",
    "resolution_noise": "resolution_noise",
    "semantic_implausibility": "semantic_implausibility",
    "compression_artifact": "compression_artifact",
    "other": "other",
}

TARGET_ALIASES = {
    None: "none",
    "": "none",
    "null": "none",
    "none": "none",
    "face": "face",
    "person": "person",
    "animal": "animal",
    "body": "object",
    "hand": "object",
    "object": "object",
    "background": "background",
    "text": "text",
    "whole_image": "whole_image",
    "whole image": "whole_image",
    "global": "whole_image",
    "mixed": "other",
    "unknown": "other",
    "other": "other",
}


PROMPT = """You are a forensic vision-language analyst.

You receive two images:
1. the full image with one red candidate box;
2. a crop around the same candidate box.

The red box is proposed by a pixel-only detector. It may be correct or a false
positive. Do not search for another region. Verify whether the marked candidate
is useful evidence, and then output one evidence card.

Output exactly one JSON object:
{
  "prediction": "real",
  "candidate_verdict": "accepted",
  "confidence": 0.0,
  "evidence_bbox": null,
  "target_scope": "none",
  "artifact_type": "none",
  "evidence_text": "short reason"
}

Allowed prediction values:
- real
- full_synthetic
- tampered

Allowed candidate_verdict values:
- accepted
- rejected
- uncertain

Allowed target_scope values:
- none
- face
- person
- animal
- object
- background
- text
- whole_image
- other

Allowed artifact_type values:
- none
- boundary_seam
- texture_smoothness
- lighting_shadow
- geometry_structure
- resolution_noise
- semantic_implausibility
- compression_artifact
- other

Decision rules:
1. Use prediction=tampered only if the red-boxed candidate contains localized
   evidence stronger than nearby context, such as pasted boundary, local texture
   mismatch, lighting/shadow mismatch, geometry inconsistency, local noise or
   compression inconsistency.
2. If prediction=tampered, set candidate_verdict=accepted and set evidence_bbox
   exactly to the provided candidate_bbox from the metadata.
3. If the image looks globally AI-generated rather than locally edited, use
   prediction=full_synthetic, target_scope=whole_image, evidence_bbox=null.
4. If there is no clear forensic evidence, use prediction=real, target_scope=none,
   artifact_type=none, evidence_bbox=null.
5. Do not mention filename, dataset label, split, or mask.
6. Output JSON only, with no markdown fence."""

PROMPT_MARKED_ONLY = """You are a forensic vision-language analyst.

You receive the full image with one red candidate box.

The red box is proposed by a pixel-only detector. It may be correct or a false
positive. Do not search for another region. Verify whether the marked candidate
is useful evidence, and then output one evidence card.

Output exactly one JSON object:
{
  "prediction": "real",
  "candidate_verdict": "accepted",
  "confidence": 0.0,
  "evidence_bbox": null,
  "target_scope": "none",
  "artifact_type": "none",
  "evidence_text": "short reason"
}

Allowed prediction values:
- real
- full_synthetic
- tampered

Allowed candidate_verdict values:
- accepted
- rejected
- uncertain

Allowed target_scope values:
- none
- face
- person
- animal
- object
- background
- text
- whole_image
- other

Allowed artifact_type values:
- none
- boundary_seam
- texture_smoothness
- lighting_shadow
- geometry_structure
- resolution_noise
- semantic_implausibility
- compression_artifact
- other

Decision rules:
1. Use prediction=tampered only if the red-boxed candidate contains localized
   evidence stronger than nearby context, such as pasted boundary, local texture
   mismatch, lighting/shadow mismatch, geometry inconsistency, local noise or
   compression inconsistency.
2. If prediction=tampered, set candidate_verdict=accepted and set evidence_bbox
   exactly to the provided candidate_bbox from the metadata.
3. If the image looks globally AI-generated rather than locally edited, use
   prediction=full_synthetic, target_scope=whole_image, evidence_bbox=null.
4. If there is no clear forensic evidence, use prediction=real, target_scope=none,
   artifact_type=none, evidence_bbox=null.
5. Do not mention filename, dataset label, split, or mask.
6. Output JSON only, with no markdown fence."""

PROMPT_CROP_ONLY = """You are a forensic vision-language analyst.

You receive one crop image around a candidate region proposed by a pixel-only
detector. You do not see the full image context in this diagnostic setting.

The crop may contain useful local forensic evidence, but it may also be a false
positive crop. Verify whether the crop contains localized evidence and then
output one evidence card.

Output exactly one JSON object:
{
  "prediction": "real",
  "candidate_verdict": "accepted",
  "confidence": 0.0,
  "evidence_bbox": null,
  "target_scope": "none",
  "artifact_type": "none",
  "evidence_text": "short reason"
}

Allowed prediction values:
- real
- full_synthetic
- tampered

Allowed candidate_verdict values:
- accepted
- rejected
- uncertain

Allowed target_scope values:
- none
- face
- person
- animal
- object
- background
- text
- whole_image
- other

Allowed artifact_type values:
- none
- boundary_seam
- texture_smoothness
- lighting_shadow
- geometry_structure
- resolution_noise
- semantic_implausibility
- compression_artifact
- other

Decision rules:
1. Use prediction=tampered only if the crop contains localized forensic
   evidence such as pasted boundary, local texture mismatch, lighting/shadow
   mismatch, geometry inconsistency, local noise, or compression inconsistency.
2. If prediction=tampered, set candidate_verdict=accepted and set evidence_bbox
   exactly to the provided candidate_bbox from the metadata.
3. If the crop does not provide enough context to decide, use
   candidate_verdict=uncertain and choose the most conservative prediction.
4. If there is no clear forensic evidence, use prediction=real, target_scope=none,
   artifact_type=none, evidence_bbox=null.
5. Do not mention filename, dataset label, split, or mask.
6. Output JSON only, with no markdown fence."""

PROMPT_TARGET_V2 = """You are a forensic vision-language analyst.

You receive two images:
1. the full image with one red candidate box;
2. a crop around the same candidate box.

The red box is proposed by a pixel-only detector. It may be correct or a false
positive. Do not search for another region. Verify whether the marked candidate
is useful evidence, and then output one evidence card.

This diagnostic prompt focuses on TARGET SCOPE. Be strict:
- target_scope=person if the boxed/cropped evidence is on a human face, head,
  body, hand, arm, leg, clothing attached to a person, or a whole person.
- target_scope=animal if the evidence is on an animal body or animal face.
- target_scope=object only for non-human, non-animal physical objects.
- target_scope=background only for scene/background regions such as sky, wall,
  floor, road, water, vegetation, or general background.
- target_scope=text only for letters, logos, signs, or printed/written text.
- Do not label a human body part as object.
- Do not label a face as object.

Output exactly one JSON object:
{
  "prediction": "tampered",
  "candidate_verdict": "accepted",
  "confidence": 0.0,
  "evidence_bbox": null,
  "target_scope": "person",
  "artifact_type": "geometry_structure",
  "evidence_text": "short reason"
}

Allowed prediction values:
- real
- full_synthetic
- tampered

Allowed candidate_verdict values:
- accepted
- rejected
- uncertain

Allowed target_scope values:
- none
- face
- person
- animal
- object
- background
- text
- whole_image
- other

Allowed artifact_type values:
- none
- boundary_seam
- texture_smoothness
- lighting_shadow
- geometry_structure
- resolution_noise
- semantic_implausibility
- compression_artifact
- other

Decision rules:
1. Use prediction=tampered only if the red-boxed candidate contains localized
   evidence stronger than nearby context.
2. If prediction=tampered, set candidate_verdict=accepted and set evidence_bbox
   exactly to the provided candidate_bbox from the metadata.
3. First decide the visible target_scope using the strict rules above; then
   decide artifact_type.
4. If there is no clear forensic evidence, use prediction=real, target_scope=none,
   artifact_type=none, evidence_bbox=null.
5. Do not mention filename, dataset label, split, or mask.
6. Output JSON only, with no markdown fence."""

PROMPT_ARTIFACT_V2 = """You are a forensic vision-language analyst.

You receive two images:
1. the full image with one red candidate box;
2. a crop around the same candidate box.

The red box is proposed by a pixel-only detector. It may be correct or a false
positive. Do not search for another region. Verify whether the marked candidate
is useful evidence, and then output one evidence card.

This diagnostic prompt focuses on ARTIFACT TYPING. Before choosing
artifact_type, check these evidence types in order:
1. boundary_seam: pasted edge, halo, cut line, abrupt local boundary.
2. texture_smoothness: local texture, smoothness, blur, or surface pattern
   differs from nearby context.
3. lighting_shadow: local light direction, shadow, reflection, color
   temperature, or illumination is inconsistent.
4. geometry_structure: distorted shape, wrong anatomy, impossible object
   structure, impossible pose, or structural mismatch.
5. resolution_noise: local sharpness, resolution, sensor noise, or grain differs.
6. compression_artifact: local JPEG/block/compression artifacts differ.
7. semantic_implausibility: object identity, interaction, or scene meaning is
   locally implausible even if pixels are smooth.

Do not default to geometry_structure. Use geometry_structure only when shape,
anatomy, pose, or object structure is the main visible evidence. If the main
cue is a boundary, texture, lighting, noise, or compression mismatch, choose
that specific type.

Output exactly one JSON object:
{
  "prediction": "tampered",
  "candidate_verdict": "accepted",
  "confidence": 0.0,
  "evidence_bbox": null,
  "target_scope": "object",
  "artifact_type": "boundary_seam",
  "evidence_text": "short reason"
}

Allowed prediction values:
- real
- full_synthetic
- tampered

Allowed candidate_verdict values:
- accepted
- rejected
- uncertain

Allowed target_scope values:
- none
- face
- person
- animal
- object
- background
- text
- whole_image
- other

Allowed artifact_type values:
- none
- boundary_seam
- texture_smoothness
- lighting_shadow
- geometry_structure
- resolution_noise
- semantic_implausibility
- compression_artifact
- other

Decision rules:
1. Use prediction=tampered only if the red-boxed candidate contains localized
   evidence stronger than nearby context.
2. If prediction=tampered, set candidate_verdict=accepted and set evidence_bbox
   exactly to the provided candidate_bbox from the metadata.
3. Choose the most specific artifact_type supported by visible evidence.
4. If evidence is weak or normal, use prediction=real, target_scope=none,
   artifact_type=none, evidence_bbox=null.
5. Do not mention filename, dataset label, split, or mask.
6. Output JSON only, with no markdown fence."""

PROMPT_BBOX_REFINE_V1 = """You are a forensic vision-language analyst.

You receive two images:
1. the full image with one red coarse candidate box;
2. a crop around the same coarse candidate box.

The red box is proposed by a pixel-only detector. It may cover the true
tampered evidence, miss it, or be a false positive. Use the red box as a
coarse visual prior, but do not blindly copy its coordinates.

Output exactly one JSON object:
{
  "prediction": "tampered",
  "candidate_verdict": "accepted",
  "confidence": 0.0,
  "evidence_bbox": [0, 0, 10, 10],
  "target_scope": "object",
  "artifact_type": "boundary_seam",
  "evidence_text": "short reason"
}

Allowed prediction values:
- real
- full_synthetic
- tampered

Allowed candidate_verdict values:
- accepted
- rejected
- uncertain

Allowed target_scope values:
- none
- face
- person
- animal
- object
- background
- text
- whole_image
- other

Allowed artifact_type values:
- none
- boundary_seam
- texture_smoothness
- lighting_shadow
- geometry_structure
- resolution_noise
- semantic_implausibility
- compression_artifact
- other

Decision rules:
1. First decide whether the red-boxed candidate contains local forensic evidence
   stronger than nearby context.
2. If prediction=tampered, output your best refined evidence_bbox in FULL-IMAGE
   coordinates [x1, y1, x2, y2]. The bbox may be smaller than, shifted from, or
   partially outside the red box if the visible evidence supports it.
3. Do not copy candidate_bbox unless it is already the best evidence box.
4. If the image looks globally AI-generated rather than locally edited, use
   prediction=full_synthetic, target_scope=whole_image, evidence_bbox=null.
5. If there is no clear forensic evidence, use prediction=real, target_scope=none,
   artifact_type=none, evidence_bbox=null.
6. Do not mention filename, dataset label, split, or mask.
7. Output JSON only, with no markdown fence."""

PROMPT_ORACLE_SEMANTIC_V1 = """You are a forensic vision-language analyst.

This is an ORACLE diagnostic, not a deployable inference setting.

You receive two images:
1. the full image with one red oracle evidence box;
2. a crop around the same oracle evidence box.

The red box is known to cover the manipulated evidence region for evaluation
purposes. Do not decide whether the image is real. Assume the boxed region is
the local tampering evidence and focus only on semantic attribution:
- what target_scope is manipulated;
- what artifact_type best explains the visible evidence;
- a short evidence_text.

Output exactly one JSON object:
{
  "prediction": "tampered",
  "candidate_verdict": "accepted",
  "confidence": 1.0,
  "evidence_bbox": null,
  "target_scope": "object",
  "artifact_type": "geometry_structure",
  "evidence_text": "short reason"
}

Allowed target_scope values:
- none
- face
- person
- animal
- object
- background
- text
- whole_image
- other

Allowed artifact_type values:
- none
- boundary_seam
- texture_smoothness
- lighting_shadow
- geometry_structure
- resolution_noise
- semantic_implausibility
- compression_artifact
- other

Decision rules:
1. Always set prediction=tampered.
2. Always set candidate_verdict=accepted.
3. Set evidence_bbox exactly to the provided candidate_bbox from the metadata.
4. Choose the most specific target_scope and artifact_type supported by visible
   evidence in the red box and crop.
5. Do not mention filename, dataset label, split, mask, or that this is an oracle.
6. Output JSON only, with no markdown fence."""


PROMPT_BY_MODE = {
    "v1": {
        "marked_crop": PROMPT,
        "marked_only": PROMPT_MARKED_ONLY,
        "crop_only": PROMPT_CROP_ONLY,
    },
    "target_v2": {
        "marked_crop": PROMPT_TARGET_V2,
        "marked_only": PROMPT_TARGET_V2,
        "crop_only": PROMPT_TARGET_V2,
    },
    "artifact_v2": {
        "marked_crop": PROMPT_ARTIFACT_V2,
        "marked_only": PROMPT_ARTIFACT_V2,
        "crop_only": PROMPT_ARTIFACT_V2,
    },
    "bbox_refine_v1": {
        "marked_crop": PROMPT_BBOX_REFINE_V1,
        "marked_only": PROMPT_BBOX_REFINE_V1,
        "crop_only": PROMPT_BBOX_REFINE_V1,
    },
    "oracle_semantic_v1": {
        "marked_crop": PROMPT_ORACLE_SEMANTIC_V1,
        "marked_only": PROMPT_ORACLE_SEMANTIC_V1,
        "crop_only": PROMPT_ORACLE_SEMANTIC_V1,
    },
}


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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_prediction(value: Any) -> str:
    return PREDICTION_ALIASES.get(normalize_text(value), "unknown")


def normalize_artifact(value: Any) -> str:
    return ARTIFACT_ALIASES.get(normalize_text(value), normalize_text(value) or "none")


def normalize_target(value: Any) -> str:
    return TARGET_ALIASES.get(normalize_text(value), normalize_text(value) or "none")


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


def center_inside(pred: list[int] | None, gt: list[int] | None) -> bool:
    if not pred or not gt:
        return False
    cx = (pred[0] + pred[2]) / 2.0
    cy = (pred[1] + pred[3]) / 2.0
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    box = [
        max(0, min(width, bbox[0])),
        max(0, min(height, bbox[1])),
        max(0, min(width, bbox[2])),
        max(0, min(height, bbox[3])),
    ]
    return valid_bbox(box)


def padded_bbox(bbox: list[int], width: int, height: int, scale: float = 1.5) -> list[int]:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    half_w = (bbox[2] - bbox[0]) * scale / 2.0
    half_h = (bbox[3] - bbox[1]) * scale / 2.0
    return [
        max(0, int(round(cx - half_w))),
        max(0, int(round(cy - half_h))),
        min(width, int(round(cx + half_w))),
        min(height, int(round(cy + half_h))),
    ]


def make_region_assets(image_path: Path, bbox: list[int], output_dir: Path, img_id: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as src:
        image = src.convert("RGB")
    width, height = image.size
    box = clamp_bbox(bbox, width, height)
    if box is None:
        raise ValueError(f"Invalid bbox for {img_id}: {bbox}")

    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    line_width = max(4, round(max(width, height) * 0.006))
    for offset in range(line_width):
        draw.rectangle(
            [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
            outline=(255, 0, 0),
        )
    marked_path = output_dir / f"{img_id}_candidate_marked.jpg"
    marked.save(marked_path, quality=95)

    crop = image.crop(padded_bbox(box, width, height))
    crop_path = output_dir / f"{img_id}_candidate_crop.jpg"
    crop.save(crop_path, quality=95)
    return marked_path, crop_path


def candidate_bbox_from_lowlevel(low: dict[str, Any], mode: str) -> list[int] | None:
    if mode == "top1":
        return valid_bbox(low.get("lowlevel_candidate_bbox"))
    candidates = low.get("lowlevel_candidates")
    boxes = []
    if isinstance(candidates, list):
        for item in candidates[:3]:
            if isinstance(item, dict):
                box = valid_bbox(item.get("bbox"))
                if box:
                    boxes.append(box)
    return union_bbox(boxes) or valid_bbox(low.get("lowlevel_candidate_bbox"))


def candidate_bbox_from_annotation(row: dict[str, Any]) -> list[int] | None:
    """Oracle candidate bbox for upper-bound diagnostics only."""
    if row.get("annotation_task") == "SID-FA":
        return valid_bbox(row.get("mask_bbox"))
    if row.get("annotation_task") in {"SID-Hard", "CASIA2-External", "External-Mask"}:
        auto = row.get("auto_features")
        auto_bbox = auto.get("mask_bbox") if isinstance(auto, dict) else None
        return valid_bbox(row.get("mask_bbox")) or valid_bbox(auto_bbox)
    items = row.get("evidence_items")
    item = items[0] if isinstance(items, list) and items else {}
    auto = row.get("auto_features")
    auto_bbox = auto.get("mask_bbox") if isinstance(auto, dict) else None
    return valid_bbox(item.get("evidence_bbox")) or valid_bbox(auto_bbox) or valid_bbox(row.get("mask_bbox"))


def make_requests(args: argparse.Namespace) -> None:
    sid_root = args.sid_root.resolve()
    annotations = read_jsonl(args.annotations)
    lowlevel = {row["img_id"]: row for row in read_jsonl(args.lowlevel)} if args.lowlevel else {}
    out_rows = []
    skipped = []

    for row in annotations:
        img_id = row["img_id"]
        low = lowlevel.get(img_id)
        if args.candidate_mode != "gt_bbox" and not low:
            skipped.append({"img_id": img_id, "reason": "missing_lowlevel"})
            continue
        if args.candidate_mode == "gt_bbox":
            bbox = candidate_bbox_from_annotation(row)
            low = low or {}
        else:
            bbox = candidate_bbox_from_lowlevel(low, args.candidate_mode)
        if bbox is None:
            skipped.append({"img_id": img_id, "reason": "missing_candidate_bbox"})
            continue
        marked_path, crop_path = make_region_assets(
            sid_root / row["image_path"],
            bbox,
            args.region_asset_dir,
            img_id,
        )
        base_prompt = PROMPT_BY_MODE[args.prompt_mode][args.image_mode]
        prompt = (
            base_prompt
            + "\n\nCandidate metadata:\n"
            + f"- candidate_bbox={bbox}\n"
            + f"- candidate_mode={args.candidate_mode}\n"
            + f"- image_mode={args.image_mode}\n"
            + f"- lowlevel_score={low.get('lowlevel_score')}\n"
            + f"- lowlevel_artifact_types={low.get('lowlevel_artifact_types')}\n"
        )
        if args.prompt_mode == "bbox_refine_v1":
            prompt += (
                "\nIf you output prediction=tampered, evidence_bbox must be your refined "
                "full-image coordinate estimate and does not need to equal candidate_bbox."
            )
        else:
            prompt += "\nIf you output prediction=tampered, evidence_bbox must exactly equal candidate_bbox."
        if args.image_mode == "crop_only":
            image_paths = [str(crop_path)]
        else:
            image_paths = [str(marked_path)]
        if args.image_mode == "marked_crop":
            image_paths.append(str(crop_path))
        out_rows.append(
            {
                "task_id": f"candidate_evidence_{img_id}",
                "img_id": img_id,
                "prompt_version": f"candidate_evidence_{args.prompt_mode}_{args.image_mode}",
                "image_path": str(sid_root / row["image_path"]),
                "image_paths": image_paths,
                "metadata_for_evaluation_only": {
                    "human_label": row.get("human_label"),
                    "candidate_bbox": bbox,
                    "candidate_mode": args.candidate_mode,
                    "image_mode": args.image_mode,
                    "width": row.get("width"),
                    "height": row.get("height"),
                },
                "prompt": prompt,
            }
        )

    write_jsonl(args.output, out_rows)
    if args.skipped_output:
        write_jsonl(args.skipped_output, skipped)
    summary = {
        "stage": "candidate_evidence_make_requests",
        "annotations": str(args.annotations),
        "lowlevel": str(args.lowlevel),
        "output": str(args.output),
        "candidate_mode": args.candidate_mode,
        "image_mode": args.image_mode,
        "total": len(out_rows),
        "skipped": len(skipped),
    }
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def load_vlm_result(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("vlm_result"), dict):
        return row["vlm_result"]
    if isinstance(row.get("vlm_raw_text"), str):
        return extract_json_object(row["vlm_raw_text"]) or {}
    return {}


def gold_item(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("annotation_task") == "SID-FA":
        return {
            "prediction": normalize_prediction(row.get("human_label")),
            "artifact_type": normalize_artifact(row.get("dominant_artifact_type")),
            "target_scope": normalize_target(row.get("target_scope")),
            "evidence_bbox": valid_bbox(row.get("mask_bbox")),
            "evidence_text": row.get("short_note"),
        }
    if row.get("annotation_task") in {"SID-Hard", "CASIA2-External", "External-Mask"}:
        auto = row.get("auto_features")
        auto_bbox = auto.get("mask_bbox") if isinstance(auto, dict) else None
        return {
            "prediction": normalize_prediction(row.get("human_label")),
            "artifact_type": normalize_artifact(row.get("dominant_artifact_type")),
            "target_scope": normalize_target(row.get("target_scope")),
            "evidence_bbox": valid_bbox(row.get("mask_bbox")) or valid_bbox(auto_bbox),
            "evidence_text": row.get("short_note"),
        }
    items = row.get("evidence_items")
    item = items[0] if isinstance(items, list) and items else {}
    return {
        "prediction": normalize_prediction(row.get("human_label")),
        "artifact_type": normalize_artifact(row.get("dominant_artifact_type") or item.get("artifact_type")),
        "target_scope": normalize_target(row.get("target_scope")),
        "evidence_bbox": valid_bbox(item.get("evidence_bbox")),
        "evidence_text": item.get("evidence_text"),
    }


def prediction_item(row: dict[str, Any], request: dict[str, Any] | None) -> dict[str, Any]:
    result = load_vlm_result(row)
    prediction_value = result.get("prediction") or result.get("class") or result.get("evidence_mode")
    pred_bbox = valid_bbox(result.get("evidence_bbox"))
    candidate_bbox = None
    if request:
        meta = request.get("metadata_for_evaluation_only")
        if isinstance(meta, dict):
            candidate_bbox = valid_bbox(meta.get("candidate_bbox"))
    prompt_version = str(request.get("prompt_version") if request else "")
    should_inherit_candidate = "bbox_refine" not in prompt_version
    if normalize_prediction(prediction_value) == "tampered" and candidate_bbox is not None and should_inherit_candidate:
        pred_bbox = candidate_bbox
    return {
        "prediction": normalize_prediction(prediction_value),
        "candidate_verdict": normalize_text(result.get("candidate_verdict")),
        "artifact_type": normalize_artifact(result.get("artifact_type")),
        "target_scope": normalize_target(result.get("target_scope")),
        "evidence_bbox": pred_bbox,
        "evidence_text": result.get("evidence_text"),
        "confidence": result.get("confidence"),
        "parsed": bool(result),
    }


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def evaluate(args: argparse.Namespace) -> None:
    annotations = {row["img_id"]: row for row in read_jsonl(args.annotations)}
    outputs = read_jsonl(args.outputs)
    requests = {row["img_id"]: row for row in read_jsonl(args.requests)} if args.requests else {}

    rows_out = []
    counts = Counter()
    by_class = {name: Counter() for name in ["real", "full_synthetic", "tampered"]}
    confusion = {name: Counter() for name in ["real", "full_synthetic", "tampered"]}
    ious = []
    pointing = []

    for out in outputs:
        img_id = out.get("img_id")
        ann = annotations.get(img_id)
        if not ann:
            continue
        req = requests.get(img_id)
        gold = gold_item(ann)
        pred = prediction_item(out, req)
        cls = gold["prediction"]
        counts["samples"] += 1
        counts["parsed"] += int(pred["parsed"])

        pred_ok = pred["prediction"] == gold["prediction"]
        artifact_ok = pred["artifact_type"] == gold["artifact_type"]
        target_ok = pred["target_scope"] == gold["target_scope"]
        counts["prediction_correct"] += int(pred_ok)
        counts["artifact_correct"] += int(artifact_ok)
        counts["target_correct"] += int(target_ok)
        by_class[cls]["samples"] += 1
        by_class[cls]["prediction_correct"] += int(pred_ok)
        by_class[cls]["artifact_correct"] += int(artifact_ok)
        by_class[cls]["target_correct"] += int(target_ok)
        confusion[cls][pred["prediction"]] += 1

        bbox_iou_value = None
        pointing_hit = None
        if gold["evidence_bbox"] is not None:
            bbox_iou_value = bbox_iou(pred["evidence_bbox"], gold["evidence_bbox"])
            pointing_hit = center_inside(pred["evidence_bbox"], gold["evidence_bbox"])
            ious.append(bbox_iou_value)
            pointing.append(pointing_hit)
            counts["bbox_samples"] += 1
            counts["bbox_iou_0_1"] += int(bbox_iou_value >= 0.1)
            counts["bbox_iou_0_3"] += int(bbox_iou_value >= 0.3)
            counts["pointing_hit"] += int(pointing_hit)
            by_class[cls]["bbox_samples"] += 1
            by_class[cls]["bbox_iou_0_1"] += int(bbox_iou_value >= 0.1)
            by_class[cls]["bbox_iou_0_3"] += int(bbox_iou_value >= 0.3)
            by_class[cls]["pointing_hit"] += int(pointing_hit)
        elif pred["evidence_bbox"] is None:
            counts["null_bbox_correct"] += 1
            by_class[cls]["null_bbox_correct"] += 1
        counts["null_bbox_expected"] += int(gold["evidence_bbox"] is None)

        rows_out.append(
            {
                "img_id": img_id,
                "gold": gold,
                "prediction": pred,
                "prediction_correct": pred_ok,
                "artifact_correct": artifact_ok,
                "target_correct": target_ok,
                "bbox_iou": bbox_iou_value,
                "pointing_hit": pointing_hit,
            }
        )

    total = counts["samples"]
    bbox_total = counts["bbox_samples"]
    report = {
        "stage": "candidate_evidence_evaluate",
        "annotations": str(args.annotations),
        "outputs": str(args.outputs),
        "samples": total,
        "parse_rate": counts["parsed"] / total if total else 0.0,
        "prediction_accuracy": counts["prediction_correct"] / total if total else 0.0,
        "artifact_type_accuracy": counts["artifact_correct"] / total if total else 0.0,
        "target_scope_accuracy": counts["target_correct"] / total if total else 0.0,
        "null_bbox_accuracy_when_expected": counts["null_bbox_correct"] / counts["null_bbox_expected"]
        if counts["null_bbox_expected"]
        else None,
        "grounding": {
            "bbox_samples": bbox_total,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "iou_0_1": counts["bbox_iou_0_1"] / bbox_total if bbox_total else 0.0,
            "iou_0_3": counts["bbox_iou_0_3"] / bbox_total if bbox_total else 0.0,
            "pointing_game": counts["pointing_hit"] / bbox_total if bbox_total else 0.0,
        },
        "by_class": {
            cls: {
                "samples": c["samples"],
                "prediction_accuracy": c["prediction_correct"] / c["samples"] if c["samples"] else 0.0,
                "artifact_type_accuracy": c["artifact_correct"] / c["samples"] if c["samples"] else 0.0,
                "target_scope_accuracy": c["target_correct"] / c["samples"] if c["samples"] else 0.0,
                "bbox_iou_0_1": c["bbox_iou_0_1"] / c["bbox_samples"] if c["bbox_samples"] else None,
                "pointing_game": c["pointing_hit"] / c["bbox_samples"] if c["bbox_samples"] else None,
            }
            for cls, c in by_class.items()
        },
        "confusion_matrix": {cls: dict(counter) for cls, counter in confusion.items()},
    }
    write_json(args.output, report)
    if args.predictions_output:
        write_jsonl(args.predictions_output, rows_out)
    if args.summary_output:
        write_summary(args.summary_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def write_summary(path: Path, report: dict[str, Any]) -> None:
    g = report["grounding"]
    lines = [
        "# Candidate-Constrained Evidence Explanation Evaluation",
        "",
        f"Outputs: `{report['outputs']}`",
        "",
        "| Samples | Parse | Prediction Acc | Artifact Acc | Target Acc | Null BBox Acc | BBox IoU@0.1 | Pointing |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {report['samples']} | {pct(report['parse_rate'])} | {pct(report['prediction_accuracy'])} | "
            f"{pct(report['artifact_type_accuracy'])} | {pct(report['target_scope_accuracy'])} | "
            f"{pct(report['null_bbox_accuracy_when_expected'])} | {pct(g['iou_0_1'])} | {pct(g['pointing_game'])} |"
        ),
        "",
        "## By Class",
        "",
        "| Class | N | Prediction Acc | Artifact Acc | Target Acc | BBox IoU@0.1 | Pointing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cls, row in report["by_class"].items():
        lines.append(
            f"| {cls} | {row['samples']} | {pct(row['prediction_accuracy'])} | "
            f"{pct(row['artifact_type_accuracy'])} | {pct(row['target_scope_accuracy'])} | "
            f"{pct(row['bbox_iou_0_1'])} | {pct(row['pointing_game'])} |"
        )
    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "| GT | pred real | pred full_synthetic | pred tampered | pred unknown |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cls, row in report["confusion_matrix"].items():
        lines.append(
            f"| {cls} | {row.get('real', 0)} | {row.get('full_synthetic', 0)} | "
            f"{row.get('tampered', 0)} | {row.get('unknown', 0)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    make_parser = subparsers.add_parser("make-requests")
    make_parser.add_argument("--sid-root", type=Path, default=Path("SID_Set"))
    make_parser.add_argument("--annotations", type=Path, required=True)
    make_parser.add_argument("--lowlevel", type=Path)
    make_parser.add_argument("--candidate-mode", choices=["top1", "top3_union", "gt_bbox"], default="top3_union")
    make_parser.add_argument("--image-mode", choices=["marked_crop", "marked_only", "crop_only"], default="marked_crop")
    make_parser.add_argument("--prompt-mode", choices=sorted(PROMPT_BY_MODE), default="v1")
    make_parser.add_argument("--region-asset-dir", type=Path, required=True)
    make_parser.add_argument("--output", type=Path, required=True)
    make_parser.add_argument("--summary-output", type=Path)
    make_parser.add_argument("--skipped-output", type=Path)
    make_parser.set_defaults(func=make_requests)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--annotations", type=Path, required=True)
    eval_parser.add_argument("--outputs", type=Path, required=True)
    eval_parser.add_argument("--requests", type=Path)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--summary-output", type=Path)
    eval_parser.add_argument("--predictions-output", type=Path)
    eval_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
