#!/usr/bin/env python3
"""
Training-free SID-Set experiment utilities.

The pipeline separates VLM observation from low-level forensic signals so the
method can run without training SIDA or changing the original dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


CLASS_NAMES = ["real", "full_synthetic", "tampered"]
LOWLEVEL_GRID_SIZE = 16
LOWLEVEL_FEATURE_VERSION = "pixel5d-v2"
LOWLEVEL_FEATURE_NAMES = ["rgb_std", "gray_mean", "gray_std", "edge_mean", "lap_var"]
ARTIFACT_TYPES = [
    "boundary_seam",
    "texture_smoothness",
    "lighting_shadow",
    "geometry_structure",
    "resolution_noise",
    "semantic_implausibility",
    "compression_artifact",
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


def select_rows(
    rows: list[dict[str, Any]],
    limit: int | None,
    per_class: int | None,
) -> list[dict[str, Any]]:
    if per_class is None:
        return rows[:limit] if limit else rows

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["class_dir"]].append(row)

    selected: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        selected.extend(by_class[class_name][:per_class])
    return selected[:limit] if limit else selected


def vlm_prompt() -> str:
    return """
You are a forensic vision-language analyst. Analyze the image without using any
ground-truth mask or dataset label.

Return valid JSON only. Use this schema:
{
  "structured_visual_observation": {
    "objects": ["short object names"],
    "attributes": ["visual attributes"],
    "relationships": ["object-scene relationships"],
    "suspicious_regions": [
      {"bbox": [x1, y1, x2, y2], "description": "short region description"}
    ]
  },
  "artifact_evidence": [
    {
      "bbox": [x1, y1, x2, y2],
      "artifact_type": "boundary_seam | texture_smoothness | lighting_shadow | geometry_structure | resolution_noise | semantic_implausibility | compression_artifact",
      "evidence_text": "specific visual evidence",
      "confidence": 1
    }
  ],
  "final_decision": {
    "class": "real | full_synthetic | tampered",
    "tampered_bbox": [x1, y1, x2, y2],
    "confidence": 0.0,
    "short_reason": "one sentence"
  }
}

Rules:
- If no suspicious local region exists, use an empty suspicious_regions list.
- For real or full_synthetic images, tampered_bbox must be null.
- For tampered images, tampered_bbox should cover the manipulated region.
- Do not mention that a mask or label was available.
""".strip()


def local_first_compact_prompt() -> str:
    return """
You are a forensic vision-language analyst. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

Your first priority is to detect LOCAL manipulation. Inspect whether any region
is visually inconsistent with its surrounding context.

Check these artifact types:
- boundary_seam: unnatural boundary, pasted edge, halo, cutout seam
- texture_smoothness: local texture differs from nearby texture
- lighting_shadow: local lighting, shadow, reflection, or color temperature mismatch
- geometry_structure: distorted object shape, face/body geometry, impossible structure
- resolution_noise: local blur, noise, sharpness, or resolution mismatch
- semantic_implausibility: locally implausible object, identity, interaction, or context
- compression_artifact: local compression or block artifact mismatch

Decision rules:
- real: no clear global or local artifact.
- full_synthetic: the whole image looks AI-generated or globally unnatural.
- tampered: one or several local regions are inconsistent with the surrounding image.
- If there is any credible localized inconsistency, prefer "tampered" over "full_synthetic".

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "local_inconsistency": true,
  "suspicious_bbox": [x1, y1, x2, y2],
  "artifact_type": "boundary_seam",
  "class": "tampered",
  "confidence": 0.75,
  "reason": "short evidence, max 20 words"
}

Rules:
- If there is no local suspicious region, set local_inconsistency to false and suspicious_bbox to null.
- suspicious_bbox must be null unless a concrete local region is visible.
- artifact_type must be one of: boundary_seam, texture_smoothness, lighting_shadow, geometry_structure, resolution_noise, semantic_implausibility, compression_artifact, none.
- class must be one of: real, full_synthetic, tampered.
- Keep reason short.
""".strip()


def balanced_compact_prompt() -> str:
    return """
You are a forensic vision-language analyst. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

Analyze in this order:
1. Global AI-generation check: Does the whole image show globally synthetic
   artifacts, unnatural rendering, impossible object details, repeated textures,
   over-smoothed surfaces, distorted hands/faces/text, or globally inconsistent
   lighting? If yes, mark global_ai_artifact as true.
2. Local manipulation check: Does one concrete region look inconsistent with
   nearby context? Check boundary seams, texture mismatch, lighting/shadow
   mismatch, geometry distortion, local blur/noise/sharpness mismatch,
   semantic implausibility, and local compression mismatch.
3. Final class:
   - real: no clear global AI artifact and no clear local inconsistency.
   - full_synthetic: global AI artifact is visible across the image, and no
     single pasted/manipulated local region stands out.
   - tampered: a concrete local region is inconsistent with surrounding context.

Important:
- Do not call an image real if it has clear global AI-generation artifacts.
- Do not call an image tampered unless you can name one concrete local region.
- If both global and local artifacts exist, choose tampered only when the local
  region stands out from the rest of the image.

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "global_ai_artifact": false,
  "local_inconsistency": true,
  "suspicious_bbox": [x1, y1, x2, y2],
  "artifact_type": "boundary_seam",
  "class": "tampered",
  "confidence": 0.75,
  "reason": "short evidence, max 20 words"
}

Rules:
- suspicious_bbox must be null unless local_inconsistency is true.
- artifact_type must be one of: boundary_seam, texture_smoothness, lighting_shadow, geometry_structure, resolution_noise, semantic_implausibility, compression_artifact, none.
- class must be one of: real, full_synthetic, tampered.
- Keep reason short.
""".strip()


def lowlevel_assisted_compact_prompt() -> str:
    return """
You are a forensic vision-language analyst. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

A pixel-only forensic module provides a candidate suspicious region below. It
was computed from image statistics only and may be wrong. Inspect both the full
image and the candidate region.

Task:
1. Decide whether at least one candidate region is truly inconsistent with its
   nearby context.
2. Check whether the whole image instead looks globally AI-generated.
3. Decide the final class.

Class rules:
- real: no clear global AI artifact and candidate region is not truly suspicious.
- full_synthetic: the whole image has global AI-generation artifacts, and the
  candidate does not stand out as a pasted/local manipulation.
- tampered: the candidate or another concrete local region is inconsistent with
  surrounding context.

Artifact types:
boundary_seam, texture_smoothness, lighting_shadow, geometry_structure,
resolution_noise, semantic_implausibility, compression_artifact, none.

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "candidate_supported": true,
  "selected_candidate_index": 1,
  "global_ai_artifact": false,
  "local_inconsistency": true,
  "suspicious_bbox": [x1, y1, x2, y2],
  "artifact_type": "boundary_seam",
  "class": "tampered",
  "confidence": 0.75,
  "reason": "short evidence, max 20 words"
}

Rules:
- If one candidate is supported, use its bbox as suspicious_bbox and report its index.
- If none of the candidates are supported, set candidate_supported to false.
- If no local suspicious region is visible, suspicious_bbox must be null.
- class must be one of: real, full_synthetic, tampered.
""".strip()


def lowlevel_assisted_compact_v1_prompt() -> str:
    return """
You are a forensic vision-language analyst. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

A pixel-only forensic module provides a candidate suspicious region below. It
was computed from image statistics only and may be wrong. Inspect both the full
image and the candidate region.

Task:
1. Decide whether the candidate region is truly inconsistent with its nearby
   context.
2. Check whether the whole image instead looks globally AI-generated.
3. Decide the final class.

Class rules:
- real: no clear global AI artifact and candidate region is not truly suspicious.
- full_synthetic: the whole image has global AI-generation artifacts, and the
  candidate does not stand out as a pasted/local manipulation.
- tampered: the candidate or another concrete local region is inconsistent with
  surrounding context.

Artifact types:
boundary_seam, texture_smoothness, lighting_shadow, geometry_structure,
resolution_noise, semantic_implausibility, compression_artifact, none.

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "candidate_supported": true,
  "global_ai_artifact": false,
  "local_inconsistency": true,
  "suspicious_bbox": [x1, y1, x2, y2],
  "artifact_type": "boundary_seam",
  "class": "tampered",
  "confidence": 0.75,
  "reason": "short evidence, max 20 words"
}

Rules:
- If the candidate is supported, use the candidate bbox as suspicious_bbox.
- If a different local region is more suspicious, use that bbox.
- If no local suspicious region is visible, suspicious_bbox must be null.
- class must be one of: real, full_synthetic, tampered.
""".strip()


def global_synthesis_compact_prompt() -> str:
    return """
You are an AI-generated image detector. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

Only answer this question: does the whole image look globally AI-generated?
Be conservative. A normal real photo with blur, compression, unusual pose,
or a single local artifact is not enough.

Inspect the whole image for:
- distorted hands, fingers, faces, eyes, limbs, teeth, or ears
- impossible object structure or malformed small objects
- unreadable or corrupted text/logos/signs
- repeated, melted, over-smoothed, or plastic-like textures
- inconsistent perspective, reflections, shadows, or lighting across the scene
- globally unnatural background details

Do not focus on one isolated pasted region. This prompt is for global synthesis.
Mark global_ai_artifact as true only if at least two independent global cues are
visible across the image.

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "global_ai_artifact": true,
  "global_artifact_type": "distorted_anatomy",
  "global_cue_count": 2,
  "class": "full_synthetic",
  "confidence": 0.75,
  "reason": "short evidence, max 20 words"
}

Rules:
- class must be full_synthetic if global_ai_artifact is true.
- class must be real if global_ai_artifact is false.
- global_ai_artifact must be false if there is only one weak cue.
- global_cue_count must be 0, 1, 2, or 3.
- global_artifact_type must be one of: distorted_anatomy, malformed_objects, corrupted_text, repeated_texture, oversmoothing, lighting_perspective, unnatural_background, none.
""".strip()


def global_synthesis_compact_v1_prompt() -> str:
    return """
You are an AI-generated image detector. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

Only answer this question: does the whole image look globally AI-generated?

Inspect the whole image for:
- distorted hands, fingers, faces, eyes, limbs, teeth, or ears
- impossible object structure or malformed small objects
- unreadable or corrupted text/logos/signs
- repeated, melted, over-smoothed, or plastic-like textures
- inconsistent perspective, reflections, shadows, or lighting across the scene
- globally unnatural background details

Do not focus on one isolated pasted region. This prompt is for global synthesis.

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "global_ai_artifact": true,
  "global_artifact_type": "distorted_anatomy",
  "class": "full_synthetic",
  "confidence": 0.75,
  "reason": "short evidence, max 20 words"
}

Rules:
- class must be full_synthetic if global_ai_artifact is true.
- class must be real if global_ai_artifact is false.
- global_artifact_type must be one of: distorted_anatomy, malformed_objects, corrupted_text, repeated_texture, oversmoothing, lighting_perspective, unnatural_background, none.
""".strip()


def real_guard_compact_v1_prompt() -> str:
    return """
You are a forensic false-positive reviewer. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

A pixel-only forensic module provides a candidate suspicious region below. Other
branches may over-call tampering when they see normal texture, compression,
lighting, reflections, blur, or clutter. Your task is to decide whether the
candidate region is truly tamper-specific or can be explained naturally.

Inspect the candidate region and its nearby context. Be conservative:
- Natural explanations include normal object boundary, depth-of-field blur,
  motion blur, JPEG compression, sensor noise, shadows, reflection, highlights,
  background clutter, low resolution, or normal material texture.
- Tamper-specific evidence requires a localized pasted edge, inconsistent
  object geometry, incompatible lighting/shadow, local resolution/noise mismatch
  that is not shared by nearby regions, or a semantically impossible insertion.

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "candidate_is_realistic": true,
  "natural_explanation": "compression_artifact",
  "tamper_specific_evidence": false,
  "local_manipulation_likelihood": 0.15,
  "should_block_tampered": true,
  "class_hint": "real",
  "reason": "short evidence, max 20 words"
}

Rules:
- should_block_tampered should be true when the candidate is better explained by
  natural image formation or common image degradation.
- should_block_tampered should be false only when there is concrete
  tamper-specific evidence.
- local_manipulation_likelihood must be between 0 and 1.
- natural_explanation must be one of: normal_boundary, normal_texture,
  compression_artifact, lighting_shadow, reflection_highlight, motion_blur,
  depth_of_field, low_resolution, background_clutter, global_ai_artifact, none.
- class_hint must be one of: real, full_synthetic, tampered.
""".strip()


def real_guard_compact_v2_prompt() -> str:
    return """
You are a forensic candidate verifier. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

A pixel-only module provides one candidate region. The candidate may be a true
local manipulation, a normal real-image artifact, or part of a globally
AI-generated image. Verify the candidate instead of assuming it is suspicious.

Decide one verdict:
- natural_false_positive: the candidate is better explained by normal texture,
  object boundary, blur, compression, reflection, shadow, low resolution, or
  background clutter.
- tamper_specific: the candidate shows concrete local manipulation evidence,
  such as pasted seam, local lighting mismatch, local resolution/noise mismatch,
  impossible inserted object, or distorted local geometry.
- global_synthetic: the image mainly looks globally AI-generated, not locally
  pasted or edited.
- uncertain: evidence is weak or ambiguous.

Use these likelihood ranges:
- natural_false_positive: 0.00 to 0.30
- uncertain: 0.31 to 0.60
- tamper_specific: 0.61 to 1.00
- global_synthetic: 0.20 to 0.60 unless a local manipulation is also visible

Return compact JSON only. No markdown. No extra text.
Keys and allowed values:
- verdict: natural_false_positive | tamper_specific | global_synthetic | uncertain
- natural_explanation: normal_boundary | normal_texture | compression_artifact | lighting_shadow | reflection_highlight | motion_blur | depth_of_field | low_resolution | background_clutter | global_ai_artifact | none
- tamper_cue: boundary_seam | texture_smoothness | lighting_shadow | geometry_structure | resolution_noise | semantic_implausibility | compression_artifact | none
- local_manipulation_likelihood: number from 0 to 1
- class_hint: real | full_synthetic | tampered
- reason: short evidence, max 20 words

Important:
- Do not output the same likelihood for every image.
- If the candidate overlaps a face, object, text, or inserted region with a
  clear seam or mismatch, prefer tamper_specific.
- If the whole image has synthetic artifacts but no isolated local mismatch,
  prefer global_synthetic.
""".strip()


def region_verifier_compact_v1_prompt() -> str:
    return """
You are a local manipulation region verifier. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

You are given two images:
1. The full image with a red rectangle marking a candidate region.
2. A crop of the candidate region.

Verify only the marked candidate. Decide whether the candidate contains local
manipulation evidence that cannot be explained by normal photography, texture,
compression, lighting, reflection, blur, or background clutter.

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "region_verdict": "supported",
  "artifact_type": "boundary_seam",
  "natural_explanation": "none",
  "support_level": 0.75,
  "reason": "short evidence, max 20 words"
}

Rules:
- region_verdict must be one of: supported, rejected, uncertain.
- Use supported only for concrete local manipulation evidence in the red-boxed
  region or crop.
- Use rejected when the region is better explained by natural texture,
  compression, lighting, reflection, blur, low resolution, or clutter.
- Use uncertain when evidence is weak or mixed.
- artifact_type must be one of: boundary_seam, texture_smoothness,
  lighting_shadow, geometry_structure, resolution_noise,
  semantic_implausibility, compression_artifact, none.
- natural_explanation must be one of: normal_boundary, normal_texture,
  compression_artifact, lighting_shadow, reflection_highlight, motion_blur,
  depth_of_field, low_resolution, background_clutter, global_ai_artifact, none.
- support_level must be between 0 and 1.
""".strip()


def region_verifier_compact_v2_prompt() -> str:
    return """
You are a local manipulation region verifier. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

You are given two images:
1. The full image with a red rectangle marking a candidate region.
2. A crop of the candidate region.

Verify only the marked candidate. Compare the red-boxed region with nearby
surrounding context in the full image, then inspect the crop for local details.

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "region_verdict": "uncertain",
  "artifact_type": "none",
  "natural_explanation": "none",
  "support_level": 0.50,
  "reason": "short evidence, max 20 words"
}

Decision rules:
- supported: use when the marked region has localized evidence stronger than
  surrounding regions, such as a pasted edge, local lighting mismatch, local
  resolution/noise mismatch, distorted local geometry, or impossible inserted
  content.
- rejected: use only when the marked region is clearly normal and the same
  texture, compression, lighting, blur, reflection, or clutter appears outside
  the red box too.
- uncertain: use when the evidence is weak, ambiguous, low-resolution, or could
  plausibly be either natural degradation or manipulation.

Important:
- Do not reject a candidate only because compression, blur, or texture is
  possible. Reject only when the natural explanation is clearly supported by
  surrounding context.
- If there is any localized mismatch stronger than surrounding regions, prefer
  supported or uncertain over rejected.
- If the crop lacks enough context, use uncertain.

Allowed values:
- region_verdict: supported, rejected, uncertain
- artifact_type: boundary_seam, texture_smoothness, lighting_shadow,
  geometry_structure, resolution_noise, semantic_implausibility,
  compression_artifact, none
- natural_explanation: normal_boundary, normal_texture, compression_artifact,
  lighting_shadow, reflection_highlight, motion_blur, depth_of_field,
  low_resolution, background_clutter, global_ai_artifact, none
- support_level: number from 0 to 1
""".strip()


def region_verifier_compact_v3_prompt() -> str:
    return """
You are a local artifact comparison verifier. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

You are given two images:
1. The full image with a red rectangle marking a candidate region.
2. A crop of the candidate region.

Your task is not to directly classify the image. Compare the red-boxed region
against nearby unboxed context in the full image.

Answer these questions:
1. Is there a visible artifact inside the red box?
2. Does the same artifact also appear outside the red box at similar strength?
3. Is the inside artifact clearly stronger or more localized than the outside
   context?

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "inside_box_artifact": true,
  "outside_box_same_artifact": false,
  "inside_stronger_than_outside": true,
  "region_verdict": "supported",
  "artifact_type": "boundary_seam",
  "natural_explanation": "none",
  "support_level": 0.75,
  "reason": "short evidence, max 20 words"
}

Decision rules:
- supported: inside_box_artifact is true, and the artifact is clearly stronger,
  sharper, more localized, or more semantically suspicious inside the red box
  than outside.
- rejected: the same artifact appears outside the red box with similar strength,
  or the boxed region looks like normal boundary, texture, compression, blur,
  reflection, shadow, or clutter.
- uncertain: evidence is weak, crop lacks context, or inside/outside comparison
  is ambiguous.

Important:
- Do not use supported if outside_box_same_artifact is true and similar strength.
- Do not use rejected only because a natural explanation is possible. Reject
  only when outside context supports that natural explanation.
- If inside artifact exists but outside comparison is unclear, use uncertain.

Allowed values:
- region_verdict: supported, rejected, uncertain
- artifact_type: boundary_seam, texture_smoothness, lighting_shadow,
  geometry_structure, resolution_noise, semantic_implausibility,
  compression_artifact, none
- natural_explanation: normal_boundary, normal_texture, compression_artifact,
  lighting_shadow, reflection_highlight, motion_blur, depth_of_field,
  low_resolution, background_clutter, global_ai_artifact, none
- support_level: number from 0 to 1
""".strip()


def global_local_router_compact_v1_prompt() -> str:
    return """
You are a global-vs-local forensic router. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

You are given two images:
1. The full image with a red rectangle marking a candidate suspicious region.
2. A crop of the candidate region.

Your task is not to make a generic fake/real decision. Decide whether the
marked candidate is an independent LOCAL manipulation, or only part of a
GLOBAL AI-generated/synthetic appearance shared by the whole image.

Definitions:
- global_synthetic: synthetic artifacts appear broadly across the image; the
  red-boxed candidate does not stand out as an independently pasted/edited
  region.
- local_tampered: the red-boxed candidate is more suspicious than surrounding
  content and shows concrete local manipulation evidence, such as pasted seam,
  local lighting mismatch, local resolution/noise mismatch, impossible inserted
  object, or distorted local geometry.
- real_false_positive: the red-boxed candidate looks like normal photography,
  normal object boundary, texture, shadow, blur, compression, reflection, or
  clutter.
- uncertain: evidence is weak or mixed.

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "global_artifact_strength": 0.75,
  "localized_candidate_is_part_of_global_artifact": true,
  "single_region_stands_out": false,
  "global_vs_local_verdict": "global_synthetic",
  "candidate_tamper_evidence": false,
  "candidate_natural_explanation": false,
  "confidence": 0.70,
  "reason": "short evidence, max 20 words"
}

Decision rules:
- Use local_tampered only when the candidate has localized evidence stronger
  than nearby context.
- Use global_synthetic only when global synthetic artifacts are visible and the
  candidate is not uniquely suspicious.
- Use real_false_positive when the candidate is better explained naturally.
- Do not route to global_synthetic merely because the whole image looks odd if
  the candidate is still a clear independent local manipulation.

Allowed values:
- global_vs_local_verdict: global_synthetic, local_tampered, real_false_positive, uncertain
- global_artifact_strength: number from 0 to 1
- confidence: number from 0 to 1
""".strip()


def region_evidence_consistency_v1_prompt() -> str:
    return """
You are a local evidence consistency verifier. Do not use any ground-truth mask,
dataset label, file name, or external metadata.

You are given two images:
1. The full image with a red rectangle marking a candidate region.
2. A crop of the candidate region.

Your task is not to directly classify the whole image. Verify whether the
red-boxed candidate has LOCAL evidence that is inconsistent with its nearby
context, or whether the apparent artifact is better explained by global style,
normal photography, compression, blur, lighting, texture, reflection, or clutter.

Inspect the red box and compare it with nearby unboxed context. Answer each
evidence type separately.

Return compact JSON only. No markdown. No extra text.
Use exactly this schema:
{
  "boundary_seam_evidence": false,
  "texture_noise_mismatch": false,
  "lighting_shadow_conflict": false,
  "geometry_semantic_conflict": false,
  "global_style_consistency": true,
  "natural_explanation_supported": true,
  "localized_evidence_strength": 0.25,
  "context_consistency_strength": 0.75,
  "evidence_verdict": "global_or_natural",
  "dominant_artifact_type": "none",
  "confidence": 0.70,
  "reason": "short evidence, max 20 words"
}

Definitions:
- boundary_seam_evidence: pasted edge, halo, cut line, abrupt boundary only at the candidate.
- texture_noise_mismatch: local texture, noise, sharpness, blur, or resolution differs from nearby context.
- lighting_shadow_conflict: local lighting, shadow, reflection, color temperature, or illumination is inconsistent.
- geometry_semantic_conflict: distorted local shape, impossible object relation, face/body/object inconsistency.
- global_style_consistency: similar artifact or synthetic style appears broadly outside the red box.
- natural_explanation_supported: nearby context supports a normal explanation such as boundary, blur, lighting, compression, reflection, or clutter.

Decision rules:
- strong_local_evidence: at least one local evidence type is clear, localized, and stronger inside the red box than outside.
- weak_local_evidence: local evidence may exist but is weak, ambiguous, or not clearly stronger than context.
- global_or_natural: the red-boxed artifact is consistent with global style or has a supported natural explanation.
- uncertain: evidence is mixed or image quality/context is insufficient.

Important:
- Do not use global_or_natural merely because the whole image looks synthetic if the red box still has clear independent local evidence.
- Do not use strong_local_evidence unless the evidence is localized and stronger inside the red box than nearby context.
- Scores must be calibrated: use 0.85 only for very clear evidence, 0.50 for ambiguous evidence, and 0.20 for weak evidence.

Allowed values:
- evidence_verdict: strong_local_evidence, weak_local_evidence, global_or_natural, uncertain
- dominant_artifact_type: boundary_seam, texture_smoothness, lighting_shadow,
  geometry_structure, resolution_noise, semantic_implausibility,
  compression_artifact, none
- localized_evidence_strength: number from 0 to 1
- context_consistency_strength: number from 0 to 1
- confidence: number from 0 to 1
""".strip()


def get_prompt(prompt_version: str) -> str:
    if prompt_version == "original":
        return vlm_prompt()
    if prompt_version == "local_first_compact":
        return local_first_compact_prompt()
    if prompt_version == "balanced_compact":
        return balanced_compact_prompt()
    if prompt_version == "lowlevel_assisted_compact":
        return lowlevel_assisted_compact_prompt()
    if prompt_version == "lowlevel_assisted_compact_v1":
        return lowlevel_assisted_compact_v1_prompt()
    if prompt_version == "global_synthesis_compact":
        return global_synthesis_compact_prompt()
    if prompt_version == "global_synthesis_compact_v1":
        return global_synthesis_compact_v1_prompt()
    if prompt_version == "real_guard_compact_v1":
        return real_guard_compact_v1_prompt()
    if prompt_version == "real_guard_compact_v2":
        return real_guard_compact_v2_prompt()
    if prompt_version == "region_verifier_compact_v1":
        return region_verifier_compact_v1_prompt()
    if prompt_version == "region_verifier_compact_v2":
        return region_verifier_compact_v2_prompt()
    if prompt_version == "region_verifier_compact_v3":
        return region_verifier_compact_v3_prompt()
    if prompt_version == "global_local_router_compact_v1":
        return global_local_router_compact_v1_prompt()
    if prompt_version == "region_evidence_consistency_v1":
        return region_evidence_consistency_v1_prompt()
    raise ValueError(f"Unsupported prompt version: {prompt_version}")


def clamp_bbox(bbox: list[int] | None, width: int, height: int) -> list[int] | None:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return [x1, y1, x2, y2]


def padded_bbox(bbox: list[int], width: int, height: int, pad_ratio: float = 0.15) -> list[int]:
    x1, y1, x2, y2 = bbox
    pad = int(round(max(x2 - x1, y2 - y1) * pad_ratio))
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    ]


def make_region_assets(
    image_path: Path,
    bbox: list[int],
    output_dir: Path,
    img_id: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as src:
        image = src.convert("RGB")
    width, height = image.size
    bbox = clamp_bbox(bbox, width, height)
    if bbox is None:
        raise ValueError(f"Invalid bbox for {img_id}")

    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    line_width = max(4, round(max(width, height) * 0.006))
    for offset in range(line_width):
        draw.rectangle(
            [bbox[0] - offset, bbox[1] - offset, bbox[2] + offset, bbox[3] + offset],
            outline=(255, 0, 0),
        )
    marked_path = output_dir / f"{img_id}_marked.jpg"
    marked.save(marked_path, quality=95)

    crop_box = padded_bbox(bbox, width, height)
    crop = image.crop(crop_box)
    crop_path = output_dir / f"{img_id}_crop.jpg"
    crop.save(crop_path, quality=95)
    return marked_path, crop_path


def make_requests(args: argparse.Namespace) -> None:
    sid_root = args.sid_root.resolve()
    rows = select_rows(read_jsonl(args.manifest), args.limit, args.per_class)
    base_prompt = get_prompt(args.prompt_version)
    lowlevel_by_id = {row["img_id"]: row for row in read_jsonl(args.lowlevel)} if args.lowlevel else {}
    region_asset_dir = args.region_asset_dir.resolve() if args.region_asset_dir else None
    out_rows = []
    for row in rows:
        prompt = base_prompt
        low = lowlevel_by_id.get(row["img_id"])
        image_paths = None
        if low:
            candidate_list = low.get("lowlevel_candidates")
            if isinstance(candidate_list, list) and candidate_list:
                candidate_text = "\n".join(
                    [
                        f"- candidate_{idx}: bbox={item.get('bbox')}, score={item.get('score')}, artifact_types={item.get('artifact_types')}"
                        for idx, item in enumerate(candidate_list, start=1)
                    ]
                )
            else:
                candidate_text = (
                    f"- candidate_1: bbox={low.get('lowlevel_candidate_bbox')}, "
                    f"score={low.get('lowlevel_score')}, "
                    f"artifact_types={low.get('lowlevel_artifact_types')}"
                )
            prompt += (
                "\n\nPixel-only candidate for this image:\n"
                f"{candidate_text}\n"
                "Remember: this candidate may be a false positive. Verify visually."
            )
            if args.prompt_version in {
                "region_verifier_compact_v1",
                "region_verifier_compact_v2",
                "region_verifier_compact_v3",
                "global_local_router_compact_v1",
                "region_evidence_consistency_v1",
            }:
                bbox = valid_bbox(low.get("lowlevel_candidate_bbox"))
                if bbox is None:
                    continue
                if region_asset_dir is None:
                    raise ValueError("--region-asset-dir is required for region/candidate router prompts")
                marked_path, crop_path = make_region_assets(
                    sid_root / row["image_path"],
                    bbox,
                    region_asset_dir,
                    row["img_id"],
                )
                image_paths = [str(marked_path), str(crop_path)]
                prompt += (
                    "\n\nMarked candidate metadata:\n"
                    f"- candidate_bbox={bbox}\n"
                    f"- lowlevel_score={low.get('lowlevel_score')}\n"
                    f"- lowlevel_artifact_types={low.get('lowlevel_artifact_types')}\n"
                    "Use the first image for full context and the second image for local detail."
                )
        request_row = {
            "task_id": row["img_id"],
            "img_id": row["img_id"],
            "prompt_version": args.prompt_version,
            "image_path": str(sid_root / row["image_path"]),
            "metadata_for_evaluation_only": {
                "class_dir": row["class_dir"],
                "label": row["label"],
                "mask_path": str(sid_root / row["mask_path"]) if row.get("mask_path") else None,
                "width": row.get("width"),
                "height": row.get("height"),
            },
            "prompt": prompt,
        }
        if image_paths is not None:
            request_row["image_paths"] = image_paths
        out_rows.append(request_row)
    write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} VLM request rows to {args.output}")


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


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True) + 1e-6
    return np.abs((values - median) / (1.4826 * mad))


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
    x2 = width if max(xs) == LOWLEVEL_GRID_SIZE - 1 else min((max(xs) + 1) * block_w, width)
    y2 = height if max(ys) == LOWLEVEL_GRID_SIZE - 1 else min((max(ys) + 1) * block_h, height)
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
    for score_idx in order:
        center = indices[int(score_idx)]
        if center in used:
            continue
        cy, cx = center
        local_blocks = [
            (y, x)
            for y in range(max(0, cy - 1), min(LOWLEVEL_GRID_SIZE, cy + 2))
            for x in range(max(0, cx - 1), min(LOWLEVEL_GRID_SIZE, cx + 2))
            if (y, x) in indices
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


def infer_artifact_types(top_feature_names: list[str], row: dict[str, Any]) -> list[str]:
    artifacts: list[str] = []
    feature_set = set(top_feature_names)
    if {"edge_mean", "lap_var"} & feature_set:
        artifacts.append("boundary_seam")
    if {"rgb_std", "lap_var"} & feature_set:
        artifacts.append("texture_smoothness")
    if "gray_mean" in feature_set:
        artifacts.append("lighting_shadow")
    if row["image_path"].lower().endswith((".jpg", ".jpeg")):
        artifacts.append("compression_artifact")
    return artifacts[:3] or ["resolution_noise"]


def lowlevel_block_features(
    patch: np.ndarray,
    patch_gray: np.ndarray,
    patch_grad: np.ndarray,
    patch_lap: np.ndarray,
) -> list[float]:
    """Return the five distinct pixel statistics used by pixel5d-v2."""
    return [
        float(patch.std()),
        float(patch_gray.mean()),
        float(patch_gray.std()),
        float(patch_grad.mean()),
        float(patch_lap.var()),
    ]


def lowlevel_evidence_for_row(row: dict[str, Any], sid_root: Path) -> dict[str, Any]:
    arr, scale_x, scale_y = image_array(sid_root / row["image_path"])
    height, width = arr.shape[:2]
    gray = arr.mean(axis=2)
    grad, lap = gray_edges(gray)

    grid = LOWLEVEL_GRID_SIZE
    block_h = max(height // grid, 1)
    block_w = max(width // grid, 1)
    features: list[list[float]] = []
    indices: list[tuple[int, int]] = []

    for gy in range(grid):
        for gx in range(grid):
            y1 = gy * block_h
            x1 = gx * block_w
            y2 = height if gy == grid - 1 else min((gy + 1) * block_h, height)
            x2 = width if gx == grid - 1 else min((gx + 1) * block_w, width)
            patch = arr[y1:y2, x1:x2]
            patch_gray = gray[y1:y2, x1:x2]
            patch_grad = grad[y1:y2, x1:x2]
            patch_lap = lap[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            features.append(lowlevel_block_features(patch, patch_gray, patch_grad, patch_lap))
            indices.append((gy, gx))

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(LOWLEVEL_FEATURE_NAMES):
        raise ValueError(
            f"{LOWLEVEL_FEATURE_VERSION} feature contract violated: "
            f"matrix shape={matrix.shape}, names={len(LOWLEVEL_FEATURE_NAMES)}"
        )
    z = robust_z(matrix)
    block_scores = z.mean(axis=1)
    score_threshold = max(float(np.percentile(block_scores, 95)), 2.0)
    picked = [indices[i] for i, score in enumerate(block_scores) if score >= score_threshold]

    if len(picked) > 8:
        top_indices = np.argsort(block_scores)[-8:]
        picked = [indices[i] for i in top_indices]

    candidate_bbox = bbox_from_blocks(picked, block_h, block_w, height, width, scale_x, scale_y)
    candidates = candidate_bboxes_from_scores(
        block_scores,
        indices,
        block_h,
        block_w,
        height,
        width,
        scale_x,
        scale_y,
    )
    top_feature_ids = np.argsort(z.mean(axis=0))[-3:]
    top_feature_names = [LOWLEVEL_FEATURE_NAMES[i] for i in reversed(top_feature_ids.tolist())]
    artifacts = infer_artifact_types(top_feature_names, row)
    for candidate in candidates:
        candidate["artifact_types"] = artifacts

    return {
        "img_id": row["img_id"],
        "image_path": row["image_path"],
        "class_dir_for_evaluation_only": row.get("class_dir", row.get("human_label")),
        "lowlevel_candidate_bbox": candidate_bbox,
        "lowlevel_candidates": candidates,
        "lowlevel_artifact_types": artifacts,
        "lowlevel_score": round(float(block_scores.max()), 4) if len(block_scores) else 0.0,
        "top_lowlevel_features": top_feature_names,
        "lowlevel_feature_version": LOWLEVEL_FEATURE_VERSION,
        "method_note": "This uses only image pixels. It does not use SID mask or class label for prediction.",
    }


def run_lowlevel(args: argparse.Namespace) -> None:
    sid_root = args.sid_root.resolve()
    rows = select_rows(read_jsonl(args.manifest), args.limit, args.per_class)
    out_rows = []
    for idx, row in enumerate(rows, start=1):
        out_rows.append(lowlevel_evidence_for_row(row, sid_root))
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(rows)}")
    write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} low-level evidence rows to {args.output}")


def scaled_bbox_to_array(
    bbox: list[int] | None,
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> list[int] | None:
    if bbox is None:
        return None
    scaled = [
        int(round(bbox[0] / scale_x)),
        int(round(bbox[1] / scale_y)),
        int(round(bbox[2] / scale_x)),
        int(round(bbox[3] / scale_y)),
    ]
    return clamp_bbox(scaled, width, height)


def safe_mean(arr: np.ndarray) -> float:
    return float(arr.mean()) if arr.size else 0.0


def safe_std(arr: np.ndarray) -> float:
    return float(arr.std()) if arr.size else 0.0


def log_ratio_abs(a: float, b: float) -> float:
    return abs(math.log((a + 1e-6) / (b + 1e-6)))


def jpeg_blockiness(gray: np.ndarray) -> float:
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return 0.0
    vertical_boundary = np.abs(gray[:, 8::8] - gray[:, 7:-1:8])
    horizontal_boundary = np.abs(gray[8::8, :] - gray[7:-1:8, :])
    vertical_inner = np.abs(gray[:, 4::8] - gray[:, 3:-1:8])
    horizontal_inner = np.abs(gray[4::8, :] - gray[3:-1:8, :])
    boundary = safe_mean(vertical_boundary) + safe_mean(horizontal_boundary)
    inner = safe_mean(vertical_inner) + safe_mean(horizontal_inner)
    return max(0.0, boundary - inner)


def region_stats(arr: np.ndarray, grad: np.ndarray, lap: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        return {
            "rgb_mean": 0.0,
            "rgb_std": 0.0,
            "gray_mean": 0.0,
            "gray_std": 0.0,
            "edge_mean": 0.0,
            "lap_var": 0.0,
            "lap_abs_mean": 0.0,
            "blockiness": 0.0,
        }
    pixels = arr[mask]
    gray = pixels.mean(axis=1)
    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max() + 1)
    x1, x2 = int(xs.min()), int(xs.max() + 1)
    return {
        "rgb_mean": safe_mean(pixels),
        "rgb_std": safe_std(pixels),
        "gray_mean": safe_mean(gray),
        "gray_std": safe_std(gray),
        "edge_mean": safe_mean(grad[mask]),
        "lap_var": float(lap[mask].var()) if mask.any() else 0.0,
        "lap_abs_mean": safe_mean(np.abs(lap[mask])),
        "blockiness": jpeg_blockiness(arr[y1:y2, x1:x2].mean(axis=2)),
    }


def contrast_features_for_row(
    row: dict[str, Any],
    low: dict[str, Any],
    sid_root: Path,
    context_pad_ratio: float,
) -> dict[str, Any]:
    bbox = valid_bbox(low.get("lowlevel_candidate_bbox"))
    if bbox is None:
        return {
            "img_id": row["img_id"],
            "image_path": row["image_path"],
            "class_dir_for_evaluation_only": row["class_dir"],
            "candidate_bbox": None,
            "valid_region_contrast": False,
            "region_contrast_score": 0.0,
            "contrast_verdict": "invalid",
            "method_note": "No valid low-level candidate bbox was available.",
        }

    arr, scale_x, scale_y = image_array(sid_root / row["image_path"], max_side=768)
    height, width = arr.shape[:2]
    scaled_bbox = scaled_bbox_to_array(bbox, scale_x, scale_y, width, height)
    if scaled_bbox is None:
        return {
            "img_id": row["img_id"],
            "image_path": row["image_path"],
            "class_dir_for_evaluation_only": row["class_dir"],
            "candidate_bbox": bbox,
            "valid_region_contrast": False,
            "region_contrast_score": 0.0,
            "contrast_verdict": "invalid",
            "method_note": "Candidate bbox became invalid after image scaling.",
        }

    gray = arr.mean(axis=2)
    grad, lap = gray_edges(gray)
    x1, y1, x2, y2 = scaled_bbox
    inside = np.zeros((height, width), dtype=bool)
    inside[y1:y2, x1:x2] = True

    pad = max(16, int(round(max(x2 - x1, y2 - y1) * context_pad_ratio)))
    ring_x1 = max(0, x1 - pad)
    ring_y1 = max(0, y1 - pad)
    ring_x2 = min(width, x2 + pad)
    ring_y2 = min(height, y2 + pad)
    context = np.zeros((height, width), dtype=bool)
    context[ring_y1:ring_y2, ring_x1:ring_x2] = True
    context &= ~inside
    if int(context.sum()) < max(64, int(inside.sum() * 0.15)):
        context = ~inside

    inside_stats = region_stats(arr, grad, lap, inside)
    context_stats = region_stats(arr, grad, lap, context)
    rgb_delta = abs(inside_stats["rgb_mean"] - context_stats["rgb_mean"]) / 64.0
    gray_delta = abs(inside_stats["gray_mean"] - context_stats["gray_mean"]) / 64.0
    edge_contrast = log_ratio_abs(inside_stats["edge_mean"], context_stats["edge_mean"])
    lap_contrast = log_ratio_abs(inside_stats["lap_var"], context_stats["lap_var"])
    lap_abs_contrast = log_ratio_abs(inside_stats["lap_abs_mean"], context_stats["lap_abs_mean"])
    texture_contrast = log_ratio_abs(inside_stats["gray_std"], context_stats["gray_std"])
    block_contrast = log_ratio_abs(inside_stats["blockiness"], context_stats["blockiness"])
    contrast_score = (
        0.24 * edge_contrast
        + 0.24 * lap_contrast
        + 0.18 * lap_abs_contrast
        + 0.16 * texture_contrast
        + 0.12 * min(rgb_delta, 3.0)
        + 0.06 * block_contrast
    )
    area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / max(
        float(row.get("width", 0) * row.get("height", 0)),
        1.0,
    )
    if contrast_score >= 0.75:
        verdict = "strong"
    elif contrast_score <= 0.35:
        verdict = "weak"
    else:
        verdict = "medium"

    return {
        "img_id": row["img_id"],
        "image_path": row["image_path"],
        "class_dir_for_evaluation_only": row["class_dir"],
        "candidate_bbox": bbox,
        "scaled_candidate_bbox": scaled_bbox,
        "valid_region_contrast": True,
        "area_ratio": round(float(area_ratio), 6),
        "region_contrast_score": round(float(contrast_score), 6),
        "contrast_verdict": verdict,
        "contrast_features": {
            "edge_log_abs_ratio": round(float(edge_contrast), 6),
            "lap_var_log_abs_ratio": round(float(lap_contrast), 6),
            "lap_abs_log_abs_ratio": round(float(lap_abs_contrast), 6),
            "texture_log_abs_ratio": round(float(texture_contrast), 6),
            "rgb_mean_delta_norm": round(float(rgb_delta), 6),
            "gray_mean_delta_norm": round(float(gray_delta), 6),
            "blockiness_log_abs_ratio": round(float(block_contrast), 6),
        },
        "inside_stats": {key: round(float(value), 6) for key, value in inside_stats.items()},
        "context_stats": {key: round(float(value), 6) for key, value in context_stats.items()},
        "method_note": "Inside-vs-context contrast uses only image pixels and the low-level candidate bbox; it does not use mask or class label for prediction.",
    }


def run_region_contrast(args: argparse.Namespace) -> None:
    sid_root = args.sid_root.resolve()
    rows = select_rows(read_jsonl(args.manifest), args.limit, args.per_class)
    lowlevel_by_id = {row["img_id"]: row for row in read_jsonl(args.lowlevel)}
    out_rows = []
    for idx, row in enumerate(rows, start=1):
        low = lowlevel_by_id.get(row["img_id"], {})
        out_rows.append(contrast_features_for_row(row, low, sid_root, args.context_pad_ratio))
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(rows)}")
    write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} region contrast rows to {args.output}")


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


def evaluate_lowlevel(args: argparse.Namespace) -> None:
    sid_root = args.sid_root.resolve()
    manifest_by_id = {row["img_id"]: row for row in read_jsonl(args.manifest)}
    predictions = read_jsonl(args.predictions)
    tampered = []
    hit_01 = 0
    hit_03 = 0
    ious = []

    for pred in predictions:
        row = manifest_by_id.get(pred["img_id"])
        if not row or row["class_dir"] != "tampered" or not row.get("mask_path"):
            continue
        gt_bbox = mask_bbox(sid_root / row["mask_path"])
        iou = bbox_iou(pred.get("lowlevel_candidate_bbox"), gt_bbox)
        ious.append(iou)
        hit_01 += int(iou >= 0.1)
        hit_03 += int(iou >= 0.3)
        tampered.append(pred["img_id"])

    metrics = {
        "prediction_file": str(args.predictions),
        "manifest_file": str(args.manifest),
        "tampered_samples": len(tampered),
        "mean_bbox_iou": round(float(np.mean(ious)), 6) if ious else 0.0,
        "hit_rate_iou_0_1": round(hit_01 / len(ious), 6) if ious else 0.0,
        "hit_rate_iou_0_3": round(hit_03 / len(ious), 6) if ious else 0.0,
        "note": "Low-level evidence is a supporting signal, not the full training-free VLM method.",
    }
    write_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def normalized_class(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "real": "real",
        "authentic": "real",
        "genuine": "real",
        "full_synthetic": "full_synthetic",
        "synthetic": "full_synthetic",
        "ai_generated": "full_synthetic",
        "fully_synthetic": "full_synthetic",
        "tampered": "tampered",
        "part_tampered": "tampered",
        "manipulated": "tampered",
    }
    return aliases.get(value)


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


def artifact_from_item(item: dict[str, Any]) -> str | None:
    artifact = item.get("artifact_type")
    if not isinstance(artifact, str):
        return None
    artifact = artifact.strip()
    return artifact if artifact in ARTIFACT_TYPES else None


def load_vlm_result(row: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(row.get("vlm_result"), dict):
        return row["vlm_result"]
    if isinstance(row.get("response_json"), dict):
        return row["response_json"]
    if isinstance(row.get("result"), dict):
        return row["result"]
    if isinstance(row.get("vlm_raw_text"), str):
        return extract_json_object(row["vlm_raw_text"])
    if isinstance(row.get("response_text"), str):
        return extract_json_object(row["response_text"])
    return None


def aggregate(args: argparse.Namespace) -> None:
    vlm_rows = read_jsonl(args.vlm_output)
    lowlevel_by_id = {row["img_id"]: row for row in read_jsonl(args.lowlevel)}
    out_rows = []

    for row in vlm_rows:
        img_id = row["img_id"]
        vlm = load_vlm_result(row) or {}
        final = vlm.get("final_decision") if isinstance(vlm.get("final_decision"), dict) else {}
        low = lowlevel_by_id.get(img_id, {})

        pred_class = normalized_class(final.get("class")) or normalized_class(vlm.get("class")) or "unknown"
        vlm_bbox = valid_bbox(final.get("tampered_bbox")) or valid_bbox(vlm.get("suspicious_bbox"))
        low_bbox = valid_bbox(low.get("lowlevel_candidate_bbox"))
        pred_bbox = vlm_bbox if pred_class == "tampered" else None
        if pred_class == "tampered" and pred_bbox is None:
            pred_bbox = low_bbox

        vlm_artifacts = []
        for item in vlm.get("artifact_evidence", []) if isinstance(vlm.get("artifact_evidence"), list) else []:
            if isinstance(item, dict):
                artifact = artifact_from_item(item)
                if artifact:
                    vlm_artifacts.append(artifact)
        if isinstance(vlm.get("artifact_type"), str) and vlm["artifact_type"] in ARTIFACT_TYPES:
            vlm_artifacts.append(vlm["artifact_type"])
        low_artifacts = [
            item for item in low.get("lowlevel_artifact_types", []) if item in ARTIFACT_TYPES
        ]
        artifacts = list(dict.fromkeys(vlm_artifacts + low_artifacts))

        evidence_card = {
            "vlm_short_reason": final.get("short_reason") or vlm.get("reason"),
            "local_inconsistency": vlm.get("local_inconsistency"),
            "vlm_artifacts": vlm_artifacts,
            "lowlevel_artifacts": low_artifacts,
            "lowlevel_score": low.get("lowlevel_score"),
            "lowlevel_bbox": low_bbox,
            "aggregation_rule": "Use VLM class as the primary decision; use low-level bbox only when VLM predicts tampered without a valid bbox.",
        }

        out_rows.append(
            {
                "img_id": img_id,
                "image_path": row.get("image_path"),
                "pred_class": pred_class,
                "pred_tampered_bbox": pred_bbox,
                "pred_artifact_types": artifacts,
                "confidence": final.get("confidence"),
                "evidence_card": evidence_card,
            }
        )

    write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} aggregated prediction rows to {args.output}")


def evaluate_predictions(args: argparse.Namespace) -> None:
    sid_root = args.sid_root.resolve()
    manifest_by_id = {row["img_id"]: row for row in read_jsonl(args.manifest)}
    predictions = read_jsonl(args.predictions)

    total = 0
    correct = 0
    by_class_total = {name: 0 for name in CLASS_NAMES}
    by_class_correct = {name: 0 for name in CLASS_NAMES}
    confusion = {
        true_name: {pred_name: 0 for pred_name in CLASS_NAMES + ["unknown"]}
        for true_name in CLASS_NAMES
    }
    tampered_ious = []
    tampered_hit_01 = 0
    tampered_hit_03 = 0
    tampered_hit_05 = 0

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
            gt_bbox = mask_bbox(sid_root / row["mask_path"])
            iou = bbox_iou(valid_bbox(pred.get("pred_tampered_bbox")), gt_bbox)
            tampered_ious.append(iou)
            tampered_hit_01 += int(iou >= 0.1)
            tampered_hit_03 += int(iou >= 0.3)
            tampered_hit_05 += int(iou >= 0.5)

    metrics = {
        "prediction_file": str(args.predictions),
        "manifest_file": str(args.manifest),
        "samples": total,
        "classification_accuracy": round(correct / total, 6) if total else 0.0,
        "per_class_accuracy": {
            name: round(by_class_correct[name] / by_class_total[name], 6)
            if by_class_total[name]
            else 0.0
            for name in CLASS_NAMES
        },
        "confusion_matrix": confusion,
        "tampered_localization": {
            "samples": len(tampered_ious),
            "mean_bbox_iou": round(float(np.mean(tampered_ious)), 6)
            if tampered_ious
            else 0.0,
            "hit_rate_iou_0_1": round(tampered_hit_01 / len(tampered_ious), 6)
            if tampered_ious
            else 0.0,
            "hit_rate_iou_0_3": round(tampered_hit_03 / len(tampered_ious), 6)
            if tampered_ious
            else 0.0,
            "hit_rate_iou_0_5": round(tampered_hit_05 / len(tampered_ious), 6)
            if tampered_ious
            else 0.0,
        },
    }
    write_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    req = subparsers.add_parser("make-requests")
    req.add_argument("--sid-root", type=Path, default=Path("SID_Set"))
    req.add_argument("--manifest", type=Path, required=True)
    req.add_argument("--output", type=Path, required=True)
    req.add_argument("--limit", type=int)
    req.add_argument("--per-class", type=int)
    req.add_argument(
        "--prompt-version",
        choices=[
            "original",
            "local_first_compact",
            "balanced_compact",
            "lowlevel_assisted_compact",
            "lowlevel_assisted_compact_v1",
            "global_synthesis_compact",
            "global_synthesis_compact_v1",
            "real_guard_compact_v1",
            "real_guard_compact_v2",
            "region_verifier_compact_v1",
            "region_verifier_compact_v2",
            "region_verifier_compact_v3",
            "global_local_router_compact_v1",
            "region_evidence_consistency_v1",
        ],
        default="original",
    )
    req.add_argument("--lowlevel", type=Path)
    req.add_argument("--region-asset-dir", type=Path)
    req.set_defaults(func=make_requests)

    low = subparsers.add_parser("lowlevel")
    low.add_argument("--sid-root", type=Path, default=Path("SID_Set"))
    low.add_argument("--manifest", type=Path, required=True)
    low.add_argument("--output", type=Path, required=True)
    low.add_argument("--limit", type=int)
    low.add_argument("--per-class", type=int)
    low.set_defaults(func=run_lowlevel)

    contrast = subparsers.add_parser("region-contrast")
    contrast.add_argument("--sid-root", type=Path, default=Path("SID_Set"))
    contrast.add_argument("--manifest", type=Path, required=True)
    contrast.add_argument("--lowlevel", type=Path, required=True)
    contrast.add_argument("--output", type=Path, required=True)
    contrast.add_argument("--limit", type=int)
    contrast.add_argument("--per-class", type=int)
    contrast.add_argument("--context-pad-ratio", type=float, default=0.5)
    contrast.set_defaults(func=run_region_contrast)

    ev = subparsers.add_parser("evaluate-lowlevel")
    ev.add_argument("--sid-root", type=Path, default=Path("SID_Set"))
    ev.add_argument("--manifest", type=Path, required=True)
    ev.add_argument("--predictions", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.set_defaults(func=evaluate_lowlevel)

    ag = subparsers.add_parser("aggregate")
    ag.add_argument("--vlm-output", type=Path, required=True)
    ag.add_argument("--lowlevel", type=Path, required=True)
    ag.add_argument("--output", type=Path, required=True)
    ag.set_defaults(func=aggregate)

    full_ev = subparsers.add_parser("evaluate")
    full_ev.add_argument("--sid-root", type=Path, default=Path("SID_Set"))
    full_ev.add_argument("--manifest", type=Path, required=True)
    full_ev.add_argument("--predictions", type=Path, required=True)
    full_ev.add_argument("--output", type=Path, required=True)
    full_ev.set_defaults(func=evaluate_predictions)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
