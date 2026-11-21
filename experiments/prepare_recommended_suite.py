#!/usr/bin/env python3
"""Prepare assets for the reviewer-recommended audit experiments.

The script is intentionally data-only: it never changes SID-Set images or
annotations.  It creates a frozen SID-Explain manifest and a stratified,
deranged shuffled-proposal condition for the JSG-Core tampered stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from candidate_evidence_explanation import PROMPT, make_region_assets, valid_bbox


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "experiments" / "runs" / "recommended_suite"
DEFAULT_REPORT_DIR = ROOT / "experiments" / "reports" / "recommended_suite"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(args: argparse.Namespace) -> None:
    annotations = read_jsonl(args.annotations)
    validation = {row["img_id"]: row for row in read_jsonl(args.validation_manifest)}
    missing = [row["img_id"] for row in annotations if row["img_id"] not in validation]
    if missing:
        raise ValueError(f"Missing {len(missing)} SID-Explain IDs from validation manifest")

    manifest = [validation[row["img_id"]] for row in annotations]
    counts = Counter(row["class_dir"] for row in manifest)
    expected = {"real": 100, "full_synthetic": 100, "tampered": 100}
    if dict(counts) != expected:
        raise ValueError(f"Unexpected SID-Explain class counts: {dict(counts)}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / "sid_explain300_manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    for class_name in expected:
        write_jsonl(
            args.run_dir / f"sid_explain300_{class_name}100_annotations.jsonl",
            [row for row in annotations if row.get("human_label") == class_name],
        )

    report = {
        "stage": "build_recommended_sid_explain_manifest",
        "annotations": str(args.annotations.relative_to(ROOT)),
        "validation_manifest": str(args.validation_manifest.relative_to(ROOT)),
        "output": str(manifest_path.relative_to(ROOT)),
        "rows": len(manifest),
        "class_counts": dict(counts),
        "annotation_sha256": sha256(args.annotations),
        "validation_manifest_sha256": sha256(args.validation_manifest),
        "output_sha256": sha256(manifest_path),
    }
    write_json(args.report_dir / "sid_explain300_manifest_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_large_negative(args: argparse.Namespace) -> None:
    selection = read_jsonl(args.selection_requests)
    validation = {row["img_id"]: row for row in read_jsonl(args.validation_manifest)}
    selected_ids = [row["img_id"] for row in selection]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Large-negative selection contains duplicate image IDs")
    missing = [img_id for img_id in selected_ids if img_id not in validation]
    if missing:
        raise ValueError(f"Missing {len(missing)} selected IDs from validation manifest")
    excluded_ids = {row["img_id"] for row in read_jsonl(args.exclude_annotations)}
    overlap = sorted(set(selected_ids) & excluded_ids)
    if overlap:
        raise ValueError(f"Large-negative selection overlaps excluded annotation pool: {overlap[:5]}")
    manifest = [validation[img_id] for img_id in selected_ids]
    counts = Counter(row["class_dir"] for row in manifest)
    if dict(counts) != {"real": 1000, "full_synthetic": 1000}:
        raise ValueError(f"Unexpected large-negative class counts: {dict(counts)}")
    annotations = []
    for row in manifest:
        class_name = row["class_dir"]
        annotations.append(
            {
                "annotation_task": "SID-Negative-Routing-2000",
                "annotation_status": "class_label_only",
                "do_not_use_for_tuning": True,
                "source_split": "frozen_val_2000_aigc_binary_selection",
                "img_id": row["img_id"],
                "human_label": class_name,
                "image_path": row["image_path"],
                "mask_path": None,
                "width": row.get("width"),
                "height": row.get("height"),
                "evidence_items": [],
                "dominant_artifact_type": "none" if class_name == "real" else "semantic_implausibility",
                "target_scope": "none" if class_name == "real" else "whole_image",
                "evaluation_scope": "classification_only; semantic fields are schema diagnostics",
            }
        )
    manifest_path = args.run_dir / "negative2000_manifest.jsonl"
    annotations_path = args.run_dir / "negative2000_annotations.jsonl"
    write_jsonl(manifest_path, manifest)
    write_jsonl(annotations_path, annotations)
    report = {
        "stage": "build_large_negative_candidate_audit",
        "selection_requests": str(args.selection_requests.relative_to(ROOT)),
        "validation_manifest": str(args.validation_manifest.relative_to(ROOT)),
        "manifest_output": str(manifest_path.relative_to(ROOT)),
        "annotations_output": str(annotations_path.relative_to(ROOT)),
        "rows": len(manifest),
        "class_counts": dict(counts),
        "excluded_annotation_pool": str(args.exclude_annotations.relative_to(ROOT)),
        "overlap_with_excluded_annotation_pool": 0,
        "selection_sha256": sha256(args.selection_requests),
        "manifest_sha256": sha256(manifest_path),
        "annotations_sha256": sha256(annotations_path),
    }
    write_json(args.report_dir / "negative2000_manifest_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _mask_bucket(annotation: dict[str, Any]) -> str:
    auto = annotation.get("auto_features")
    if isinstance(auto, dict) and auto.get("mask_area_bucket"):
        return str(auto["mask_area_bucket"])
    return "unknown"


def _derangement(ids: list[str], rng: random.Random) -> list[str]:
    if len(ids) < 2:
        raise ValueError("A derangement requires at least two samples")
    donors = list(ids)
    for _ in range(10000):
        rng.shuffle(donors)
        if all(target != donor for target, donor in zip(ids, donors)):
            return list(donors)
    raise RuntimeError("Could not construct a derangement")


def _scale_bbox(
    bbox: list[int], donor_width: int, donor_height: int, target_width: int, target_height: int
) -> list[int]:
    if min(donor_width, donor_height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    scaled = [
        int(round(bbox[0] * target_width / donor_width)),
        int(round(bbox[1] * target_height / donor_height)),
        int(round(bbox[2] * target_width / donor_width)),
        int(round(bbox[3] * target_height / donor_height)),
    ]
    scaled[0] = max(0, min(target_width - 1, scaled[0]))
    scaled[1] = max(0, min(target_height - 1, scaled[1]))
    scaled[2] = max(scaled[0] + 1, min(target_width, scaled[2]))
    scaled[3] = max(scaled[1] + 1, min(target_height, scaled[3]))
    checked = valid_bbox(scaled)
    if checked is None:
        raise ValueError(f"Invalid scaled bbox: {scaled}")
    return checked


def _prompt_value(prompt: str, key: str) -> str:
    match = re.search(rf"^- {re.escape(key)}=(.*)$", prompt, flags=re.MULTILINE)
    return match.group(1).strip() if match else "None"


def _replace_prompt_line(prompt: str, key: str, value: str) -> str:
    updated, count = re.subn(
        rf"^- {re.escape(key)}=.*$", f"- {key}={value}", prompt, count=1, flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError(f"Expected exactly one legacy prompt line for {key}; found {count}")
    return updated


def _adapt_legacy_view_prompt(prompt: str, view_mode: str) -> str:
    if view_mode in {"matched_crop", "shuffled_crop", "shuffled_proposal"}:
        return prompt
    original = (
        "You receive two images:\n"
        "1. the full image with one red candidate box;\n"
        "2. a crop around the same candidate box."
    )
    if original not in prompt:
        raise ValueError("Frozen legacy prompt does not contain the expected two-view preamble")
    if view_mode == "marked_only":
        replacement = "You receive one image: the full image with one red candidate box."
    elif view_mode == "crop_only":
        replacement = (
            "You receive one image: a crop around a candidate region proposed on the corresponding "
            "full image. Only the candidate crop is visible."
        )
    else:
        raise ValueError(f"Unsupported view mode: {view_mode}")
    return prompt.replace(original, replacement, 1)


def make_legacy_requests(args: argparse.Namespace) -> None:
    frozen_rows = read_jsonl(args.frozen_template_requests)
    if not frozen_rows:
        raise ValueError("Frozen template request file is empty")
    frozen_prompt = str(frozen_rows[0]["prompt"])
    generated_rows = read_jsonl(args.generated_requests)
    output_rows = []
    for generated in generated_rows:
        meta = generated.get("metadata_for_evaluation_only") or {}
        candidate_bbox = valid_bbox(meta.get("candidate_bbox"))
        if candidate_bbox is None:
            raise ValueError(f"Generated request {generated.get('img_id')} has no valid candidate bbox")
        generated_prompt = str(generated.get("prompt") or "")
        prompt = frozen_prompt
        prompt = _replace_prompt_line(prompt, "candidate_bbox", str(candidate_bbox))
        prompt = _replace_prompt_line(prompt, "lowlevel_score", _prompt_value(generated_prompt, "lowlevel_score"))
        prompt = _replace_prompt_line(
            prompt,
            "lowlevel_artifact_types",
            _prompt_value(generated_prompt, "lowlevel_artifact_types"),
        )
        prompt = _adapt_legacy_view_prompt(prompt, args.view_mode)
        image_paths = list(generated.get("image_paths") or [])
        if args.view_mode == "marked_only":
            image_paths = image_paths[:1]
        elif args.view_mode == "crop_only":
            image_paths = image_paths[-1:]
        if not image_paths:
            raise ValueError(f"No image path remains for {generated.get('img_id')}")
        prompt_version = "candidate_evidence_v1"
        if args.view_mode != "matched_crop":
            prompt_version += f"_legacy_{args.view_mode}"
        output_rows.append(
            {
                "task_id": f"candidate_evidence_{generated['img_id']}",
                "img_id": generated["img_id"],
                "prompt_version": prompt_version,
                "image_path": generated["image_path"],
                "image_paths": image_paths,
                "metadata_for_evaluation_only": {
                    "human_label": meta.get("human_label"),
                    "candidate_bbox": candidate_bbox,
                    "candidate_mode": "top3_union",
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                },
                "experimental_provenance": {
                    "view_mode": args.view_mode,
                    "source_request_sha256": canonical_request_hash(generated),
                    "secondary_view_provenance": meta.get("secondary_view_provenance"),
                    "crop_donor_img_id": meta.get("crop_donor_img_id"),
                    "proposal_provenance": meta.get("proposal_provenance"),
                    "proposal_donor_img_id": meta.get("donor_img_id"),
                },
                "prompt": prompt,
            }
        )
    write_jsonl(args.output, output_rows)
    report = {
        "stage": "make_legacy_prompt_controlled_requests",
        "view_mode": args.view_mode,
        "rows": len(output_rows),
        "frozen_template_requests": str(args.frozen_template_requests.relative_to(ROOT)),
        "generated_requests": str(args.generated_requests.relative_to(ROOT)),
        "output": str(args.output.relative_to(ROOT)),
        "frozen_template_sha256": sha256(args.frozen_template_requests),
        "generated_sha256": sha256(args.generated_requests),
        "output_sha256": sha256(args.output),
        "legacy_target_label_space_excludes_person_and_animal": (
            "- person" not in frozen_prompt and "- animal" not in frozen_prompt
        ),
        "legacy_prompt_excludes_image_mode_metadata": "- image_mode=" not in frozen_prompt,
    }
    # A view mode can be reused across cohorts (for example, SID-Explain-300
    # and the disjoint negative-2000 audit).  Anchor the report name to the
    # actual output artifact so a later preparation step cannot overwrite an
    # earlier cohort's provenance summary.
    write_json(args.report_dir / f"{args.output.stem}_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def canonical_request_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_shuffled_requests(args: argparse.Namespace) -> None:
    annotations = {row["img_id"]: row for row in read_jsonl(args.annotations)}
    baseline_rows = read_jsonl(args.baseline_requests)
    baseline = {row["img_id"]: row for row in baseline_rows}
    ids = [row["img_id"] for row in baseline_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Baseline requests contain duplicate img_id values")

    by_bucket: dict[str, list[str]] = defaultdict(list)
    for img_id in ids:
        by_bucket[_mask_bucket(annotations[img_id])].append(img_id)

    rng = random.Random(args.seed)
    donor_for: dict[str, str] = {}
    for bucket in sorted(by_bucket):
        bucket_ids = sorted(by_bucket[bucket])
        for target, donor in zip(bucket_ids, _derangement(bucket_ids, rng)):
            donor_for[target] = donor

    output_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    for target_id in ids:
        target_request = baseline[target_id]
        donor_id = donor_for[target_id]
        donor_request = baseline[donor_id]
        target_meta = target_request.get("metadata_for_evaluation_only") or {}
        donor_meta = donor_request.get("metadata_for_evaluation_only") or {}
        donor_bbox = valid_bbox(donor_meta.get("candidate_bbox"))
        if donor_bbox is None:
            raise ValueError(f"Donor {donor_id} has no valid proposal")

        donor_width = int(donor_meta.get("width") or 0)
        donor_height = int(donor_meta.get("height") or 0)
        target_width = int(target_meta.get("width") or 0)
        target_height = int(target_meta.get("height") or 0)
        shuffled_bbox = _scale_bbox(
            donor_bbox, donor_width, donor_height, target_width, target_height
        )

        image_path = Path(str(target_request["image_path"]))
        marked_path, crop_path = make_region_assets(
            image_path, shuffled_bbox, args.asset_dir, target_id
        )
        donor_prompt = str(donor_request.get("prompt") or "")
        prompt = (
            PROMPT
            + "\n\nCandidate metadata:\n"
            + f"- candidate_bbox={shuffled_bbox}\n"
            + "- candidate_mode=top3_union\n"
            + "- image_mode=marked_crop\n"
            + f"- lowlevel_score={_prompt_value(donor_prompt, 'lowlevel_score')}\n"
            + f"- lowlevel_artifact_types={_prompt_value(donor_prompt, 'lowlevel_artifact_types')}\n"
            + "\nIf you output prediction=tampered, evidence_bbox must exactly equal candidate_bbox."
        )
        output_rows.append(
            {
                "task_id": f"candidate_evidence_shuffled_{target_id}",
                "img_id": target_id,
                "prompt_version": "candidate_evidence_v1_marked_crop_shuffled_proposal_package",
                "image_path": str(image_path),
                "image_paths": [str(marked_path), str(crop_path)],
                "metadata_for_evaluation_only": {
                    "human_label": target_meta.get("human_label"),
                    "candidate_bbox": shuffled_bbox,
                    "candidate_mode": "top3_union",
                    "image_mode": "marked_crop",
                    "width": target_width,
                    "height": target_height,
                    "proposal_provenance": "stratified_deranged_donor_package",
                    "donor_img_id": donor_id,
                    "donor_candidate_bbox": donor_bbox,
                },
                "prompt": prompt,
            }
        )
        permutation_rows.append(
            {
                "target_img_id": target_id,
                "donor_img_id": donor_id,
                "mask_area_bucket": _mask_bucket(annotations[target_id]),
                "donor_bbox": donor_bbox,
                "target_scaled_bbox": shuffled_bbox,
                "fixed_point": target_id == donor_id,
            }
        )

    if any(row["fixed_point"] for row in permutation_rows):
        raise AssertionError("Shuffled proposal permutation contains a fixed point")
    write_jsonl(args.output, output_rows)
    write_jsonl(args.permutation_output, permutation_rows)
    report = {
        "stage": "make_stratified_shuffled_proposal_requests",
        "seed": args.seed,
        "rows": len(output_rows),
        "fixed_points": 0,
        "stratum_counts": dict(Counter(row["mask_area_bucket"] for row in permutation_rows)),
        "baseline_requests": str(args.baseline_requests.relative_to(ROOT)),
        "output": str(args.output.relative_to(ROOT)),
        "permutation_output": str(args.permutation_output.relative_to(ROOT)),
        "baseline_sha256": sha256(args.baseline_requests),
        "output_sha256": sha256(args.output),
        "permutation_sha256": sha256(args.permutation_output),
    }
    write_json(args.report_dir / "shuffled_proposal_requests_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _proposal_geometry(request: dict[str, Any]) -> tuple[float, float]:
    meta = request.get("metadata_for_evaluation_only") or {}
    bbox = valid_bbox(meta.get("candidate_bbox"))
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    if bbox is None or min(width, height) <= 0:
        raise ValueError(f"Invalid proposal geometry for {request.get('img_id')}")
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    area_ratio = box_width * box_height / (width * height)
    aspect_ratio = box_width / box_height
    return area_ratio, aspect_ratio


def _local_geometry_derangement(
    ids: list[str], requests: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Return a minimum-cost, no-fixed-point geometry assignment."""
    ordered = sorted(ids)
    if len(ordered) < 2:
        raise ValueError("Geometry-matching stratum must contain at least two rows")
    raw_features = [
        tuple(math.log(max(value, 1e-12)) for value in _proposal_geometry(requests[img_id]))
        for img_id in ordered
    ]
    means = [sum(row[index] for row in raw_features) / len(raw_features) for index in range(2)]
    scales = []
    for index in range(2):
        variance = sum((row[index] - means[index]) ** 2 for row in raw_features) / len(raw_features)
        scales.append(max(math.sqrt(variance), 1e-12))
    features = [
        tuple((row[index] - means[index]) / scales[index] for index in range(2))
        for row in raw_features
    ]
    costs: list[list[float]] = []
    for target_index, target in enumerate(features):
        row_costs = []
        for donor_index, donor in enumerate(features):
            if target_index == donor_index:
                row_costs.append(1e9)
            else:
                row_costs.append(sum((target[k] - donor[k]) ** 2 for k in range(2)))
        costs.append(row_costs)

    # Hungarian assignment, with rows as targets and columns as crop donors.
    size = len(costs)
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row_index in range(1, size + 1):
        p[0] = row_index
        column0 = 0
        min_value = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            active_row = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = costs[active_row - 1][column - 1] - u[active_row] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assigned_column = [0] * size
    for column in range(1, size + 1):
        assigned_column[p[column] - 1] = column - 1
    mapping = {
        ordered[target_index]: ordered[donor_index]
        for target_index, donor_index in enumerate(assigned_column)
    }
    if any(target == donor for target, donor in mapping.items()):
        raise AssertionError("Geometry assignment contains a fixed point")
    return mapping


def make_shuffled_crop_requests(args: argparse.Namespace) -> None:
    annotations = {row["img_id"]: row for row in read_jsonl(args.annotations)}
    baseline_rows = read_jsonl(args.baseline_requests)
    baseline = {row["img_id"]: row for row in baseline_rows}
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for row in baseline_rows:
        annotation = annotations[row["img_id"]]
        stratum = f"{annotation.get('human_label', 'unknown')}|{_mask_bucket(annotation)}"
        by_bucket[stratum].append(row["img_id"])

    donor_for: dict[str, str] = {}
    for bucket in sorted(by_bucket):
        donor_for.update(_local_geometry_derangement(by_bucket[bucket], baseline))

    output_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    for target_request in baseline_rows:
        target_id = target_request["img_id"]
        donor_id = donor_for[target_id]
        donor_request = baseline[donor_id]
        target_paths = target_request.get("image_paths") or []
        donor_paths = donor_request.get("image_paths") or []
        if len(target_paths) < 2 or len(donor_paths) < 2:
            raise ValueError("Matched marked+crop requests must contain two image paths")
        target_area, target_aspect = _proposal_geometry(target_request)
        donor_area, donor_aspect = _proposal_geometry(donor_request)
        meta = dict(target_request.get("metadata_for_evaluation_only") or {})
        meta.update(
            {
                "secondary_view_provenance": "geometry_matched_deranged_donor_crop",
                "crop_donor_img_id": donor_id,
                "crop_match_stratum": (
                    f"{annotations[target_id].get('human_label', 'unknown')}|"
                    f"{_mask_bucket(annotations[target_id])}"
                ),
            }
        )
        output = dict(target_request)
        output["task_id"] = f"candidate_evidence_shuffled_crop_{target_id}"
        output["prompt_version"] = "candidate_evidence_v1_marked_matched_bbox_shuffled_crop"
        output["image_paths"] = [str(target_paths[0]), str(donor_paths[1])]
        output["metadata_for_evaluation_only"] = meta
        output_rows.append(output)
        permutation_rows.append(
            {
                "target_img_id": target_id,
                "donor_img_id": donor_id,
                "matching_stratum": (
                    f"{annotations[target_id].get('human_label', 'unknown')}|"
                    f"{_mask_bucket(annotations[target_id])}"
                ),
                "target_candidate_area_ratio": target_area,
                "donor_candidate_area_ratio": donor_area,
                "absolute_log_area_difference": abs(math.log(target_area) - math.log(donor_area)),
                "target_candidate_aspect_ratio": target_aspect,
                "donor_candidate_aspect_ratio": donor_aspect,
                "absolute_log_aspect_difference": abs(math.log(target_aspect) - math.log(donor_aspect)),
                "fixed_point": target_id == donor_id,
            }
        )

    if any(row["fixed_point"] for row in permutation_rows):
        raise AssertionError("Shuffled-crop permutation contains a fixed point")
    write_jsonl(args.output, output_rows)
    write_jsonl(args.permutation_output, permutation_rows)
    report = {
        "stage": "make_geometry_matched_shuffled_crop_requests",
        "rows": len(output_rows),
        "fixed_points": 0,
        "stratum_counts": dict(Counter(row["matching_stratum"] for row in permutation_rows)),
        "mean_absolute_log_area_difference": sum(
            row["absolute_log_area_difference"] for row in permutation_rows
        )
        / len(permutation_rows),
        "mean_absolute_log_aspect_difference": sum(
            row["absolute_log_aspect_difference"] for row in permutation_rows
        )
        / len(permutation_rows),
        "prompt_bytes_unchanged": all(
            output["prompt"] == baseline[output["img_id"]]["prompt"] for output in output_rows
        ),
        "baseline_requests": str(args.baseline_requests.relative_to(ROOT)),
        "output": str(args.output.relative_to(ROOT)),
        "permutation_output": str(args.permutation_output.relative_to(ROOT)),
        "baseline_sha256": sha256(args.baseline_requests),
        "output_sha256": sha256(args.output),
        "permutation_sha256": sha256(args.permutation_output),
    }
    write_json(args.report_dir / "shuffled_crop_requests_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument(
        "--annotations", type=Path, default=ROOT / "SID_Set" / "annotations" / "SID-Explain-300.jsonl"
    )
    manifest.add_argument(
        "--validation-manifest", type=Path, default=ROOT / "SID_Set" / "validation" / "manifest.jsonl"
    )
    manifest.set_defaults(func=build_manifest)

    large_negative = subparsers.add_parser("build-large-negative")
    large_negative.add_argument(
        "--selection-requests",
        type=Path,
        default=ROOT / "experiments" / "runs" / "val_2000_aigc_binary_requests.jsonl",
    )
    large_negative.add_argument(
        "--validation-manifest", type=Path, default=ROOT / "SID_Set" / "validation" / "manifest.jsonl"
    )
    large_negative.add_argument(
        "--exclude-annotations",
        type=Path,
        default=ROOT / "SID_Set" / "annotations" / "SID-Explain-300.jsonl",
    )
    large_negative.set_defaults(func=build_large_negative)

    shuffled = subparsers.add_parser("make-shuffled-requests")
    shuffled.add_argument(
        "--annotations", type=Path, default=ROOT / "SID_Set" / "annotations" / "SID-Explain-300.jsonl"
    )
    shuffled.add_argument("--baseline-requests", type=Path, required=True)
    shuffled.add_argument("--asset-dir", type=Path, required=True)
    shuffled.add_argument("--output", type=Path, required=True)
    shuffled.add_argument("--permutation-output", type=Path, required=True)
    shuffled.add_argument("--seed", type=int, default=20260729)
    shuffled.set_defaults(func=make_shuffled_requests)

    shuffled_crop = subparsers.add_parser("make-shuffled-crop-requests")
    shuffled_crop.add_argument(
        "--annotations", type=Path, default=ROOT / "SID_Set" / "annotations" / "SID-Explain-300.jsonl"
    )
    shuffled_crop.add_argument("--baseline-requests", type=Path, required=True)
    shuffled_crop.add_argument("--output", type=Path, required=True)
    shuffled_crop.add_argument("--permutation-output", type=Path, required=True)
    shuffled_crop.set_defaults(func=make_shuffled_crop_requests)

    legacy = subparsers.add_parser("make-legacy-requests")
    legacy.add_argument(
        "--frozen-template-requests",
        type=Path,
        default=ROOT / "experiments" / "runs" / "jsg_core_requests_pixel5d_v2_legacy_prompt_controlled.jsonl",
    )
    legacy.add_argument("--generated-requests", type=Path, required=True)
    legacy.add_argument(
        "--view-mode",
        choices=["matched_crop", "marked_only", "crop_only", "shuffled_crop", "shuffled_proposal"],
        required=True,
    )
    legacy.add_argument("--output", type=Path, required=True)
    legacy.set_defaults(func=make_legacy_requests)

    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.report_dir = args.report_dir.resolve()
    for name in [
        "annotations",
        "validation_manifest",
        "selection_requests",
        "exclude_annotations",
        "frozen_template_requests",
        "generated_requests",
        "baseline_requests",
        "asset_dir",
        "output",
        "permutation_output",
    ]:
        if hasattr(args, name):
            value = getattr(args, name)
            if isinstance(value, Path):
                setattr(args, name, value.resolve())
    args.func(args)


if __name__ == "__main__":
    main()
