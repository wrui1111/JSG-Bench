#!/usr/bin/env python3
"""Revision experiments for the JSG-Bench manuscript (no new annotation).

Subcommands
-----------
make-requests  Build controlled request variants from the frozen primary requests.
               Each variant changes exactly one declared component:
                 oracle       proposal box := reference (mask-tight) bbox; hints kept
                 gray         both visual inputs := uniform gray images; prompt kept
                 ontology     Core: add person/animal to the target list;
                              Scale: remove person/animal from the target list
                 shuffled     Scale: within-band deranged donor proposal package
score          Score variant outputs with the frozen normalizer and gate rules,
               and compare them with the primary per-report conditions.

All paths are relative to the repository root (parent of ``experiments/``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_evidence_explanation import (  # noqa: E402
    gold_item,
    make_region_assets,
    normalize_artifact,
    normalize_target,
    prediction_item,
    valid_bbox,
)
from prepare_recommended_suite import (  # noqa: E402
    _derangement,
    _prompt_value,
    _replace_prompt_line,
    _scale_bbox,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
REV = RUNS / "revision"
REPORTS = ROOT / "experiments" / "reports" / "revision"
IOU_THRESHOLD = 0.1
SHUFFLE_SEED = 20260729

COHORTS: dict[str, dict[str, Path]] = {
    "core": {
        "annotations": RUNS / "sid_explain300_tampered100_annotations_revised.jsonl",
        "requests": RUNS / "jsg_core_requests_pixel5d_v2_legacy_prompt_controlled.jsonl",
        "outputs": RUNS / "jsg_core_outputs_pixel5d_v2_legacy_prompt_controlled.jsonl",
        "trace": RUNS / "frozen_reference_v1/experiments/runs/jsg_core_td_egfa_audit_trace_pixel5d_v2.jsonl",
    },
    "scale": {
        "annotations": RUNS / "sid_fa600_annotations_postreview_eval.jsonl",
        "requests": RUNS / "jsg_scale_requests_pixel5d_v2_legacy_prompt_controlled.jsonl",
        "outputs": RUNS / "jsg_scale_outputs_pixel5d_v2_legacy_prompt_controlled.jsonl",
        "trace": RUNS / "frozen_reference_v1/experiments/runs/jsg_scale_td_egfa_audit_trace_pixel5d_v2.jsonl",
    },
}

RESTRICTED_TARGETS = "- none\n- face\n- object\n- background\n"
FULL_TARGETS = "- none\n- face\n- person\n- animal\n- object\n- background\n"
HARMONIZE = {"person": "object", "animal": "object"}
CONDITIONS = ["parse", "prediction", "artifact", "target", "spatial"]


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def stratum_of(annotation: dict[str, Any]) -> str:
    auto = annotation.get("auto_features") or {}
    return str(annotation.get("mask_area_bucket") or auto.get("mask_area_bucket"))


# ----------------------------------------------------------------------------
# Request variants
# ----------------------------------------------------------------------------

def _variant_row(base: dict[str, Any], variant: str, **changes: Any) -> dict[str, Any]:
    row = json.loads(json.dumps(base))
    row.update(changes)
    row["revision_variant"] = variant
    row["source_request_sha256"] = hashlib.sha256(
        json.dumps(base, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return row


def make_oracle(cohort: str) -> list[dict[str, Any]]:
    # Oracle proposals use the image-aligned (corrected) reference bbox.
    references = corrected_reference_boxes(cohort)
    asset_dir = REV / f"assets_{cohort}_oracle"
    rows = []
    for base in read_jsonl(COHORTS[cohort]["requests"]):
        img_id = base["img_id"]
        oracle = valid_bbox(references[img_id]["corrected"])
        if oracle is None:
            raise ValueError(f"No reference bbox for {img_id}")
        marked, crop = make_region_assets(Path(base["image_path"]), oracle, asset_dir, img_id)
        meta = dict(base["metadata_for_evaluation_only"])
        meta.update({"candidate_bbox": oracle, "candidate_mode": "oracle_reference_bbox",
                     "b_top3_bbox": base["metadata_for_evaluation_only"]["candidate_bbox"]})
        rows.append(_variant_row(
            base, f"{cohort}_oracle",
            prompt=_replace_prompt_line(base["prompt"], "candidate_bbox", str(oracle)),
            image_paths=[rel(marked), rel(crop)],
            metadata_for_evaluation_only=meta,
        ))
    return rows


def _gray_like(path: Path, cache: dict[tuple[int, int], Path], out_dir: Path) -> Path:
    with Image.open(ROOT / path) as image:
        size = image.size
    if size not in cache:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"gray_{size[0]}x{size[1]}.jpg"
        Image.new("RGB", size, (128, 128, 128)).save(target, quality=95)
        cache[size] = target
    return cache[size]


def make_gray(cohort: str) -> list[dict[str, Any]]:
    cache: dict[tuple[int, int], Path] = {}
    out_dir = REV / "assets_gray"
    rows = []
    for base in read_jsonl(COHORTS[cohort]["requests"]):
        paths = [rel(_gray_like(Path(p), cache, out_dir)) for p in base["image_paths"]]
        rows.append(_variant_row(base, f"{cohort}_gray", image_paths=paths))
    return rows


def make_ontology(cohort: str) -> list[dict[str, Any]]:
    old, new = (RESTRICTED_TARGETS, FULL_TARGETS) if cohort == "core" else (FULL_TARGETS, RESTRICTED_TARGETS)
    rows = []
    for base in read_jsonl(COHORTS[cohort]["requests"]):
        if base["prompt"].count(old) != 1:
            raise ValueError(f"Target list not found exactly once for {base['img_id']}")
        rows.append(_variant_row(base, f"{cohort}_ontology_swap", prompt=base["prompt"].replace(old, new, 1)))
    return rows


def make_shuffled(cohort: str, seed: int = SHUFFLE_SEED,
                  asset_directory: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotations = {row["img_id"]: row for row in read_jsonl(COHORTS[cohort]["annotations"])}
    base_rows = read_jsonl(COHORTS[cohort]["requests"])
    base = {row["img_id"]: row for row in base_rows}
    by_band: dict[str, list[str]] = defaultdict(list)
    for img_id in base:
        by_band[stratum_of(annotations[img_id])].append(img_id)
    rng = random.Random(seed)
    donor_for: dict[str, str] = {}
    for band in sorted(by_band):
        ids = sorted(by_band[band])
        donor_for.update(zip(ids, _derangement(ids, rng)))
    asset_dir = asset_directory or REV / f"assets_{cohort}_shuffled"
    rows, perm = [], []
    for row in base_rows:
        target_id = row["img_id"]
        donor = base[donor_for[target_id]]
        tm, dm = row["metadata_for_evaluation_only"], donor["metadata_for_evaluation_only"]
        bbox = _scale_bbox(dm["candidate_bbox"], int(dm["width"]), int(dm["height"]),
                           int(tm["width"]), int(tm["height"]))
        marked, crop = make_region_assets(Path(row["image_path"]), bbox, asset_dir, target_id)
        prompt = _replace_prompt_line(row["prompt"], "candidate_bbox", str(bbox))
        for key in ("lowlevel_score", "lowlevel_artifact_types"):
            prompt = _replace_prompt_line(prompt, key, _prompt_value(donor["prompt"], key))
        meta = dict(tm)
        meta.update({"candidate_bbox": bbox, "proposal_provenance": "stratified_deranged_donor_package",
                     "donor_img_id": donor["img_id"], "b_top3_bbox": tm["candidate_bbox"]})
        rows.append(_variant_row(row, f"{cohort}_shuffled_package", prompt=prompt,
                                 image_paths=[rel(marked), rel(crop)], metadata_for_evaluation_only=meta))
        perm.append({"target_img_id": target_id, "donor_img_id": donor["img_id"],
                     "band": stratum_of(annotations[target_id])})
    if any(p["target_img_id"] == p["donor_img_id"] for p in perm):
        raise AssertionError("Derangement has a fixed point")
    return rows, perm


def cmd_make_requests(_: argparse.Namespace) -> None:
    built: dict[str, Path] = {}
    jobs = {
        "scale_oracle": lambda: make_oracle("scale"),
        "core_oracle": lambda: make_oracle("core"),
        "core_ontology_swap": lambda: make_ontology("core"),
        "scale_ontology_swap": lambda: make_ontology("scale"),
        "core_gray": lambda: make_gray("core"),
        "scale_gray": lambda: make_gray("scale"),
    }
    for name, job in jobs.items():
        path = REV / f"requests_{name}.jsonl"
        write_jsonl(path, job())
        built[name] = path
    rows, perm = make_shuffled("scale")
    built["scale_shuffled_package"] = REV / "requests_scale_shuffled_package.jsonl"
    write_jsonl(built["scale_shuffled_package"], rows)
    write_jsonl(REV / "scale_shuffled_permutation.jsonl", perm)
    manifest = {name: {"path": rel(path), "rows": len(read_jsonl(path)), "sha256": sha256(path)}
                for name, path in built.items()}
    manifest["_shuffle_seed"] = SHUFFLE_SEED
    write_json(REPORTS / "request_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def bbox_iou(a: list[int] | None, b: list[int] | None) -> float:
    if not a or not b:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def corrected_reference_boxes(cohort: str) -> dict[str, dict[str, Any]]:
    """Map the frozen mask-coordinate reference bbox into image coordinates.

    The frozen SID-Set references were computed as the tight ``mask > 0`` bbox in
    the mask's own pixel grid. For masks whose size differs from the image, the
    mask is a resized copy of the image-aligned mask, so the bbox must be scaled
    by (W_image / W_mask, H_image / H_mask). Masks with matching size are unchanged.
    """
    import math

    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(COHORTS[cohort]["annotations"]):
        image_path = ROOT / "SID_Set" / row["image_path"]
        mask_path = ROOT / "SID_Set" / row["mask_path"]
        with Image.open(image_path) as im:
            W, H = im.size
        with Image.open(mask_path) as mk:
            arr = np.asarray(mk.convert("L")) > 0
        h, w = arr.shape
        ys, xs = np.where(arr)
        native = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        current = valid_bbox(gold_item(row)["evidence_bbox"])
        geometry = row.get("reference_geometry") or {}
        frozen = geometry.get("frozen_mask_grid_bbox") or current
        if native != frozen:
            raise ValueError(f"Frozen reference does not match mask>0 rule for {row['img_id']}")
        if (w, h) == (W, H):
            corrected = native
        else:
            sx, sy = W / w, H / h
            corrected = [int(math.floor(native[0] * sx)), int(math.floor(native[1] * sy)),
                         min(W, int(math.ceil(native[2] * sx))), min(H, int(math.ceil(native[3] * sy)))]
        if current not in (frozen, corrected):
            raise ValueError(f"Annotation bbox is neither frozen nor corrected for {row['img_id']}")
        out[row["img_id"]] = {"frozen": frozen, "corrected": corrected,
                              "image_size": [W, H], "mask_size": [w, h], "changed": corrected != frozen}
    return out


def score_rows(cohort: str, outputs_path: Path, requests_path: Path,
               harmonize: bool = False, threshold: float = IOU_THRESHOLD,
               reference_override: dict[str, list[int]] | None = None) -> dict[str, dict[str, Any]]:
    """Per-report gate outcomes using the frozen normalizer and B_top3 inheritance rule."""
    annotations = {row["img_id"]: row for row in read_jsonl(COHORTS[cohort]["annotations"])}
    outputs = {row["img_id"]: row for row in read_jsonl(outputs_path)}
    requests = {row["img_id"]: row for row in read_jsonl(requests_path)}
    if set(outputs) != set(annotations) or set(requests) != set(annotations):
        raise ValueError(f"Unaligned IDs for {outputs_path}")
    scored = {}
    for img_id, ann in annotations.items():
        gold = gold_item(ann)
        if reference_override is not None:
            gold["evidence_bbox"] = reference_override[img_id]
        req = requests[img_id]
        meta = req.get("metadata_for_evaluation_only") or {}
        pred = prediction_item(outputs[img_id], req, meta.get("width"), meta.get("height"))
        parsed = bool(pred["parsed"])
        g_target, p_target = gold["target_scope"], pred["target_scope"]
        if harmonize:
            g_target, p_target = HARMONIZE.get(g_target, g_target), HARMONIZE.get(p_target, p_target)
        iou = bbox_iou(pred["evidence_bbox"], gold["evidence_bbox"])
        cond = {
            "parse": parsed,
            "prediction": parsed and pred["prediction_present"] and pred["prediction"] == gold["prediction"],
            "artifact": parsed and pred["artifact_present"] and pred["artifact_type"] == gold["artifact_type"],
            "target": parsed and pred["target_present"] and p_target == g_target,
            "spatial": parsed and pred["evidence_bbox"] is not None and iou >= threshold,
        }
        cond = {k: bool(v) for k, v in cond.items()}
        scored[img_id] = {
            "stratum": stratum_of(ann),
            "conditions": cond,
            "joint": all(cond.values()),
            "semantic_gate": all(cond[k] for k in ("parse", "prediction", "artifact", "target")),
            "iou": iou,
            "pred_artifact": pred["artifact_type"],
            "pred_target": pred["target_scope"],
            "pred_prediction": pred["prediction"],
            "gold_artifact": gold["artifact_type"],
            "gold_target": gold["target_scope"],
            "evaluated_bbox": pred["evidence_bbox"],
            "candidate_bbox": valid_bbox(meta.get("candidate_bbox")),
            "reference_bbox": gold["evidence_bbox"],
        }
    return scored


def strat_rate(values: dict[str, float], strata: dict[str, str]) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for key, value in values.items():
        groups[strata[key]].append(value)
    return float(np.mean([np.mean(v) for v in groups.values()]))


def paired_bootstrap(a: dict[str, float], b: dict[str, float], strata: dict[str, str],
                     reps: int = 10000, seed: int = 20260718) -> tuple[float, float, float]:
    """Stratified paired bootstrap of mean(a-b) with fixed design weights n_s/N.

    Equal-size strata (Core, Scale) reduce to equal weights; JSG-Xfer uses its
    16/16/18 sample-count weights, matching the manuscript's benchmark mixture.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in sorted(a):
        groups[strata[key]].append(key)
    diffs = {s: np.array([a[k] - b[k] for k in ids], dtype=float) for s, ids in groups.items()}
    total = sum(len(d) for d in diffs.values())
    weight = {s: len(d) / total for s, d in diffs.items()}
    point = float(sum(weight[s] * d.mean() for s, d in diffs.items()))
    draws = np.zeros(reps)
    for s, d in diffs.items():
        idx = rng.integers(0, len(d), size=(reps, len(d)))
        draws += weight[s] * d[idx].mean(axis=1)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def exact_mcnemar(b: int, c: int) -> float:
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(scored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    strata = {k: v["stratum"] for k, v in scored.items()}
    out = {"n": len(scored)}
    for name in CONDITIONS:
        out[name] = round(100 * strat_rate({k: float(v["conditions"][name]) for k, v in scored.items()}, strata), 2)
    out["joint"] = round(100 * strat_rate({k: float(v["joint"]) for k, v in scored.items()}, strata), 2)
    out["joint_count"] = sum(v["joint"] for v in scored.values())
    return out


def compare(variant: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]) -> dict[str, Any]:
    strata = {k: v["stratum"] for k, v in reference.items()}
    a = {k: float(variant[k]["joint"]) for k in reference}
    b = {k: float(reference[k]["joint"]) for k in reference}
    point, lo, hi = paired_bootstrap(a, b, strata)
    pa = {k for k, v in variant.items() if v["joint"]}
    pb = {k for k, v in reference.items() if v["joint"]}
    union = pa | pb
    gains, losses = len(pa - pb), len(pb - pa)
    return {
        "delta_joint_pp": round(100 * point, 2),
        "delta_ci_pp": [round(100 * lo, 2), round(100 * hi, 2)],
        "gains": gains,
        "losses": losses,
        "exact_mcnemar_p": round(exact_mcnemar(gains, losses), 4),
        "passing_set_jaccard": round(len(pa & pb) / len(union), 3) if union else 1.0,
    }


VARIANTS = {
    # name: (cohort, requests, outputs)
    "core_primary_rescored": ("core", None, None),
    "scale_primary_rescored": ("scale", None, None),
    "scale_oracle": ("scale", "requests_scale_oracle.jsonl", "outputs_scale_oracle.jsonl"),
    "core_oracle": ("core", "requests_core_oracle.jsonl", "outputs_core_oracle.jsonl"),
    "core_ontology_swap": ("core", "requests_core_ontology_swap.jsonl", "outputs_core_ontology_swap.jsonl"),
    "scale_ontology_swap": ("scale", "requests_scale_ontology_swap.jsonl", "outputs_scale_ontology_swap.jsonl"),
    "core_gray": ("core", "requests_core_gray.jsonl", "outputs_core_gray.jsonl"),
    "scale_gray": ("scale", "requests_scale_gray.jsonl", "outputs_scale_gray.jsonl"),
    "scale_shuffled_package": ("scale", "requests_scale_shuffled_package.jsonl",
                               "outputs_scale_shuffled_package.jsonl"),
    "core_repeat_mps": ("core", None, "outputs_core_repeat_mps.jsonl"),
    "core_cpu_fp32": ("core", None, "outputs_core_cpu_fp32.jsonl"),
}


def resolve(name: str) -> tuple[str, Path, Path]:
    cohort, req, out = VARIANTS[name]
    requests = REV / req if req else COHORTS[cohort]["requests"]
    outputs = REV / out if out else COHORTS[cohort]["outputs"]
    return cohort, requests, outputs


def conformance(cohort: str, scored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The rescoring must reproduce every archived per-report condition."""
    trace = {row["sample_id"]: row for row in read_jsonl(COHORTS[cohort]["trace"])}
    mismatches = [k for k, v in scored.items()
                  if v["conditions"] != trace[k]["audit_profiles"]["primary_strict"]["conditions"]]
    return {"reports": len(scored), "condition_mismatches": len(mismatches), "examples": mismatches[:5]}


def raw_identity(path_a: Path, path_b: Path) -> dict[str, Any]:
    a = {r["img_id"]: r.get("vlm_raw_text") for r in read_jsonl(path_a)}
    b = {r["img_id"]: r.get("vlm_raw_text") for r in read_jsonl(path_b)}
    same = sum(a[k] == b.get(k) for k in a)
    return {"reports": len(a), "byte_identical_raw_text": same}


def cmd_score(args: argparse.Namespace) -> None:
    results: dict[str, Any] = {}
    corrected = {c: {k: v["corrected"] for k, v in corrected_reference_boxes(c).items()} for c in COHORTS}
    frozen = {c: {k: v["frozen"] for k, v in corrected_reference_boxes(c).items()} for c in COHORTS}
    modes = {"frozen_reference": lambda c: frozen[c], "corrected_reference": lambda c: corrected[c]}
    for mode, ref_for in modes.items():
        block: dict[str, Any] = {}
        reference = {c: score_rows(c, COHORTS[c]["outputs"], COHORTS[c]["requests"],
                                   reference_override=ref_for(c)) for c in COHORTS}
        reference_h = {c: score_rows(c, COHORTS[c]["outputs"], COHORTS[c]["requests"], harmonize=True,
                                     reference_override=ref_for(c)) for c in COHORTS}
        for cohort in COHORTS:
            block[f"{cohort}_primary_rescored"] = {
                "summary": summarize(reference[cohort]),
                "harmonized_target_summary": summarize(reference_h[cohort]),
            }
            if mode == "frozen_reference":
                block[f"{cohort}_primary_rescored"]["conformance_with_archived_trace"] = conformance(
                    cohort, reference[cohort])
            else:
                block[f"{cohort}_primary_rescored"]["vs_frozen_reference"] = compare(
                    reference[cohort], results["frozen_reference"]["_ref"][cohort])
        for name in VARIANTS:
            if name.endswith("primary_rescored"):
                continue
            cohort, requests, outputs = resolve(name)
            if not outputs.exists() or len(read_jsonl(outputs)) < len(reference[cohort]):
                block[name] = {"status": "missing_or_incomplete", "outputs": rel(outputs)}
                continue
            scored = score_rows(cohort, outputs, requests, reference_override=ref_for(cohort))
            entry: dict[str, Any] = {
                "outputs": rel(outputs),
                "outputs_sha256": sha256(outputs),
                "summary": summarize(scored),
                "vs_primary": compare(scored, reference[cohort]),
            }
            if "ontology" in name:
                harmonized = score_rows(cohort, outputs, requests, harmonize=True,
                                        reference_override=ref_for(cohort))
                entry["harmonized_target_summary"] = summarize(harmonized)
                entry["harmonized_vs_primary_harmonized"] = compare(harmonized, reference_h[cohort])
                entry["pred_target_distribution"] = dict(Counter(v["pred_target"] for v in scored.values()))
            if name in {"core_repeat_mps", "core_cpu_fp32"}:
                entry["raw_identity_vs_archived"] = raw_identity(outputs, COHORTS[cohort]["outputs"])
            if "gray" in name or "oracle" in name or "shuffled" in name:
                entry["pred_artifact_distribution"] = dict(Counter(v["pred_artifact"] for v in scored.values()))
                entry["pred_target_distribution"] = dict(Counter(v["pred_target"] for v in scored.values()))
                entry["pred_class_distribution"] = dict(Counter(v["pred_prediction"] for v in scored.values()))
                entry["semantic_gate_rate"] = round(100 * float(np.mean(
                    [v["semantic_gate"] for v in scored.values()])), 2)
                entry["semantic_gate_vs_primary"] = compare(
                    {k: {**v, "joint": v["semantic_gate"]} for k, v in scored.items()},
                    {k: {**v, "joint": v["semantic_gate"]} for k, v in reference[cohort].items()})
            write_jsonl(REV / f"scored_{name}_{mode}.jsonl",
                        [{"img_id": k, **v} for k, v in sorted(scored.items())])
            block[name] = entry
        block["_ref"] = reference
        results[mode] = block
    for mode in results:
        results[mode].pop("_ref")
    results["reference_correction"] = {
        c: {"changed_reports": sorted(k for k, v in corrected_reference_boxes(c).items() if v["changed"])}
        for c in COHORTS}
    write_json(REPORTS / "variant_scores.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False)[:6000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-requests").set_defaults(func=cmd_make_requests)
    sub.add_parser("score").set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
