#!/usr/bin/env python3
"""Zero-inference revision analyses (A1-A8 plus fixed-output IAA rescoring).

A1  Label-prior baselines that inherit B_top3 (constant, band-majority, prior-random)
A2  Within-stratum independence null for semantic-gate x spatial-gate co-passing
A3  Class-aware semantic diagnostics (majority baseline, macro recall, kappa, confusion)
A4  Target-ontology harmonization (person/animal -> object)
A5  JSG-Xfer inverse-probability reweighting to the natural candidate pools
A6  Label-free marginals on the complete external candidate pools
A7  Mask-pixel spatial references (coverage, purity, box-mask IoU)
A8  Model-emitted bbox readout under the proposal-conditioned prompt
IAA Fixed-output rescoring against second-pass labels (Core-20 and Scale-120)

Outputs: experiments/reports/revision/zero_inference_analyses.{json,md}
"""

from __future__ import annotations

import ast
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from revision_experiments import (  # noqa: E402
    COHORTS,
    corrected_reference_boxes,
    HARMONIZE,
    REPORTS,
    ROOT,
    RUNS,
    bbox_iou,
    exact_mcnemar,
    paired_bootstrap,
    read_jsonl,
    score_rows,
    write_json,
)
from stratified_statistics import mover_wilson_interval, wilson_interval  # noqa: E402

SEED = 20260718
REPS = 10000
T = 0.1
SEMANTIC = ("parse", "prediction", "artifact", "target")


# ----------------------------------------------------------------------------
# Unified per-report records
# ----------------------------------------------------------------------------

def load_core_scale(cohort: str, corrected: bool = False) -> list[dict[str, Any]]:
    boxes = corrected_reference_boxes(cohort)
    override = {k: v["corrected" if corrected else "frozen"] for k, v in boxes.items()}
    scored = score_rows(cohort, COHORTS[cohort]["outputs"], COHORTS[cohort]["requests"],
                        reference_override=override)
    records = []
    for img_id, v in sorted(scored.items()):
        records.append({
            "id": img_id,
            "stratum": v["stratum"],
            **{k: v["conditions"][k] for k in ("parse", "prediction", "artifact", "target", "spatial")},
            "iou": v["iou"],
            "candidate_iou": bbox_iou(v["candidate_bbox"], v["reference_bbox"]),
            "candidate_bbox": v["candidate_bbox"],
            "reference_bbox": v["reference_bbox"],
            "evaluated_bbox": v["evaluated_bbox"],
            "gold_artifact": v["gold_artifact"],
            "gold_target": v["gold_target"],
            "pred_artifact": v["pred_artifact"] if v["conditions"]["parse"] else "<unparsed>",
            "pred_target": v["pred_target"] if v["conditions"]["parse"] else "<unparsed>",
        })
    return records


def load_xfer() -> list[dict[str, Any]]:
    trace = read_jsonl(RUNS / "jsg_xfer_td_egfa_audit_trace_pixel5d_v2.jsonl")
    semantic = {r["img_id"]: r for r in read_jsonl(RUNS / "jsg_xfer_semantic_rows_pixel5d_v2.jsonl")}
    records = []
    for row in trace:
        sid = row["sample_id"]
        sem = semantic[sid]
        cond = row["audit_profiles"]["primary_strict"]["conditions"]
        acceptable = sem.get("gold_acceptable_artifacts")
        if isinstance(acceptable, str):
            acceptable = ast.literal_eval(acceptable)
        records.append({
            "id": sid,
            "stratum": row["sampling_stratum"],
            "source": row["external_dataset"],
            **{k: bool(cond[k]) for k in ("parse", "prediction", "artifact", "target", "spatial")},
            "iou": float(row["evaluated_region"]["iou"]),
            # prediction is tampered for every Xfer report, so the evaluated box is B_top3.
            "candidate_iou": float(row["evaluated_region"]["iou"]),
            "gold_artifact": sem["gold_artifact"],
            "gold_target": sem["gold_target"],
            "pred_artifact": sem["pred_artifact"] if cond["parse"] else "<unparsed>",
            "pred_target": sem["pred_target"] if cond["parse"] else "<unparsed>",
            "acceptable_artifacts": acceptable or [sem["gold_artifact"]],
        })
    if any(r["candidate_iou"] != r["iou"] or not r["prediction"] for r in records):
        raise AssertionError("Xfer assumption violated: every report must inherit B_top3")
    return records


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------

def weights(records: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(r["stratum"] for r in records)
    return {s: n / len(records) for s, n in counts.items()}


def rate(records: list[dict[str, Any]], fn: Callable[[dict[str, Any]], float]) -> float:
    """Benchmark-weighted rate (stratum weights proportional to fixed design counts)."""
    return float(np.mean([fn(r) for r in records]))


def joint(r: dict[str, Any]) -> bool:
    return all(r[k] for k in ("parse", "prediction", "artifact", "target", "spatial"))


def semantic_gate(r: dict[str, Any]) -> bool:
    return all(r[k] for k in SEMANTIC)


def pct(x: float) -> float:
    return round(100 * x, 2)


def ci_pct(ci: list[float] | tuple[float, float]) -> list[float]:
    return [round(100 * ci[0], 2), round(100 * ci[1], 2)]


def binary_summary(records: list[dict[str, Any]], fn: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    values = [float(fn(r)) for r in records]
    return {"rate": pct(np.mean(values)), "count": int(sum(values)),
            "ci95_wilson_mover": ci_pct(mover_wilson_interval(values, [r["stratum"] for r in records]))}


def paired(records: list[dict[str, Any]], fa: Callable, fb: Callable) -> dict[str, Any]:
    a = {r["id"]: float(fa(r)) for r in records}
    b = {r["id"]: float(fb(r)) for r in records}
    strata = {r["id"]: r["stratum"] for r in records}
    point, lo, hi = paired_bootstrap(a, b, strata, reps=REPS, seed=SEED)
    pa = {k for k, v in a.items() if v}
    pb = {k for k, v in b.items() if v}
    gains, losses = len(pa - pb), len(pb - pa)
    return {
        "delta_pp": pct(point), "delta_ci95_pp": ci_pct((lo, hi)),
        "a_only": gains, "b_only": losses, "both": len(pa & pb),
        "exact_mcnemar_p": round(exact_mcnemar(gains, losses), 4),
        "passing_set_jaccard": round(len(pa & pb) / len(pa | pb), 3) if pa | pb else 1.0,
    }


def frechet(records: list[dict[str, Any]]) -> list[float]:
    m = [rate(records, lambda r, k=k: float(r[k])) for k in ("prediction", "artifact", "target", "spatial")]
    return [pct(max(0.0, sum(m) - 3)), pct(min(m))]


# ----------------------------------------------------------------------------
# A1  Label-prior baselines
# ----------------------------------------------------------------------------

def a1_baselines(records: list[dict[str, Any]]) -> dict[str, Any]:
    art_major = Counter(r["gold_artifact"] for r in records).most_common(1)[0][0]
    tgt_major = Counter(r["gold_target"] for r in records).most_common(1)[0][0]
    cand_pass = lambda r: r["candidate_iou"] >= T  # noqa: E731

    def constant(r: dict[str, Any]) -> bool:
        return r["gold_artifact"] == art_major and r["gold_target"] == tgt_major and cand_pass(r)

    band_art, band_tgt = {}, {}
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_band[r["stratum"]].append(r)
    for band, rows in by_band.items():
        band_art[band] = Counter(r["gold_artifact"] for r in rows).most_common(1)[0][0]
        band_tgt[band] = Counter(r["gold_target"] for r in rows).most_common(1)[0][0]

    def band_major(r: dict[str, Any]) -> bool:
        return r["gold_artifact"] == band_art[r["stratum"]] and r["gold_target"] == band_tgt[r["stratum"]] and cand_pass(r)

    # Prior-random: labels drawn independently from the cohort gold marginals.
    pa = Counter(r["gold_artifact"] for r in records)
    pt = Counter(r["gold_target"] for r in records)
    n = len(records)
    expected = np.mean([cand_pass(r) * pa[r["gold_artifact"]] / n * pt[r["gold_target"]] / n for r in records])
    rng = np.random.default_rng(SEED)
    arts, tgts = list(pa), list(pt)
    pa_v = np.array([pa[a] / n for a in arts])
    pt_v = np.array([pt[t] / n for t in tgts])
    gold_a = np.array([arts.index(r["gold_artifact"]) for r in records])
    gold_t = np.array([tgts.index(r["gold_target"]) for r in records])
    cp = np.array([cand_pass(r) for r in records])
    draws = []
    for _ in range(2000):
        sa = rng.choice(len(arts), size=n, p=pa_v)
        st = rng.choice(len(tgts), size=n, p=pt_v)
        draws.append(np.mean((sa == gold_a) & (st == gold_t) & cp))
    return {
        "majority_labels": {"artifact": art_major, "target": tgt_major},
        "band_majority_labels": {b: [band_art[b], band_tgt[b]] for b in sorted(band_art)},
        "model_joint": binary_summary(records, joint),
        "constant_label_joint": binary_summary(records, constant),
        "model_minus_constant": paired(records, joint, constant),
        "band_majority_joint": binary_summary(records, band_major),
        "model_minus_band_majority": paired(records, joint, band_major),
        "prior_random_joint_expected": pct(expected),
        "prior_random_joint_95range": ci_pct(np.percentile(draws, [2.5, 97.5])),
        "note": "Baselines use gold label frequencies (oracle prior) and inherit B_top3 for every report.",
    }


# ----------------------------------------------------------------------------
# A2  Independence null
# ----------------------------------------------------------------------------

def a2_independence(records: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_band[r["stratum"]].append(r)
    sem = {b: np.array([semantic_gate(r) for r in rows], dtype=float) for b, rows in by_band.items()}
    sp = {b: np.array([r["spatial"] for r in rows], dtype=float) for b, rows in by_band.items()}
    n = len(records)
    observed = sum((sem[b] * sp[b]).sum() for b in by_band) / n
    expected = sum(len(by_band[b]) * sem[b].mean() * sp[b].mean() for b in by_band) / n
    null = np.zeros(REPS)
    for b in by_band:
        s = sem[b]
        perm = np.array([rng.permutation(s) for _ in range(REPS)])
        null += (perm * sp[b]).sum(axis=1)
    null /= n
    p_two = float(np.mean(np.abs(null - expected) >= abs(observed - expected) - 1e-12))
    return {
        "observed_joint": pct(observed),
        "independence_expectation": pct(expected),
        "permutation_null_95": ci_pct(np.percentile(null, [2.5, 97.5])),
        "two_sided_permutation_p": round(p_two, 4),
        "frechet_bounds": frechet(records),
        "semantic_gate_rate": pct(rate(records, lambda r: float(semantic_gate(r)))),
        "spatial_gate_rate": pct(rate(records, lambda r: float(r["spatial"]))),
    }


# ----------------------------------------------------------------------------
# A3  Class-aware diagnostics
# ----------------------------------------------------------------------------

def cohen_kappa(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    n = len(gold)
    po = sum(g == p for g, p in zip(gold, pred)) / n
    cg, cp = Counter(gold), Counter(pred)
    pe = sum(cg[label] * cp[label] for label in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def field_diagnostics(records: list[dict[str, Any]], field: str, mapper: Callable[[str], str] | None = None) -> dict[str, Any]:
    m = mapper or (lambda x: x)
    gold = [m(r[f"gold_{field}"]) for r in records]
    pred = [m(r[f"pred_{field}"]) for r in records]
    support = Counter(gold)
    per_class = {}
    for label, count in support.most_common():
        hit = sum(g == label and p == label for g, p in zip(gold, pred))
        per_class[label] = {"support": count, "recall": pct(hit / count), "emitted": sum(p == label for p in pred)}
    majority_label, majority_count = support.most_common(1)[0]
    macro = np.mean([v["recall"] for v in per_class.values()])
    non_major = [v["recall"] for k, v in per_class.items() if k != majority_label]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gold, pred):
        confusion[g][p] += 1
    return {
        "accuracy": pct(np.mean([g == p for g, p in zip(gold, pred)])),
        "majority_label": majority_label,
        "majority_baseline": pct(majority_count / len(gold)),
        "macro_recall": round(float(macro), 2),
        "non_majority_macro_recall": round(float(np.mean(non_major)), 2) if non_major else None,
        "cohen_kappa_vs_gold": round(cohen_kappa(gold, pred), 3),
        "per_class": per_class,
        "confusion_gold_to_pred": {g: dict(v) for g, v in confusion.items()},
    }


def harm(label: str) -> str:
    return HARMONIZE.get(label, label)


# ----------------------------------------------------------------------------
# A4  Harmonized target ontology
# ----------------------------------------------------------------------------

def a4_harmonized(records: list[dict[str, Any]]) -> dict[str, Any]:
    def tgt_h(r: dict[str, Any]) -> bool:
        return r["parse"] and harm(r["pred_target"]) == harm(r["gold_target"])

    def joint_h(r: dict[str, Any]) -> bool:
        return r["parse"] and r["prediction"] and r["artifact"] and tgt_h(r) and r["spatial"]

    return {
        "target_raw": pct(rate(records, lambda r: float(r["target"]))),
        "target_harmonized": pct(rate(records, lambda r: float(tgt_h(r)))),
        "joint_raw": binary_summary(records, joint),
        "joint_harmonized": binary_summary(records, joint_h),
        "harmonized_minus_raw": paired(records, joint_h, joint),
        "target_diagnostics_harmonized": field_diagnostics(records, "target", harm),
    }


# ----------------------------------------------------------------------------
# A5 / A6  External pools
# ----------------------------------------------------------------------------

POOLS = {
    "CASIA2": RUNS / "casia2_candidate996_top3_predictions.jsonl",
    "IMD2020": RUNS / "imd2020_candidate2010_top3_predictions.jsonl",
}
BANDS = ["candidate_iou<0.1", "candidate_iou[0.1,0.3)", "candidate_iou>=0.3"]


def band_of(iou: float) -> str:
    return BANDS[0] if iou < 0.1 else BANDS[1] if iou < 0.3 else BANDS[2]


def a5_a6_external(xfer: list[dict[str, Any]]) -> dict[str, Any]:
    pools = {src: read_jsonl(path) for src, path in POOLS.items()}
    selection = {r["img_id"]: r for r in read_jsonl(RUNS / "external_semantic_eval_rows.jsonl")}
    shares, pool_rows = {}, {}
    for src, rows in pools.items():
        counts = Counter(band_of(float(r.get("bbox_iou") or 0.0)) for r in rows)
        shares[src] = {b: counts[b] / len(rows) for b in BANDS}
        n = len(rows)
        tp = sum(r.get("prediction") == "tampered" for r in rows)
        ious = [float(r.get("bbox_iou") or 0.0) for r in rows]
        pool_rows[src] = {
            "n": n,
            "tampered_recall": pct(tp / n), "tampered_recall_ci95": ci_pct(wilson_interval(tp, n)),
            "mean_iou": round(float(np.mean(ious)), 4),
            **{f"bbox_pass@{t}": pct(np.mean([i >= t for i in ious])) for t in (0.1, 0.3, 0.5)},
            **{f"bbox_pass@{t}_ci95": ci_pct(wilson_interval(sum(i >= t for i in ious), n)) for t in (0.1, 0.3, 0.5)},
            "band_shares": {b: round(shares[src][b], 4) for b in BANDS},
        }
    # Selection consistency: Xfer stratum must equal the band of the pool IoU.
    pool_iou = {r["img_id"]: float(r.get("bbox_iou") or 0.0) for rows in pools.values() for r in rows}
    band_mismatch = [r["id"] for r in xfer if r["stratum"].split("|")[1] != band_of(pool_iou[r["id"]])]
    iou_mismatch = [k for k in selection if abs(float(selection[k].get("bbox_iou") or 0) - pool_iou[k]) > 1e-9]

    metrics: dict[str, Callable[[dict[str, Any]], float]] = {
        "joint": lambda r: float(joint(r)),
        "artifact": lambda r: float(r["artifact"]),
        "target": lambda r: float(r["target"]),
        "bbox_pass@0.1": lambda r: float(r["spatial"]),
    }
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in xfer:
        by_cell[r["stratum"]].append(r)

    def ipw(src: str, fn: Callable, cells: dict[str, list[dict[str, Any]]]) -> float:
        return sum(shares[src][b] * np.mean([fn(r) for r in cells[f"{src}|{b}"]]) for b in BANDS)

    def design(src: str, fn: Callable) -> float:
        return float(np.mean([fn(r) for r in xfer if r["source"] == src]))

    rng = np.random.default_rng(SEED)
    boot: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(REPS):
        cells = {k: [v[i] for i in rng.integers(0, len(v), len(v))] for k, v in by_cell.items()}
        for src in POOLS:
            for name, fn in metrics.items():
                boot[(src, name)].append(ipw(src, fn, cells))
        for name, fn in metrics.items():
            boot[("pooled_size_weighted", name)].append(
                sum(len(pools[s]) * ipw(s, fn, cells) for s in POOLS) / sum(len(p) for p in pools.values()))
    reweighted = {}
    for src in list(POOLS) + ["pooled_size_weighted"]:
        reweighted[src] = {}
        for name, fn in metrics.items():
            if src == "pooled_size_weighted":
                point = sum(len(pools[s]) * ipw(s, fn, by_cell) for s in POOLS) / sum(len(p) for p in pools.values())
                design_value = float(np.mean([fn(r) for r in xfer]))
            else:
                point, design_value = ipw(src, fn, by_cell), design(src, fn)
            reweighted[src][name] = {
                "design_weighted": pct(design_value),
                "pool_reweighted": pct(point),
                "pool_reweighted_ci95": ci_pct(np.percentile(boot[(src, name)], [2.5, 97.5])),
            }
    return {
        "pool_label_free_marginals": pool_rows,
        "selection_consistency": {"band_mismatches": band_mismatch,
                                  "selection_vs_pool_iou_mismatches": len(iou_mismatch)},
        "inverse_probability_reweighting": reweighted,
        "note": ("Pool bands use the legacy proposal IoU that defined the Xfer sampling strata. "
                 "Reweighting assumes within-band exchangeability between selected and unselected reports."),
    }


# ----------------------------------------------------------------------------
# A7  Mask-pixel spatial references
# ----------------------------------------------------------------------------

def load_mask(cohort: str) -> dict[str, np.ndarray]:
    ann = {r["img_id"]: r for r in read_jsonl(COHORTS[cohort]["annotations"])}
    masks = {}
    for img_id, row in ann.items():
        with Image.open(ROOT / "SID_Set" / row["mask_path"]) as im:
            arr = np.asarray(im.convert("L")) > 127
        with Image.open(ROOT / "SID_Set" / row["image_path"]) as image:
            size = image.size
        if (arr.shape[1], arr.shape[0]) != size:
            # Resized masks are mapped back onto the image grid (see reference correction).
            with Image.open(ROOT / "SID_Set" / row["mask_path"]) as im:
                arr = np.asarray(im.convert("L").resize(size, Image.NEAREST)) > 127
        masks[img_id] = arr
    return masks


def box_mask_stats(box: list[int] | None, mask: np.ndarray) -> tuple[float, float, float]:
    total = mask.sum()
    if not box or total == 0:
        return 0.0, 0.0, 0.0
    x1, y1, x2, y2 = box
    inside = mask[y1:y2, x1:x2].sum()
    area = (x2 - x1) * (y2 - y1)
    coverage = inside / total
    purity = inside / area if area else 0.0
    iou = inside / (area + total - inside)
    return float(coverage), float(purity), float(iou)


def a7_mask_reference(records: list[dict[str, Any]], cohort: str) -> dict[str, Any]:
    masks = load_mask(cohort)
    rows = []
    for r in records:
        cov, pur, miou = box_mask_stats(r["evaluated_bbox"], masks[r["id"]])
        rows.append({**r, "mask_coverage": cov, "mask_purity": pur, "mask_iou": miou})

    def sem(r: dict[str, Any]) -> bool:
        return semantic_gate(r)

    variants = {
        "bbox_iou>=0.1 (primary)": lambda r: sem(r) and r["spatial"],
        "box-mask IoU>=0.1": lambda r: sem(r) and r["mask_iou"] >= 0.1,
        "mask coverage>=0.5": lambda r: sem(r) and r["mask_coverage"] >= 0.5,
        "mask coverage>=0.5 and purity>=0.1": lambda r: sem(r) and r["mask_coverage"] >= 0.5 and r["mask_purity"] >= 0.1,
    }
    primary = variants["bbox_iou>=0.1 (primary)"]
    out = {
        "mean_mask_coverage": round(float(np.mean([r["mask_coverage"] for r in rows])), 4),
        "mean_mask_purity": round(float(np.mean([r["mask_purity"] for r in rows])), 4),
        "mean_box_mask_iou": round(float(np.mean([r["mask_iou"] for r in rows])), 4),
        "spatial_rates": {
            "bbox_iou>=0.1": pct(np.mean([r["spatial"] for r in rows])),
            "box-mask IoU>=0.1": pct(np.mean([r["mask_iou"] >= 0.1 for r in rows])),
            "mask coverage>=0.5": pct(np.mean([r["mask_coverage"] >= 0.5 for r in rows])),
        },
        "joint_variants": {},
    }
    for name, fn in variants.items():
        entry = binary_summary(rows, fn)
        if name != "bbox_iou>=0.1 (primary)":
            entry["vs_primary"] = paired(rows, fn, primary)
        out["joint_variants"][name] = entry
    return out


# ----------------------------------------------------------------------------
# A8  Model-emitted bbox
# ----------------------------------------------------------------------------

def a8_emitted_bbox(cohort: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = {r["img_id"]: r for r in read_jsonl(COHORTS[cohort]["outputs"])}
    rec = {r["id"]: r for r in records}
    equal = differs = missing = 0
    emitted_pass = 0
    for img_id, r in rec.items():
        result = outputs[img_id].get("vlm_result") or {}
        box = result.get("evidence_bbox")
        valid = isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)
        if not valid:
            missing += 1
            continue
        box = [int(round(v)) for v in box]
        if box == r["candidate_bbox"]:
            equal += 1
        else:
            differs += 1
        emitted_pass += int(semantic_gate(r) and bbox_iou(box, r["reference_bbox"]) >= T)
    n = len(rec)
    return {
        "emitted_equals_candidate": equal, "emitted_differs": differs, "emitted_missing_or_invalid": missing,
        "joint_with_emitted_bbox": pct(emitted_pass / n),
        "note": "The frozen prompt instructs the model to copy candidate_bbox, so this readout is near-degenerate.",
    }


# ----------------------------------------------------------------------------
# IAA fixed-output rescoring
# ----------------------------------------------------------------------------

def iaa_core(core: list[dict[str, Any]]) -> dict[str, Any]:
    rec = {r["id"]: r for r in core}
    second = [r for r in read_jsonl(ROOT / "SID_Set/annotations/iaa/SID-Explain-IAA-60_template.jsonl")
              if r["human_label"] == "tampered"]
    first_space = ["none", "face", "object", "background", "text", "whole_image", "other"]
    second_space = sorted(set(second[0]["allowed_target_scopes"]))
    g1t = [rec[r["source_img_id"]]["gold_target"] for r in second]
    g2t = [r["target_scope"] for r in second]
    g1a = [rec[r["source_img_id"]]["gold_artifact"] for r in second]
    g2a = [r["dominant_artifact_type"] for r in second]
    disagreements = Counter(f"{a}->{b}" for a, b in zip(g1t, g2t) if a != b)

    def joint_with(rid: str, art: str, tgt: str, mapper: Callable[[str], str] = lambda x: x) -> bool:
        r = rec[rid]
        return bool(r["parse"] and r["prediction"] and r["spatial"]
                    and r["pred_artifact"] == art and mapper(r["pred_target"]) == mapper(tgt))

    face_person = lambda x: "face_or_person" if x in {"face", "person"} else x  # noqa: E731
    ids = [r["source_img_id"] for r in second]
    return {
        "n": len(second),
        "first_pass_target_space": first_space,
        "second_pass_target_space": second_space,
        "target_agreement": pct(np.mean([a == b for a, b in zip(g1t, g2t)])),
        "target_kappa": round(cohen_kappa(g1t, g2t), 3),
        "target_disagreements": dict(disagreements),
        "target_agreement_where_second_pass_label_exists_in_first_space": (
            f"{sum(a == b for a, b in zip(g1t, g2t) if b in first_space)}/{sum(b in first_space for b in g2t)}"),
        "target_agreement_post_hoc_person_to_object": pct(np.mean([harm(a) == harm(b) for a, b in zip(g1t, g2t)])),
        "artifact_agreement": pct(np.mean([a == b for a, b in zip(g1a, g2a)])),
        "artifact_kappa": round(cohen_kappa(g1a, g2a), 3),
        "artifact_disagreements": dict(Counter(f"{a}->{b}" for a, b in zip(g1a, g2a) if a != b)),
        "joint_first_pass": sum(joint(rec[i]) for i in ids),
        "joint_second_pass": sum(joint_with(i, a, t) for i, a, t in zip(ids, g2a, g2t)),
        "joint_second_pass_target_harmonized_person_object": sum(
            joint_with(i, a, t, harm) for i, a, t in zip(ids, g2a, g2t)),
        "joint_second_pass_artifact_only_change": sum(
            joint_with(i, a, rec[i]["gold_target"]) for i, a in zip(ids, g2a)),
    }


def iaa_scale(scale: list[dict[str, Any]]) -> dict[str, Any]:
    rec = {r["id"]: r for r in scale}
    second = read_jsonl(ROOT / "SID_Set/annotations/iaa/SID-FA-IAA-120_cross60x2_template.jsonl")
    fix = {"perosn": "person"}
    ids = [r["source_img_id"] for r in second]
    g2t = [fix.get(r["target_scope"], r["target_scope"]) for r in second]
    g2a = [r["dominant_artifact_type"] for r in second]
    g1t = [rec[i]["gold_target"] for i in ids]
    g1a = [rec[i]["gold_artifact"] for i in ids]

    def joint_with(rid: str, art: str, tgt: str) -> bool:
        r = rec[rid]
        return bool(r["parse"] and r["prediction"] and r["spatial"]
                    and r["pred_artifact"] == art and r["pred_target"] == tgt)

    first_pass = {i for i in ids if joint(rec[i])}
    second_pass = {i for i, a, t in zip(ids, g2a, g2t) if joint_with(i, a, t)}
    return {
        "n": len(second),
        "target_agreement": pct(np.mean([a == b for a, b in zip(g1t, g2t)])),
        "target_kappa": round(cohen_kappa(g1t, g2t), 3),
        "artifact_agreement": pct(np.mean([a == b for a, b in zip(g1a, g2a)])),
        "artifact_kappa": round(cohen_kappa(g1a, g2a), 3),
        "joint_frozen_labels": len(first_pass),
        "joint_second_annotator_labels": len(second_pass),
        "passing_set_jaccard": round(len(first_pass & second_pass) / len(first_pass | second_pass), 3),
        "note": "Both annotation forms offered eight targets. Normalizing one 'perosn' to 'person' yields 115/120 agreements.",
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    report: dict[str, Any] = {"seed": SEED, "bootstrap_reps": REPS, "threshold": T}
    xfer = load_xfer()
    for mode, corrected in (("frozen_reference", False), ("corrected_reference", True)):
        data = {"JSG-Core": load_core_scale("core", corrected),
                "JSG-Scale": load_core_scale("scale", corrected), "JSG-Xfer": xfer}
        block: dict[str, Any] = {}
        for name, records in data.items():
            block[name] = {
                "primary": {k: binary_summary(records, (lambda r, k=k: r[k]) if k != "joint" else joint)
                            for k in ("prediction", "artifact", "target", "spatial", "joint")},
                "frechet_bounds": frechet(records),
                "A1_label_prior_baselines": a1_baselines(records),
                "A2_independence_null": a2_independence(records),
                "A3_artifact_diagnostics": field_diagnostics(records, "artifact"),
                "A3_target_diagnostics": field_diagnostics(records, "target"),
                "A4_harmonized_target": a4_harmonized(records),
            }
        for name, cohort in (("JSG-Core", "core"), ("JSG-Scale", "scale")):
            block[name]["A7_mask_reference"] = a7_mask_reference(data[name], cohort)
            block[name]["A8_emitted_bbox"] = a8_emitted_bbox(cohort, data[name])
        block["IAA_fixed_output"] = {"JSG-Core-20": iaa_core(data["JSG-Core"]),
                                     "JSG-Scale-120": iaa_scale(data["JSG-Scale"])}
        report[mode] = block
    report["JSG-Xfer_external_pools"] = a5_a6_external(xfer)
    write_json(REPORTS / "zero_inference_analyses.json", report)
    print("wrote", REPORTS / "zero_inference_analyses.json")


if __name__ == "__main__":
    main()
