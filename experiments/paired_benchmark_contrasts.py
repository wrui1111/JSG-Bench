#!/usr/bin/env python3
"""Stratified paired Core interface and Scale proposal-policy contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from stratified_statistics import (
    core_strata_by_id,
    is_binary_vector,
    mover_wilson_interval,
    scale_strata_by_id,
    stratified_percentile_interval_list,
    stratum_counts,
    validate_expected_strata,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
REPORTS = ROOT / "experiments" / "reports"

CORE_FREE_FILE = RUNS / "sid_explain300_evidence_explanation_v2_revised_predictions.jsonl"
CORE_CONDITIONED_FILE = RUNS / "jsg_core_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl"
SCALE_THRESHOLD_FILE = RUNS / "sid_fa600_candidate_top1_postreview_predictions.jsonl"
SCALE_TOP3_FILE = RUNS / "jsg_scale_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl"

METRICS = (
    "parse_rate",
    "prediction_accuracy",
    "artifact_accuracy",
    "target_accuracy",
    "mean_iou",
    "iou_at_0.1",
    "iou_at_0.3",
    "iou_at_0.5",
    "joint_at_0.1",
    "joint_at_0.3",
    "joint_at_0.5",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: str) -> int:
    key = "\0".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def parsed(row: dict[str, Any]) -> bool:
    prediction = row.get("prediction") or {}
    return bool(row.get("parsed", prediction.get("parsed", True)))


def metric_values(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    output: dict[str, list[float]] = {name: [] for name in METRICS}
    for row in rows:
        parse_pass = parsed(row)
        prediction_pass = bool(row.get("prediction_correct"))
        artifact_pass = bool(row.get("artifact_correct"))
        target_pass = bool(row.get("target_correct"))
        iou = float(row.get("bbox_iou") or 0.0)
        semantic_gate = parse_pass and prediction_pass and artifact_pass and target_pass
        output["parse_rate"].append(float(parse_pass))
        output["prediction_accuracy"].append(float(prediction_pass))
        output["artifact_accuracy"].append(float(artifact_pass))
        output["target_accuracy"].append(float(target_pass))
        output["mean_iou"].append(iou)
        for threshold in (0.1, 0.3, 0.5):
            spatial = iou >= threshold
            output[f"iou_at_{threshold}"].append(float(spatial))
            output[f"joint_at_{threshold}"].append(float(semantic_gate and spatial))
    return {name: np.asarray(values, dtype=np.float64) for name, values in output.items()}


def summarize(
    values: dict[str, np.ndarray],
    strata: list[str],
    seed: int,
    reps: int,
    context: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, items in values.items():
        if name != "mean_iou" and is_binary_vector(items):
            interval = mover_wilson_interval(items, strata)
            method = "stratum-specific Wilson intervals combined by MOVER"
        else:
            interval = stratified_percentile_interval_list(
                items,
                strata,
                stable_seed(seed, context, name),
                reps,
            )
            method = "stratified percentile bootstrap over matched IDs"
        output[name] = {
            "estimate": float(items.mean()),
            "ci_95": interval,
            "interval_method": method,
        }
    return output


def paired_differences(
    threshold_values: dict[str, np.ndarray],
    top3_values: dict[str, np.ndarray],
    strata: list[str],
    seed: int,
    reps: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in METRICS:
        difference = top3_values[name] - threshold_values[name]
        output[name] = {
            "estimate": float(difference.mean()),
            "paired_stratified_bootstrap_ci_95": stratified_percentile_interval_list(
                difference,
                strata,
                stable_seed(seed, "B_top3-minus-B_thr", name),
                reps,
            ),
            "interval_method": "paired stratified percentile bootstrap over matched IDs",
        }
    return output


def mcnemar_exact(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    smaller = min(first_only, second_only)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * tail * (0.5 ** discordant))


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stratified Paired Benchmark Contrasts",
        "",
        "## JSG-Core free versus candidate-conditioned interface",
        "",
        "The same 100 Core IDs are compared. Matched IDs are resampled within the four "
        "mask-area bands; the contrast changes the complete reporting interface and is not "
        "a single-component causal effect.",
        "",
        "| Interface | N | Parse | Prediction | Artifact | Target | Mean IoU | IoU@0.1 | Joint@0.1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    core = report["contrasts"]["JSG-Core_interface"]
    for key, label in (("free", "Free evidence card"), ("conditioned", "B_top3-conditioned")):
        item = core["arms"][key]
        metrics = item["metrics"]
        lines.append(
            f"| {label} | {item['samples']} | {percent(metrics['parse_rate']['estimate'])} | "
            f"{percent(metrics['prediction_accuracy']['estimate'])} | "
            f"{percent(metrics['artifact_accuracy']['estimate'])} | "
            f"{percent(metrics['target_accuracy']['estimate'])} | "
            f"{metrics['mean_iou']['estimate']:.4f} | "
            f"{percent(metrics['iou_at_0.1']['estimate'])} | "
            f"{percent(metrics['joint_at_0.1']['estimate'])} |"
        )
    core_joint = core["paired_differences"]["joint_at_0.1"]
    core_low, core_high = core_joint["paired_stratified_bootstrap_ci_95"]
    lines.extend(
        [
            "",
            f"Conditioned minus free Joint@0.1: {percent(core_joint['estimate'])} pp "
            f"[{percent(core_low)}, {percent(core_high)}]; exact McNemar "
            f"p={core['joint_at_0.1_mcnemar_exact_p']:.12g}.",
            "",
            "## JSG-Scale thresholded-box versus top-3 policy",
        "",
        "The comparison is conditional on the 544 frozen IDs for which the thresholded box is non-null. "
        "Matched IDs are resampled within mask-area bands, preserving the common-set band counts.",
        "",
        "| Policy | N | Parse | Prediction | Artifact | Target | Mean IoU | IoU@0.1 | IoU@0.3 | Joint@0.1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    scale = report["contrasts"]["JSG-Scale_policy"]
    for key, label in (("B_thr", "B_thr"), ("B_top3", "B_top3")):
        item = scale["arms"][key]
        metrics = item["metrics"]
        lines.append(
            f"| {label} | {item['samples']} | {percent(metrics['parse_rate']['estimate'])} | "
            f"{percent(metrics['prediction_accuracy']['estimate'])} | "
            f"{percent(metrics['artifact_accuracy']['estimate'])} | "
            f"{percent(metrics['target_accuracy']['estimate'])} | "
            f"{metrics['mean_iou']['estimate']:.4f} | "
            f"{percent(metrics['iou_at_0.1']['estimate'])} | "
            f"{percent(metrics['iou_at_0.3']['estimate'])} | "
            f"{percent(metrics['joint_at_0.1']['estimate'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired B_top3 minus B_thr differences",
            "",
            "| Metric | Difference | 95% stratified bootstrap interval |",
            "|---|---:|---:|",
        ]
    )
    for name in ("mean_iou", "iou_at_0.1", "iou_at_0.3", "joint_at_0.1"):
        item = scale["paired_differences"][name]
        low, high = item["paired_stratified_bootstrap_ci_95"]
        if name == "mean_iou":
            estimate_text = f"{item['estimate']:.4f}"
            interval_text = f"[{low:.4f}, {high:.4f}]"
        else:
            estimate_text = percent(item["estimate"])
            interval_text = f"[{percent(low)}, {percent(high)}] pp"
        lines.append(f"| {name} | {estimate_text} | {interval_text} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS / "paired_benchmark_contrasts.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPORTS / "paired_benchmark_contrasts.md",
    )
    args = parser.parse_args()
    if args.bootstrap_reps <= 0:
        parser.error("--bootstrap-reps must be positive")

    core_free_map = {str(row["img_id"]): row for row in read_jsonl(CORE_FREE_FILE)}
    core_conditioned_map = {
        str(row["img_id"]): row for row in read_jsonl(CORE_CONDITIONED_FILE)
    }
    if len(core_free_map) != 300 or len(core_conditioned_map) != 100:
        raise ValueError("Expected 300 free-interface and 100 conditioned Core records")
    core_ids = sorted(core_conditioned_map)
    if not set(core_ids).issubset(core_free_map):
        raise ValueError("Every conditioned Core ID must have a paired free-interface record")
    core_stratum_map = core_strata_by_id()
    core_strata = [core_stratum_map[img_id] for img_id in core_ids]
    validate_expected_strata("JSG-Core", core_strata)
    core_free_values = metric_values([core_free_map[img_id] for img_id in core_ids])
    core_conditioned_values = metric_values(
        [core_conditioned_map[img_id] for img_id in core_ids]
    )
    core_paired = paired_differences(
        core_free_values,
        core_conditioned_values,
        core_strata,
        args.seed,
        args.bootstrap_reps,
    )
    core_free_joint = core_free_values["joint_at_0.1"]
    core_conditioned_joint = core_conditioned_values["joint_at_0.1"]
    conditioned_only = int(
        np.sum((core_conditioned_joint == 1.0) & (core_free_joint == 0.0))
    )
    free_only = int(
        np.sum((core_conditioned_joint == 0.0) & (core_free_joint == 1.0))
    )

    threshold_map = {
        str(row["img_id"]): row for row in read_jsonl(SCALE_THRESHOLD_FILE)
    }
    top3_map = {str(row["img_id"]): row for row in read_jsonl(SCALE_TOP3_FILE)}
    if len(threshold_map) != 544 or len(top3_map) != 600:
        raise ValueError("Expected 544 thresholded-box and 600 top-3 records")
    common_ids = sorted(set(threshold_map) & set(top3_map))
    if len(common_ids) != 544 or set(common_ids) != set(threshold_map):
        raise ValueError("Thresholded-box IDs must be the complete 544-ID common set")

    scale_stratum_map = scale_strata_by_id()
    scale_strata = [scale_stratum_map[img_id] for img_id in common_ids]
    scale_counts = stratum_counts(scale_strata)
    threshold_rows = [threshold_map[img_id] for img_id in common_ids]
    top3_rows = [top3_map[img_id] for img_id in common_ids]
    threshold_values = metric_values(threshold_rows)
    top3_values = metric_values(top3_rows)

    report = {
        "stage": "stratified_paired_benchmark_contrasts",
        "configuration": {
            "seed": args.seed,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_unit": "matched img_id resampled within mask-area band",
            "binary_interval": "stratum-specific Wilson intervals combined by MOVER",
            "continuous_and_paired_interval": "stratified percentile bootstrap",
        },
        "contrasts": {
            "JSG-Core_interface": {
                "sampling_design": {
                    "estimand": "paired interface difference under the frozen equal-band Core mix",
                    "stratification": "mask-area band",
                    "stratum_counts": stratum_counts(core_strata),
                    "design_weights": {
                        key: value / len(core_ids)
                        for key, value in stratum_counts(core_strata).items()
                    },
                    "population_inference": False,
                },
                "inputs": {
                    "free": {
                        "path": str(CORE_FREE_FILE.relative_to(ROOT)),
                        "sha256": sha256(CORE_FREE_FILE),
                    },
                    "conditioned": {
                        "path": str(CORE_CONDITIONED_FILE.relative_to(ROOT)),
                        "sha256": sha256(CORE_CONDITIONED_FILE),
                    },
                },
                "arms": {
                    "free": {
                        "samples": len(core_ids),
                        "metrics": summarize(
                            core_free_values,
                            core_strata,
                            args.seed,
                            args.bootstrap_reps,
                            "Core/free",
                        ),
                    },
                    "conditioned": {
                        "samples": len(core_ids),
                        "metrics": summarize(
                            core_conditioned_values,
                            core_strata,
                            args.seed,
                            args.bootstrap_reps,
                            "Core/conditioned",
                        ),
                    },
                },
                "paired_differences": core_paired,
                "joint_at_0.1_discordance": {
                    "conditioned_only": conditioned_only,
                    "free_only": free_only,
                },
                "joint_at_0.1_mcnemar_exact_p": mcnemar_exact(
                    conditioned_only,
                    free_only,
                ),
                "interpretation": "complete interface contrast, not a single-component causal effect",
            },
            "JSG-Scale_policy": {
                "sampling_design": {
                    "estimand": (
                        "performance conditional on threshold-policy coverage in the frozen "
                        "benchmark mix"
                    ),
                    "stratification": "mask-area band",
                    "stratum_counts": scale_counts,
                    "design_weights": {
                        key: value / len(common_ids) for key, value in scale_counts.items()
                    },
                    "population_inference": False,
                },
                "inputs": {
                    "B_thr": {
                        "path": str(SCALE_THRESHOLD_FILE.relative_to(ROOT)),
                        "sha256": sha256(SCALE_THRESHOLD_FILE),
                        "legacy_file_label": "candidate_top1",
                    },
                    "B_top3": {
                        "path": str(SCALE_TOP3_FILE.relative_to(ROOT)),
                        "sha256": sha256(SCALE_TOP3_FILE),
                    },
                },
                "arms": {
                    "B_thr": {
                        "samples": len(common_ids),
                        "metrics": summarize(
                            threshold_values,
                            scale_strata,
                            args.seed,
                            args.bootstrap_reps,
                            "Scale/B_thr",
                        ),
                    },
                    "B_top3": {
                        "samples": len(common_ids),
                        "metrics": summarize(
                            top3_values,
                            scale_strata,
                            args.seed,
                            args.bootstrap_reps,
                            "Scale/B_top3",
                        ),
                    },
                },
                "paired_differences": paired_differences(
                    threshold_values,
                    top3_values,
                    scale_strata,
                    args.seed,
                    args.bootstrap_reps,
                ),
            },
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "md": str(args.output_md)}))


if __name__ == "__main__":
    main()
