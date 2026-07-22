#!/usr/bin/env python3
"""Offline spatial-construct controls for frozen JSG-Core/Scale reports.

The script never runs model inference. It keeps each frozen report's parse and
semantic decisions fixed while replacing only its proposal box with deterministic
null boxes. Random-box placement is sampled without consulting the GT box.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from stratified_statistics import (
    design_metadata,
    is_binary_vector,
    mover_wilson_interval,
    stratified_percentile_interval,
    stratified_percentile_interval_list,
    validate_expected_strata,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
REPORTS = ROOT / "experiments" / "reports"
THRESHOLDS = (0.1, 0.3, 0.5)
CURVE_THRESHOLDS = np.linspace(0.0, 1.0, 101, dtype=np.float64)

DATASETS = {
    "JSG-Core": {
        "predictions": RUNS / "jsg_core_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl",
        "annotations": RUNS / "sid_explain300_tampered100_annotations_revised.jsonl",
        "requests": RUNS / "jsg_core_requests_pixel5d_v2_legacy_prompt_controlled.jsonl",
    },
    "JSG-Scale": {
        "predictions": RUNS / "jsg_scale_predictions_pixel5d_v2_legacy_prompt_controlled.jsonl",
        "annotations": RUNS / "sid_fa600_annotations_postreview_eval.jsonl",
        "requests": RUNS / "jsg_scale_requests_pixel5d_v2_legacy_prompt_controlled.jsonl",
    },
}

METRIC_LABELS = {
    "area_ratio": "Area ratio",
    "mean_iou": "Mean IoU",
    "spatial_at_0.1": "Spatial@0.1",
    "spatial_at_0.3": "Spatial@0.3",
    "spatial_at_0.5": "Spatial@0.5",
    "joint_at_0.1": "Joint@0.1",
    "joint_at_0.3": "Joint@0.3",
    "joint_at_0.5": "Joint@0.5",
    "joint_auc": "Joint-AUC",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def bbox_area(box: list[int]) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))


def bbox_iou(first: list[int], second: list[int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = bbox_area(first) + bbox_area(second) - intersection
    return float(intersection / union) if union else 0.0


def annotation_bbox(row: dict[str, Any]) -> list[int] | None:
    items = row.get("evidence_items") or []
    if items:
        box = valid_bbox(items[0].get("evidence_bbox"))
        if box:
            return box
    for candidate in (
        row.get("mask_bbox"),
        (row.get("auto_features") or {}).get("mask_bbox"),
    ):
        box = valid_bbox(candidate)
        if box:
            return box
    return None


def parsed(row: dict[str, Any]) -> bool:
    prediction = row.get("prediction") or {}
    return bool(row.get("parsed", prediction.get("parsed", True)))


def semantic_gate(row: dict[str, Any]) -> bool:
    return bool(
        parsed(row)
        and row.get("prediction_correct")
        and row.get("artifact_correct")
        and row.get("target_correct")
    )


def image_center_box(proposal: list[int], width: int, height: int) -> list[int]:
    box_width = proposal[2] - proposal[0]
    box_height = proposal[3] - proposal[1]
    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2
    return [x1, y1, x1 + box_width, y1 + box_height]


def per_sample_rng(seed: int, dataset: str, img_id: str) -> np.random.Generator:
    key = f"{seed}\0{dataset}\0{img_id}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(derived_seed)


def matched_random_boxes(
    proposal: list[int],
    width: int,
    height: int,
    draws: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    box_width = proposal[2] - proposal[0]
    box_height = proposal[3] - proposal[1]
    max_x = width - box_width
    max_y = height - box_height
    xs = rng.integers(0, max_x + 1, size=draws)
    ys = rng.integers(0, max_y + 1, size=draws)
    return [
        [int(x), int(y), int(x) + box_width, int(y) + box_height]
        for x, y in zip(xs, ys)
    ]


def assert_box_in_image(box: list[int], width: int, height: int, context: str) -> None:
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError(f"Out-of-image bbox for {context}: {box} versus {width}x{height}")


def load_aligned_rows(dataset: str, spec: dict[str, Path]) -> list[dict[str, Any]]:
    predictions = read_jsonl(spec["predictions"])
    annotations = read_jsonl(spec["annotations"])
    requests = read_jsonl(spec["requests"])
    prediction_map = {str(row["img_id"]): row for row in predictions}
    annotation_map = {str(row["img_id"]): row for row in annotations}
    request_map = {str(row["img_id"]): row for row in requests}
    if len(prediction_map) != len(predictions):
        raise ValueError(f"Duplicate prediction img_id in {dataset}")
    if len(annotation_map) != len(annotations):
        raise ValueError(f"Duplicate annotation img_id in {dataset}")
    if len(request_map) != len(requests):
        raise ValueError(f"Duplicate request img_id in {dataset}")
    if set(prediction_map) != set(annotation_map) or set(prediction_map) != set(request_map):
        missing = sorted(set(prediction_map) - set(annotation_map))
        extra = sorted(set(annotation_map) - set(prediction_map))
        request_difference = sorted(set(prediction_map) ^ set(request_map))
        raise ValueError(
            f"Unaligned {dataset} IDs: missing_annotations={missing[:5]}, "
            f"extra_annotations={extra[:5]}, request_difference={request_difference[:5]}"
        )

    aligned: list[dict[str, Any]] = []
    for img_id in sorted(prediction_map):
        prediction = prediction_map[img_id]
        annotation = annotation_map[img_id]
        request = request_map[img_id]
        proposal = valid_bbox((request.get("metadata_for_evaluation_only") or {}).get("candidate_bbox"))
        output_bbox = valid_bbox((prediction.get("prediction") or {}).get("evidence_bbox"))
        frozen_gt = valid_bbox((prediction.get("gold") or {}).get("evidence_bbox"))
        annotation_gt = annotation_bbox(annotation)
        if not proposal or not frozen_gt or not annotation_gt:
            raise ValueError(f"Missing required bbox for {dataset}/{img_id}")
        if frozen_gt != annotation_gt:
            raise ValueError(
                f"Frozen/annotation GT mismatch for {dataset}/{img_id}: "
                f"{frozen_gt} versus {annotation_gt}"
            )
        width = int(annotation.get("width") or 0)
        height = int(annotation.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for {dataset}/{img_id}")
        assert_box_in_image(proposal, width, height, f"{dataset}/{img_id}/proposal")
        assert_box_in_image(frozen_gt, width, height, f"{dataset}/{img_id}/GT")
        if output_bbox and output_bbox != proposal:
            raise ValueError(
                f"Frozen output/request proposal mismatch for {dataset}/{img_id}: "
                f"{output_bbox} versus {proposal}"
            )
        recomputed_iou = bbox_iou(proposal, frozen_gt)
        stored_iou = float(prediction.get("bbox_iou") or 0.0)
        if output_bbox and not math.isclose(recomputed_iou, stored_iou, abs_tol=1e-12):
            raise ValueError(
                f"Frozen IoU mismatch for {dataset}/{img_id}: "
                f"stored={stored_iou}, recomputed={recomputed_iou}"
            )
        aligned.append(
            {
                "img_id": img_id,
                "width": width,
                "height": height,
                "proposal": proposal,
                "gt": frozen_gt,
                "parsed": parsed(prediction),
                "prediction_correct": bool(prediction.get("prediction_correct")),
                "artifact_correct": bool(prediction.get("artifact_correct")),
                "target_correct": bool(prediction.get("target_correct")),
                "semantic_gate": semantic_gate(prediction),
                "output_bbox_available": output_bbox is not None,
                "sampling_stratum": str(
                    (annotation.get("auto_features") or {}).get("mask_area_bucket")
                    or annotation.get("mask_area_bucket")
                    or ""
                ),
            }
        )
    strata = [row["sampling_stratum"] for row in aligned]
    validate_expected_strata(dataset, strata)
    return aligned


def metric_values(
    rows: list[dict[str, Any]],
    dataset: str,
    method: str,
    seed: int,
    random_draws: int,
) -> dict[str, np.ndarray]:
    sample_metrics = {name: [] for name in METRIC_LABELS}
    for row in rows:
        proposal = row["proposal"]
        width = row["width"]
        height = row["height"]
        if method == "proposal":
            boxes = [proposal]
        elif method == "center_matched":
            boxes = [image_center_box(proposal, width, height)]
        elif method == "whole_image":
            boxes = [[0, 0, width, height]]
        elif method == "matched_random":
            boxes = matched_random_boxes(
                proposal,
                width,
                height,
                random_draws,
                per_sample_rng(seed, dataset, row["img_id"]),
            )
        else:
            raise ValueError(method)

        for box in boxes:
            assert_box_in_image(box, width, height, f"{dataset}/{row['img_id']}/{method}")
        ious = np.asarray([bbox_iou(box, row["gt"]) for box in boxes], dtype=np.float64)
        areas = np.asarray([bbox_area(box) / (width * height) for box in boxes], dtype=np.float64)
        gate = float(row["semantic_gate"])
        sample_metrics["area_ratio"].append(float(areas.mean()))
        sample_metrics["mean_iou"].append(float(ious.mean()))
        for threshold in THRESHOLDS:
            sample_metrics[f"spatial_at_{threshold}"].append(float((ious >= threshold).mean()))
            sample_metrics[f"joint_at_{threshold}"].append(float(((ious >= threshold) * gate).mean()))
        # Integral_0^1 1[IoU >= t] dt equals IoU for each sample/draw.
        sample_metrics["joint_auc"].append(float((ious * gate).mean()))
    return {name: np.asarray(values, dtype=np.float64) for name, values in sample_metrics.items()}


def joint_curve_values(
    rows: list[dict[str, Any]],
    dataset: str,
    method: str,
    seed: int,
    random_draws: int,
) -> np.ndarray:
    """Return image-level Joint values over the fixed threshold grid."""
    values = np.empty((len(rows), len(CURVE_THRESHOLDS)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        proposal = row["proposal"]
        width = row["width"]
        height = row["height"]
        if method == "proposal":
            boxes = [proposal]
        elif method == "center_matched":
            boxes = [image_center_box(proposal, width, height)]
        elif method == "whole_image":
            boxes = [[0, 0, width, height]]
        elif method == "matched_random":
            boxes = matched_random_boxes(
                proposal,
                width,
                height,
                random_draws,
                per_sample_rng(seed, dataset, row["img_id"]),
            )
        else:
            raise ValueError(method)
        ious = np.asarray([bbox_iou(box, row["gt"]) for box in boxes], dtype=np.float64)
        gate = float(row["semantic_gate"])
        values[row_index] = gate * (ious[:, None] >= CURVE_THRESHOLDS[None, :]).mean(axis=0)
    return values


def bootstrap_curve_ci(
    values: np.ndarray,
    strata: list[str],
    seed: int,
    reps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile CI with images resampled within benchmark strata."""
    return stratified_percentile_interval(values, strata, seed, reps)


def summarize_curve(
    values: np.ndarray,
    strata: list[str],
    seed: int,
    reps: int,
    context: str,
) -> dict[str, Any]:
    low, high = bootstrap_curve_ci(
        values,
        strata,
        stable_bootstrap_seed(seed, context, "joint-threshold-curve"),
        reps,
    )
    return {
        "estimate": values.mean(axis=0).tolist(),
        "bootstrap_ci_95_low": low.tolist(),
        "bootstrap_ci_95_high": high.tolist(),
        "interval_method": "stratified percentile bootstrap over image IDs",
    }


def write_curve_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "box_rule", "threshold", "joint", "ci95_low", "ci95_high"])
        for dataset, dataset_report in report["datasets"].items():
            for method, method_report in dataset_report["methods"].items():
                curve = method_report["joint_threshold_curve"]
                for threshold, estimate, low, high in zip(
                    report["configuration"]["curve_thresholds"],
                    curve["estimate"],
                    curve["bootstrap_ci_95_low"],
                    curve["bootstrap_ci_95_high"],
                ):
                    writer.writerow([dataset, method, threshold, estimate, low, high])


def plot_joint_threshold_curves(report: dict[str, Any], output_stem: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )
    labels = {
        "proposal": "Proposal",
        "center_matched": "Center, size-matched",
        "matched_random": "Random, size-matched",
        "whole_image": "Whole image",
    }
    styles = {
        "proposal": {"color": "#1f1f1f", "linestyle": "-", "linewidth": 1.6},
        "center_matched": {"color": "#D55E00", "linestyle": "--", "linewidth": 1.35},
        "matched_random": {"color": "#0072B2", "linestyle": ":", "linewidth": 1.5},
        "whole_image": {"color": "#009E73", "linestyle": "-.", "linewidth": 1.35},
    }
    order = ("proposal", "center_matched", "matched_random", "whole_image")
    thresholds = np.asarray(report["configuration"]["curve_thresholds"], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(7.18, 2.55), sharex=True, sharey=True)
    for panel, (ax, (dataset, dataset_report)) in enumerate(zip(axes, report["datasets"].items())):
        for method in order:
            curve = dataset_report["methods"][method]["joint_threshold_curve"]
            estimate = 100.0 * np.asarray(curve["estimate"])
            low = 100.0 * np.asarray(curve["bootstrap_ci_95_low"])
            high = 100.0 * np.asarray(curve["bootstrap_ci_95_high"])
            ax.fill_between(thresholds, low, high, color=styles[method]["color"], alpha=0.08, linewidth=0)
            ax.plot(thresholds, estimate, label=labels[method], **styles[method])
        ax.axvline(0.1, color="#777777", linestyle=(0, (2, 2)), linewidth=0.7, zorder=0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 72.0)
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        ax.set_yticks(np.arange(0.0, 71.0, 10.0))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.set_xlabel("IoU threshold, $t$")
        panel_label = chr(ord("a") + panel)
        ax.text(
            0.5,
            -0.27,
            f"({panel_label})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Fixed-semantic Joint@t (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.91), w_pad=1.2)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def stable_bootstrap_seed(seed: int, *parts: str) -> int:
    key = "\0".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def summarize_metrics(
    values: dict[str, np.ndarray],
    strata: list[str],
    seed: int,
    reps: int,
    context: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, items in values.items():
        binary_rate = name.startswith(("spatial_at_", "joint_at_")) and is_binary_vector(items)
        if binary_rate:
            interval = mover_wilson_interval(items, strata)
            method = "stratum-specific Wilson intervals combined by MOVER"
        else:
            interval = stratified_percentile_interval_list(
                items,
                strata,
                stable_bootstrap_seed(seed, context, name),
                reps,
            )
            method = "stratified percentile bootstrap over image IDs"
        output[name] = {
            "estimate": float(items.mean()),
            "ci_95": interval,
            "interval_method": method,
        }
    return output


def paired_differences(
    baseline: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    strata: list[str],
    seed: int,
    reps: int,
    context: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in METRIC_LABELS:
        if baseline[name].shape != control[name].shape:
            raise ValueError(f"Unpaired arrays for {context}/{name}")
        difference = control[name] - baseline[name]
        output[name] = {
            "estimate": float(difference.mean()),
            "paired_stratified_bootstrap_ci_95": stratified_percentile_interval_list(
                difference,
                strata,
                stable_bootstrap_seed(seed, context, name),
                reps,
            ),
            "interval_method": "paired stratified percentile bootstrap over image IDs",
        }
    return output


def criterion_values(rows: list[dict[str, Any]], setting: str) -> np.ndarray:
    values = []
    for row in rows:
        spatial = bbox_iou(row["proposal"], row["gt"]) >= 0.1
        checks = {
            "parse": row["parsed"],
            "prediction": row["prediction_correct"],
            "artifact": row["artifact_correct"],
            "target": row["target_correct"],
            "spatial": spatial,
        }
        relaxed = {
            "observed": set(),
            "spatial_relaxed": {"spatial"},
            "artifact_relaxed": {"artifact"},
            "target_relaxed": {"target"},
            "artifact_and_target_relaxed": {"artifact", "target"},
        }[setting]
        values.append(float(all(value or name in relaxed for name, value in checks.items())))
    return np.asarray(values, dtype=np.float64)


def criterion_relaxations(
    rows: list[dict[str, Any]],
    strata: list[str],
    seed: int,
    reps: int,
    dataset: str,
) -> dict[str, Any]:
    settings = [
        "observed",
        "spatial_relaxed",
        "artifact_relaxed",
        "target_relaxed",
        "artifact_and_target_relaxed",
    ]
    observed = criterion_values(rows, "observed")
    output: dict[str, Any] = {}
    for setting in settings:
        values = criterion_values(rows, setting)
        difference = values - observed
        output[setting] = {
            "passes": int(values.sum()),
            "joint_at_0.1": float(values.mean()),
            "joint_at_0.1_mover_wilson_ci_95": mover_wilson_interval(values, strata),
            "minus_observed": float(difference.mean()),
            "minus_observed_paired_stratified_bootstrap_ci_95": stratified_percentile_interval_list(
                difference,
                strata,
                stable_bootstrap_seed(seed, dataset, "criterion", setting, "difference"),
                reps,
            ),
        }
    return output


def format_rate(value: float) -> str:
    return f"{100.0 * value:.2f}"


def format_ci(item: dict[str, Any], key: str = "ci_95") -> str:
    low, high = item[key]
    return f"{format_rate(item['estimate'])} [{format_rate(low)}, {format_rate(high)}]"


def format_float_ci(item: dict[str, Any], key: str = "ci_95") -> str:
    low, high = item[key]
    return f"{item['estimate']:.4f} [{low:.4f}, {high:.4f}]"


def format_delta(item: dict[str, Any]) -> str:
    low, high = item["paired_stratified_bootstrap_ci_95"]
    return f"{100.0 * item['estimate']:+.2f} [{100.0 * low:+.2f}, {100.0 * high:+.2f}]"


def format_float_delta(item: dict[str, Any]) -> str:
    low, high = item["paired_stratified_bootstrap_ci_95"]
    return f"{item['estimate']:+.4f} [{low:+.4f}, {high:+.4f}]"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen Spatial Construct Controls",
        "",
        "This is an offline analysis of frozen per-sample outputs; it runs no model inference. "
        "Semantic decisions are held fixed while only the evaluated bbox changes.",
        "",
        "Matched-random boxes preserve each proposal's width and height and sample their top-left "
        "coordinates uniformly over all valid in-image positions. Each `dataset + img_id` has a "
        "deterministic RNG stream. GT is read only after box generation to score overlap.",
        "",
        f"Random seed: `{report['configuration']['seed']}`; random draws per image: "
        f"`{report['configuration']['random_draws']}`; sample-cluster bootstrap replicates: "
        f"`{report['configuration']['bootstrap_reps']}`.",
        "",
        "Strictly binary aggregate rates use stratum-specific Wilson intervals combined by MOVER. "
        "Continuous summaries, paired contrasts, and curve bands use 95% stratified percentile "
        "bootstrap intervals over `img_id`. For matched-random, the fixed draws are averaged within "
        "each image before bootstrapping, so images, not draws, are the resampling unit.",
        "",
        "Raw spatial controls evaluate the frozen input proposal from request metadata even when a "
        "report failed to parse. Fixed-semantic Joint retains the parse gate. Consequently, proposal "
        "Spatial@t can be slightly higher than the parse-aware IoU pass rate in the manuscript.",
    ]
    method_labels = {
        "proposal": "Frozen proposal",
        "center_matched": "Center, size-matched",
        "whole_image": "Whole image",
        "matched_random": "Random, size-matched",
    }
    for dataset, dataset_report in report["datasets"].items():
        lines.extend(
            [
                "",
                f"## {dataset}",
                "",
                f"Aligned frozen sample count: `{dataset_report['samples']}`; fixed semantic-gate "
                f"pass rate: `{format_rate(dataset_report['semantic_gate_rate'])}%`; parsed output bbox "
                f"available for `{dataset_report['output_bbox_available']}/{dataset_report['samples']}` rows.",
                "",
                "### Raw spatial controls",
                "",
                "| Box rule | Area % | Mean IoU | Spatial@0.1 % | Spatial@0.3 % | Spatial@0.5 % |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, method_report in dataset_report["methods"].items():
            metrics = method_report["metrics"]
            lines.append(
                f"| {method_labels[method]} | {format_ci(metrics['area_ratio'])} | "
                f"{format_float_ci(metrics['mean_iou'])} | {format_ci(metrics['spatial_at_0.1'])} | "
                f"{format_ci(metrics['spatial_at_0.3'])} | {format_ci(metrics['spatial_at_0.5'])} |"
            )
        lines.extend(
            [
                "",
                "`Area %` and threshold pass rates are percentages; mean IoU is a proportion.",
                "",
                "### Fixed-semantic Joint",
                "",
                "| Box rule | Joint@0.1 % | Joint@0.3 % | Joint@0.5 % | Joint-AUC |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method, method_report in dataset_report["methods"].items():
            metrics = method_report["metrics"]
            lines.append(
                f"| {method_labels[method]} | {format_ci(metrics['joint_at_0.1'])} | "
                f"{format_ci(metrics['joint_at_0.3'])} | {format_ci(metrics['joint_at_0.5'])} | "
                f"{format_float_ci(metrics['joint_auc'])} |"
            )
        lines.extend(
            [
                "",
                "`Joint-AUC = mean(semantic_gate * IoU)`, exactly the integral of the fixed-semantic "
                "Joint-threshold curve over thresholds from 0 to 1.",
                "",
                "### Paired differences from frozen proposal",
                "",
                "| Control minus proposal | Mean IoU change | Spatial@0.1 pp | Joint@0.1 pp | Joint-AUC change |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in ("center_matched", "whole_image", "matched_random"):
            differences = dataset_report["methods"][method]["minus_proposal"]
            lines.append(
                f"| {method_labels[method]} | {format_float_delta(differences['mean_iou'])} | "
                f"{format_delta(differences['spatial_at_0.1'])} | "
                f"{format_delta(differences['joint_at_0.1'])} | "
                f"{format_float_delta(differences['joint_auc'])} |"
            )
        lines.extend(
            [
                "",
                "All paired differences, including thresholds 0.3 and 0.5 and area ratio, are retained in the JSON.",
                "",
                "### Frozen-output criterion relaxations",
                "",
                "| Setting | Passes | Joint@0.1 % | Change from observed pp (paired 95% CI) |",
                "|---|---:|---:|---:|",
            ]
        )
        relaxation_labels = {
            "observed": "Observed",
            "spatial_relaxed": "Spatial criterion relaxed",
            "artifact_relaxed": "Artifact criterion relaxed",
            "target_relaxed": "Target criterion relaxed",
            "artifact_and_target_relaxed": "Artifact + target criteria relaxed",
        }
        for setting, item in dataset_report["criterion_relaxations"].items():
            low, high = item[
                "minus_observed_paired_stratified_bootstrap_ci_95"
            ]
            lines.append(
                f"| {relaxation_labels[setting]} | {item['passes']} | "
                f"{format_rate(item['joint_at_0.1'])} | {100.0 * item['minus_observed']:+.2f} "
                f"[{100.0 * low:+.2f}, {100.0 * high:+.2f}] |"
            )
        lines.extend(
            [
                "",
                "Criterion relaxation changes evaluation only: it sets the named pass condition(s) to true "
                "while retaining the frozen parse, prediction, remaining semantic fields, and proposal. "
                "These are sensitivity/upper-bound readouts, not causal component effects.",
            ]
        )
    lines.extend(
        [
            "",
            "## Reproducibility checks",
            "",
            "- Predictions and annotations were joined one-to-one by `img_id`; duplicate, missing, or extra IDs fail the run.",
            "- Frozen request rows were joined one-to-one by `img_id`; every non-null parsed output bbox was required to equal its request proposal.",
            "- Frozen GT boxes were required to equal the aligned annotation boxes.",
            "- For non-null output bboxes, proposal IoUs were recomputed and required to equal stored IoUs to absolute tolerance `1e-12`.",
            "- Every proposal and generated control box was required to lie within image bounds.",
            "- Input SHA-256 hashes and the complete aggregate metric/CI payload are stored in the JSON report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--random-draws", type=int, default=1000)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS / "spatial_construct_controls.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPORTS / "spatial_construct_controls.md",
    )
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=REPORTS / "spatial_joint_threshold_curves.csv",
    )
    parser.add_argument(
        "--figure-stem",
        type=Path,
        default=ROOT / "ieee-journal-latex-package" / "figures" / "fig_spatial_joint_threshold_controls",
    )
    args = parser.parse_args()
    if args.random_draws <= 0 or args.bootstrap_reps <= 0:
        parser.error("--random-draws and --bootstrap-reps must be positive")

    report: dict[str, Any] = {
        "stage": "frozen_spatial_construct_controls",
        "definition": {
            "semantic_gate": "parsed AND prediction_correct AND artifact_correct AND target_correct",
            "spatial_at_t": "bbox_iou >= t",
            "joint_at_t": "semantic_gate AND bbox_iou >= t",
            "joint_auc": "mean over images of semantic_gate * bbox_iou",
            "random_box_rule": (
                "For each image and draw, preserve the frozen proposal width/height and sample x1 and y1 "
                "independently and uniformly from all integer top-left positions that keep the box in-image. "
                "The RNG depends only on global seed, dataset, and img_id; GT is not an RNG input."
            ),
            "raw_spatial_proposal_source": "request.metadata_for_evaluation_only.candidate_bbox",
            "parse_failure_handling": (
                "Raw spatial metrics still score the frozen request proposal. Fixed-semantic Joint "
                "includes parsed, so parse failures remain Joint failures."
            ),
        },
        "configuration": {
            "seed": args.seed,
            "random_draws": args.random_draws,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_unit": (
                "img_id resampled within mask-area stratum after within-image averaging "
                "of random draws"
            ),
            "binary_interval": "stratum-specific Wilson intervals combined by MOVER",
            "continuous_paired_and_curve_interval": "stratified percentile bootstrap",
            "thresholds": list(THRESHOLDS),
            "curve_thresholds": CURVE_THRESHOLDS.tolist(),
        },
        "inputs": {},
        "datasets": {},
    }

    for dataset, spec in DATASETS.items():
        rows = load_aligned_rows(dataset, spec)
        strata = [row["sampling_stratum"] for row in rows]
        report["inputs"][dataset] = {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in spec.items()
        }
        values_by_method = {
            method: metric_values(rows, dataset, method, args.seed, args.random_draws)
            for method in ("proposal", "center_matched", "whole_image", "matched_random")
        }
        curves_by_method = {
            method: joint_curve_values(rows, dataset, method, args.seed, args.random_draws)
            for method in ("proposal", "center_matched", "whole_image", "matched_random")
        }
        proposal_values = values_by_method["proposal"]
        methods: dict[str, Any] = {}
        for method, values in values_by_method.items():
            estimate_context = dataset if method == "proposal" else f"{dataset}/{method}/estimate"
            methods[method] = {
                "random_draws_per_image": args.random_draws if method == "matched_random" else 1,
                "metrics": summarize_metrics(
                    values,
                    strata,
                    args.seed,
                    args.bootstrap_reps,
                    estimate_context,
                ),
                "joint_threshold_curve": summarize_curve(
                    curves_by_method[method],
                    strata,
                    args.seed,
                    args.bootstrap_reps,
                    f"{dataset}/{method}",
                ),
            }
            if method != "proposal":
                methods[method]["minus_proposal"] = paired_differences(
                    proposal_values,
                    values,
                    strata,
                    args.seed,
                    args.bootstrap_reps,
                    f"{dataset}/{method}/minus_proposal",
                )
        report["datasets"][dataset] = {
            "samples": len(rows),
            "sampling_design": design_metadata(dataset, strata),
            "unique_img_ids": len({row["img_id"] for row in rows}),
            "semantic_gate_passes": sum(row["semantic_gate"] for row in rows),
            "semantic_gate_rate": sum(row["semantic_gate"] for row in rows) / len(rows),
            "output_bbox_available": sum(row["output_bbox_available"] for row in rows),
            "methods": methods,
            "criterion_relaxations": criterion_relaxations(
                rows,
                strata,
                args.seed,
                args.bootstrap_reps,
                dataset,
            ),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(build_markdown(report), encoding="utf-8")
    write_curve_csv(report, args.curve_csv)
    plot_joint_threshold_curves(report, args.figure_stem)
    print(
        json.dumps(
            {
                "json": str(args.output_json),
                "md": str(args.output_md),
                "curve_csv": str(args.curve_csv),
                "figure_stem": str(args.figure_stem),
            }
        )
    )


if __name__ == "__main__":
    main()
