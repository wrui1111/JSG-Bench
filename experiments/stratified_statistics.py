#!/usr/bin/env python3
"""Shared benchmark-design strata and stratified interval utilities."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"

CORE_ANNOTATIONS = RUNS / "sid_explain300_tampered100_annotations_revised.jsonl"
SCALE_ANNOTATIONS = RUNS / "sid_fa600_annotations_postreview_eval.jsonl"

EXPECTED_STRATUM_COUNTS = {
    "JSG-Core": {
        "large": 25,
        "medium": 25,
        "small": 25,
        "tiny": 25,
    },
    "JSG-Scale": {
        "large": 150,
        "medium": 150,
        "small": 150,
        "tiny": 150,
    },
    "JSG-Xfer": {
        "CASIA2|candidate_iou<0.1": 16,
        "CASIA2|candidate_iou[0.1,0.3)": 16,
        "CASIA2|candidate_iou>=0.3": 18,
        "IMD2020|candidate_iou<0.1": 16,
        "IMD2020|candidate_iou[0.1,0.3)": 16,
        "IMD2020|candidate_iou>=0.3": 18,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def core_strata_by_id(path: Path = CORE_ANNOTATIONS) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in read_jsonl(path):
        stratum = (row.get("auto_features") or {}).get("mask_area_bucket")
        if not stratum:
            raise ValueError(f"Missing Core mask-area stratum for {row.get('img_id')}")
        output[str(row["img_id"])] = str(stratum)
    return output


def scale_strata_by_id(path: Path = SCALE_ANNOTATIONS) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in read_jsonl(path):
        stratum = row.get("mask_area_bucket")
        if not stratum:
            raise ValueError(f"Missing Scale mask-area stratum for {row.get('img_id')}")
        output[str(row["img_id"])] = str(stratum)
    return output


def xfer_selection_band(candidate_iou: float) -> str:
    """Return the three bands used by the frozen Xfer selection code."""
    if candidate_iou < 0.1:
        return "candidate_iou<0.1"
    if candidate_iou < 0.3:
        return "candidate_iou[0.1,0.3)"
    return "candidate_iou>=0.3"


def xfer_stratum(source: Any, candidate_iou: float) -> str:
    source_name = str(source or "")
    if source_name not in {"CASIA2", "IMD2020"}:
        raise ValueError(f"Unexpected Xfer source: {source!r}")
    return f"{source_name}|{xfer_selection_band(float(candidate_iou))}"


def stratum_counts(strata: Sequence[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in strata).items()))


def validate_expected_strata(dataset: str, strata: Sequence[Any]) -> None:
    expected = EXPECTED_STRATUM_COUNTS.get(dataset)
    if expected is None:
        raise ValueError(f"Unknown benchmark sampling design: {dataset}")
    observed = stratum_counts(strata)
    if observed != expected:
        raise ValueError(
            f"Unexpected {dataset} stratum counts: observed={observed}, expected={expected}"
        )


def design_metadata(dataset: str, strata: Sequence[Any]) -> dict[str, Any]:
    counts = stratum_counts(strata)
    total = sum(counts.values())
    return {
        "estimand": "frozen benchmark-design weighted mean",
        "stratification": (
            "mask-area band"
            if dataset in {"JSG-Core", "JSG-Scale"}
            else "source x prespecified candidate-IoU selection band"
        ),
        "stratum_counts": counts,
        "design_weights": {key: value / total for key, value in counts.items()},
        "population_inference": False,
    }


def wilson_interval(
    successes: int,
    samples: int,
    z: float = 1.959963984540054,
) -> list[float]:
    if samples <= 0:
        raise ValueError("Wilson intervals require at least one sample")
    proportion = successes / samples
    denominator = 1.0 + z * z / samples
    center = (proportion + z * z / (2.0 * samples)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / samples
            + z * z / (4.0 * samples * samples)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _as_strata(strata: Sequence[Any], samples: int) -> np.ndarray:
    if len(strata) != samples:
        raise ValueError(f"Strata length {len(strata)} does not match samples {samples}")
    normalized = np.asarray([str(value) for value in strata], dtype=object)
    if any(not value for value in normalized):
        raise ValueError("Empty sampling stratum")
    return normalized


def is_binary_vector(values: Sequence[float] | np.ndarray) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return array.ndim == 1 and bool(np.all((array == 0.0) | (array == 1.0)))


def mover_wilson_interval(
    values: Sequence[float] | np.ndarray,
    strata: Sequence[Any],
) -> list[float]:
    """MOVER interval for a fixed-weight sum of independent stratum rates."""
    array = np.asarray(values, dtype=np.float64)
    if not is_binary_vector(array):
        raise ValueError("MOVER-Wilson requires a one-dimensional binary vector")
    normalized = _as_strata(strata, len(array))
    estimate = float(array.mean())
    lower_variance = 0.0
    upper_variance = 0.0
    for stratum in sorted(set(normalized)):
        stratum_values = array[normalized == stratum]
        samples = len(stratum_values)
        successes = int(stratum_values.sum())
        proportion = successes / samples
        lower, upper = wilson_interval(successes, samples)
        weight = samples / len(array)
        lower_variance += weight * weight * (proportion - lower) ** 2
        upper_variance += weight * weight * (upper - proportion) ** 2
    return [
        max(0.0, estimate - math.sqrt(lower_variance)),
        min(1.0, estimate + math.sqrt(upper_variance)),
    ]


def stratified_percentile_interval(
    values: Sequence[float] | np.ndarray,
    strata: Sequence[Any],
    seed: int,
    reps: int,
    batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval preserving the observed count of every stratum."""
    if reps <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or len(array) == 0:
        raise ValueError("Bootstrap values must contain at least one sample")
    normalized = _as_strata(strata, len(array))
    groups = [array[normalized == stratum] for stratum in sorted(set(normalized))]
    tail_shape = array.shape[1:]
    bootstrap_means = np.empty((reps, *tail_shape), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, reps, batch_size):
        stop = min(start + batch_size, reps)
        size = stop - start
        sums = np.zeros((size, *tail_shape), dtype=np.float64)
        for group in groups:
            indices = rng.integers(0, len(group), size=(size, len(group)))
            sums += group[indices].sum(axis=1)
        bootstrap_means[start:stop] = sums / len(array)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)
    return lower, upper


def stratified_percentile_interval_list(
    values: Sequence[float] | np.ndarray,
    strata: Sequence[Any],
    seed: int,
    reps: int,
) -> list[float]:
    lower, upper = stratified_percentile_interval(values, strata, seed, reps)
    if np.asarray(lower).ndim != 0:
        raise ValueError("Expected a one-dimensional bootstrap statistic")
    return [float(lower), float(upper)]
