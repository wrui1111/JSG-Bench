#!/usr/bin/env python3
"""Analyze Joint bounds, J-AUC, and criterion relaxations from audit traces.

This script performs no inference. It uses released versioned TD-EGFA traces
as the authoritative outcome record and frozen annotation metadata only to
recover the prespecified benchmark sampling strata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from stratified_statistics import (
    CORE_ANNOTATIONS,
    SCALE_ANNOTATIONS,
    core_strata_by_id,
    design_metadata,
    mover_wilson_interval,
    scale_strata_by_id,
    stratified_percentile_interval_list,
    validate_expected_strata,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
REPORTS = ROOT / "experiments" / "reports"

TRACE_PATHS = {
    "JSG-Core": RUNS / "jsg_core_td_egfa_audit_trace_pixel5d_v2.jsonl",
    "JSG-Scale": RUNS / "jsg_scale_td_egfa_audit_trace_pixel5d_v2.jsonl",
    "JSG-Xfer": RUNS / "jsg_xfer_td_egfa_audit_trace_pixel5d_v2.jsonl",
}

CONDITION_ORDER = ("parse", "prediction", "artifact", "target", "spatial")
MARGINAL_CONDITIONS = ("prediction", "artifact", "target", "spatial")
RELAXATIONS = {
    "observed": frozenset(),
    "spatial_relaxed": frozenset({"spatial"}),
    "artifact_relaxed": frozenset({"artifact"}),
    "target_relaxed": frozenset({"target"}),
    "artifact_and_target_relaxed": frozenset({"artifact", "target"}),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"Missing sample_id in {path}:{line_number}")
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id {sample_id} in {path}")
            sample_ids.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"No trace rows in {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: str) -> int:
    key = "\0".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def validate_trace(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    if row.get("dataset") != dataset:
        raise ValueError(
            f"Dataset mismatch for {row.get('sample_id')}: {row.get('dataset')} versus {dataset}"
        )
    profile = (row.get("audit_profiles") or {}).get("primary_strict") or {}
    conditions = profile.get("conditions") or {}
    missing = [name for name in CONDITION_ORDER if name not in conditions]
    if missing:
        raise ValueError(f"Missing strict conditions {missing} for {dataset}/{row.get('sample_id')}")
    normalized = {name: bool(conditions[name]) for name in CONDITION_ORDER}
    order = profile.get("condition_order")
    vector = profile.get("pass_vector")
    if order != list(CONDITION_ORDER):
        raise ValueError(f"Unexpected condition order for {dataset}/{row.get('sample_id')}: {order}")
    if vector != [normalized[name] for name in CONDITION_ORDER]:
        raise ValueError(f"Pass-vector mismatch for {dataset}/{row.get('sample_id')}")
    expected_joint = all(normalized.values())
    expected_failed = [name for name in CONDITION_ORDER if not normalized[name]]
    if bool(profile.get("joint")) != expected_joint:
        raise ValueError(f"Joint mismatch for {dataset}/{row.get('sample_id')}")
    if profile.get("failed_conditions") != expected_failed:
        raise ValueError(f"Failed-condition mismatch for {dataset}/{row.get('sample_id')}")
    if not normalized["parse"] and any(normalized[name] for name in CONDITION_ORDER[1:]):
        raise ValueError(f"Parse failure did not gate downstream fields for {dataset}/{row.get('sample_id')}")

    region = row.get("evaluated_region") or {}
    threshold = float(region.get("iou_threshold"))
    iou = float(region.get("iou") or 0.0)
    if not (0.0 <= iou <= 1.0):
        raise ValueError(f"Invalid IoU {iou} for {dataset}/{row.get('sample_id')}")
    expected_spatial = bool(normalized["parse"] and region.get("bbox") is not None and iou >= threshold)
    if normalized["spatial"] != expected_spatial:
        raise ValueError(f"Spatial-condition mismatch for {dataset}/{row.get('sample_id')}")
    return {
        "sample_id": str(row["sample_id"]),
        "contract_version": str(row.get("contract_version")),
        "system_id": str(row.get("system_id")),
        "conditions": normalized,
        "iou_threshold": threshold,
        "iou": iou,
        "external_dataset": row.get("external_dataset"),
        "sampling_stratum": row.get("sampling_stratum"),
    }


def setting_values(samples: list[dict[str, Any]], relaxed: frozenset[str]) -> np.ndarray:
    return np.asarray(
        [
            float(
                all(value or name in relaxed for name, value in sample["conditions"].items())
            )
            for sample in samples
        ],
        dtype=np.float64,
    )


def passing_ids(
    samples: list[dict[str, Any]],
    values: np.ndarray,
) -> list[str]:
    return [sample["sample_id"] for sample, value in zip(samples, values) if value]


def analyze_dataset(
    dataset: str,
    path: Path,
    seed: int,
    reps: int,
    core_strata: dict[str, str],
    scale_strata: dict[str, str],
) -> dict[str, Any]:
    samples = sorted(
        (validate_trace(row, dataset) for row in read_jsonl(path)),
        key=lambda sample: sample["sample_id"],
    )
    contract_versions = sorted({sample["contract_version"] for sample in samples})
    system_ids = sorted({sample["system_id"] for sample in samples})
    thresholds = sorted({sample["iou_threshold"] for sample in samples})
    if len(contract_versions) != 1 or len(system_ids) != 1 or thresholds != [0.1]:
        raise ValueError(
            f"Nonuniform trace metadata for {dataset}: versions={contract_versions}, "
            f"systems={system_ids}, thresholds={thresholds}"
        )

    if dataset == "JSG-Core":
        strata = [core_strata[sample["sample_id"]] for sample in samples]
    elif dataset == "JSG-Scale":
        strata = [scale_strata[sample["sample_id"]] for sample in samples]
    elif dataset == "JSG-Xfer":
        strata = [str(sample["sampling_stratum"] or "") for sample in samples]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    validate_expected_strata(dataset, strata)

    n = len(samples)
    marginal_passes = {
        name: sum(sample["conditions"][name] for sample in samples)
        for name in MARGINAL_CONDITIONS
    }
    marginal_rates = {name: count / n for name, count in marginal_passes.items()}
    lower_bound = max(0.0, sum(marginal_rates.values()) - (len(MARGINAL_CONDITIONS) - 1))
    upper_bound = min(marginal_rates.values())

    observed = setting_values(samples, RELAXATIONS["observed"])
    semantic_gate = np.asarray(
        [
            float(
                all(
                    sample["conditions"][name]
                    for name in ("parse", "prediction", "artifact", "target")
                )
            )
            for sample in samples
        ],
        dtype=np.float64,
    )
    ious = np.asarray([sample["iou"] for sample in samples], dtype=np.float64)
    j_auc_values = semantic_gate * ious

    relaxations: dict[str, Any] = {}
    for setting, relaxed in RELAXATIONS.items():
        values = setting_values(samples, relaxed)
        difference = values - observed
        ids = passing_ids(samples, values)
        relaxations[setting] = {
            "relaxed_conditions": sorted(relaxed),
            "passes": int(values.sum()),
            "joint_at_0.1": float(values.mean()),
            "joint_at_0.1_mover_wilson_ci_95": mover_wilson_interval(values, strata),
            "minus_observed": float(difference.mean()),
            "minus_observed_paired_stratified_bootstrap_ci_95": stratified_percentile_interval_list(
                difference,
                strata,
                stable_seed(seed, dataset, "criterion", setting, "difference"),
                reps,
            ),
            "passing_sample_ids": ids,
            "passing_set_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        }

    if relaxations["observed"]["passes"] != sum(
        all(sample["conditions"].values()) for sample in samples
    ):
        raise ValueError(f"Observed Joint cross-check failed for {dataset}")

    return {
        "samples": n,
        "sampling_design": design_metadata(dataset, strata),
        "contract_version": contract_versions[0],
        "system_id": system_ids[0],
        "iou_threshold": thresholds[0],
        "marginal_passes": marginal_passes,
        "marginal_rates": marginal_rates,
        "frechet_joint_bounds": [lower_bound, upper_bound],
        "joint_auc": {
            "estimate": float(j_auc_values.mean()),
            "stratified_bootstrap_ci_95": stratified_percentile_interval_list(
                j_auc_values,
                strata,
                stable_seed(seed, dataset, "joint_auc"),
                reps,
            ),
        },
        "criterion_relaxations": relaxations,
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Trace-First Joint Audit Analysis",
        "",
        "This offline analysis reads the released TD-EGFA audit traces, runs no model inference, "
        "and reads frozen annotation metadata only to recover sampling strata.",
        "",
        f"Bootstrap seed: `{report['configuration']['seed']}`; stratified sample-cluster bootstrap replicates: "
        f"`{report['configuration']['bootstrap_reps']}`. Binary aggregate rates use stratum-specific "
        "Wilson intervals combined by MOVER; continuous summaries and paired contrasts use 95% "
        "stratified percentile-bootstrap intervals.",
        "",
        "## Marginals, Joint bounds, and J-AUC",
        "",
        "| Dataset | N | Prediction % | Artifact % | Target % | Spatial@0.1 % | Frechet Joint bound % | Observed Joint % | J-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in report["datasets"].items():
        rates = item["marginal_rates"]
        low, high = item["frechet_joint_bounds"]
        observed = item["criterion_relaxations"]["observed"]["joint_at_0.1"]
        lines.append(
            f"| {dataset} | {item['samples']} | {percent(rates['prediction'])} | "
            f"{percent(rates['artifact'])} | {percent(rates['target'])} | "
            f"{percent(rates['spatial'])} | [{percent(low)}, {percent(high)}] | "
            f"{percent(observed)} | {item['joint_auc']['estimate']:.4f} |"
        )

    labels = {
        "observed": "Observed contract",
        "spatial_relaxed": "Spatial condition relaxed",
        "artifact_relaxed": "Artifact condition relaxed",
        "target_relaxed": "Target condition relaxed",
        "artifact_and_target_relaxed": "Artifact + target relaxed",
    }
    lines.extend(
        [
            "",
            "`J-AUC = mean(IoU * 1[parse, prediction, artifact, and target all pass])`.",
            "",
            "## Fixed-output criterion relaxations",
        ]
    )
    for dataset, item in report["datasets"].items():
        lines.extend(
            [
                "",
                f"### {dataset}",
                "",
                "| Scoring rule | Passes | Joint@0.1 % | Change from observed pp (paired 95% CI) |",
                "|---|---:|---:|---:|",
            ]
        )
        for setting, relaxation in item["criterion_relaxations"].items():
            ci_low, ci_high = relaxation[
                "minus_observed_paired_stratified_bootstrap_ci_95"
            ]
            lines.append(
                f"| {labels[setting]} | {relaxation['passes']} | "
                f"{percent(relaxation['joint_at_0.1'])} | "
                f"{100.0 * relaxation['minus_observed']:+.2f} "
                f"[{100.0 * ci_low:+.2f}, {100.0 * ci_high:+.2f}] |"
            )

    lines.extend(
        [
            "",
            "The relaxation rows set only the named scoring bit(s) to pass. Parse validity, prediction, "
            "all unrelaxed conditions, recorded IoU, and sample identity remain unchanged. They are "
            "monotone sensitivity bounds, not causal component gains.",
            "",
            "## Reproducibility checks",
            "",
            "- Trace rows are required to have unique sample IDs and uniform contract, system, and threshold metadata.",
            "- Stored pass vectors, Joint values, and failed-condition lists are recomputed from the strict condition map.",
            "- Stored spatial bits are checked against parse validity, bbox availability, recorded IoU, and the threshold.",
            "- Input paths and SHA-256 hashes, passing IDs, and passing-set hashes are retained in the JSON report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS / "criterion_relaxation_analysis.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPORTS / "criterion_relaxation_analysis.md",
    )
    args = parser.parse_args()
    if args.bootstrap_reps <= 0:
        parser.error("--bootstrap-reps must be positive")

    core_strata = core_strata_by_id()
    scale_strata = scale_strata_by_id()
    inputs = {
        dataset: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for dataset, path in TRACE_PATHS.items()
    }
    inputs["sampling_metadata"] = {
        "JSG-Core": {
            "path": str(CORE_ANNOTATIONS.relative_to(ROOT)),
            "sha256": sha256(CORE_ANNOTATIONS),
        },
        "JSG-Scale": {
            "path": str(SCALE_ANNOTATIONS.relative_to(ROOT)),
            "sha256": sha256(SCALE_ANNOTATIONS),
        },
        "JSG-Xfer": {
            "source": "prespecified sampling_stratum retained in the audit trace",
            "selection_bands": ["<0.1", "[0.1,0.3)", ">=0.3"],
        },
    }

    report = {
        "stage": "trace_first_joint_audit_analysis",
        "definition": {
            "marginal_conditions": list(MARGINAL_CONDITIONS),
            "joint": "parse AND prediction AND artifact AND target AND spatial",
            "joint_auc": "mean(IoU * indicator[parse AND prediction AND artifact AND target])",
            "criterion_relaxation": (
                "Set only the named strict-profile condition(s) to true; retain every other "
                "condition, the recorded IoU, and sample identity."
            ),
        },
        "configuration": {
            "seed": args.seed,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_unit": "sample_id resampled within benchmark-design stratum",
            "binary_interval": "stratum-specific Wilson intervals combined by MOVER",
            "continuous_and_paired_interval": "stratified percentile bootstrap",
        },
        "inputs": inputs,
        "datasets": {
            dataset: analyze_dataset(
                dataset,
                path,
                args.seed,
                args.bootstrap_reps,
                core_strata,
                scale_strata,
            )
            for dataset, path in TRACE_PATHS.items()
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
