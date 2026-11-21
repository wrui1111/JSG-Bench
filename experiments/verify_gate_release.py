#!/usr/bin/env python3
"""Reproduce primary counts, marginal bounds, intervals, and J-AUC from gate records."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from stratified_statistics import mover_wilson_interval, validate_expected_strata, wilson_interval

EXPECTED = {"JSG-Core/TD-EGFA": (100, [35,14,0], 0.09863746473091263),
            "JSG-Scale/TD-EGFA": (600, [156,79,22], 0.08876152534227272),
            "JSG-Xfer/TD-EGFA": (100, [21,11,1], 0.06954518409551305),
            "JSG-Core/SIDA": (100, [0,0,0], 0.0)}

EXPECTED_SCALE_STRATIFIED = {
    "tiny": {"n": 150, "prediction": 149, "artifact": 125, "target": 75, "spatial": 12,
             "lower": 0, "upper": 12},
    "small": {"n": 150, "prediction": 149, "artifact": 118, "target": 86, "spatial": 63,
              "lower": 0, "upper": 63},
    "medium": {"n": 150, "prediction": 148, "artifact": 127, "target": 89, "spatial": 111,
               "lower": 25, "upper": 89},
    "large": {"n": 150, "prediction": 149, "artifact": 114, "target": 91, "spatial": 131,
              "lower": 35, "upper": 91},
}


def replay(directory):
    path = directory/"gate_records.jsonl"
    manifest = json.loads((directory/"manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["gate_records_sha256"]
    groups, seen = defaultdict(list), set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        key = r["cohort"]+"/"+r["system"]
        assert (key,r["report_id"]) not in seen
        seen.add((key,r["report_id"]))
        assert 0 <= r["iou"] <= 1
        for field in ("parse","prediction","artifact","target","valid_box","artifact_set"):
            assert isinstance(r[field], bool)
        if manifest["version"] in {"jsg-gate-replay-v1.3", "jsg-gate-replay-v1.4", "jsg-gate-replay-v1.5"}:
            assert r["region_source"] in {"inherited_proposal", "model_emitted", "missing", "sida_native_mask_tight_bbox"}
            if r["region_source"] == "missing":
                assert not r["valid_box"]
            if not r["parse"]:
                assert not any(r[f] for f in ("prediction", "artifact", "target"))
        groups[key].append(r)
    assert len(seen) == manifest["records"]
    assert set(groups) == set(EXPECTED)
    output = {}
    for key, rows in groups.items():
        n = len(rows)
        strata = [r["stratum"] for r in rows]
        validate_expected_strata(rows[0]["cohort"], strata)
        def spatial(r,t): return bool(r["parse"] and r["valid_box"] and r["iou"] >= t)
        def semantic(r): return all(r[f] for f in ("parse","prediction","artifact","target"))
        counts = [sum(semantic(r) and spatial(r,t) for r in rows) for t in (0.1,0.3,0.5)]
        auc = sum(r["iou"] if semantic(r) and r["valid_box"] else 0 for r in rows)/n
        expected_n, expected_counts, expected_auc = EXPECTED[key]
        assert n == expected_n and counts == expected_counts, (key,n,counts)
        assert math.isclose(auc,expected_auc,abs_tol=1e-12)
        marginals = {f:sum(r[f] for r in rows)/n for f in ("prediction","artifact","target")}
        marginals["spatial"] = sum(spatial(r,0.1) for r in rows)/n
        shared = lambda r: all(r[f] for f in ("parse","prediction","target")) and spatial(r,0.1)
        set_count = sum(shared(r) and r["artifact_set"] for r in rows)
        output[key] = {"n":n,"marginals":marginals,"frechet_bounds":[max(0,sum(marginals.values())-3),min(marginals.values())],
            "joint_counts_at_0.1_0.3_0.5":counts,"joint_0.1_ci":mover_wilson_interval([semantic(r) and spatial(r,0.1) for r in rows],strata),
            "j_auc":auc,"shared_count":sum(shared(r) for r in rows),"set_count":set_count}
        if "artifact_primary_or_secondary" in rows[0]:
            output[key]["primary_or_secondary_count"] = sum(shared(r) and r["artifact_primary_or_secondary"] for r in rows)
            assert set_count == counts[0] == 21
            assert output[key]["primary_or_secondary_count"] == 30
        if key == "JSG-Scale/TD-EGFA":
            observed = {}
            for stratum, members in _group_by_stratum(rows).items():
                n_stratum = len(members)
                fields = {
                    "prediction": sum(r["parse"] and r["prediction"] for r in members),
                    "artifact": sum(r["parse"] and r["artifact"] for r in members),
                    "target": sum(r["parse"] and r["target"] for r in members),
                    "spatial": sum(spatial(r, 0.1) for r in members),
                }
                observed[stratum] = {
                    "n": n_stratum, **fields,
                    "lower": max(0, sum(fields.values()) - 3 * n_stratum),
                    "upper": min(fields.values()),
                }
            assert observed == EXPECTED_SCALE_STRATIFIED, observed
            assert sum(r["lower"] for r in observed.values()) == 60
            assert sum(r["upper"] for r in observed.values()) == 255
            output[key]["area_stratified_counts"] = observed
            output[key]["area_stratified_bounds"] = [60, 255]
    return output


def replay_pairs(directory, manifest, primary):
    spec = manifest["scale_pairs"]
    path = directory / spec["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
    assert spec["gate_order"] == ["parse", "prediction", "artifact", "target", "spatial"]
    assert spec["iou_threshold"] == 0.1
    matched = {r["report_id"]: r for r in primary if r["cohort"] == "JSG-Scale"}
    recipients, donors = set(), set()
    counts, changed = Counter(), []
    for line in path.read_text().splitlines():
        pair = json.loads(line)
        recipient, donor = pair["report_id"], pair["donor_report_id"]
        assert recipient != donor and recipient not in recipients and donor not in donors
        recipients.add(recipient)
        donors.add(donor)
        row = matched[recipient]
        assert pair["area_band"] == row["stratum"] == matched[donor]["stratum"]
        spatial = row["parse"] and row["valid_box"] and row["iou"] >= 0.1
        expected = [row[k] for k in spec["gate_order"][:-1]] + [spatial]
        assert pair["matched_gates"] == expected
        for side in ("matched", "permuted"):
            gates = pair[side + "_gates"]
            assert len(gates) == 5 and all(type(g) is bool for g in gates)
            assert pair[side + "_pass"] is all(gates)
        before, after = pair["matched_pass"], pair["permuted_pass"]
        transition = ("both" if after else "exit") if before else ("entry" if after else "neither")
        counts[transition] += 1
        if before != after:
            changed.append({"transition": transition, **pair})
    assert recipients == donors == set(matched) and len(recipients) == spec["rows"] == 600
    assert counts == {"both": 94, "exit": 62, "entry": 46, "neither": 398}
    exit_spatial_only = sum(pair["permuted_gates"] == [True, True, True, True, False]
                            for pair in changed if pair["transition"] == "exit")
    entry_spatial_only = sum(pair["matched_gates"] == [True, True, True, True, False]
                             for pair in changed if pair["transition"] == "entry")
    assert (exit_spatial_only, entry_spatial_only) == (37, 30)
    return dict(counts), changed


def replay_stability(directory, manifest):
    spec = manifest["scale_stability_pairs"]
    path = directory / spec["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
    assert spec["gate_order"] == ["parse", "prediction", "artifact", "target", "spatial"]
    assert spec["iou_threshold"] == 0.1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == spec["rows"] == 3600
    runs = defaultdict(list)
    for row in rows:
        assert len(row["matched_gates"]) == len(row["rerun_gates"]) == 5
        assert all(type(g) is bool for g in row["matched_gates"] + row["rerun_gates"])
        assert row["matched_pass"] is all(row["matched_gates"])
        assert row["rerun_pass"] is all(row["rerun_gates"])
        if row["matched_pass"] != row["rerun_pass"]:
            assert row["transition"] in {"exit", "entry"}
        runs[row["run"]].append(row)
    assert len(runs) == spec["runs"] == 6
    assert all(len(v) == spec["reports_per_run"] for v in runs.values())
    base_run = runs["unchanged_input"]
    if manifest.get("core_target_mapping"):
        check_stability_evidence(directory, manifest, runs)
    assert all(row["donor_seed"] is None for row in base_run)
    assert all(row["matched_gates"] == row["rerun_gates"] for row in base_run)
    assert {row["report_id"] for row in base_run} == {row["report_id"] for row in runs["donor_20260729"]}
    summary = {}
    for seed in spec["donor_seeds"]:
        sub = runs["donor_" + str(seed)]
        joint = sum(row["rerun_pass"] for row in sub)
        exits = sum(row["transition"] == "exit" for row in sub)
        entries = sum(row["transition"] == "entry" for row in sub)
        spatial_only = sum(row["spatial_only_on_failing_side"]
                           for row in sub if row["transition"] in {"exit", "entry"})
        assert joint + exits - entries == sum(row["matched_pass"] for row in sub)
        summary[str(seed)] = {"joint": joint, "exits": exits, "entries": entries,
                              "spatial_only": spatial_only}
    expected = {
        "20260729": (140, 62, 46, 67),
        "20260730": (130, 65, 39, 68),
        "20260731": (131, 64, 39, 64),
        "20260732": (139, 59, 42, 58),
        "20260733": (134, 62, 40, 57),
    }
    for seed, values in expected.items():
        row = summary[seed]
        assert (row["joint"], row["exits"], row["entries"], row["spatial_only"]) == values
        exits, entries = row["exits"], row["entries"]
        n = exits + entries
        row["both"] = row["joint"] - entries
        row["mcnemar_p"] = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(exits, entries) + 1)) / 2**n)
    cumulative = 0.0
    for i, row in enumerate(sorted(summary.values(), key=lambda x: x["mcnemar_p"])):
        cumulative = max(cumulative, min(1.0, (len(summary) - i) * row["mcnemar_p"]))
        row["holm_p"] = cumulative
    assert round(min(row["holm_p"] for row in summary.values()), 5) == 0.06918
    assert sum(row["rerun_pass"] for row in base_run) == 156
    assert all(row["transition"] in {"both", "neither"} for row in base_run)
    return summary


def check_stability_evidence(directory, manifest, runs):
    """Validate every pair against the primary record and the realized donor allocation."""
    primary = {r["report_id"]: r for r in map(json.loads, (directory / "gate_records.jsonl").read_text().splitlines())
               if r["cohort"] == "JSG-Scale"}
    first = {r["report_id"]: r for r in map(json.loads, (directory / manifest["scale_pairs"]["file"]).read_text().splitlines())}
    expected_runs = {"unchanged_input": None, **{"donor_" + str(s): s for s in range(20260729, 20260734)}}
    assert set(runs) == set(expected_runs)
    for name, seed in expected_runs.items():
        rows = runs[name]
        assert len(rows) == len(primary) == len({r["report_id"] for r in rows}) == 600
        assert {r["report_id"] for r in rows} == set(primary)
        if seed is not None:
            assert {r["donor_report_id"] for r in rows} == set(primary)
        for r in rows:
            key = r["report_id"]
            p = primary[key]
            assert r["area_band"] == p["stratum"] and r["donor_seed"] == seed
            assert r["matched_gates"] == [p[k] for k in ("parse", "prediction", "artifact", "target")] + [
                p["parse"] and p["valid_box"] and p["iou"] >= 0.1]
            a, b = r["matched_pass"], r["rerun_pass"]
            assert r["transition"] == ("both" if a and b else "exit" if a else "entry" if b else "neither")
            failing = r["rerun_gates"] if a else r["matched_gates"]
            assert r["spatial_only_on_failing_side"] is (a != b and failing == [True, True, True, True, False])
            if seed is None:
                h = r["matched_response_sha256"]
                assert h == r["rerun_response_sha256"] and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
            else:
                d = r["donor_report_id"]
                assert key != d and primary[d]["stratum"] == p["stratum"]
                if seed == 20260729:
                    assert d == first[key]["donor_report_id"] and r["rerun_gates"] == first[key]["permuted_gates"]


def replay_mapping(directory, manifest):
    spec = manifest["core_target_mapping"]
    data = (directory / spec["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == spec["sha256"]
    rows = list(map(json.loads, data.splitlines()))
    primary = {(p["system"], p["report_id"]): p for p in map(json.loads, (directory / "gate_records.jsonl").read_text().splitlines())
               if p["cohort"] == "JSG-Core"}
    assert len(rows) == spec["rows"] == len(primary) == 200
    assert {(r["system"], r["report_id"]) for r in rows} == set(primary)
    out = {}
    for mapping, expected in [("original_target", (61, 32, 25)), ("coarse_target", (61, 59, 45))]:
        passed = defaultdict(set)
        targets = Counter()
        for r in rows:
            system, key = r["system"], r["report_id"]
            p = primary[system, key]
            assert r["original_target"] is p["target"] and type(r["coarse_target"]) is bool
            assert not r["original_target"] or r["coarse_target"]
            targets[system] += r[mapping]
            if r[mapping] and p["parse"] and p["prediction"] and p["valid_box"] and p["iou"] >= 0.1:
                passed[system].add(key)
        td, sida = passed["TD-EGFA"], passed["SIDA"]
        counts = len(td), len(sida), len(td & sida)
        assert counts == expected
        assert (targets["TD-EGFA"], targets["SIDA"]) == (94, 41 if mapping == "original_target" else 79)
        out[mapping] = {"TD-EGFA": counts[0], "SIDA": counts[1], "both": counts[2], "jaccard": counts[2] / len(td | sida)}
    return out


def _group_by_stratum(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["stratum"]].append(row)
    return groups


def replay_xfer(primary, pools):
    # Preserve the archived report and stratum insertion order for the seeded bootstrap.
    cells = defaultdict(list)
    for row in primary:
        if row["cohort"] == "JSG-Xfer":
            passed = all(row[k] for k in ("parse", "prediction", "artifact", "target", "valid_box"))
            cells[row["stratum"]].append(int(passed and row["iou"] >= 0.1))
    assert set(cells) == set(pools) and sum(pools.values()) == 3006
    pool_sizes = np.array([pools[k] for k in cells], dtype=float)
    weights = pool_sizes / pool_sizes.sum()
    rates = np.array([np.mean(v) for v in cells.values()])
    point = float(weights @ rates)
    limits = np.array([wilson_interval(sum(v), len(v)) for v in cells.values()])
    mover = [point - np.linalg.norm(weights * (rates - limits[:, 0])),
             point + np.linalg.norm(weights * (limits[:, 1] - rates))]
    source_rates = []
    for source in ("CASIA2", "IMD2020"):
        mask = np.array([k.split("|")[0] == source for k in cells])
        source_rates.append(float(np.average(rates[mask], weights=pool_sizes[mask])))
    rng = np.random.default_rng(20260718)
    arrays = [np.array(v) for v in cells.values()]
    draws = [weights @ [v[rng.integers(len(v), size=len(v))].mean() for v in arrays]
             for _ in range(10000)]
    percent = lambda x: round(float(x) * 100, 2)
    return {"selected_pct": percent(sum(map(sum, cells.values())) / sum(map(len, cells.values()))),
            "pool_weighted_pct": percent(point), "equal_source_pct": percent(np.mean(source_rates)),
            "bootstrap_95_pct": list(map(percent, np.percentile(draws, [2.5, 97.5]))),
            "wilson_mover_95_pct": list(map(percent, mover))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-dir",type=Path,default=Path(__file__).resolve().parents[1]/"verification")
    parser.add_argument("--changed-ids", action="store_true",
                        help="Print the changed Scale report pairs, one JSON record per line.")
    args = parser.parse_args()
    directory = args.verification_dir
    result = replay(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if "scale_pairs" in manifest:
        primary = [json.loads(line) for line in (directory / "gate_records.jsonl").read_text().splitlines()]
        counts, changed = replay_pairs(directory, manifest, primary)
        if args.changed_ids:
            for pair in changed:
                print(json.dumps(pair))
        else:
            result["Scale transitions at 0.1"] = counts
            result["Xfer pool reweighting"] = replay_xfer(primary, manifest["xfer_pool_counts"])
    elif args.changed_ids:
        parser.error("The supplied manifest predates paired records. Use verification package v1.2.")
    if "scale_stability_pairs" in manifest and not args.changed_ids:
        result["Scale stability"] = replay_stability(directory, manifest)
    if "core_target_mapping" in manifest and not args.changed_ids:
        result["Core target mappings"] = replay_mapping(directory, manifest)
    if not args.changed_ids:
        print(json.dumps(result, indent=2))
