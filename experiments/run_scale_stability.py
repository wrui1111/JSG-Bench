#!/usr/bin/env python3
"""Prepare and resume the prespecified 3,000-inference Scale stability suite.

Each run has its own outputs and pairs. Source records remain unchanged.
Only complete 600-row runs enter the results; failure records stay in the denominator.
"""
from __future__ import annotations
import argparse
import fcntl
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import revision_experiments as revision
from export_gate_verification import report_id

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def configure(root):
    original = revision.ROOT
    def convert(value):
        if isinstance(value, Path) and value.is_relative_to(original):
            return root / value.relative_to(original)
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value
    for key, value in list(vars(revision).items()):
        if key.isupper():
            setattr(revision, key, convert(value))


def prepare(root, directory, design):
    directory.mkdir(parents=True, exist_ok=True)
    original = revision.COHORTS["scale"]["requests"]
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["design"] == design
        assert manifest["source_sha256"] == sha(original)
        for run in manifest["runs"]:
            assert sha(root / run["requests"]) == run["requests_sha256"]
        return manifest
    manifest = {"design": design, "source_sha256": sha(original),
                "created_at": datetime.now(timezone.utc).isoformat(), "runs": []}
    def register(name, requests, seed=None):
        manifest["runs"].append({
            "name": name, "seed": seed,
            "requests": str(requests.relative_to(root)), "requests_sha256": sha(requests),
            "outputs": str((directory / ("outputs_" + name + ".jsonl")).relative_to(root)),
        })
    register("unchanged_input", original)
    for seed in design["additional_donor_seeds"]:
        name = "donor_" + str(seed)
        requests, donors = revision.make_shuffled("scale", seed, directory / ("assets_" + name))
        assert len(requests) == 600
        assert len({p["donor_img_id"] for p in donors}) == 600
        path = directory / ("requests_" + name + ".jsonl")
        revision.write_jsonl(path, requests)
        revision.write_jsonl(directory / ("donors_" + name + ".jsonl"), donors)
        register(name, path, seed)
    write_json(manifest_path, manifest)
    return manifest


def summarize(root, directory, manifest):
    base = revision.score_rows("scale", revision.COHORTS["scale"]["outputs"],
                               revision.COHORTS["scale"]["requests"])
    assert sum(r["joint"] for r in base.values()) == 156
    runs = [manifest["runs"][0], {
        "name": "donor_20260729", "seed": 20260729,
        "requests": "experiments/runs/revision/requests_scale_shuffled_package.jsonl",
        "outputs": "experiments/runs/revision/outputs_scale_shuffled_package.jsonl",
    }] + manifest["runs"][1:]
    results, all_pairs = [], []
    for run in runs:
        path = root / run["outputs"]
        if not path.exists():
            continue
        output = revision.read_jsonl(path)
        if len(output) != 600:
            continue
        assert len({r["img_id"] for r in output}) == 600
        after = revision.score_rows("scale", path, root / run["requests"])
        counts = Counter({k: 0 for k in ("both", "exit", "entry", "neither",
                                         "exit_spatial_only", "entry_spatial_only")})
        for key, before in base.items():
            target = after[key]
            state = ("both" if target["joint"] else "exit") if before["joint"] else (
                "entry" if target["joint"] else "neither")
            counts[state] += 1
            failure = target if state == "exit" else before
            spatial_only = (state in ("exit", "entry") and
                            [failure["conditions"][g] for g in revision.CONDITIONS] ==
                            [True, True, True, True, False])
            if spatial_only:
                counts[state + "_spatial_only"] += 1
            all_pairs.append({
                "run": run["name"], "donor_seed": run["seed"],
                "report_id": report_id("JSG-Scale", key), "area_band": before["stratum"],
                "matched_gates": [before["conditions"][g] for g in revision.CONDITIONS],
                "rerun_gates": [target["conditions"][g] for g in revision.CONDITIONS],
                "matched_pass": before["joint"], "rerun_pass": target["joint"],
                "transition": state, "spatial_only_on_failing_side": spatial_only,
            })
        changed = counts["exit"] + counts["entry"]
        union = counts["both"] + changed
        summary = {"name": run["name"], "seed": run["seed"], "n": 600,
                   "joint_count": counts["both"] + counts["entry"], **dict(counts),
                   "changed_fraction": changed / 600, "jaccard": counts["both"] / union,
                   "spatial_only_fraction_of_changes": (
                       (counts["exit_spatial_only"] + counts["entry_spatial_only"]) / changed
                       if changed else None),
                   "runtime_errors": sum(bool(r.get("error")) for r in output),
                   "outputs_sha256": sha(path)}
        if run["seed"] == 20260729:
            assert (counts["both"], counts["exit"], counts["entry"], counts["neither"]) == (94, 62, 46, 398)
        results.append(summary)
    write_json(directory / "results.json", {"completed_runs": results})
    revision.write_jsonl(directory / "paired_records.jsonl", all_pairs)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve()
    configure(root)
    design = json.loads((REPO / "configs/scale_stability_v1.json").read_text())
    directory = root / "experiments/runs/scale_stability_v1"
    manifest = prepare(root, directory, design)
    if args.prepare_only or args.summarize_only:
        print(json.dumps(summarize(root, directory, manifest), indent=2))
        return
    lock = (directory / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("The Scale stability suite is already running.")
    # Preserve the virtualenv symlink; resolving it selects the system interpreter.
    python = (args.python or root / ".venv_qwen25vl/bin/python").absolute()
    log_path = directory / "inference.log"
    for run in manifest["runs"]:
        write_json(directory / "status.json", {
            "pid": os.getpid(), "state": "running", "run": run["name"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        outputs = root / run["outputs"]
        complete = outputs.exists() and len(revision.read_jsonl(outputs)) == 600
        if not complete:
            command = [str(python), "-B", "-u", str(REPO / "experiments/qwen25vl_sid_infer.py"),
                       "--requests", run["requests"], "--output", run["outputs"],
                       "--device", design["device"], "--max-pixels", str(design["max_pixels"]),
                       "--max-new-tokens", str(design["max_new_tokens"])]
            with log_path.open("a") as log:
                log.write("\nStarting " + run["name"] + "\n")
                log.flush()
                completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                write_json(directory / "status.json", {"pid": os.getpid(), "state": "failed",
                    "run": run["name"], "exit_code": completed.returncode})
                raise SystemExit(completed.returncode)
        assert len(revision.read_jsonl(outputs)) == 600
        summarize(root, directory, manifest)
    results = summarize(root, directory, manifest)
    assert len(results) == 6
    write_json(directory / "status.json", {"state": "complete",
        "completed_runs": len(results), "new_inferences": 3000,
        "completed_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
