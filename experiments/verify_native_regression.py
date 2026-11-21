#!/usr/bin/env python3
"""Replay archived native reports; preserve all source traces and model outputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import candidate_evidence_explanation as candidate
import evidence_explanation_experiment as free
import evaluate_sida_jsg_core as sida
import export_jsg_audit_traces as exporter


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    verification = args.verification_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    gold_path = source / "SID_Set/annotations/SID-Explain-300.jsonl"
    gold = {r["img_id"]: free.gold_item(r) for r in rows(gold_path)}
    result = {"semantic_reference_sha256": digest(gold_path), "free_reports": {},
              "td_regions": {}, "sida_parse": {}, "source_files": {}}
    for model in ("qwen25vl", "ollama7b", "minicpm45"):
        path = source / f"experiments/runs/sid_explain300_evidence_explanation_v2_{model}_outputs.jsonl"
        output = rows(path)
        assert len(output) == len({r["img_id"] for r in output}) == 300
        assert {r["img_id"] for r in output} == set(gold)
        count = Counter()
        for row in output:
            native = {k: v for k, v in row.items() if k != "vlm_result"}
            pred = free.prediction_item(native)
            count["reports"] += 1
            count["parsed"] += pred["parsed"]
            for field, flag in (("artifact_type", "artifact_present"), ("target_scope", "target_present")):
                old = pred[field] == gold[row["img_id"]][field]
                new = bool(pred["parsed"] and pred[flag] and old)
                count[field + "_correct"] += new
                count[field + "_changed"] += old != new
        result["free_reports"][model] = dict(count)
        result["source_files"][str(path.relative_to(source))] = digest(path)
    assert [(r["artifact_type_correct"], r["target_scope_correct"])
            for r in result["free_reports"].values()] == [(136, 101), (103, 107), (102, 105)]
    assert sum(r["artifact_type_changed"] + r["target_scope_changed"]
               for r in result["free_reports"].values()) == 0

    # Rebind input paths to the archive; outputs go only to an isolated temporary directory.
    def rebase(value):
        if isinstance(value, Path) and value.is_relative_to(repo):
            return source / value.relative_to(repo)
        if isinstance(value, dict):
            return {k: rebase(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rebase(v) for v in value]
        return value
    for key, value in list(vars(exporter).items()):
        if key.isupper():
            setattr(exporter, key, rebase(value))
    with tempfile.TemporaryDirectory(prefix="jsg-native-regression-") as temporary:
        trace_dir = Path(temporary)
        traces = {}
        for cohort, spec in exporter.TD_SPECS.items():
            traces[spec["output"].name] = exporter.td_trace_rows(cohort, spec)
        traces[exporter.XFER_OUTPUT.name] = exporter.xfer_trace_rows()
        for filename, trace in traces.items():
            cohort = trace[0]["dataset"]
            result["td_regions"][cohort] = dict(Counter(r["evaluated_region"]["provenance"] for r in trace))
            previous = {r["sample_id"]: r for r in rows(source / "experiments/runs" / filename)}
            for record in trace:
                old = previous[record["sample_id"]]
                assert record["evaluated_region"]["bbox"] == old["evaluated_region"]["bbox"]
                assert record["audit_profiles"] == old["audit_profiles"]
                assert record["evaluated_region"]["provenance"] == (
                    "inherited_proposal" if old["evaluated_region"]["provenance"] == "inherited_B_top3"
                    else "missing")
        native_path = source / "remote_sida/jsg_core_native100/native_outputs.jsonl"
        native = rows(native_path)
        old_parse = [r.get("runtime_status") == "ok" and bool(r.get("native_text_normalized"))
                     and sida.native_prediction(r) is not None for r in native]
        new_parse = [sida.report_parse_valid(r) for r in native]
        assert len(native) == 100 and old_parse == new_parse and all(new_parse)
        result["sida_parse"] = {"reports": 100, "changed": 0}
        traces[exporter.SIDA_OUTPUT.name] = exporter.sida_trace_rows()
        for filename, trace in traces.items():
            exporter.write_jsonl(trace_dir / filename, trace)
        subprocess.run([sys.executable, "-B", str(repo / "experiments/export_gate_verification.py"),
                        "--source-root", str(source), "--trace-dir", str(trace_dir),
                        "--output-dir", str(verification)], check=True)
    verification.mkdir(parents=True, exist_ok=True)
    (verification / "native_regression.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
