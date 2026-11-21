#!/usr/bin/env python3
"""Export gate-only verification records without images, boxes, text, or gold labels."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from stratified_statistics import core_strata_by_id, scale_strata_by_id, xfer_stratum


GATES = ["parse", "prediction", "artifact", "target", "spatial"]


def report_id(cohort, image_id):
    return hashlib.sha256((cohort + "|" + image_id).encode()).hexdigest()[:24]


def export_pairs(root, directory, records):
    archive = root / "experiments/runs/revision"
    donors = {}
    for line in (archive / "scale_shuffled_permutation.jsonl").read_text().splitlines():
        donor = json.loads(line)
        recipient = donor["target_img_id"]
        assert recipient not in donors
        donors[recipient] = donor
    matched = {r["report_id"]: r for r in records if r["cohort"] == "JSG-Scale"}
    pairs = []
    for line in (archive / "scored_scale_shuffled_package_corrected_reference.jsonl").read_text().splitlines():
        score = json.loads(line)
        recipient = score["img_id"]
        donor = donors.pop(recipient)
        key = report_id("JSG-Scale", recipient)
        row = matched[key]
        assert donor["band"] == score["stratum"] == row["stratum"]
        spatial = row["parse"] and row["valid_box"] and row["iou"] >= 0.1
        before = [row[g] for g in GATES[:-1]] + [spatial]
        after = [score["conditions"][g] for g in GATES]
        assert all(after) == score["joint"]
        pairs.append({"report_id": key, "area_band": donor["band"],
                      "donor_report_id": report_id("JSG-Scale", donor["donor_img_id"]),
                      "matched_gates": before, "permuted_gates": after,
                      "matched_pass": all(before), "permuted_pass": all(after)})
    assert not donors and len(pairs) == len(matched) == 600
    path = directory / "scale_package_pairs.jsonl"
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in pairs))
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(pairs), "gate_order": GATES, "iou_threshold": 0.1, "donor_seed": 20260729}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, help="Directory of newly normalized traces")
    args = parser.parse_args()
    root = args.source_root.resolve()
    sources = [("JSG-Core", "TD-EGFA", "jsg_core_td_egfa_audit_trace_pixel5d_v2.jsonl"),
               ("JSG-Scale", "TD-EGFA", "jsg_scale_td_egfa_audit_trace_pixel5d_v2.jsonl"),
               ("JSG-Xfer", "TD-EGFA", "jsg_xfer_td_egfa_audit_trace_pixel5d_v2.jsonl"),
               ("JSG-Core", "SIDA", "jsg_core_sida_audit_trace_v1.jsonl")]
    strata = {"JSG-Core": core_strata_by_id(root/"experiments/runs/sid_explain300_tampered100_annotations_revised.jsonl"),
              "JSG-Scale": scale_strata_by_id(root/"experiments/runs/sid_fa600_annotations_postreview_eval.jsonl")}
    records, hashes = [], {}
    for cohort, system, filename in sources:
        path = (args.trace_dir or root/"experiments/runs") / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            profiles = row["audit_profiles"]
            conditions = profiles["primary_strict"]["conditions"]
            set_conditions = profiles["set_compatible"]["conditions"]
            record = {"cohort": cohort, "system": system,
                "report_id": report_id(cohort, row["sample_id"]),
                "stratum": row.get("sampling_stratum") or strata[cohort][row["sample_id"]],
                "parse": conditions["parse"], "prediction": conditions["prediction"],
                "artifact": conditions["artifact"], "target": conditions["target"],
                "valid_box": row["evaluated_region"]["bbox"] is not None,
                "region_source": row["evaluated_region"]["provenance"],
                "iou": row["evaluated_region"]["iou"],
                "artifact_set": set_conditions.get("artifact_set_hit", set_conditions.get("artifact"))}
            if "primary_or_secondary" in profiles:
                record["artifact_primary_or_secondary"] = profiles["primary_or_secondary"]["conditions"]["artifact"]
            records.append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/"gate_records.jsonl"
    path.write_text("\n".join(json.dumps(row, separators=(",",":")) for row in records)+"\n")
    pairs = export_pairs(root, args.output_dir, records)
    pool_counts = Counter()
    for source, file in [("CASIA2", "casia2_candidate996_top3_predictions.jsonl"),
                         ("IMD2020", "imd2020_candidate2010_top3_predictions.jsonl")]:
        for line in (root / "experiments/runs" / file).read_text().splitlines():
            row = json.loads(line)
            pool_counts[xfer_stratum(source, float(row["bbox_iou"] or 0.0))] += 1
    (args.output_dir/"manifest.json").write_text(json.dumps({"version":"jsg-gate-replay-v1.3", "records":len(records),
        "scale_pairs":pairs, "xfer_pool_counts":dict(sorted(pool_counts.items())),
        "gate_records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_trace_sha256": hashes,
        "scope":"Primary gate outcomes, paired Scale transitions at IoU 0.1, and Xfer pool-weighted rates.",
        "excluded":"Images, masks, coordinates, gold semantic labels, original IDs, native text, and checkpoints."}, indent=2)+"\n")
    print(json.dumps({"records":len(records),"output":str(path)}))


if __name__ == "__main__":
    main()
