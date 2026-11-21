"""Build the anonymous stability and Core mapping records from the frozen archive."""
import argparse
import hashlib
import json
from pathlib import Path

from export_gate_verification import report_id


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_rows(directory, filename, rows):
    data = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    (directory / filename).write_bytes(data)
    return {"file": filename, "sha256": digest(data), "rows": len(rows)}


def response_hashes(path):
    rows = read_rows(path)
    hashes = {report_id("JSG-Scale", r["img_id"]): digest(r["vlm_raw_text"].encode("utf-8"))
              for r in rows}
    assert len(hashes) == len(rows) == 600
    return hashes


def export(root, directory):
    runs = root / "experiments/runs"
    stable = runs / "scale_stability_v1"
    orig = response_hashes(runs / "jsg_scale_outputs_pixel5d_v2_legacy_prompt_controlled.jsonl")
    repeat = response_hashes(stable / "outputs_unchanged_input.jsonl")
    assert orig == repeat
    mappings = {}
    seeds = list(range(20260729, 20260734))
    for seed in seeds:
        path = (runs / "revision/scale_shuffled_permutation.jsonl" if seed == seeds[0]
                else stable / f"donors_donor_{seed}.jsonl")
        donors = read_rows(path)
        mappings[seed] = {report_id("JSG-Scale", r["target_img_id"]):
                          (report_id("JSG-Scale", r["donor_img_id"]), r["band"]) for r in donors}
        assert len(donors) == 600 and set(mappings[seed]) == set(orig)
    pairs = read_rows(stable / "paired_records.jsonl")
    for r in pairs:
        key, seed = r["report_id"], r["donor_seed"]
        if seed is None:
            r["matched_response_sha256"] = orig[key]
            r["rerun_response_sha256"] = repeat[key]
        else:
            donor, band = mappings[seed][key]
            assert band == r["area_band"]
            r["donor_report_id"] = donor
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scale_stability_pairs"] = {
        **save_rows(directory, "scale_stability_pairs.jsonl", pairs), "runs": 6,
        "reports_per_run": 600, "donor_seeds": seeds, "unchanged_input_repeats": 1,
        "gate_order": manifest["scale_pairs"]["gate_order"], "iou_threshold": 0.1}

    sida = read_rows(runs / "sida_jsg_core_adapted_predictions.jsonl")
    td = read_rows(runs / "jsg_core_td_egfa_audit_trace_pixel5d_v2.jsonl")
    gold = {r["img_id"]: r["gold"]["target_scope"] for r in sida}
    assert len(sida) == len(td) == len(gold) == 100
    assert set(gold) == {r["sample_id"] for r in td}
    def coarse(value):
        return "object" if value in {"person", "animal"} else value
    mapping_rows = []
    for system, rows in (("TD-EGFA", td), ("SIDA", sida)):
        for r in rows:
            if system == "TD-EGFA":
                key, pred = r["sample_id"], r["normalized_fields"]["target_scope"]
                gates = r["audit_profiles"]["shared_fields"]["conditions"]
            else:
                key, pred, gates = r["img_id"], r["target_scope"], r["conditions"]
            original = bool(gates["parse"] and pred == gold[key])
            assert original == gates["target"]
            mapping_rows.append({"report_id": report_id("JSG-Core", key), "system": system,
                                 "original_target": original,
                                 "coarse_target": bool(gates["parse"] and coarse(pred) == coarse(gold[key]))})
    manifest["core_target_mapping"] = save_rows(directory, "core_target_mapping.jsonl", mapping_rows)
    manifest["version"] = "jsg-gate-replay-v1.5"
    manifest["scope"] = "Primary gates, Scale donor assignments and repeat digests, Core target mappings, and Xfer reweighting."
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.source_root, args.verification_dir)
