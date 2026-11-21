# Supplementary Material S1

JSG-Bench verification package v1.5 accompanies the report-level audit. Supplementary tables S1–S9 are in `docs/supplementary_tables.md`.

Run from the extracted archive root with Python 3 and NumPy 2.0.2:

```bash
python experiments/verify_gate_release.py
python experiments/verify_gate_release.py --changed-ids
python -m unittest discover -s experiments -p 'test_gate_release.py'
```

`verification/gate_records.jsonl` contains 900 records: 600 Scale, 100 Core, and 100 Xfer reports from TD-EGFA, plus 100 SIDA reports on the same Core images. Every record has a `region_source`: `inherited_proposal`, `model_emitted`, `missing`, or `sida_native_mask_tight_bbox`.

The verifier checks file digests and reproduces primary counts, pooled Fréchet bounds, four-field area-stratified bounds, Wilson–MOVER intervals, and J-AUC. The Scale area bands each retain N=150. Prediction pass counts are 149, 149, 148, 149 in tiny-to-large order; lower bounds are 0, 0, 25, 35 and upper bounds are 12, 63, 89, 91. The aggregate range is 60–255.

`verification/scale_package_pairs.jsonl` contains 600 pairs with hashed recipient and donor IDs, the area band, and both gate vectors. Every donor occurs once within the recipient's band, and no recipient is its own donor. The fixed donor seed is 20260729. Vector order is parse, prediction, artifact, target, spatial; the spatial threshold is IoU ≥ 0.1.

`verification/scale_stability_pairs.jsonl` contains 3,600 paired records: the same 600 Scale reports under five donor allocations (seeds 20260729–20260733) and one unchanged-input repeat. Each row gives the run, donor seed, report ID, area band, both gate vectors, the transition, and whether the failing side failed only the spatial check. The verifier reproduces the joint counts 156 then 140, 130, 131, 139, and 134; exits 62, 65, 64, 59, and 62; entries 46, 39, 39, 42, and 40; and spatial-only counts 67, 68, 64, 58, and 57. The unchanged repeat reproduces every matched gate vector.

The verifier reproduces 94 images passing both runs, 62 exits, 46 entries, and 398 failures in both. Among changed images, it asserts 37 spatial-only failures in the reassigned run for exits and 30 in the matched run for entries. The `--changed-ids` command prints the 108 changed pairs.

The six source-by-legacy-IoU pool counts in `verification/manifest.json` support Xfer reweighting. The verifier reproduces the 21.00% selected-sample rate, 10.92% pool-weighted estimate, 10.59% equal-source estimate, 10,000-draw percentile interval (seed 20260718), and fixed-weight Wilson–MOVER interval.

`verification/native_regression.json` records frozen-output replay: no changes to 1,800 artifact/target judgments across 900 free reports; no box-coordinate or source discrepancies among 800 TD-EGFA reports; no parse changes among 100 native SIDA reports. The replay source is `experiments/verify_native_regression.py` in the code repository. It requires the local native-output archive. Generated trace hashes in the manifest refer to the deterministic v1.1 adapters and can be rebuilt by that replay.

Each donor row includes its anonymous donor ID. The 600 repeat rows pair SHA-256 hashes of original and repeated UTF-8 response text. The 200 records in `core_target_mapping.jsonl` provide the two target decisions per system and image. The verifier joins them to the primary records to reproduce the two shared-field comparisons.

`docs/run_provenance.md` indexes the inference families. `docs/revision_reproducibility_v1.5.md` documents the export and replay commands. The literature coverage ledger is in `docs/protocol_observability_coding.md`.
