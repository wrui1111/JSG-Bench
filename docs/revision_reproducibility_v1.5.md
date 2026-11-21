# Reproducing the submission audit

Run `python experiments/verify_gate_release.py` from the extracted supplement with NumPy installed. The verification manifest identifies every record file by its SHA-256.

The 900 primary records give 156/600 strict Scale passes and a four-field area-stratified range of 60–255. Supplementary Table S8 describes the 600-image matched repeat and five fixed donor assignments. Each assignment records 600 distinct anonymous donors paired within area bands with no self-pairs. The identical-input rows pair the SHA-256 hashes of both UTF-8 response strings without trimming, reserializing, or adding a newline. All 600 pairs agree.

For Core, the 200 system-specific decisions in `verification/core_target_mapping.jsonl` join to the 100 shared image identifiers in the primary records. The original target gate uses the adapter's normalized prediction and the existing reference. The sensitivity rule maps person and animal to object for predictions and references. Parse, class, and region gates come from the same frozen primary records. The original comparison gives 61/32 shared passes, including 25 common passes. The coarse comparison gives 61/59, including 45 common passes.

Both Scale annotation forms offered face, person, animal, object, background, text, whole_image, and other. The output protocol also permits none. One second-pass spelling, perosn, was normalized to person. This changes target agreement from 114 to 115 of 120 paired annotations and κ from 0.918 to 0.931. Table S9 records the three full prompt variants used in the six-feature experiment.

To rebuild the record files from the native archive, run these from a code checkout with the paths to the archive and destination:

```bash
python experiments/verify_native_regression.py --source-root /path/to/archive --verification-dir /path/to/verification
python experiments/export_submission_evidence.py --source-root /path/to/archive --verification-dir /path/to/verification
python experiments/verify_gate_release.py --verification-dir /path/to/verification
```

The first command rescored 1,800 semantic field decisions on 900 archived reports and checked 800 region decisions and 100 SIDA parse decisions. Each agreed with its archived counterpart. The second command reads the donor maps, original and repeated raw responses, adapted SIDA predictions, TD-EGFA traces, and image-aligned paired gate vectors. The distributed files contain pseudonymous IDs, gate outcomes, and response hashes.
