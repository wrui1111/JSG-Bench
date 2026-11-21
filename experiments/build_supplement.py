#!/usr/bin/env python3
"""Package the explicitly allowed S1 files and print the archive SHA-256."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

FILES = [
    "verification/README.md", "verification/manifest.json",
    "verification/gate_records.jsonl", "verification/scale_package_pairs.jsonl",
    "verification/scale_stability_pairs.jsonl",
    "verification/native_regression.json", "verification/core_target_mapping.jsonl",
    "experiments/verify_gate_release.py", "experiments/test_gate_release.py",
    "experiments/stratified_statistics.py",
    "docs/supplementary_tables.md", "docs/protocol_observability_coding.md",
    "docs/run_provenance.md", "docs/revision_reproducibility_v1.5.md",
    "configs/jsg_audit_contract_v1.json", "configs/scale_stability_v1.json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name in FILES:
        if not (root / name).is_file():
            raise SystemExit("Missing S1 input: " + name)
    with ZipFile(args.output, "w", ZIP_DEFLATED) as archive:
        for name in FILES:
            entry = ZipInfo(name, date_time=(2026, 11, 21, 0, 0, 0))
            entry.external_attr = 0o100644 << 16
            entry.create_system = 3
            archive.writestr(entry, (root / name).read_bytes(), compress_type=ZIP_DEFLATED)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest(), args.output)


if __name__ == "__main__":
    main()
