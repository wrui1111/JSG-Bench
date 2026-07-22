#!/usr/bin/env python3
"""Build a JSONL manifest for the downloaded CASIA2 split dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def mask_name_for(image_path: Path) -> str:
    return f"{image_path.stem}_gt.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for split in ("Train", "Validation", "Test"):
        image_dir = root / split / "Tp"
        mask_dir = root / split / "GroundTruth"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            mask_path = mask_dir / mask_name_for(image_path)
            if not mask_path.exists():
                missing.append(str(image_path))
                continue
            rows.append(
                {
                    "img_id": image_path.stem,
                    "dataset": "CASIA2_split",
                    "split": split.lower(),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "class_dir": "tampered",
                    "label": "tampered",
                }
            )

    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset_root": str(root),
        "output": str(output),
        "num_pairs": len(rows),
        "num_missing_masks": len(missing),
        "missing_examples": missing[:20],
        "split_counts": {},
    }
    for row in rows:
        summary["split_counts"][row["split"]] = summary["split_counts"].get(row["split"], 0) + 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
