#!/usr/bin/env python3
"""Build a JSONL manifest for IMD2020 manipulated images and masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing_image = []
    for mask in sorted(args.root.glob("*/*_mask.png")):
        stem = mask.name[: -len("_mask.png")]
        candidates = [mask.with_name(stem + ".jpg"), mask.with_name(stem + ".png")]
        image = next((p for p in candidates if p.exists()), None)
        if not image:
            missing_image.append(str(mask))
            continue
        width, height = image_size(image)
        rel_image = image.relative_to(args.root)
        rel_mask = mask.relative_to(args.root)
        rows.append(
            {
                "dataset": "IMD2020",
                "img_id": str(rel_image.with_suffix("")).replace("/", "__"),
                "image_path": str(rel_image),
                "mask_path": str(rel_mask),
                "class_dir": "tampered",
                "human_label": "tampered",
                "width": width,
                "height": height,
            }
        )

    write_jsonl(args.output, rows)
    write_json(
        args.summary_output,
        {
            "stage": "imd2020_manifest",
            "root": str(args.root),
            "output": str(args.output),
            "samples": len(rows),
            "missing_image_count": len(missing_image),
            "missing_image_preview": missing_image[:20],
        },
    )
    print(json.dumps({"samples": len(rows), "missing_image_count": len(missing_image)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
