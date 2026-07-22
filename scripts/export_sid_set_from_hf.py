#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


LABEL_TO_DIR = {
    0: ("real", "jpg"),
    1: ("full_synthetic", "png"),
    2: ("tampered", "png"),
}


def safe_name(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def ensure_dirs(split_root: Path) -> None:
    for name in ("real", "full_synthetic", "tampered", "masks"):
        (split_root / name).mkdir(parents=True, exist_ok=True)


def export_split(split: str, output_root: Path, limit: int = 0) -> None:
    split_root = output_root / split
    ensure_dirs(split_root)

    manifest_path = split_root / "manifest.jsonl"
    dataset = load_dataset("saberzl/SID_Set", split=split, streaming=True)

    counts = {"real": 0, "full_synthetic": 0, "tampered": 0, "masks": 0, "skipped": 0}
    processed = 0

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for row in tqdm(dataset, desc=f"Exporting {split}", unit="img"):
            label = int(row["label"])
            if label not in LABEL_TO_DIR:
                raise ValueError(f"Unknown label {label} for row {row.get('img_id')}")

            cls_dir, ext = LABEL_TO_DIR[label]
            image_id = safe_name(row["img_id"])
            image_path = split_root / cls_dir / f"{image_id}.{ext}"
            mask_path = None

            if image_path.exists():
                counts["skipped"] += 1
            else:
                image = row["image"].convert("RGB")
                if ext == "jpg":
                    image.save(image_path, format="JPEG", quality=95)
                else:
                    image.save(image_path, format="PNG")
                counts[cls_dir] += 1

            if label == 2:
                mask_path = split_root / "masks" / f"{image_id}_mask.png"
                if not mask_path.exists():
                    if row["mask"] is None:
                        raise ValueError(f"Missing mask for tampered image {image_id}")
                    row["mask"].convert("L").save(mask_path, format="PNG")
                    counts["masks"] += 1

            record = {
                "img_id": image_id,
                "label": label,
                "class_dir": cls_dir,
                "image_path": str(image_path.relative_to(output_root)),
                "mask_path": str(mask_path.relative_to(output_root)) if mask_path else None,
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

            if limit and processed >= limit:
                break

    print(json.dumps({"split": split, "processed": processed, "counts": counts}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("SID_Set"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=0, help="0 means export the full split")
    args = parser.parse_args()

    export_split(args.split, args.output_root, args.limit)


if __name__ == "__main__":
    main()
