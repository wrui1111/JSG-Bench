#!/usr/bin/env python3
"""Download external datasets used for sanity checks.

This helper intentionally keeps downloads outside the main experiment scripts.
Large public datasets can take hours on a slow connection; run it under nohup
and inspect the JSON status files in external/downloads/status.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


DATASETS = {
    "casia_split": {
        "type": "kagglehub",
        "handle": "sapban2004/casia-2-0-image-tampering-split-dataset",
        "target": "external/downloads/casia_kaggle/sapban2004_casia_split",
    },
    "casia_full": {
        "type": "kagglehub",
        "handle": "divg07/casia-20-image-tampering-detection-dataset",
        "target": "external/downloads/casia_kaggle/divg07_casia_full",
    },
}


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_tree(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != item.stat().st_size:
                shutil.copy2(item, target)


def write_status(status_dir: Path, name: str, payload: dict) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (status_dir / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def download_kagglehub(name: str, cfg: dict, project_root: Path) -> None:
    import kagglehub

    status_dir = project_root / "external/downloads/status"
    target = project_root / cfg["target"]
    write_status(
        status_dir,
        name,
        {"name": name, "state": "running", "handle": cfg["handle"], "target": str(target)},
    )
    try:
        cache_path = Path(kagglehub.dataset_download(cfg["handle"]))
        copy_tree(cache_path, target)
        write_status(
            status_dir,
            name,
            {
                "name": name,
                "state": "completed",
                "handle": cfg["handle"],
                "cache_path": str(cache_path),
                "target": str(target),
                "bytes": dir_size(target),
            },
        )
    except Exception as exc:  # noqa: BLE001 - status file should capture the real failure.
        write_status(
            status_dir,
            name,
            {
                "name": name,
                "state": "failed",
                "handle": cfg["handle"],
                "target": str(target),
                "error": repr(exc),
                "bytes": dir_size(target),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    for name in args.names:
        cfg = DATASETS[name]
        if cfg["type"] == "kagglehub":
            download_kagglehub(name, cfg, project_root)
        else:
            raise ValueError(f"unsupported dataset type: {cfg['type']}")


if __name__ == "__main__":
    main()
