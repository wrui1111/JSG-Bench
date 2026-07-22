#!/usr/bin/env python3
"""Download Qwen2.5-VL-3B-Instruct from ModelScope into the local model folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="ModelScope model id.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("models/Qwen2.5-VL-3B-Instruct"),
        help="Project-local target directory.",
    )
    args = parser.parse_args()

    args.local_dir.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        args.model_id,
        local_dir=str(args.local_dir),
    )
    print(f"Downloaded {args.model_id} to {path}")


if __name__ == "__main__":
    main()
