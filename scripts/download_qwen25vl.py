#!/usr/bin/env python3
"""Download Qwen2.5-VL-3B-Instruct into the project-local models directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("models/Qwen2.5-VL-3B-Instruct"),
        help="Project-local target directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".hf_cache/hub"),
        help="Project-local Hugging Face cache directory.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Use 1 on unstable networks; increase only if Hugging Face is fast.",
    )
    args = parser.parse_args()

    args.local_dir.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
    )
    print(f"Downloaded {args.model_id} to {path}")


if __name__ == "__main__":
    main()
