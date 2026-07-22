#!/usr/bin/env python3
"""Build pixel5d requests while preserving the frozen legacy prompt surface.

Only the candidate bbox, low-level score, artifact hints, and image assets are
updated. Static instructions, label spaces, prompt version, and metadata fields
remain byte-for-byte aligned with the frozen request wherever possible.
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def replace_once(prompt: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, prompt, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one prompt replacement for {pattern!r}, found {count}")
    return updated


def build_row(
    frozen: dict[str, Any],
    generated: dict[str, Any],
    lowlevel: dict[str, Any],
) -> dict[str, Any]:
    candidate_bbox = generated["metadata_for_evaluation_only"]["candidate_bbox"]
    prompt = frozen["prompt"]
    prompt = replace_once(prompt, r"^- candidate_bbox=.*$", f"- candidate_bbox={candidate_bbox}")
    prompt = replace_once(prompt, r"^- lowlevel_score=.*$", f"- lowlevel_score={lowlevel['lowlevel_score']}")
    prompt = replace_once(
        prompt,
        r"^- lowlevel_artifact_types=.*$",
        f"- lowlevel_artifact_types={lowlevel['lowlevel_artifact_types']}",
    )

    row = deepcopy(frozen)
    row["prompt"] = prompt
    row["image_path"] = generated["image_path"]
    row["image_paths"] = generated["image_paths"]
    metadata = deepcopy(frozen["metadata_for_evaluation_only"])
    metadata["candidate_bbox"] = candidate_bbox
    row["metadata_for_evaluation_only"] = metadata
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-requests", type=Path, required=True)
    parser.add_argument("--generated-requests", type=Path, required=True)
    parser.add_argument("--lowlevel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frozen_rows = read_jsonl(args.frozen_requests)
    generated = {row["img_id"]: row for row in read_jsonl(args.generated_requests)}
    lowlevel = {row["img_id"]: row for row in read_jsonl(args.lowlevel)}
    frozen_ids = {row["img_id"] for row in frozen_rows}
    if frozen_ids != set(generated) or frozen_ids != set(lowlevel):
        raise ValueError("Frozen, generated, and low-level request IDs must match exactly")

    rows = [build_row(row, generated[row["img_id"]], lowlevel[row["img_id"]]) for row in frozen_rows]
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} legacy-prompt-controlled requests to {args.output}")


if __name__ == "__main__":
    main()
