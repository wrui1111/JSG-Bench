#!/usr/bin/env python3
"""Run Ollama VLM inference for request JSONL files.

The output schema matches experiments/qwen25vl_sid_infer.py so existing
evaluation scripts can be reused.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row.get("img_id") for row in read_jsonl(path) if isinstance(row.get("img_id"), str)}


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def image_paths(row: dict[str, Any]) -> list[str]:
    paths = row.get("image_paths")
    if isinstance(paths, list) and paths:
        return [str(path) for path in paths]
    return [str(row["image_path"])]


def image_to_b64(path: str) -> str:
    with Path(path).open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def infer_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    paths = image_paths(row)
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": row["prompt"],
                "images": [image_to_b64(path) for path in paths],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": args.max_new_tokens,
        },
    }
    response = requests.post(
        args.ollama_url.rstrip("/") + "/api/chat",
        json=payload,
        timeout=args.timeout,
    )
    response.raise_for_status()
    data = response.json()
    raw = str(data.get("message", {}).get("content", "")).strip()
    parsed = extract_json_object(raw)
    out = {
        "img_id": row.get("img_id"),
        "image_path": row.get("image_path"),
        "image_paths": paths,
        "vlm_raw_text": raw,
        "model": args.model,
    }
    if parsed is not None:
        out["vlm_result"] = parsed
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.requests)
    if args.limit:
        rows = rows[: args.limit]
    if args.overwrite and args.output.exists():
        args.output.unlink()
    seen = done_ids(args.output)
    rows = [row for row in rows if row.get("img_id") not in seen]

    for row in tqdm(rows, desc=f"Ollama {args.model}"):
        try:
            out = infer_one(row, args)
        except Exception as exc:  # Keep long runs resumable.
            out = {
                "img_id": row.get("img_id"),
                "image_path": row.get("image_path"),
                "error": f"{type(exc).__name__}: {exc}",
                "model": args.model,
            }
        append_jsonl(args.output, out)
        if args.sleep:
            time.sleep(args.sleep)

    print(f"Wrote Ollama outputs to {args.output}")


if __name__ == "__main__":
    main()
