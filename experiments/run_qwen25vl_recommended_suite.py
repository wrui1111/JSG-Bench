#!/usr/bin/env python3
"""Run multiple frozen Qwen2.5-VL request sets with one model load.

Generation delegates to the preserved main-experiment runner so preprocessing,
chat templating, greedy decoding, and output parsing remain identical.  This
wrapper adds job-level runtime provenance and makes long multi-condition runs
resumable without silently retrying failures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor

from qwen25vl_sid_infer import (
    append_jsonl,
    choose_device,
    generate_one,
    load_done_ids,
    load_model,
    read_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def parse_job(value: str) -> tuple[str, Path, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("--job must be NAME|REQUESTS_JSONL|OUTPUT_JSONL")
    return parts[0], Path(parts[1]), Path(parts[2])


def model_hashes(model_path: Path) -> dict[str, str]:
    names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "preprocessor_config.json",
        "generation_config.json",
    ]
    missing = [name for name in names if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is incomplete; missing: {missing}")
    return {name: sha256(model_path / name) for name in names}


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", type=parse_job, required=True)
    parser.add_argument(
        "--model-path", type=Path, default=ROOT / "models" / "Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--limit-per-job", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "experiments" / "reports" / "recommended_suite" / "inference_manifest.json",
    )
    parser.add_argument("--provenance-id", default="qwen25vl-transformers-mps-recommended-v1")
    args = parser.parse_args()

    args.model_path = args.model_path.resolve()
    args.manifest_output = args.manifest_output.resolve()
    jobs = [(name, request.resolve(), output.resolve()) for name, request, output in args.job]
    for _, request, _ in jobs:
        if not request.is_file():
            raise FileNotFoundError(request)

    set_seeds(args.seed)
    checkpoint_hashes = model_hashes(args.model_path)
    device = choose_device(args.device)
    started_at = utc_now()
    suite_started = time.perf_counter()
    manifest: dict[str, Any] = {
        "manifest_version": "recommended-experiment-inference-v1.0",
        "provenance_id": args.provenance_id,
        "started_at_utc": started_at,
        "command": [sys.executable, *sys.argv],
        "runner": {
            "wrapper_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "wrapper_sha256": sha256(Path(__file__).resolve()),
            "preserved_runner_path": "experiments/qwen25vl_sid_infer.py",
            "preserved_runner_sha256": sha256(ROOT / "experiments" / "qwen25vl_sid_infer.py"),
        },
        "checkpoint": {
            "path": str(args.model_path.relative_to(ROOT)),
            "hashes": checkpoint_hashes,
        },
        "generation": {
            "strategy": "greedy",
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "max_pixels": args.max_pixels,
            "seed_recorded_despite_greedy_decoding": args.seed,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_brand": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "physical_memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
            "device": device,
            "dtype": "float16" if device == "mps" else ("bfloat16" if device == "cuda" else "float32"),
            "attention_implementation": "sdpa",
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "qwen_vl_utils": package_version("qwen-vl-utils"),
            "numpy": np.__version__,
            "pillow": package_version("pillow"),
            "modelscope": package_version("modelscope"),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "jobs": [],
    }
    write_json(args.manifest_output, manifest)

    print(f"Loading Qwen2.5-VL from {args.model_path} on {device}", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path, max_pixels=args.max_pixels, use_fast=False
    )
    model = load_model(args.model_path, device)

    for job_name, requests_path, output_path in jobs:
        rows = read_jsonl(requests_path)
        if args.limit_per_job:
            rows = rows[: args.limit_per_job]
        if args.overwrite and output_path.exists():
            output_path.unlink()
        completed_ids = load_done_ids(output_path)
        pending = [row for row in rows if row.get("img_id") not in completed_ids]
        job_started_at = utc_now()
        job_start = time.perf_counter()
        error_count = 0
        for row_index, row in enumerate(pending, start=1):
            try:
                output = generate_one(
                    model=model,
                    processor=processor,
                    row=row,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
                output["inference_provenance_id"] = args.provenance_id
                output["request_sha256"] = canonical_hash(row)
            except Exception as exc:
                error_count += 1
                output = {
                    "img_id": row.get("img_id"),
                    "image_path": row.get("image_path"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "inference_provenance_id": args.provenance_id,
                    "request_sha256": canonical_hash(row),
                }
            append_jsonl(output_path, output)
            if row_index == 1 or row_index % args.progress_every == 0 or row_index == len(pending):
                elapsed = time.perf_counter() - job_start
                rate = row_index / elapsed if elapsed else 0.0
                print(
                    f"[{job_name}] {row_index}/{len(pending)} rows; "
                    f"{rate:.3f} rows/s; errors={error_count}",
                    flush=True,
                )
        duration = time.perf_counter() - job_start
        output_rows = read_jsonl(output_path) if output_path.exists() else []
        job_record = {
            "name": job_name,
            "started_at_utc": job_started_at,
            "finished_at_utc": utc_now(),
            "duration_seconds": duration,
            "requests_path": str(requests_path.relative_to(ROOT)),
            "requests_sha256": sha256(requests_path),
            "declared_rows": len(rows),
            "rows_pending_at_start": len(pending),
            "output_path": str(output_path.relative_to(ROOT)),
            "output_rows_after_job": len(output_rows),
            "errors_in_this_invocation": error_count,
            "output_sha256": sha256(output_path) if output_path.exists() else None,
        }
        manifest["jobs"].append(job_record)
        write_json(args.manifest_output, manifest)
        print(json.dumps(job_record, ensure_ascii=False, indent=2), flush=True)
        if device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    manifest["finished_at_utc"] = utc_now()
    manifest["duration_seconds"] = time.perf_counter() - suite_started
    write_json(args.manifest_output, manifest)
    print(f"Wrote inference manifest to {args.manifest_output}", flush=True)


if __name__ == "__main__":
    main()
