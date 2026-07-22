#!/usr/bin/env python3
"""
Run local Qwen2.5-VL-3B-Instruct inference for SID-Set VLM request JSONL files.

Input rows should come from:
  python experiments/training_free_pipeline.py make-requests ...

Output rows are compatible with:
  python experiments/training_free_pipeline.py aggregate --vlm-output ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in read_jsonl(path):
        img_id = row.get("img_id")
        if isinstance(img_id, str):
            done.add(img_id)
    return done


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


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: Path, device: str) -> Qwen2_5_VLForConditionalGeneration:
    if device == "cuda":
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )

    dtype = torch.float16 if device == "mps" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    return model


def build_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    image_paths = row.get("image_paths")
    if not isinstance(image_paths, list) or not image_paths:
        image_paths = [row["image_path"]]
    content = [{"type": "image", "image": image_path} for image_path in image_paths]
    content.append({"type": "text", "text": row["prompt"]})
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def generate_one(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    row: dict[str, Any],
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    messages = build_messages(row)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device if device == "cuda" else device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    parsed = extract_json_object(raw_text)

    out = {
        "img_id": row["img_id"],
        "image_path": row["image_path"],
        "vlm_raw_text": raw_text,
    }
    if isinstance(row.get("image_paths"), list):
        out["image_paths"] = row["image_paths"]
    if parsed is not None:
        out["vlm_result"] = parsed
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/runs/qwen25vl_outputs.jsonl"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=768 * 28 * 28,
        help="Qwen-VL visual token budget. Lower this if memory is tight.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.requests)
    if args.limit:
        rows = rows[: args.limit]

    if args.overwrite and args.output.exists():
        args.output.unlink()
    done_ids = load_done_ids(args.output)
    rows = [row for row in rows if row.get("img_id") not in done_ids]

    device = choose_device(args.device)
    print(f"Loading Qwen2.5-VL from {args.model_path} on {device}")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=args.max_pixels,
        use_fast=False,
    )
    model = load_model(args.model_path, device)

    for row in tqdm(rows, desc="Qwen2.5-VL SID inference"):
        try:
            out = generate_one(
                model=model,
                processor=processor,
                row=row,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:  # Keep long batch runs resumable.
            out = {
                "img_id": row.get("img_id"),
                "image_path": row.get("image_path"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        append_jsonl(args.output, out)

    print(f"Wrote Qwen2.5-VL outputs to {args.output}")


if __name__ == "__main__":
    main()
