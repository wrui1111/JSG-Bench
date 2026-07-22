#!/usr/bin/env python3
"""Compute simple agreement and Cohen's kappa for IAA overlap files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def norm(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(sorted(str(v).strip().lower() for v in value))
    if value is None:
        return ""
    return str(value).strip().lower()


def cohen_kappa(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return {"samples": 0, "agreement": None, "kappa": None}
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    labels = set(left) | set(right)
    expected = sum((left[label] / n) * (right[label] / n) for label in labels)
    kappa = (agree - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "samples": n,
        "agreement": round(agree, 6),
        "expected_agreement": round(expected, 6),
        "cohen_kappa": round(kappa, 6),
        "labels": sorted(labels),
    }


def index_reference(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        img_id = row.get("img_id") or row.get("source_img_id")
        if img_id:
            indexed[str(img_id)] = row
    return indexed


def source_id(row: dict[str, Any]) -> str:
    return str(row.get("source_img_id") or row.get("img_id") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--second-pass", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = index_reference(read_jsonl(args.reference))
    second = read_jsonl(args.second_pass)
    field_pairs: dict[str, list[tuple[str, str]]] = {field: [] for field in args.fields}
    missing = []
    for row in second:
        img_id = source_id(row)
        ref = reference.get(img_id)
        if not ref:
            missing.append(img_id)
            continue
        for field in args.fields:
            field_pairs[field].append((norm(ref.get(field)), norm(row.get(field))))

    report = {
        "stage": "iaa_agreement",
        "reference": str(args.reference),
        "second_pass": str(args.second_pass),
        "fields": args.fields,
        "matched_samples": len(second) - len(missing),
        "missing_reference": missing,
        "metrics": {field: cohen_kappa(pairs) for field, pairs in field_pairs.items()},
        "note": "Use only after the second-pass template has been independently filled. Do not use IAA overlap samples for method tuning.",
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
