"""Shared field, geometry, and manifest validation for audit evaluators."""
from __future__ import annotations

import math
from typing import Any


def valid_bbox(value: Any, width: int | None = None, height: int | None = None) -> list[int] | None:
    """Normalize coordinates, then reject nonfinite, empty, or out-of-image boxes.

    Dimensions are optional for proposal-generation utilities. Evaluation must
    supply the original image dimensions; boxes are never silently clipped.
    """
    if not isinstance(value, list) or len(value) != 4 or any(isinstance(v, bool) for v in value):
        return None
    try:
        coordinates = [float(v) for v in value]
        if not all(math.isfinite(v) for v in coordinates):
            return None
        x1, y1, x2, y2 = [int(round(v)) for v in coordinates]
    except (TypeError, ValueError, OverflowError):
        return None
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        return None
    if width is not None and (width <= 0 or x2 > width):
        return None
    if height is not None and (height <= 0 or y2 > height):
        return None
    return [x1, y1, x2, y2]


def field_valid(raw: Any, normalized: str, allowed: set[str]) -> bool:
    return isinstance(raw, str) and bool(raw.strip()) and normalized in allowed


def gate_conditions(parse: bool, **fields: bool) -> dict[str, bool]:
    """Apply structural availability to every field in a shared final gate."""
    return {"parse": bool(parse), **{key: bool(parse and value) for key, value in fields.items()}}


def index_rows(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        identifier = row.get("img_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Missing or invalid img_id in {source}")
        if identifier in indexed:
            raise ValueError(f"Duplicate img_id {identifier} in {source}")
        indexed[identifier] = row
    return indexed


def check_output_ids(annotations: dict, outputs: dict) -> None:
    extra = sorted(set(outputs) - set(annotations))
    if extra:
        raise ValueError(f"Outputs outside the declared annotation manifest: {extra[:5]}")


def image_dimensions(annotation: dict[str, Any]) -> tuple[int, int]:
    values = (annotation.get("width"), annotation.get("height"))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or v <= 0 or int(v) != v for v in values):
        raise ValueError(f"Missing or invalid image dimensions for {annotation.get('img_id')}")
    return int(values[0]), int(values[1])
