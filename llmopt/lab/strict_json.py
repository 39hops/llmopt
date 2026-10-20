"""Receipts that a strict JSON parser accepts.

Python's json.dump emits the bare constants Infinity, -Infinity and NaN
for non-finite floats (allow_nan=True by default). They are not JSON:
a strict parser refuses them, and a receipt with a margin-bin boundary
of Infinity cannot be read by such a parser. The functions here encode a
non-finite float as the string "inf", "-inf" or "nan", convert numpy
scalars and arrays to Python values, and dump with allow_nan=False so a
constant that escaped the conversion fails at write time instead of
landing in a receipt.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


def strictify(obj):
    """A copy of obj with non-finite floats as strings and numpy values as
    Python values; dict keys are left as they are (json.dump converts
    them), tuples become lists."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a lab dependency
        np = None
    if np is not None and isinstance(obj, np.ndarray):
        return strictify(obj.tolist())
    if np is not None and isinstance(obj, np.generic):
        return strictify(obj.item())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, dict):
        return {k: strictify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strictify(v) for v in obj]
    return obj


def dumps_strict(obj, indent=1) -> str:
    return json.dumps(strictify(obj), indent=indent, allow_nan=False)


def write_strict(obj, path, indent=1, overwrite=False) -> None:
    """Write obj as strict JSON; refuses an existing path unless overwrite.
    The text is written to a sibling temporary file and renamed, so a
    reader never sees a partial receipt."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    text = dumps_strict(obj, indent=indent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
