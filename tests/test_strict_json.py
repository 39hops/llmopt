"""llmopt.lab.strict_json: receipts that a strict JSON parser accepts.

The booked E0 disagreement receipt carries a bare Infinity margin-bin
boundary (Python's json.dump default); a strict parser refuses it.
Future receipts encode non-finite floats as strings and refuse to emit
the bare constants.
"""
import json
import math

import pytest

from llmopt.lab.strict_json import dumps_strict, strictify, write_strict


def _refuse(name):
    raise ValueError(f"non-standard constant {name}")


def test_non_finite_floats_become_strings():
    doc = {"edges": [0.0, 0.5, math.inf], "neg": -math.inf, "nan": math.nan, "ok": 1.5}
    out = strictify(doc)
    assert out == {"edges": [0.0, 0.5, "inf"], "neg": "-inf", "nan": "nan", "ok": 1.5}


def test_dumps_passes_a_strict_parser_and_keeps_integers():
    doc = {"n": 3, "edges": (0.0, math.inf), "nested": {"k": [math.nan]}, "flag": True, "none": None}
    text = dumps_strict(doc, indent=1)
    back = json.loads(text, parse_constant=_refuse)
    assert back == {"n": 3, "edges": [0.0, "inf"], "nested": {"k": ["nan"]}, "flag": True, "none": None}
    assert "Infinity" not in text and "NaN" not in text


def test_numpy_scalars_and_arrays_are_converted():
    np = pytest.importorskip("numpy")
    doc = {"a": np.float32(2.5), "b": np.int64(7), "c": np.array([1.0, np.inf]), "d": np.bool_(True)}
    out = strictify(doc)
    assert out == {"a": 2.5, "b": 7, "c": [1.0, "inf"], "d": True}
    assert type(out["b"]) is int and type(out["a"]) is float


def test_write_strict_refuses_an_existing_path(tmp_path):
    p = tmp_path / "r.json"
    write_strict({"x": math.inf}, p)
    assert json.loads(p.read_text(), parse_constant=_refuse) == {"x": "inf"}
    with pytest.raises(FileExistsError):
        write_strict({"x": 1}, p)
    write_strict({"x": 2}, p, overwrite=True)
    assert json.loads(p.read_text()) == {"x": 2}
