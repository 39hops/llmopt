"""Materialize the gen-4 diet as paired prefix/infix jsonl files —
native-transformer rung 1 (spec 2026-07-25-native-transformer).

Writes data/gen4_diet_infix.jsonl (materialized control — byte-for-
byte the rows the --gen4 loader yields, so both twins train through
the identical --diet path) and data/gen4_diet_prefix.jsonl (cur/nxt
re-serialized prefix). Rows whose expressions fail sympify or
strict-encode are skipped WHOLE from BOTH files (paired arms stay
row-identical) and reported — never silently dropped.

    .venv/bin/python scripts/convert_diet_prefix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sympy as sp

from llmopt.mathgen.prefix import from_prefix, to_prefix
from llmopt.train.mathnative import MathTokenizer
from train_mathnative import load_rows


def main() -> None:
    x = sp.Symbol("x")
    env = {"Integral": sp.Integral, "x": x, "sqrt": sp.sqrt,
           "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
           "log": sp.log, "atan": sp.atan, "asin": sp.asin,
           "pi": sp.pi, "E": sp.E}
    tok = MathTokenizer()
    rows = load_rows(gen4=True)
    kept, skipped = 0, 0
    with open("data/gen4_diet_infix.jsonl", "w") as fi, \
            open("data/gen4_diet_prefix.jsonl", "w") as fp:
        for r in rows:
            try:
                pre = {}
                for key in ("cur", "nxt"):
                    e = sp.sympify(r[key], locals=env)
                    p = to_prefix(e)
                    assert from_prefix(p) == e or \
                        sp.simplify(from_prefix(p) - e) == 0
                    tok.encode(p)      # strict: raises on gap
                    tok.encode(r[key])
                    pre[key] = p
            except Exception as ex:
                skipped += 1
                if skipped <= 5:
                    print(f"skip: {r['cur'][:50]!r} ({ex})", flush=True)
                continue
            fi.write(json.dumps({"cur": r["cur"], "nxt": r["nxt"],
                                 "level": r.get("level")}) + "\n")
            fp.write(json.dumps({"cur": pre["cur"], "nxt": pre["nxt"],
                                 "level": r.get("level")}) + "\n")
            kept += 1
    print(f"kept {kept} rows, skipped {skipped} "
          f"(paired: both files row-identical)", flush=True)


if __name__ == "__main__":
    main()
