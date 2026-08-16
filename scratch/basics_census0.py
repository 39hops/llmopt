"""BASICS-CENSUS-0 — how much EXPLICIT arithmetic/algebra the stock
diet already states on its own rows (desk census, no model).

Prices the BASICS-DIET bank (RIFF-LEDGER 4916), whose premise is
that the stock diet teaches arithmetic and algebra only implicitly.
Counts the diet as birth19m_curric.load_excised_rows() delivers it,
by source tag and by structure (Integral-wrapped v bare
expression), in BOTH rows and encoded tokens (tokens are what
training spends).

Usage: .venv/bin/python -u scratch/basics_census0.py
"""
import os
import re
import sys
from collections import Counter

os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "3")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import birth19m_curric as C  # noqa: E402
import train_mathnative as TM  # noqa: E402

NUMERIC = re.compile(r"^[-+*/(). 0-9]+$")


def main():
    tok = TM.MathTokenizer()
    rows = C.load_excised_rows()
    print(f"[diet] {len(rows)} excised rows", flush=True)

    by_source = Counter()
    tok_by_source = Counter()
    kind_rows = Counter()
    kind_toks = Counter()
    total_toks = 0

    for r in rows:
        cur, nxt = str(r["cur"]), str(r["nxt"])
        src = r.get("source") or "untagged"
        text = f"Current: {cur}\nHints: none\nStep: {nxt}\n"
        try:
            n = len(tok.encode(text)) + 1
        except ValueError:
            continue
        total_toks += n
        by_source[src] += 1
        tok_by_source[src] += n

        wrapped = "Integral(" in cur or "Derivative(" in cur
        if wrapped:
            kind = "calculus (wrapped)"
        elif NUMERIC.match(cur):
            kind = "arithmetic explicit (bare numeric)"
        else:
            kind = "algebra explicit (bare symbolic)"
        kind_rows[kind] += 1
        kind_toks[kind] += n

    print(f"\n[total] {sum(by_source.values())} encodable rows, "
          f"{total_toks} tokens")
    print("\n== by source (rows, row%, tokens, token%)")
    for s, c in by_source.most_common():
        print(f"  {s:28s} {c:7d} {100*c/sum(by_source.values()):6.2f}%"
              f" {tok_by_source[s]:9d} {100*tok_by_source[s]/total_toks:6.2f}%")
    print("\n== by kind (structural)")
    for k, c in kind_rows.most_common():
        print(f"  {k:36s} {c:7d} {100*c/sum(kind_rows.values()):6.2f}%"
              f" {kind_toks[k]:9d} {100*kind_toks[k]/total_toks:6.2f}%")

    # sample the bare-expression families so the classification is auditable
    print("\n== samples per source, bare-expression rows")
    seen = Counter()
    for r in rows:
        cur = str(r["cur"])
        if "Integral(" in cur or "Derivative(" in cur:
            continue
        src = r.get("source") or "untagged"
        if seen[src] < 2:
            seen[src] += 1
            print(f"  [{src}] {cur[:56]!r} -> {str(r['nxt'])[:40]!r}")


if __name__ == "__main__":
    main()
