"""PHASE 1 of RESULTS-HARDENING (Artin GO 2026-08-07): exclude-union
audit of LEGACY diets against the 120-problem forward gate band.

Reuses D2's exact instrument (tenet_d2_revdiet.gate_band_exprs +
norm) — the law that fired 21/120 on gen-4. READ-ONLY: reports
per-diet collision counts (prompt-side cur, target-side nxt,
distinct band expressions touched, rows affected, level histogram);
excises NOTHING (legacy diets are frozen evidence — the audit's
output decides whether re-births need clean rebuilds).

Usage: .venv/bin/python scratch/legacy_diet_audit.py data/*.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

band = set(gate_band_exprs())
print(f"[audit] band: {len(band)} distinct expressions", flush=True)
for path in sys.argv[1:]:
    p = Path(path)
    if not p.exists():
        print(f"{p}: MISSING")
        continue
    rows = cur_hit = nxt_hit = 0
    touched, lvl = set(), Counter()
    bad_rows = 0
    for line in p.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows += 1
        c = norm(str(r.get("cur", "")))
        n = norm(str(r.get("nxt", "")))
        hit = False
        if c in band:
            cur_hit += 1; touched.add(c); hit = True
        if n in band:
            nxt_hit += 1; touched.add(n); hit = True
        if hit:
            bad_rows += 1
            lvl[r.get("level", "?")] += 1
    print(f"{p.name}: rows {rows} | band-touching rows {bad_rows} "
          f"({100*bad_rows/max(rows,1):.2f}%) | cur-hits {cur_hit} "
          f"nxt-hits {nxt_hit} | distinct band exprs touched "
          f"{len(touched)}/120 | levels {dict(lvl)}", flush=True)
