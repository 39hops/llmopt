#!/bin/bash
# one-command pipeline for poly_chain3 arrival: pull, audit, diet, birth, probe
set -e
cd ~/code/llmopt
[ -f data/poly_chain3.jsonl ] || scp -i ~/.ssh/winwsl -o BatchMode=yes a@10.0.0.184:/mnt/c/Users/a/Documents/code/axiom/data/qual/poly_chain3.jsonl data/
wc -l data/poly_chain3.jsonl
timeout 7200 .venv/bin/python - << 'PYEOF'
import json
import sympy as sp
x = sp.Symbol("x")
def integrand(s):
    e = sp.sympify(s)
    if isinstance(e, sp.Integral):
        return e.function
    # sums/scaled Integrals: differentiate to recover integrand
    return sp.diff(e, x)
bad = n = 0
for line in open("data/poly_chain3.jsonl"):
    r = json.loads(line); n += 1
    k = r["kind"]
    try:
        if k in ("ibridge", "iclose", "close"):
            # integral-grammar rewrite: equal integrands (differentiate
            # any closed parts; Integral wrappers contribute functions)
            d = sp.cancel(integrand(r["cur"]) - integrand(r["nxt"]))
            ok = (d == 0)
        else:
            ok = (sp.cancel(sp.sympify(r["cur"]) - sp.sympify(r["nxt"])) == 0)
    except Exception:
        ok = False
    if not ok:
        bad += 1
        if bad < 4: print("BAD:", k, r["cur"][:70])
print(f"FULL AUDIT {bad}/{n}")
assert bad == 0, "audit failed - do not train"
PYEOF
.venv/bin/python - << 'PYEOF'
import json, sys
sys.path.insert(0, "scripts")
from train_mathnative import load_rows
base = load_rows(gen4=True)
train, probe = [], []
for line in open("data/poly_chain3.jsonl"):
    r = json.loads(line)
    if r["cur"].replace(" ", "") == r["nxt"].replace(" ", ""): continue
    row = {"cur": r["cur"], "nxt": r["nxt"], "level": r["level"]}
    if 17 <= r["seed"] <= 19:
        probe.append({**row, "family": r["family"] + "/" + r["kind"], "seed": r["seed"]})
    else:
        train.append(row)
with open("data/poly3_diet.jsonl", "w") as f:
    for r in base + train: f.write(json.dumps(r) + "\n")
with open("data/poly3_probe.jsonl", "w") as f:
    for r in probe: f.write(json.dumps(r) + "\n")
print(f"poly3 train {len(train)} probe {len(probe)}")
PYEOF
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py --diet data/poly3_diet.jsonl --epochs 3 --out checkpoints/mathnative_19m_poly3.pt > logs/poly3_birth.log 2>&1
sed "s/series_probe.jsonl/poly3_probe.jsonl/" scratch/series_probe.py > /tmp/poly3_probe.py
.venv/bin/python /tmp/poly3_probe.py checkpoints/mathnative_19m_poly3.pt > logs/poly3_probe.log 2>&1
echo POLY3_DONE >> logs/poly3_probe.log
grep -E "SERIES|gate" logs/poly3_probe.log | grep -v "ep[0-9]"
