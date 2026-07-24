#!/bin/bash
# poly5: bridge law's third knock. Diet = chain4 minus poly_ibridge
# family + chain5 (re-spelled ibridge family w/ icancel). One
# variable vs the poly4 birth. Audit: icancel joins the integral-
# grammar kinds (integrand equality).
source "$(dirname "$0")/remote.env.sh"
set -e
cd ~/code/llmopt
[ -f data/poly_chain5.jsonl ] || scp -i "$WSL_KEY" -o BatchMode=yes \
  "$WSL_REMOTE":/mnt/c/Users/a/Documents/code/axiom/data/qual/poly_chain5.jsonl data/
wc -l data/poly_chain5.jsonl
timeout 7200 .venv/bin/python - << 'PYEOF'
import json
import sympy as sp
x = sp.Symbol("x")
def integrand(s):
    e = sp.sympify(s)
    if isinstance(e, sp.Integral):
        return e.function
    return sp.diff(e, x)
bad = n = 0
for line in open("data/poly_chain5.jsonl"):
    r = json.loads(line); n += 1
    k = r["kind"]
    try:
        if k in ("ibridge", "iclose", "close", "icancel"):
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
def take(path, family_filter):
    for line in open(path):
        r = json.loads(line)
        if not family_filter(r.get("family", "")):
            continue
        if r["cur"].replace(" ", "") == r["nxt"].replace(" ", ""):
            continue
        row = {"cur": r["cur"], "nxt": r["nxt"], "level": r["level"]}
        if 17 <= r["seed"] <= 19:
            probe.append({**row, "family": r["family"] + "/" + r["kind"],
                          "seed": r["seed"]})
        else:
            train.append(row)
take("data/poly_chain4.jsonl", lambda f: f != "poly_ibridge")
take("data/poly_chain5.jsonl", lambda f: True)
with open("data/poly5_diet.jsonl", "w") as f:
    for r in base:
        f.write(json.dumps({"cur": r["cur"], "nxt": r["nxt"],
                            "level": r["level"]}) + "\n")
    for r in train:
        f.write(json.dumps(r) + "\n")
with open("data/poly5_probe.jsonl", "w") as f:
    for r in probe:
        f.write(json.dumps(r) + "\n")
print(f"poly5 train {len(train)} probe {len(probe)}")
PYEOF
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py \
  --diet data/poly5_diet.jsonl --epochs 3 \
  --out checkpoints/mathnative_19m_poly5.pt > logs/poly5_birth.log 2>&1
sed "s/series_probe.jsonl/poly5_probe.jsonl/" scratch/series_probe.py > /tmp/poly5_probe.py
.venv/bin/python /tmp/poly5_probe.py checkpoints/mathnative_19m_poly5.pt \
  > logs/poly5_probe.log 2>&1
echo POLY5_DONE >> logs/poly5_probe.log
grep -E "SERIES|gate|ibridge|icancel" logs/poly5_probe.log | grep -v "ep[0-9]"
