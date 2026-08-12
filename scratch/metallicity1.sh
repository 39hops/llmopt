#!/bin/bash
# METALLICITY-1 driver, 3080. Pre-reg: RESULTS PRE-REG METALLICITY-1.
# Ignition width per refinement grade: 4 grades (z0 vacuum, z1 pop3,
# z2 polluted, z3 verified — scratch/metallicity_diets.py, string-
# seeded, shas in data/metallicity/manifest.json) x 4 widths
# (d32/48/56/64, ffn=4d, L8 H4), micro-star recipe (3ep, bf16,
# budget 6144), BIRTH_SEED=1. STREAMED: every gate prints as it
# lands, so a wall-kill still leaves bookable cells. Gates are
# 3080-family only (msearch family; never compare to Mac).
set -eo pipefail
cd ~/code/llmopt
mkdir -p logs/metallicity1 data/metallicity

# grades regenerate deterministically; shas must match the committed
# manifest (verify deps at arm time — remote-ops doctrine)
.venv/bin/python scratch/metallicity_diets.py \
  > logs/metallicity1/diets.log 2>&1
.venv/bin/python - << 'EOF'
import hashlib, json
man = json.load(open("data/metallicity/manifest.json"))
ref = json.load(open("data/metallicity/manifest.ref.json"))
for k, v in ref.items():
    got = man[k]
    assert got == v, f"grade {k}: sha {got} != committed {v}"
print("grade shas verified:", ref)
EOF

for Z in z3 z0 z2 z1; do            # z3 first (anchor), z0 second (control)
  for D in 64 56 48 32; do          # widest first: ignition read earliest
    FFN=$((D * 4))
    CK="checkpoints/metal_${Z}_d${D}.pt"
    BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py \
      --diet data/metallicity/${Z}.jsonl --fast --epochs 3 \
      --budget 6144 --d $D --layers 8 --ffn $FFN --heads 4 \
      --out "$CK" \
      > "logs/metallicity1/${Z}_d${D}_birth.log" 2>&1
    .venv/bin/python scratch/gate_ckpt.py "$CK" $D 8 $FFN 4 \
      "metal_${Z}_d${D}" \
      > "logs/metallicity1/${Z}_d${D}_gate.log" 2>&1
    grep -h "gate" "logs/metallicity1/${Z}_d${D}_gate.log" | tail -1
    echo "CELL DONE ${Z} d${D}"
  done
done
echo "METALLICITY-1 done"
