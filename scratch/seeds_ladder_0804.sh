#!/bin/bash
# GRAVMOE-SEEDS-LADDER (PRE-REG 2026-08-04, GO received): the COND+QK
# gate win at three FRESH birth seeds, paired, regression-gated.
# Arm 0 must reproduce the GRB1 pin (1fcfd187) or NOTHING is read.
# Pattern: p4_arms_0801.sh (1 torch thread/arm, all parallel, one wait;
# marker semantics live in the launcher, exit code here).
set -u
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_DISABLE_NATIVE_JIT=1
# CONTRACT_ENV hygiene (llmopt/reproduce.py): every knob the contract
# tracks is explicitly unset here; arms set only what they declare.
unset ANSWER_ONLY BIRTH_SEED COND QK TAU GATE SS LN LD K E STEPS SHIFT \
      SCHED EXPORT TWIN TRAJECTORY_ONLY WINDOWS_BIN WINDOWS_CONTRACT
mkdir -p logs/seedslad

# PRE-FLIGHT (registered): the gate diet is UNTRACKED — git pull does
# not carry it; assert its sha before arming anything.
DIET=data/micromodel_gen4_sidecar.jsonl
WANT_DIET=809bce4215a24164ecbf5e951d77507d455bfd1923d08fe39aa02942b11a200b
got=$(sha256sum "$DIET" 2>/dev/null | awk '{print $1}')
if [[ "$got" != "$WANT_DIET" ]]; then
  echo "PRE-FLIGHT FAIL: diet sha ${got:-MISSING} != $WANT_DIET"
  exit 3
fi
echo "pre-flight: diet sha OK, HEAD $(git rev-parse --short HEAD)"

# arm-name | env | expected sha (empty = fresh seed, no pin exists)
ARMS=(
  "REG17|GATE=1 COND=1 QK=1 BIRTH_SEED=17|1fcfd187873d980c7c082a56c0f380ce2c40a859eab1e8a0c9dcf6baa4853eca"
  "S41C|GATE=1 COND=1 BIRTH_SEED=41|"
  "S41CQ|GATE=1 COND=1 QK=1 BIRTH_SEED=41|"
  "S47C|GATE=1 COND=1 BIRTH_SEED=47|"
  "S47CQ|GATE=1 COND=1 QK=1 BIRTH_SEED=47|"
  "S53C|GATE=1 COND=1 BIRTH_SEED=53|"
  "S53CQ|GATE=1 COND=1 QK=1 BIRTH_SEED=53|"
)

for a in "${ARMS[@]}"; do
  IFS='|' read -r name envv sha <<< "$a"
  echo "launch $name ($envv)"
  env $envv $PY scratch/detbwd_gravmoe.py > "logs/seedslad/$name.log" 2>&1 &
done
wait

echo "=== SEEDS LADDER TABLE ==="
fail=0
reg=$(grep "FINAL trajectory sha" logs/seedslad/REG17.log | awk '{print $NF}')
if [[ "$reg" == "1fcfd187873d980c7c082a56c0f380ce2c40a859eab1e8a0c9dcf6baa4853eca" ]]; then
  echo "REGRESSION PASS $reg"
else
  echo "REGRESSION FAIL got ${reg:-MISSING} — NOTHING ELSE IS READ"
  fail=1
fi
for a in "${ARMS[@]}"; do
  IFS='|' read -r name envv sha <<< "$a"
  echo "--- $name"
  grep -E "solves|token-acc|FINAL trajectory sha|loss" "logs/seedslad/$name.log" | tail -8
done
exit $fail
