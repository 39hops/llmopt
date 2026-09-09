#!/bin/bash
# WRITER-DFA-1 post-discovery driver (pre-reg RESULTS L68321 item 12,
# AMENDMENT -PRECISION P4 / P5b): trajectory census (zero gates), then
# the dependence instrument in DEPEND_SET=dfa mode (two full gates, the
# FUNCTION-BAND decided mechanically, reverts and swaps only on pass),
# then on band pass the mandatory 8 x 5 depth x class census, then ACT,
# alignment and the independent verifier. Under the liverun interlock;
# marker on success only.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
[ -z "$(git status --porcelain)" ] || { echo "driver: dirty tree"; git status --porcelain; exit 2; }
[ -f logs/writerdfa1/act_envelope.json ] || { echo "driver: act_envelope.json missing"; exit 3; }
rc=0
echo "=== trajectory census $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python scratch/dfa_trajcensus.py 2>&1 | tee logs/writerdfa1/trajcensus.log || rc=$?
[ "$rc" -eq 0 ] || { mark_done logs/writerdfa1_post.DONE "$rc"; exit "$rc"; }
echo "=== dependence + band + swaps $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
DEPEND_SET=dfa .venv/bin/python scratch/writertraj_depend.py 2>&1 | tee logs/writerdfa1/depend.log || rc=$?
[ "$rc" -eq 0 ] || { mark_done logs/writerdfa1_post.DONE "$rc"; exit "$rc"; }
BAND=$(.venv/bin/python -c 'import json;print(json.load(open("logs/writerdfa1/depend.json"))["band"]["band_pass"])')
echo "=== FUNCTION-BAND pass: $BAND ==="
if [ "$BAND" = "True" ]; then
  echo "=== depth x class census $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  .venv/bin/python scratch/dfa_depthclass.py 2>&1 | tee logs/writerdfa1/depthclass.log || rc=$?
  [ "$rc" -eq 0 ] || { mark_done logs/writerdfa1_post.DONE "$rc"; exit "$rc"; }
fi
echo "=== ACT $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ACT_SET=dfa .venv/bin/python scratch/dfa_act.py 2>&1 | tee logs/writerdfa1/act.log || rc=$?
[ "$rc" -eq 0 ] || { mark_done logs/writerdfa1_post.DONE "$rc"; exit "$rc"; }
echo "=== alignment $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python scratch/dfa_align.py 2>&1 | tee logs/writerdfa1/align.log || rc=$?
[ "$rc" -eq 0 ] || { mark_done logs/writerdfa1_post.DONE "$rc"; exit "$rc"; }
echo "=== verifier $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python scratch/dfa_verify.py 2>&1 | tee logs/writerdfa1/verify.log || rc=$?
mark_done logs/writerdfa1_post.DONE "$rc"
