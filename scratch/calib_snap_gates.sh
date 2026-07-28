#!/bin/bash
# Rung-1 validation v2 (spec 2026-07-28): probe + Q=16 DIRECT rational
# snap gate per crystal — rational_snap.py, the instrument that
# measured the 49->26 ground truth (v1 wrongly used rat_deploy's
# scaled snap: gentler lattice, all-parity gates, no variance).
# Serial on purpose (one MPS device, paired reads).
# gate_ckpt args: ckpt d layers ffn heads label.
set -e
cd ~/code/llmopt
PY=.venv/bin/python

for name in wfloor_d256 wfloor_d256_s2 wfloor_d256_s3 \
            wfloor_d256_stream4 wfloor_d256_muon; do
  ck=checkpoints/mathnative_${name}.pt
  $PY scratch/calib_probe.py $ck 256 8 1024 4 16 \
      | tee -a logs/calib_r1v2_probe.log
  $PY scratch/rational_snap.py $ck 16 checkpoints/calib_${name}_q16.pt
  $PY scratch/gate_ckpt.py checkpoints/calib_${name}_q16.pt \
      256 8 1024 4 calib_${name}_q16 \
      2>&1 | tee logs/calib_${name}_q16v2_gate.log
done

# Mac-19M cracker (mathnative_19m.pt, control 49, Q=16 gate 26
# already booked): probe only
$PY scratch/calib_probe.py checkpoints/mathnative_19m.pt \
    384 8 1536 6 16 | tee -a logs/calib_r1v2_probe.log
touch logs/calib_r1v2_done.marker
