#!/bin/bash
# Night 2026-07-28 Mac chain: Muon-3ep revival, revpairs-10%
# stupid-corner, Z[i] born-rational (complex lineage = Mac gates).
cd ~/code/llmopt
PY=.venv/bin/python

$PY scratch/muon_3ep_d256.py > logs/n28_muon3ep.log 2>&1 \
  && $PY scratch/gate_ckpt.py checkpoints/muon3ep_d256.pt \
     256 8 1024 4 muon3ep >> logs/n28_muon3ep.log 2>&1 \
  && touch logs/n28_muon3ep.marker

FORMAT=revpairs10 SCHED=1p BIRTH_SEED=1 \
  $PY scratch/format_ladder.py > logs/n28_revpairs10.log 2>&1 \
  && $PY scratch/gate_ckpt.py checkpoints/fmt_revpairs10_1p.pt \
     256 8 1024 4 revpairs10-1p >> logs/n28_revpairs10.log 2>&1 \
  && touch logs/n28_revpairs10.marker

CPLX_ALPHA=ZI BIRTH_SEED=1 $PY scratch/complex_birth.py --epochs 3 \
  --tag _zi > logs/n28_zi.log 2>&1 \
  && $PY scratch/gate_cplx.py checkpoints/cplx_ZI_zi.pt ZI ZI-latent \
     384 8 1536 6 >> logs/n28_zi.log 2>&1 \
  && $PY scratch/gate_cplx.py checkpoints/cplx_ZI_zi_dep.pt none ZI-dep \
     384 8 1536 6 >> logs/n28_zi.log 2>&1 \
  && touch logs/n28_zi.marker

touch logs/night_28_mac_done.marker
