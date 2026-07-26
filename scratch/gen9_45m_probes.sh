#!/bin/bash
set -e
cd ~/code/llmopt
probe_one () {  # ckpt d layers ffn heads
  local CKPT=$1 D=$2 L=$3 F=$4 H=$5
  for P in series_probe_1e phys_energy_probe poly3_probe; do
    sed -e "s/series_probe.jsonl/$P.jsonl/" \
        -e "s/build_model(len(tok.vocab))/build_model(len(tok.vocab), d=$D, layers=$L, ffn=$F, heads=$H)/" \
        -e "s/\"mps\" if torch.backends.mps.is_available() else \"cpu\"/\"cuda\" if torch.cuda.is_available() else \"cpu\"/" \
        scratch/series_probe.py > /tmp/g9_probe.py
    VOCAB_EXTRA=t .venv/bin/python /tmp/g9_probe.py checkpoints/$CKPT.pt \
      > logs/${CKPT}_$P.log 2>&1
  done
  VOCAB_EXTRA=t .venv/bin/python scratch/gate_rarity.py \
    checkpoints/$CKPT.pt $D $L $F $H ${CKPT}_r > logs/${CKPT}_rarity.log 2>&1 || true
}
probe_one mathnative_45m_gen9B 512 12 2048 8
probe_one mathnative_19m_gen9B 384 8 1536 6
echo GEN9_45M_PROBES_DONE >> logs/gen9_45m_probes.log
