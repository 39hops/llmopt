# The rabbit hole — five linear-algebra tunnels (2026-07-17 night)

All cheap, all gated, in priority order. Production ref: gen4_std
69/120 @ 67.13.

## 2. Gauge-calibration table (RUNNING tonight)
Four gauges (CV, floor, shelf, R) over every checkpoint with a
known GATE_N=24 chain gate (~7 distinct births + lineage points).
Question: which gauge predicts capability, and how well? Known
confound to report honestly: within-lineage the lattice is frozen,
so gauges should predict BIRTH quality, not RL polish — the table
tests exactly that split.

## 1. GPTQ-int3 (error-compensated rounding)
House methods.py GPTQ needs calibration activations: harvest ~2k
gate-band prompt activations once, quantize with compensation,
chain-gate vs naive MX-int3's 67/120. Moves the floor to 3 bits or
confirms 4 is fundamental.

## 3. Graph modularity / small-worldness
Threshold |W| into adjacency; clustering coefficient, modularity
(greedy Louvain-ish, pure python), path lengths. Compare crystal
vs Qwen vs coder (prediction: coder most modular — its R=0.215
clumps should resolve into countable modules).

## 4. Relational storage (geometry without coordinates)
Per-layer: k anchor neurons, store each neuron as distances/inner
products to anchors (n*k vs n*d). Reconstruct via MDS, gate.
Question: at what k does the gate survive? (Rank floor says
k>=~128 likely — this measures it differently.)

## 5. Principal curricula
The transfer matrix (curriculum A/B deltas per level) + phylogeny
distances as an operator; eigenvectors = the "natural courses."
Thin data (3 A/Bs) — mostly a frame until more curriculum points
exist; write up, don't over-fit.
