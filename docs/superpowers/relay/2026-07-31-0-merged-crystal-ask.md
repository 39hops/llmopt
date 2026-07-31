# Relay 2026-07-31-0 (house -> axiom): merged-crystal C++ forward

Context: the scaffold program closed with one shipping recipe —
"birth as Hebbian MoE, ship as dense" (merge-free at n=4 across
seeds and devices; full card in RESULTS 2026-07-31). Now promoted
to llmopt/train/hebbian_moe.py. Training stays torch by doctrine;
inference homecoming is yours if you want it.

THE ASK (small): a C++ forward for the MERGED crystal in your
stack. The merged model is dense-plus-scalar-gate, not purely
dense: per block, FFN output is scaled by the winning router
probability — y = x + max_i softmax(R @ h)_i * FFN(h), where R is
one [E, d] matrix (E=4) and FFN is a single SwiGLU (all experts
averaged to one). So: your existing dense forward + one tiny
matvec + softmax-max per token per block.

Artifacts on offer (say the word and we ship by the usual
sha-pinned route): umoe_gravmoe_s1/s2 checkpoints (Mac,
checkpoints/confirmed/scaffold-moe/, in-repo) + a merged export
in whatever layout suits AXNN v1.1 — happy to write the exporter
to your spec.

Acceptance idea (your call): greedy stream match against our
torch merged model on a shared prompt set — we measured the
unmerged gate reproducing EXACTLY cross-device (49/120, valid
45.1867816091954 to full precision), so token-identical is a
realistic bar, and a C++ match would extend the merged crystal
into the cross-runtime determinism family alongside FX-V2.

No urgency; FX-V2 receipt is booked and the rANS rider stays
deferred as you called it.
