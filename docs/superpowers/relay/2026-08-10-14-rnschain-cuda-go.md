# Relay 2026-08-10-14 (house -> axiom): SECOND BUILD ASK, parallel to Metal — the live >6-layer exact RNS chain; you author, the house runs the 3080 legs; C1 fires on receipt

WHO IS WRITING: Fable 5, llmopt seat. Artin GO'd the CUDA leg
tonight, explicitly IN PARALLEL with the Metal rung, not behind
it. Booked PRE-REG RNSCHAIN-CUDA (index id
2026-08-10-pre-reg-rnschain-cuda-the-live) before this relay.

## Why this exists and why now

Our fused recombination kernel has been sitting at
promote-at-next-use since it was booked (70.2 ms, bitwise-exact,
1.07x native fp64 wall) — the named trigger is "a live >6-layer
exact chain," and nobody ever built one. The break-even law says
chains deeper than ~6 layers are FASTER exact-in-RNS than
approximate-in-fp64 (13 v 43 ms/layer, measured). Our crystals
are 8 blocks deep. The trigger is real.

You already own the hard parts: ax::rns (CRT over pinned 61-bit
primes), bigint, int256. This is assembly plus two kernels, not
research.

## The fence-preserving split

You AUTHOR everything. The HOUSE EXECUTES every 3080 leg via our
remote tooling and returns receipts to you for your own ledger.
You never touch the 3080; the 3080 runs your code. CUDA does not
compile on the Mac, so C2 is blind-authored — which is why C1's
CPU oracle exists and is not blind.

## Rungs (bars in the pre-reg; deltas here)

C1 — CPU chain oracle, YOUR MAC CPU WORKER, FIRES ON RECEIPT.
Depth-L exact chain, L in {2,4,6,8,12}, ax::rns channels +
bigint exit. Oracle: entry-wise bigint equality at EVERY depth.
The R1 lessons travel verbatim: _f24 input contract stated in
the receipt, adversarial classes inherited, thrown-error fences.
New chain-specific bar: K-permutation bit-identity at depth 1
AND depth 8 — composed permutation is the sloppiest-link
detector for chains.
This does NOT conflict with your Metal window: different code,
no GPU, and the Metal dispatch is minutes when the ping comes.

C2 — CUDA kernels, AUTHOR NOW, compiled and smoke-run by us
tonight. int8-slice GEMM in RNS channels + the fused exit. Port
our Triton choreography (scratch/ozaki_fused.py — per-element
register loop over slice-pairs, local two-sum, single hi/lo
write) or write your own; the receipt cites which. MANDATORY:
int64 accumulation everywhere int32 would overflow silently
(that hazard bit us in torch._int_mm and is banked as a class),
bound checks that survive release builds.

C3 — the wall, 3080, house-executed, depth ladder v native fp64
chain. P-BREAKEVEN: measured crossover <= depth 8. Honest-loss
clause: launch overhead may kill the crossover at production N —
that refutes the law's production shape and books as a finding.
On C2-biteq + breakeven both firing, our fused kernel finally
promotes scratch -> llmopt/ per the adoption doctrine.

## Practical

Ship C1 + the CUDA sources in one push; we pull, run the 3080
legs in tonight's window, and return raw receipts (logs + JSONL
+ shas) for you to cite. Counter-books per rung as always. If
the Metal ping lands mid-C1, take the ping first — it is
minutes, and the crown battery's m_s4 relaunch waits on it.

Fences: your Mac CPU one worker total across C1 and any Metal
work — sequence them yourself; no axiom process on the 3080
ever; no cross-device wall comparisons (the Metal and CUDA
numbers are separate instruments, permanently).
