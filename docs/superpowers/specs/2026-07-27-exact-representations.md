# SPEC: exact representations (Artin's infinite-precision push, 2026-07-27)

Origin: Artin's evening riff ("represent 1/3 exactly, not
64-bit-exactly"; rotational/complex/qubit weights; the black-hole
compression frame — both banked in RIFF-LEDGER 2026-07-27).
Measured grounding read before firing (full RESULTS sweep):
disagreement #2 (exact-dd masters bit-identical to fp64 — exact
training is a SPEED lever), ozaki v5/v6 (RNS stay-in-residue,
exact beats every finite format on wall), the complex program
(interior neutral on math; NNUE null phase-free; surviving
justifications = phase-carrying data + G16), precision AMENDMENT
(doctrine holds above instrument sigma; E3 = sole retest).

## Rungs

- **(a) Rational-snap distillation — FIRED 2026-07-27 (Mac)**:
  snap mathnative_19m weights to best p/q, q <= Q in {4,16,64};
  four arms (control + 3) all on the Mac MPS gate, paired.
  Compression framing. Pre-reg in RESULTS. scratch/rational_snap.py,
  runner scratch/run_snap_gates.sh, logs logs/snap_*.log.
- **(c) G9 roots-of-unity cell — CONDITIONAL, not queued** (the
  RESULTS sweep caught the 07-26 closure before arming: rotation
  wing CLOSED at G5-dep 32 v M5 31, and the 19M ZX seed fence
  drowns any 19M cell). Fires ONLY as Artin's declared reopening,
  at 45M-class, IF the in-flight 45M union's ZX gate clears its
  bar; else banks unfired. Conditional pre-reg in RESULTS.
  Quantizer generalized in scratch/complex_model.py (gn_quantize;
  G5 route byte-preserved, sanity-tested); scratch/night_g9.sh =
  19M template only, DO NOT LAUNCH as-is. Follow-up if it ever
  pays: G17, then per-layer alphabet mixing.
- **(b) RNS optimizer step — SPEC ONLY (next 3080 free slot)**:
  whole AdamW step carry-free in residue space, one exit
  (ozaki v5's banked endgame, named as a training rung).
  Disagreement #2 fixes the outcome (bit-identical); this is a
  WALL rung: target = beat the fp32 optimizer step wall on a
  19M birth at exactness. Build on scratch/ozaki_cuda5.py's
  channel machinery. Pre-reg wall numbers before firing.

## Fences

- Precision doctrine stays CLOSED; nothing here reopens it
  (E3 is the sole named retest). (a) is compression; (c) is
  alphabet/geometry (ZX thread); (b) is wall.
- Mac gate numbers never compare to cuda gate numbers (all
  paired arms same-device).
- Phase-free complex cells are measured null twice — no new
  ones without a phase-carrying diet.

## The long line (banked, not scheduled)

Roots-of-unity weights whose ring (Z[zeta_8, 1/sqrt(2)]) is the
Clifford+T synthesis ring make "qubit weights" literal: a weight
matrix as an exact multi-qubit gate word. If G9/G17 pays on ZX,
the natural rung after per-layer mixing is exact cyclotomic
inference (integer coords per weight, zero rounding in the
rotation algebra) — FX-V1's exact-GEMM machinery is the
substrate. The black-hole/Bekenstein frame stays a frame until
a rung produces a measured packing-density number.

## Addendum (late 2026-07-27, Artin's three upgrades)

- **Function-aware snap = COMPUTE the actual number**: per-layer
  least squares ||X W_lat - X W|| on calibration activations,
  GPTQ-greedy over the exact-rational lattice — the minimizer is
  computed, not approximated. Predicted: parity Q drops 48 ->
  16-24 shelf. (Dual of the born arm: training computes the
  number forward; this computes it backward from the function.)
- **Only the bits needed = water-filling**: per-tensor Q against
  the anatomy's sensitivity map (head/emb = 73% of divergence
  share at 0.16% of params -> high Q free; FFN interiors Q=6).
  Target: mean bits/weight toward M5's ~2.3 class, exact.
- **The factory**: template (coarse stats, time lever) -> short
  train -> function-aware snap (seconds) -> thin precise film
  (snap+repair rung). All legs measured or pre-reg'd; if the
  born-rational read says lattice-from-step-0 wins, the factory
  simplifies to template + born-lattice + free exact deploy.
- Order: morning reads -> snap+repair -> GPTQ-rational build ->
  mixed-Q water-filling.
