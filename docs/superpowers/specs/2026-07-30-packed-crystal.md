# Spec: THE PACKED CRYSTAL (2026-07-30; drafted 07-29 eve)

Artin's README catch promoted to the front of the queue: the
month's snap laws become a REAL deploy artifact. Everything we
quantify is fake-quant (grid values, fp storage) except the
integer twin; this spec ships actual bytes, an actual kernel,
and an honest comparison against the repo's own GPTQ/AWQ/HQQ
implementations (quantize/). Publication candidate IF the
honest triple lands.

## The claims under test (each measured, none assumed)
1. CALIBRATION-FREE ALLOCATION: per-tensor bit-width read off
   sigma alone (Q ~ 2/sigma, the sigma-law knee) + one global
   safety margin from the flips/token probe (k_c, rho .883 at
   ~1% of a gate's cost). GPTQ/AWQ/HQQ all spend calibration
   passes + Hessian/scale search to find what we compute in
   closed form. THIS is the differentiator — not raw speed.
2. AT-THE-BOUND BITS: the integer twin measured 6.65 bits/wt
   v Gaussian capacity 6.755 — packed size should land within
   ~2% of the entropy bound (arith-coded check rides along).
3. DETERMINISM FOR FREE (the cross-device tie-in): an integer
   forward (int codes x one shared denominator per tensor,
   Kulisch-style wide accumulate) has NO reduction-order or
   FMA ambiguity — the SAME tokens should decode on Mac and
   3080. Our entire device-fence pain class vanishes inside
   the packed path. Flagship claim if it lands; Ozaki/RNS arc
   says the cost is affordable.
4. TIERS BECOME MEMORY: the matryoshka/escalation ladder on a
   packed tensor = real bytes and real bandwidth at decode
   time, not just FLOPs accounting. Greedy-first (90% of
   solves at 12% of tokens) is the decode default.

## Cells
- C0 FORMAT: generalized integer twin — per-tensor denominator
  q_t = ceil(2/sigma_t) (rational-snap rule), codes packed to
  ceil(log2(range)) bits, one fp scale per tensor. Emit .npz +
  a 40-line reader. Entropy-coded size reported next to raw
  packed size (bound check, claim 2).
- C1 PACK + PARITY (Mac, desk): pack layers-4 d56 (smallest)
  AND d64 EMA. Bar: gate within sigma of the fp control
  (deploy-tax law says rational lattices deploy at ZERO tax;
  phase lattices paid -4 — we are on the safe geometry).
  Report bits/wt, bytes v fp32/fp16.
- C2 KERNEL (Mac, MLX/Metal): dequant-fused GEMV in
  kernels/metal.py style (split-K scar: mx.eval every timed
  iteration; honest numbers incl. losses). Bench decode
  tok/s v fp16 GEMV at crystal shapes; bandwidth model
  prediction printed next to measured.
- C3 BASELINES (Mac, desk): quantize/'s own GPTQ, AWQ, HQQ at
  matched avg bits on the SAME crystal -> gate + DeltaKL +
  calibration wall-time. Honest table, wins and losses.
- C4 DETERMINISM (needs both machines; 3080 = short test):
  packed integer forward, fixed prompt battery -> token
  streams + logit hashes on MPS v cuda. Claim 3 lands only on
  hash equality. (Integer accumulate in int32/int64 — no fp in
  the matmul path; norms/softmax stay fp and are AFTER the
  argmax-relevant margins? NO — they are not; so cell reads
  BOTH: full-forward hash (expected to differ) and
  integer-GEMM-output hash (must match). Honest split.)
- C5 TIERED PACK: pack the 3-tier matryoshka; escalation
  decode with measured bytes touched per tier; effective
  bandwidth per gate row v dense.
- C6 TRANSFER (optional, 0.5B-era tooling): sigma-law
  allocation v HQQ on the legacy 0.5B (quantize/ harness,
  DeltaKL + perplexity) — the external-validity cell a paper
  needs. [HOLD until C1-C3 read; runs on Artin's GO.]

## Order + budget
C0+C1 desk (one evening); C2 the kernel day; C3 desk behind
C1; C4 next 3080 short-test window; C5 behind C1; C6 HOLD.
Pre-reg each cell in RESULTS before it fires. Fences travel:
sigma per-tensor (never transported), gates same-device as
their comparators, n=1 cells read against sigma ~3.5.

## Tie-ins consumed (from the RESULTS/FINDINGS/RIFF pass)
sigma-priced knee + width-bound Q fences; distortion collapse
two-parameter equation (k_c meter = flips probe); integer twin
(shared-denominator integer GEMM, 6.65 bits/wt); born-rational
(born-on-lattice 2x fewer bits than post-hoc — C1b arm if C1
disappoints); deploy-tax lattice-geometry law; alphabet
tournament (P2 3.17-bit beats fp32 = a born-packed 3-bit arm
exists); matryoshka joint loss + escalation policy; greedy-
first economics; Ozaki int8-exact + stay-in-RNS (determinism
affordable); Metal split-K + MLX lazy-eval scars; quantize/
GPTQ/AWQ/HQQ (baselines in-tree); cell-sparse head map
(optional structured-sparsity rider AFTER replication on a
second crystal — n=1 today, not consumed yet).
