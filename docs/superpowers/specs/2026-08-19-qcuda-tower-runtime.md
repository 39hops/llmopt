# Spec 2026-08-19: reusable CUDA tower runtime (qcuda dispatcher + FusedS16Linear)

Trigger: QWEN-BLE-FREEGEN-1-ABORT — scratch rung4's module surgery
routes only w4 to compressed execution; BLe's 48 promoted s16
decoder tensors materialized as 6.875 GiB dense FP32 and the screen
paced at 0.56 tok/s. Decision (Artin GO 2026-08-19): make the CUDA
runtime reusable in llmopt/lab instead of patching one experiment.
scratch/qwen_cuda_rung4.py stays FROZEN as receipt-cited evidence.

## The executable invariant this incident buys

A compressed 2D layer tensor must NEVER silently fall through to
dense nn.Linear. The dispatcher FAILS CLOSED on any manifest codec
without an explicit route.

## Build plan (order matters; parity gate at every step)

1. **FusedS16Linear** in llmopt/lab (new symbols only — adopted
   historical symbols in qcuda are never mutated):
   - decode step, flat rows == 1 -> existing parity-gated
     S16Gpu.gemv;
   - prefill, flat rows > 1 -> GPU s16 row-decode kernel +
     transient row chunks + matmul (fused multirow only if
     profiling later warrants).
2. **Codec dispatcher** (qcuda_tower / qruntime): w4 ->
   FusedW4Linear, s16 -> FusedS16Linear, raw -> explicit dense
   policy, anything else -> refuse. Fail-closed check that no
   compressed 2D layer tensor is left on nn.Linear after surgery.
3. **Runtime residency planning**: artifact bytes != runtime
   bytes. Pre-build, estimate the selected representation's bytes
   for EVERY manifest tensor, compare against
   torch.cuda.mem_get_info with a safety reserve, refuse
   unexpected dense expansion. Receipt records planned + observed
   alloc/reserved/free.
4. **Generic run-provenance helper** (from the RESIDUAL
   completion-commit deviation): capture start_commit /
   start_tree_dirty / interpreter at process ENTRY; optional
   completion_commit recorded separately. Scientific provenance
   keys off start state.

## Qualification ladder before any freegen relaunch

a. synthetic s16 codec parity fixtures incl. exponent edges;
b. REAL BLe s16 tensor GEMV v canonical dec_s16 @ x on
   representative qkv/z/out shapes;
c. microbench: current dense-FP32 fallback v compressed s16 GEMV;
d. 2-layer mechanism smoke;
e. full BLe forward-1 NUMERICAL parity against the old runtime:
   finite + traversal + top1 identical + full-logit max error /
   rel-L2 under a frozen tolerance (top1 alone can agree while
   the other 248k logits diverge materially);
f. 2-token cached check;
g. gen32 speed check.
Plus, before any 3072-token freegen relaunch: qualify MEMORY
GROWTH, not just startup fit — M(N) = weights + KV/state(N) +
workspace + reserve; measure alloc/reserved/free at increasing
generation lengths first. Only then RE-REGISTER the 30+30 freegen
screen naming the NEW runtime, on a NEW output path.

Routing invariant is EXACT CONSERVATION (r2 hardening): the set
of compressed 2D manifest keys must equal the fused-module keys,
each on its codec's exact class, no missing/duplicate/unexpected/
wrong-codec routes; per-key route map lands in the qualification
receipt (verify_routes; the earlier no-detected-fallthrough sweep
could be evaded by a bad name_fn or omitted module). For the
old-v-new BLe equivalence rung, raw tensors STAY dense FP32 so
s16 routing is the only variable; the placement-aware
generalization (role x codec x placement x representation table —
embed CPU row-lookup, decoder/lm_head GPU fused, raw dense-dtype
policy) comes AFTER equivalence is banked. The ~2.8x GEMV
microbench is kernel-level only — the whole-model recovery
prediction stays unquantified until gen32. Old and new rows never merge; the
old B numbers remain behavioral context until runtime equivalence
is proven.

## Profiling + optimization (phase 2, after routing is correct)

- Nsight on gen32: kernel time v launch overhead v memory
  transfers/page faults (the WSL/WDDM paging question from the
  abort). CUDA-graph/static decode-step work only after
  compressed s16 routing lands — never optimize launch overhead
  while 6.875 GiB is represented wrongly.
- Kernel geometry: sweep BLK_C / num_warps on the real shapes
  rather than assuming 512 is optimal.
- Optional optimized s16 variant: consume packed code PAIRS (one
  u8 = two weights: load once, unpack hi/lo, reuse the shared
  E8M0 block scale, multiply two x elements). Strict parity gate.
- BANKED (RIFF-LEDGER): the analogous w4 group-of-4 kernel
  optimization — one index selects a 4-vector but the current
  kernel logically reloads it per scalar element; one-index/
  4-weight dot may cut gather overhead. Benchmark against current
  qcuda w4, never promote from intuition.
