# AXIOM-SURFACE — the house-side map of what axiom already ships

Standing rule (Artin, 2026-08-14): use axiom whenever the machinery
exists there. Order of operations for any engine-shaped need:
(1) this surface, (2) grep docs/superpowers/relay/ and RESULTS for
delivered or adopted primitives, (3) only a true gap earns a relay
ask. Origin incident: the 2026-08-14 atom farm rebuilt one-ply
emission in sympy while `emit_chain` (source tag `axiom-oneply`)
sat bound on the Mac; relay 2026-08-14-0 carries the correction.

Surveyed 2026-08-14 (two read-only Opus passes, anchors verified by
the session model). Axiom checkout: `~/code/axiom` (Mac). Line
anchors refer to that repo and drift with their HEAD; symbol names
are the stable key.

## Python bindings (the in-process surface)

`axiom_sym` — source `bindings/axiom_sym.cpp`, `INTERFACE_VERSION
= 6` (axiom a53590b + ca052f4, landed 2026-08-14), 18 names pinned
in `INTERFACE`:
parse_sstr, diff, canonical, equivalent, equivalent_mod_const,
verify_edge, dead_mask, dead_reason, predecessors, successors,
successors_dist, solve, solve_batch, emit_chain, frontier_eval,
gate_battery, PyRand, count_ops.

IV6 additions (house counter-verified 2026-08-14, relay -4):
- `PyRand(seed_str_or_int)` — bit-exact CPython random.Random twin:
  random/randint/choice/shuffle/getrandbits. Parity 16 house seeds
  (string "kind-{level}-{seed}" shapes, 0, negative, >64-bit,
  10**50) x all methods: 0 fails. LIMIT: getrandbits k>64 raises
  ValueError (CPython allows any k).
- `count_ops(expr: Expr) -> int` — sympy-1.14-exact. Takes an Expr,
  not a string: call `ax.count_ops(ax.parse_sstr(sstr))`. Score
  BOTH sides on the same sstr (axiom's parse distributes numeric
  coefficients over sums; pre-print sympy trees show representation
  drift). House parity 204 exprs incl. exp(-x)*sin(x) fraction
  shapes: 0 mismatches. Pin sympy 1.14 for any check (1.13 differs
  on exactly these shapes; house venv is 1.14.0).
- `predecessors` now returns `{rows, expired}` — the silent-partial
  defect is FIXED; deadline_ms>0 is safe. Signature takes an Expr
  (`predecessors(parse_sstr(s), max_candidates=160, deadline_ms=0,
  use_macros=True)`).
- Provenance attrs `GIT_SHA` / `BUILD_TIME` baked at compile time
  on axiom_sym (verified: ca052f4..., 2026-08-14T20:06:00Z).

IV7 (axiom 5a8ae70, landed 2026-08-14 in fresh `build-iv7/`;
house counter-verified same day, VERDICT AXIOM-IV7-ACCEPT):
axiom_sym goes to 23 names, intbirth gains INTERFACE_VERSION = 1.
All six 2026-08-11-1 exposures:
- `replay_verify(root, history, deadline_ms=0)` — the search's own
  chain verifier; `verify_size_reject_count()` distinguishes size
  rejects from censoring.
- RNS/CRT on intbirth: `rns_to_res(num, den, k)` /
  `rns_reconstruct(num_res, den_res, k)` -> {ok, num, den} with
  ok=False on exhausted modulus; addm/subm/mulm/powm/invm/res_of,
  is_prime, factor, modinv; `crt([(res, mod), ...])` -> (value,
  modulus). All traffic exact Python ints.
- `rns_primes(k)` — regenerates the pinned deterministic 61-bit
  basis (head 2**61-1).
- `anchor2_init(nprimes, prec_bits)`, `anchor2_fb_counters()`,
  `anchor2_sense()`; `anchor2_ledger()` RAISES in default builds —
  LEDGER CENSUS needs a -DAX_ANCHOR2_TRACE probe build.
- Lean: `LeanCert`, `to_lean(lhs, rhs, var='x')`,
  `sidecar_line(id, cert)`. Pass ORIGINAL sstr text (fractional-pow
  fence is lexical, pre-parse). FENCE: `eligible=True` is a lexical
  pre-filter, NOT provability — non-ring identities (sin**2+cos**2,
  exp(x)*exp(-x)) emit certs that fail `by ring`; lean4 stays the
  final rejector.
- `gemm_acc(a, w)` (w is [N,K], computes A@W.T — torch Linear
  shape), `gemm_nt_acc` (w[K,N]), `gemm_xty_acc` (X.T@Y): exact
  Python-int outputs, compose exactly across gemms;
  `finalize_rdiv(acc, d)` places ONE RoundHalfAway rounding,
  int64-narrow output guard throws on overflow. `sha256` matches
  hashlib.

PIN DISCIPLINE: the house pin is `~/code/axiom/build-iv7`
(axiom_sym IV7 + intbirth IV1, GIT_SHA 5a8ae70) — re-pinned
2026-08-14 after AXIOM-IV7-ACCEPT, the deliberate act this
paragraph requires. `build-rel` (IV6, ca052f4) remains frozen as
the artifact the IV6 acceptance cites. Never import both in one
process; assert the GIT_SHA attr at arm time. The Lean
`eligible` fence is documented upstream (their doc-only c154ac1):
eligible = emitted, lean4 kernel = rejector of record.

Semantics that bite (all verified in source):
- `equivalent`/`equivalent_mod_const` return three-valued strings;
  UNDECIDED means fall back to sympy, never treat as valid.
- `solve`, `solve_batch`, `successors`, `successors_dist` carry an
  `expired` flag: censored != fact, never fossilize a censored
  solved=False (E4 amendment doctrine, in the binding docstrings).
- `solve_batch` PLY SEMANTICS: plies == len(history) INCLUDING
  whole-expression algebra moves (cancel/expand/subs_eval);
  house carrier-rewrite count = plies minus those. PARITY FENCE in
  the docstring: E4 FAILED — labels live under engine=axiom
  permanently; the two label families never mix.
- `emit_chain` returns farm_v22-shaped rows {cur,nxt,level,source,
  hints,think}; source is `axiom-oneply` for 1-step solves,
  `axiom-chain` otherwise; every pair re-verified via verify_edge,
  rejects counted in dropped_pairs, replay_ok=False roots skipped.
  THIS IS THE ATOM EMITTER.
- `successors` children are verify_edge-certified (verify_p=1.0);
  `frontier_eval` enumerates UNVERIFIED (verify_p=0.0) and defers
  the oracle to top-k — rows with verified=None are unchecked, drop
  or hard-flag them.
- `predecessors` (IV6): returns `{rows, expired}`; deadline_ms>0
  is now safe. (Pre-IV6 silent-partial fence retired 2026-08-14.)
- Scoped adoption of record (RESULTS L8281): axiom bridge =
  default enumerator for soundness-consumers; house
  derivation.successors stays the semantic reference for house-set
  replication; sympy stays oracle-of-record at final verification.

`intbirth` — source `bindings/intbirth.cpp`: int_gemm/_nt/_xty,
rdiv, Block (fwd/bwd/body/moe/rms/softmax_rows), AdamW (params
mutated IN PLACE), FullBirth, ExactAnchor (bit ceiling is a
PROCESS-GLOBAL static — never run two anchors concurrently),
MultiBirth, MoeBirth. NO INTERFACE_VERSION on this module — no
version handshake; assert shapes/behavior at arm time instead.

DUAL-.SO HAZARD (downgraded 2026-08-14): repo root still carries a
Jul 28 IV5 build; `build-rel/` is the Aug 14 IV6 build with baked
GIT_SHA/BUILD_TIME. Pin `~/code/axiom/build-rel` and record
GIT_SHA in any booking that cites axiom numbers — provenance is
now one attr read, mtime/size no longer needed. An import that
answers IV5 or lacks GIT_SHA grabbed the stale root .so.

## Rule table

`default_rules()` in `src/search/rules.cpp` (+rules2/rules3): core
d_* rules; integral tranche i_const, i_power, i_sum,
i_const_factor, i_table, i_usub, i_parts, i_apart, i_log_power,
i_transcend_div, i_inverse_trig, i_sqrt_basis, i_cyclic, i_unprod,
i_ansatz_exp, i_linear_basis; algebra cancel/expand/subs_eval.
`i_heurisch` is NOT native — injected per-call as an external
Python slot by solve/emit_chain.

## CLI tools (build-rel/, flat — bin/ is an empty trap)

axiom-oracle (JSONL parity/verdict harness), axiom-qual-gate,
axiom-chain-emit (farm-shard chain emission with hints/think),
axiom-inverse-gate, axiom-boards, axiom-ode-sample,
axiom-series-sample, axiom-series-chain, axiom-poly-chain,
axiom-phys-chain, axiom-zx-chain, axiom-nt-chain,
axiom-nt-callspan, axiom-nn-logits, axiom-nn-exact,
axiom-nn-greedy, axiom-nn-moe-greedy, ax_bench, ax_tests.
Out-of-CMake rigs: tools/fp32limb (Metal exact-arithmetic rig +
receipts), tools/exact_anchor, tools/engine_scale (names our
RESULTS L22317 pre-reg), tools/int_adamw, tools/fx_v2, tools/fx_v3,
tools/moe_merge. cuda/rns_chain.cu builds standalone (sm_86);
its receipts never sit next to Metal numbers (both labs' fence).

## Capabilities with NO python binding (relay-ask candidates)

- Lean 4 certificate emitter (`sym/print_lean.hpp`) — machine-
  checkable proofs for EQUIVALENT verdicts, no house consumer yet.
- RNS/int256/dyadic exact core, fp32limb GEMM + Metal kernels.
  (pyrand and count_ops BOUND at IV6 — see bindings section.)
- `sym/budget` semantics ANSWERED (relay 2026-08-14-2): cooperative
  thread-local deadline, work_expired throw at polls, conservative
  rejection, never a partial. Bounded modulo poll gaps, not a hard
  OS wall. House policy: bridge calls in-process for desk/gate use;
  farm loops keep the fork wall; overshoots past ~one poll stride
  are bug reports to axiom.

## Cross-lab obligations (checked 2026-08-14: none open)

Axiom's specs list three waits on llmopt; all were delivered:
64 ref-side rows adjudicated (RESULTS L2576), successors sample
band pinned + acceptance run (RESULTS L8232/L8281), Fourier
verdict relayed (relay 2026-07-27-2). Their spec text is stale,
not our debt — say so in the next relay.
