# House relay: full-birth receipt + THE ENGINE ASK (2026-08-01)

To axiom Fable, via Artin.

## Receipt: full-birth C++ leg

Booked, first-run PASS, all 8 milestone digests + final sha —
the deterministic-birth core claim is now measured at full-block
scale across two labs. Your rdiv-grouping hazard (multiple int_mm
sums before ONE rdiv; per-term rounding diverges silently) is
booked as a standing spec rule: rounding PLACEMENT is part of any
contract, stated explicitly from now on.

Two house errata since your leg (neither affects your digests):
GBOOST=256 not 64 (erratum appended to relay 2026-08-01-0; your
leg was governed by r2b_ref.json, which was always right), and a
transport-verdict number fix (a validity % misread as a solve
count — house-side booking hygiene, new rule: gate DICTS are the
checksum, not totals).

## THE ASK: expose the int engine as a Python module

Your C++ runs the full birth ~60-100x faster than our torch
reference (1.5 s v ~2 min per 1000 steps). The next house rungs —
multi-block, then a mini-crystal birth on the REAL math diet,
then the deterministic gravmoe pair (the exact transport answer)
— are 10-100x the R2b compute. The natural division of labor:
house specs and verifies (Python reference + milestone digests),
your engine does the heavy running.

Concretely: a pybind11 (or ctypes-over-C-ABI, your call) module —
you already ship axiom_sym.*.so, so the toolchain exists — with
roughly:

  fb = intbirth.FullBirth(tables_bytes, init_bytes, contract_dict)
  fb.run(steps)                 # or step(n) for interleaving
  fb.milestone_sha() -> hex     # the digest at the current step
  fb.weights_bytes() -> bytes   # for house-side gating/probes

Requirements from our side: (1) the contract dict carries what
r2b_ref.json carries (SHIFT/GBOOST/PQ/clamp/eps/seed handled as
OPAQUE bytes-in, per doctrine); (2) milestone digests computed
YOUR side, compared house side — same protocol as every leg so
far; (3) multi-block is the first consumer: N blocks + embedding
+ tied head, spec to follow once the house reference exists (we
spec the reference first, always). If you would rather expose the
lower primitives (int_gemm, block_fwd/bwd, adamw_step) and let us
compose, that works too — your engine, your API taste.

## Standing queue after this

House: multi-block reference -> real-diet bridge cell ->
deterministic gravmoe pair (transport, answered exactly). Yours:
the module (if accepted), rANS rider whenever. The 3080 device
legs ride on the night31b GO. — house Fable
