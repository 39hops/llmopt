# Relay 2026-08-11-1 (house -> axiom): pybinding coverage audit — six exposure asks, counter-book path first

## WHO IS WRITING

Fable 5, llmopt main seat. This relay rides on no new verdict — it is
an EXPOSURE ASK, grounded in a read-only audit of your checkout at
`bindings/axiom_sym.cpp` + `bindings/intbirth.cpp` (2026-08-11
evening, house side; file:line cites below are your tree). DRAFT —
Artin has not sent this yet; nothing here is a commitment.

## Verified, not accepted

House-side audit (read-only courtesy pass over your working tree):

- Binding surface: `axiom_sym` (Expr, parse_sstr, diff, canonical,
  equivalent[_mod_const], verify_edge, dead_mask/reason,
  predecessors/successors[_dist], solve[_batch], emit_chain,
  frontier_eval, gate_battery; INTERFACE_VERSION=5) and `intbirth`
  (int_gemm/_nt/_xty, rdiv, BlockCache, Block, AdamW, FullBirth,
  ExactAnchor, MultiBirth, MoeBirth). No .pyi stubs, no version attr
  on intbirth.
- Everything below is checked to exist unbound in your headers at the
  cited lines. If a line moved since our read, the symbol name is the
  anchor.

## The ask — six exposures, ranked by counter-book value

The pattern we keep hitting: every time the house needs to
INDEPENDENTLY RECOMPUTE one of your results (counter-book doctrine:
we never accept tables), the needed primitive is on the C++ side of
the wall, so we end up re-implementing it in Python or reading your
tool's own printout — the exact failure mode counter-booking exists
to prevent.

1. **`replay_verify`** (`include/ax/search/search.hpp:161`) — full
   chain replay verification. Only the single-edge `verify_edge` is
   bound; house-side chain verification currently composes edges in
   Python, which is not your verifier. Also expose
   `verify_size_reject_count()` (`search.hpp:109`) so a reject is
   distinguishable from size-guard censoring.
2. **RNS/CRT reconstruction** (`include/ax/core/rns.hpp:77`
   `rns::reconstruct`, `:57 to_res`, `:22-37 addm/subm/mulm/powm/
   invm/res_of`; `include/ax/core/nt.hpp:27 crt`, `:23 modinv`) —
   the counter-book primitive for every exact-arithmetic claim. We
   want to hand your reconstruct OUR residues and vice versa.
3. **NPRIMES ladder control** (`include/ax/core/rns.hpp:46-55`
   `rns::ctx::make(k)`; `nt.hpp:13,15 is_prime`, `:18 factor`) — the
   prime basis is currently pinned C++-side; the house cannot
   regenerate or check the ladder independently.
4. **Anchor-v2 ledger readout** (`include/ax/nn/exact_anchor2.hpp:105
   Exact2::init(nprimes, prec_bits)`, `:165 reconstruct_rat(why)`,
   `:44-60` fallback counters, `:78-84` cofactor-witness ledger) —
   the COFACTOR-WITNESS census numbers are today readable only from
   your CLI's own output. Binding the ledger struct makes the census
   counter-bookable.
5. **`to_lean` / `sidecar_line` / `lean_cert`**
   (`include/ax/sym/print_lean.hpp:51,56,32`) — the certification
   path has zero Python surface; house Lean-corpus work re-derives
   what your emitter already guarantees.
6. **Wide-accumulator GEMMs** (`include/ax/nn/intbirth_core.hpp:300
   gemm_acc`, `:315 gemm_nt_acc`, `:330 gemm_xty_acc`) — Python can
   only compose the per-gemm-ROUNDED forms, so multi-gemm
   compositions cannot reproduce your single-rounding placement.
   Also a standalone `sha256` (`intbirth.hpp:38`) so we can digest
   byte streams with YOUR hasher instead of trusting ours matches.

Small riders, zero urgency: a version attr on `intbirth`
(axiom_sym has INTERFACE_VERSION; intbirth has none), and .pyi stubs
if cheap.

## Fences

- This is an interface ask, not a run request: no machine time
  implied, no [HOLD] created, nothing gates on it.
- House side will counter-verify any new binding on landing (same
  bar as FX-V3: decode YOUR shipped artifacts, never regenerated).
- Mac allocation unchanged; 3080 unaffected; no cross-device claims
  ride on this relay.
