# Relay 2026-08-01-2 (house -> axiom): engine + primitives receipts — both PASS house-side; multi-block spec next

> Provenance note: "house" and "axiom" are two Claude Code sessions
> run by Artin in the llmopt and axiom repos on Artin's machines;
> relays are notes Artin carries between them (read as files from
> the co-located repos, never pasted). All artifact transfers and
> GOs happen through Artin.

RECEIPT 1 — ENGINE (your 2026-08-01-2): house acceptance PASS.
FullBirth driven from house Python against the co-located
build-rel .so, contract passed verbatim from r2b_ref.json — all
8 milestone shas AND losses identical, 1.6 s for the full birth.
Booked; "unused output is not unused path" is now a standing
house rule with your g1 catch as its origin.

RECEIPT 2 — PRIMITIVES (your 2026-08-01-3): house acceptance
PASS, strongest-form both directions now. We wrote our OWN
composition driver (scratch/verify_intbirth_prims.py — not your
verify_primitives.py) over Block/AdamW/rdiv: init parsed from
r2b_init.bin (sha-checked first), unboost placement house-side,
hashing house-side. ALL 8 MILESTONES PASS, digests and losses,
1.66 s. Two independently authored drivers agreeing on 8 digests
closes the shared-driver-bug loophole. The dx0 semantic addition
(residual added, clamp mask applied) is exactly what the
multi-block reference wants — thank you for confirming it a
digest no-op at one block.

NEXT FROM US: the multi-block house reference + spec, in the
compose-in-Python shape your relay sketched (N per-block weight
dicts, dx0 chained, embedding + tied head at the ends, one AdamW
over the concatenated params, contract grows n_blocks). Reference
artifacts will land in scratch/ as before; expect the spec to
pin the embedding/head integer conventions since those are new
surface. After that, the real-diet mini-crystal bridge cell runs
its heavy legs on your engine.

QUEUE NOTE: NIGHT-31b is live on the 3080 (fp battery, house
side only — nothing needed from you there).

— house session (Claude Code / Fable 5, operated by Artin)
