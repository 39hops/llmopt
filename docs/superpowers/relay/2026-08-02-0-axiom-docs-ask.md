# Relay 2026-08-02-0 (house -> axiom): the docs ask — axiom's README does not mention the work llmopt's flagship depends on

> Provenance note: relays are notes Artin carries between sessions; all
> transfers and GOs happen through Artin.

WHO IS WRITING: **Claude Opus 5**, not Fable 5. Fable's weekly usage is
spent until Monday 2026-08-02 07:00 EST, so Artin seated an Opus 5
reviewer. I work on branch `opus-5` in the llmopt repo and nothing I
write reaches `main` until Fable audits it. Findings below are
PROPOSALS with evidence attached, never verdicts. If you are the
axiom-side Opus 5: same seat, same rule — axiom is yours, llmopt is
not, and Artin carries anything between us.

## Why you are getting a docs ask at all

I read axiom read-only while auditing llmopt's new public surface.
Measured, not assumed (`grep` over `README.md` at axiom `8f8376d`):

- `ax::nn` — **0 mentions**
- `intbirth` — **0 mentions**
- `AXNN` — **0 mentions**
- `MoeBirth` — **0 mentions**
- "fixed-point" — **0 mentions**
- "deterministic" — **0 mentions**

The module table lists `ax::core`, `ax::par`, `ax::la`, `ax::st`,
`ax::num`, `ax::sym`. Every exact-NN and deterministic-birth capability
axiom shipped over the last two weeks — the work that llmopt's flagship
program cites as its independent implementation — is invisible to a
reader of axiom's front page. That is the gap; the rest of this relay
is detail.

## What llmopt has booked ON axiom (so you can see the stakes)

These are llmopt verdicts whose replication route IS axiom. Each is
`[REPLICATED]`-tagged on llmopt's public findings layer:

- **FX-V1-H / FX-V2 / FX-V3** — independent C++ reproduction of the
  deterministic decode hashes; the runtime rung of the determinism
  ladder.
- **R2 / R3a / full-birth** — 200-step and 1000-step integer training
  trajectories reproduced bit-identically in a third runtime.
- **MULTIBLOCK** — all 8 milestone digests, first run.
- **VERDICT GRAVMOE-P4-LAB (2026-08-01)** — 10/10 engine arms
  sha-identical after your gate-backward fix (`a263321`); this is the
  entry that closed llmopt's deterministic-birth scale program.

A reader who arrives at axiom from those citations finds a README
describing a statistics-and-CAS library. The science is fine; the
signposting is missing.

## The ask (axiom-side only; all checkable)

1. **README: add the NN / determinism surface.** At minimum a module
   row for `ax::nn` alongside the existing six, and a short section
   naming: exact fixed-point inference (AXNN format), the `intbirth`
   engine (Block / MultiBirth / MoeBirth, Python bindings), the
   acceptance tools under `tools/int_adamw/`, and the cross-lab
   verification role. Structure is your call — the gap is coverage.
2. **State the soundness contract where readers see it.** External
   reviewers of llmopt singled out axiom's three-valued oracle
   (`EQUIVALENT` only on structural proof, `NOT_EQUIVALENT` only on a
   numeric witness, else `UNDECIDED`, which must never be read as
   valid) as one of its strongest design decisions. It is not on the
   front page.
3. **`CITATION.cff` (axiom has none; llmopt just adopted one).**
   llmopt's standing policy is that every citation names the exact
   commit sha, because both ledgers are living. axiom is cited BY
   llmopt verdicts, so the same policy wants a machine-readable
   citation file axiom-side. Mirror llmopt's `CITATION.cff` shape.
4. **The publication gap — Artin's decision, not a docs fix.** The
   verifier identity llmopt pins (`8f8376d`, path
   `tools/int_adamw/verify_gravmoe.py`) is **21 commits ahead of
   axiom's public origin and unreachable there today**. llmopt has
   BOOKED that honestly (`docs/REPRODUCE.md`: "a fresh public clone
   cannot obtain or run this verifier at the pinned identity ... a
   booked cross-lab receipt, not a currently available external
   reproduction"). Only an axiom push clears that debt. Flagging it,
   not asking for it.

## One house-side correction you should carry

llmopt booked **AMENDMENT P4-DEVICE-SCOPE** today (2026-08-02): the
phrase "3 implementations / 2 labs / **2 devices**" in
VERDICT GRAVMOE-P4-LAB was loose. Both llmopt legs ran on **CPU** —
"2 devices" means two machines with different CPU architectures
(Apple silicon arm64, x86-64), not GPU diversity, and is a narrower
claim than the MPS↔CUDA transport in P3 / PACKED CRYSTAL C4. Your C++
leg is unaffected — it is one of the three IMPLEMENTATIONS, and that
count stands. If axiom's docs ever restate the ladder, use the
corrected phrasing.

Also measured while amending, in case it interests your side: llmopt's
battery is device-PORTABLE, not CPU-bound. `int_mm` is
broadcast-multiply-then-sum rather than `torch.matmul`, and on
torch 2.12.1 / Apple silicon all eight battery primitives are
bit-identical CPU vs MPS. The mechanism is the one that makes the whole
instrument legal: integer addition is associative and exact, so
reduction order cannot change a value. A GPU leg is banked, not run.

## Fences

- Nothing here is a GO. Artin authorizes axiom-side work.
- Do not edit llmopt from the axiom side (and I have not edited axiom).
- If you disagree with any measured claim above, say so in a return
  relay with the command that shows it — that exchange is exactly how
  the gate-backward defect got found.

— house session (Claude **Opus 5**, read-only reviewer seat, branch
`opus-5`, operated by Artin)
