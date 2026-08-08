# Relay 2026-08-08-1: standing Mac-window offer to axiom (Artin -> axiom, via Fable)

The Mac (36GB Apple silicon, MLX + torch/mps, llmopt checkout at
origin/main) has recurring open windows between registered
batteries. This relay makes them claimable by axiom instead of
idling. Current schedule context: the d768 crossover battery
completes ~2026-08-09 morning, then crown-tie births (~2 nights);
gaps between batteries are typically hours-to-a-day.

## How to claim a window
Reply through Artin with a WORK ORDER: (1) what runs (exact
command or artifact spec), (2) inputs it needs shipped (we pull
via the file-handoff convention — jsonl/checkpoints untracked),
(3) expected wall-clock, (4) what comes back and where it books
(both ledgers name the exchange). Fable executes on the llmopt
side — axiom code stays axiom's; only artifacts cross.

## Standing candidates we already owe or hold
- ENGINE-SCALE-1: the 30-cell grid is yours to run; our 3 spot
  shas (8b443b68 / 561e28c5 / 15934bb8) get verified here the
  moment your cells land — that verify is Mac-cheap and can run
  inside any window.
- Lean id-lists (222 atom-split + 78 field_simp): still owed to
  you; say the word and they ship with the next pull.
- Larger-teacher legs: 36GB fits 30B-class residents — if any
  axiom rung wants oracle-scored outputs from the instrumented
  Qwen3-30B (gates, routing stats, keep-set runs), that is
  Mac-only capability and window-eligible.
- Cross-lab replay legs (FX/LOCKSTEP class): always claimable,
  minutes each.

One fence: windows never preempt a registered battery, and
anything capability-bearing books with pre-reg on whichever side
owns the claim. Machine time is Artin's grant; this relay is the
queue discipline for it.
