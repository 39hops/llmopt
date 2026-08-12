---
name: probe
description: Use when the user proposes any multi-hour run — "train a model", "queue a run", "start a birth", "how long will that take" — to check first whether a probe on existing checkpoints answers the same question. Also for "write a probe", "effective context", "k_eff", "loss floor", "can we test this faster".
---

# Probe first (measurement-cost triage)

Origin: 2026-08-11, Artin — "is there a quicker/more granular way to
test instead of spending 3 hours training a model?" There was. A
five-hour width-ladder question became a ten-minute-per-checkpoint
curve, and the probe answered something the ladder could not.

**Ask this before every long run.** Not to avoid the run — some
questions need training — but because the answer is often reachable
without one, and probes are granular where training gives a point.

## The triage

1. **Name the observable.** What number changes if the hypothesis is
   true? If you cannot name it, no instrument helps yet.
2. **Ask whether an existing checkpoint already carries it.**
   `data/catalog/models.jsonl` lists every checkpoint with arch and
   shas. Anything about what a trained model *has learned* is often
   readable off weights that already exist.
3. **Cost the ladder.** Training run (hours) > probe on existing
   checkpoints (minutes) > synthetic micro-task (minutes, and
   isolates one faculty). Move the question down the ladder if it
   will go.
4. **Then say what only the long run can answer** — usually
   capability (the gate) and the training floor itself. Run it for
   those, not for what the probe already covers.

## House probes

- **`scratch/keff_probe.py` — effective context.** Loss at deep
  positions as a function of how many trailing tokens the model may
  see (k = 4..128, plus full). The knee is measured k_eff. ~10 min
  per checkpoint on CPU, no GPU, no training. Works on anything in
  the catalogue, retroactively.
- **`scratch/loss_floor_census.py` — corpus entropy.** H_k of the
  diet, the yardstick a floor is read against. Corpus-side, so one
  run serves every model trained on that diet.
- **The standard 120 gate** (`llmopt.lab.gate.gate_checkpoint`) —
  capability on any checkpoint, no training.
- **`scratch/softprompt_sampler_probe.py` — harness v stock.** When a
  wrapped or rebuilt model gates differently from the checkpoint it
  was built from, this separates the two candidate causes in seconds:
  part 1 compares logits and every state-dict tensor (is the model
  identical?), part 2 compares `torch.multinomial` draws at two
  category counts (is the sampler identical?). CPU, no training.
  Reach for it before theorizing about any harness discrepancy.

## Building a new probe — the rules that bit

- **Positions must be identical across conditions.** If condition A
  scores different positions than condition B, the difference is a
  position-mix artifact, not the effect. Score the same set every
  time.
- **Truncation must preserve absolute positions.** Left-pad the
  window so kept tokens keep their real indices; renumbering them
  moves RoPE and overstated loss by ~0.36 nats at k=128 when it was
  measured. Masked pads, not renumbered windows.
- **Fixed sample across models**, string-seeded and label-free, or
  the curves are not comparable.
- **Add an anchor cell** (full context, or a known-good condition).
  It is how you catch a broken probe before it produces a verdict.
- **Never compare probe numbers to training-floor numbers.**
  Different position mixtures by construction — say so in the fence.
- **When a reading changes, isolate the model from the sampler before
  explaining it.** They fail independently and the fix differs. A
  bit-exact forward with a different gate number is a sampler
  problem, and no amount of weight comparison will show it — the
  weights sha and the logits both agree while the trajectories
  diverge. Two cheap checks, in that order, beat one plausible story.

## Booking

A probe result books like any other: pre-reg with bars first (`/rung`
step 2), fences named, then `/book`. Smoke output is labeled `smoke`
and is never a verdict — a probe deserves the same discipline as a
five-hour run, and costs far less to repeat when it is wrong.
