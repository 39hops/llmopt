# Spec 2026-08-07: ENGINE-SCALE — the joint-scaling sweep on the intbirth engine

Status: DESIGN (Artin GO for the spec; the sweep itself fires after
a Fable+axiom readiness check). Motivation: VERDICT P-CAPACITY-2
closed the single-knob space — at the 264-token diet, steps regress,
windows saturate, params in both directions land above the anchor,
only decay bends (11,777 v 12,518). The plateau is a JOINT
(params x windows x steps x schedule) surface, and joint sweeps are
dozens of cells — unaffordable at Python speed (~1-2 min/cell x
dozens x repetitions), trivial at the intbirth engine's measured
60x (1.5 s per 1000 steps, axiom C++, full-birth digest-certified).

## The question

Where does the deterministic integer birth's loss floor actually
move when capacity, data, horizon, and schedule scale TOGETHER —
and does the MB-S14 law (width stabilizes, decay converts) hold
across the joint surface or break at a named corner?

## Grid (v1 proposal, ~36 cells, engine-side ~minutes total)

- params: 31k / 60k / 110k (the P-CAPACITY-2 ladder, FFN/NBLK only,
  DIM=64 fenced by the clamp law)
- windows: 8 / 32 / 128 (128 needs a diet-file sufficiency check
  at draw time)
- steps: 1000 / 4000 / 16000
- schedule: const / SCHED (lrd x2 @ quarter-points, the MB-S14 arm-B
  shape generalized to the horizon)
Not full-factorial: registered slices — (a) the diagonal (scale all
together), (b) each axis at the largest joint point (leave-one-small),
(c) the anchor row for continuity. ~36 cells.

## Bars (pre-reg before fire, this spec is not the pre-reg)

- P-JOINT: the diagonal's largest cell beats 11,266 (the 10% class
  no single knob reached) -> joint scaling converts; the binder was
  interaction, not any axis.
- P-DIET-FLOOR: no cell beats 11,266 but leave-one-small shows the
  WINDOWS axis binding (only cells with 128 windows improve) ->
  the floor is data; the diet itself becomes the lever (gen-4 has
  the rows; NWIN=128 is 4,224 tokens, still tiny).
- Schedule interaction booked descriptively at every cell (the
  decay law's joint-surface test).

## Contracts

- Engine leg is axiom's (their C++ intbirth engine, full-birth
  digest-certified f8aa16f); house leg = spot-verify N cells
  digest-identical via the Python stack (the LOCKSTEP pattern:
  digest equality, exact at n=1). Cell outputs = milestone losses +
  FINAL trajectory sha per cell, one jsonl row each.
- Relay: this spec travels to axiom with the cell list; their
  engine runs the grid, house re-derives the sampled cells. Same
  verification shape as every cross-lab rung.
- Fences: deterministic integer battery scope; seed 17 (a seed
  column is a v2 extension, not v1); wall-clock never a readout;
  the diet file's first-N-encodable rule is the frozen window
  draw (ids shas booked per cell).
