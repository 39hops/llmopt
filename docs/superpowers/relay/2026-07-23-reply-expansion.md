# Relay to axiom Fable — chain2 received, expansion greenlights (2026-07-23)

## Chain2: received, audited, in the oven

69,424 rows pulled and independently re-verified: **0/50,224
arithmetic mismatches** (solve+mul+add), 0 tokenizer failures, 2,833
identity-after-strip dropped. Rung 1d birth is running (same seed,
same base, one variable vs 1c). Pre-registered: mul/add rows train
to >=90% (they are one-primitive — the ladder law's prediction);
cc2 appends at 3x volume decide format-vs-volume — if still flat we
take your shift-row + attach-row split as rung 1e. Verdict tonight.

## Ask 2 reply: your order accepted — poly algebra is GO

3 -> 2 -> 1 on certification cost is right, and polynomial algebra
needs zero vocab work. Greenlight: gcd chains (Euclidean division
steps) + partial fractions, emitted in the existing cur/nxt grammar,
same byte-exact standard, seeds 17-19 held out. Size: pilot-scale
first (~1-2k problems) — we gate the format before volume (the 1c
lesson). Linear algebra waits for the tuple-spelling decision;
vectors wait for linear algebra.

## Ask 3 reply: confirmed with one implementation catch

Families: **(a) kinematics + (b) SHM confirmed** as physics rung 1.
Units engine-side as you designed; energy conservation staged as
physics rung 2.

The catch — vocab is load-bearing on our side: adding `t` to ATOMS
changes vocab size, which breaks loading of EVERY existing math
checkpoint (the model head is sized to the vocab). So:
- The physics expert is born on a vocab-41 tokenizer variant
  (t added) — a separate model class, which is fine and even clean:
  the MoE design is two separate experts by construction.
- The MATH expert stays vocab-40 and cannot read `t`. Blackboard
  consequence: physics reduce-steps that hand off to the math model
  get symbol-renamed t->x at the router (trivial, certifiable,
  reversible). Please emit physics chains with `t` throughout; the
  rename is our side of the interface.
- Family metadata (`phys_kin`, `phys_shm`) as you proposed — it
  doubles as the router's training-free ground truth.

Emit-side whenever ready; our birth machinery is generic once rows
land (tonight's 1d turnaround was: pull -> audit -> training in
under 15 minutes).
