# Proposal: sealing amendment for SYNTHETIC-GRADIENT-WRITER-1 qualification (arena: top four blocks over the frozen backbone)

Status: PROPOSAL for Artin. Not sealed, not registered, nothing armed.
On Artin's word it is booked verbatim as AMENDMENT
SYNTHETIC-GRADIENT-WRITER-1-SEAL (target L69765 + L70126), the birth
driver is written and smoked, the prereg-auditor runs on the clean
tree, and a separate GO fires the seed-27 births.

Basis: PRE-REG SYNTHETIC-GRADIENT-WRITER-1 (L69765), AMENDMENT -ARENA
(L70126, Artin's seven folds), OBSERVATION SG-PREDICTOR-AUDIT-0
(L70301, constants), VERDICT FROZEN-BACKBONE-1 (L70024, the arena's
same-writer context), the integrity smoke receipt
logs/sgwriter1/integrity_smoke.jsonl (4/4 cases pass at ea4b51ca).

## 1. Writer law (frozen)

- Arena: emb + blocks 0..3 frozen at W_0 (freeze_lower(model, 4), the
  FROZEN-BACKBONE-1 law, re-verified bit-identical at the end of each
  birth). SG blocks 4..7. head.weight and norm.g: true CE gradient with
  x_8 detached (unchanged from L69765).
- Credit: hat_delta_l = s_l * G_phi_l(stopgrad(x_{l+1}), stopgrad(e_t))
  per token; block l receives J_{f_l}^T hat_delta_l only, through the
  surrogate sum_l <hat_delta_l.detach(), x_{l+1}> with every SG block
  input detached (scratch/sg_credit.py writer_forward / sg_step_terms).
- Constants s_l: the booked arena constants s_4 = 5.464e-6, s_5 =
  4.479e-6, s_6 = 3.564e-6, s_7 = 2.441e-6 (L70301), fixed for the
  whole birth; credit applied in delta^BP units by construction
  (de-normalized deterministically). No runtime gain, no running mean.
- Targets: delta^BP_l = dL/dx_{l+1} from the parameter-detached
  teacher pass (functional_call over {name: p.detach()}, x_0 a leaf
  activation) at the same pre-update state; used only as the predictor
  regression target delta^BP_l / s_l.
- Predictor regression: L_phi = sum_l mean over mask=1 tokens of
  ||G_phi_l - delta^BP_l / s_l||^2; AdamW on phi only, wd 0, clip 1.0 on
  phi separately, OneCycle of the stock shape at the predictor LR.
- Step order (frozen): writer forward; hat_delta from the CURRENT phi;
  model surrogate backward, clip 1.0, model step (blocks 4..7 + norm +
  head, stock AdamW lr 3e-4 wd 0.01 OneCycle); teacher targets;
  predictor loss backward; predictor step. Skipping the predictor step
  leaves the model gradients bit-identical (smoked, check g).
- Predictor init: output layer zero (Jaderberg); LINEAR G = hA + eB + C
  (163k params per block) or MLP-256 (one ReLU hidden layer of width
  256, 175k per block); predictor init seed = 1000 + birth seed.
- Registered design property, corrected from L69765: with zero-output
  predictors the first step's block GRADIENTS are exactly zero; the
  blocks still move by AdamW's decoupled weight decay (p *= 1 - lr wd),
  the stock optimizer's own motion, not credit (smoke check
  first_step_block_move_is_pure_decay). "The first steps move only
  head / norm" in L69765 is therefore restated as "the first steps
  carry zero credit to the blocks".

## 2. Integrity law (resolved per fold 5, preferred implementation)

Structural: the true loss graph contains activations only (parameters
enter as detached constants), so true gradients cannot reach any
block parameter by construction. Smoked mechanically on the
frozenbb1_smoke finals for (arena, full) x (LINEAR, MLP-256), all
seven checks (a) to (g) of L70126 fold 5 pass with bit-exact equality
where registered (receipt logs/sgwriter1/integrity_smoke.jsonl,
tolerance (a) 1e-5 on fp32 CPU logits, measured 0.0). The same smoke
reruns on the launch commit before the seed-27 births and its rows
are receipts of the qualification.

## 3. Ladder (frozen before seed 27)

- Control first: BP-top-over-frozen at seed 27 (MODE=zero K_BP=4 under
  the FROZEN-BACKBONE-1 driver law), gate = c. ADEQUATE-CONTROL: finite
  and c >= 24, else NOT-RESOLVABLE-CONTROL, no SG cell adjudicated.
- Ordered SG cells at seed 27, same W_0 / stream / schedule / device:
  1. LINEAR, predictor lr 3e-4
  2. LINEAR, predictor lr 3e-5
  3. MLP-256, predictor lr 3e-4
  4. MLP-256, predictor lr 3e-5
  Minimal-sufficient stopping law: cells run in this order; the ladder
  STOPS at the first cell whose gate g satisfies c - 7 <= g <= c + 7
  (FUNCTION-MATCH) and that cell's recipe is the selection; later
  cells are not born. If no cell matches, the ladder books
  ACCESSIBILITY-ONLY with every cell's gate (STABLE iff finite loss and
  g > 0; FLOOR g >= 24 descriptive), and no discovery follows.
  Selection is by the frozen order, never by the highest gate.
- Function match uses c from the seed-27 control only; the replicated
  FROZEN 59 to 63 is contextual control-sanity evidence (fold 2).
- Descriptive per cell: predictor regression loss per block over
  training, cos(hat_delta_l, delta^BP_l) per block on the frozen probe
  at the 17 snapshots (the alignment desk law), ACT vector on the
  final, wall per step (priced about 2x the FROZEN wall, to be
  measured in the driver smoke).

## 4. What qualification does and does not establish (fold 1)

A FUNCTION-MATCH qualifies the SG credit law and the selected predictor
recipe on blocks 4..7 over a frozen random backbone. It does not
establish full-stack SG. On success the selected family, predictor LR,
constants, integrity implementation and every writer-law detail above
are FROZEN and the full-stack discovery at seed 2 (full-stack
constants s_0..7 of L70301, SG blocks 0..7, emb via the identity path)
is pre-registered as the accessibility / discovery test with the paired
same-W_0 BP control and no tuning from its outcome.

## 5. Cost, receipts, gates

- Five births at most (control + up to four SG cells), FROZEN-class
  walls: control 28 min; SG cells about 2x (two forwards + two
  backwards per step) about 55 to 60 min each; worst case 4.5 h;
  storage 5 x 18 x 75.7 MB = 6.8 GB (disk 29 GiB free after the prune).
- Receipts: logs/sgwriter1/qual.jsonl (birth, gate, ladder rows),
  logs/sgwriter1/selection.json, per-cell train logs, gate.log,
  integrity smoke rerun rows, logs/liverun/sgwriter1q.jsonl; smoke
  path-isolated (checkpoints/sgwriter1_smoke/, logs/sgwriter1/
  smoke.jsonl).
- Gates before launch: driver written as a sibling of birth19m_fb.py
  (MODE=sg, FAMILY, PLR), source-invariant test, endpoint identity
  test (SG with hat_delta forced to delta^BP reproduces the BP-top-
  over-frozen gradient bit-exactly through one step; SG with zero
  predictors reproduces "head / norm only" gradients), 300-step smoke
  of every cell shape, integrity smoke rerun, clean-tree
  prereg-auditor, Artin GO.

## 6. Registered prior (proposed)

Control c 55 to 65 (p 0.8). LINEAR lr 3e-4 FUNCTION-MATCH p 0.35;
ladder matches at some cell p 0.55; ACCESSIBILITY-ONLY p 0.45. If a
match occurs, the predictor regression loss falls below 0.5 (normalized
units) by step 1028 on the matched cell (p 0.6). cos(hat_delta,
delta^BP) at the final above 0.3 on every SG block of a matched cell
(p 0.6).

## 7. Open points for Artin

1. The stopping law: stop at the FIRST matching cell (proposed) v run
   all four and select the first matching in the frozen order (costs
   up to 2 h more, yields the full ladder picture).
2. Predictor LR ladder order: 3e-4 before 3e-5 (proposed: the stock LR
   first; Jaderberg's 3e-5 second) v the reverse.
3. Whether the alignment diagnostic cos(hat_delta, delta^BP) on the
   snapshots should carry a bar (proposed: descriptive only).
