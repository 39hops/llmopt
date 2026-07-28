# The Calibration Program — Implementation Plan

> **For agentic workers:** House policy overrides the default here: NO
> subagents in this repo (Fable-only, Artin's standing policy). Execute
> inline via superpowers:executing-plans. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** Make near-tie density a measured instrument (flips/token under
Q=16 snap), measure the mass leg of the branching-entropy floor, test
distribution rows as a training lever, and pilot judge-collapsed
decoding — per `docs/superpowers/specs/2026-07-28-calibration-program.md`.

**Architecture:** Four scratch experiment scripts (the lab-notebook
convention: `scratch/*.py` committed, checkpoints untracked), each with
a built-in control arm instead of pytest (house convention — scratch
scripts are experiments; controls are the tests). Every run is
pre-registered in RESULTS.md BEFORE it fires and booked the moment it
lands (`scripts/gen_results_index.py` same commit).

**Tech Stack:** torch (MPS), sympy, existing house APIs:
`llmopt/search/derivation.py:successors/verify_edge`,
`llmopt/search/engine.py:MarkovPrior`, `scripts/bench_verify_fast.py:
verify_wave`, `scratch/rat_deploy.py` (Q-snap), `scratch/gate_ckpt.py`
(the d256/19M gate harness), `scripts/train_mathnative.py`.

## Global Constraints

- All runs on the Mac (`dev = "mps"`); the 3080 is not touched.
- Pre-registration in RESULTS.md before each run fires; verdicts booked
  with `scripts/gen_results_index.py` regenerated in the same commit.
- Sympy on model text only through `verify_wave` / the engine's
  timeboxed `successors` (never bare sympy on generated strings).
- Probe sets use stable STRING seeds; probe-state seed space =
  99_000_000 (fresh; 77M and 88M are used by prior studies).
- d256 model config everywhere: `d=256, layers=8, heads=4, ffn=1024`;
  19M config: `d=384, layers=8, heads=8, ffn=1536`.
- Same-device fence: all cross-crystal probe comparisons read on MPS
  in one script run; never mix devices inside a comparison.
- Commit messages end with `Co-Authored-By: Claude Fable 5
  <noreply@anthropic.com>`; no session URLs (public repo).

---

### Task 1: Rung 1 instrument — `scratch/calib_probe.py`

**Files:**
- Create: `scratch/calib_probe.py`

**Interfaces:**
- Consumes: `scratch/rat_deploy.py`'s snap logic (inlined as a
  function — rat_deploy is argv-driven), `llmopt.train.mathnative`
  `MathTokenizer`/`build_model`, `train_mathnative.load_rows`.
- Produces: CLI `calib_probe.py <ckpt> <d> <layers> <ffn> <heads> [Q]`
  printing `flips_per_token=<float> n_tokens=<int> margin_median=<f>
  margin_at_flips=<f>`; also importable
  `flips_per_token(ckpt, d, layers, ffn, heads, Q=16, dev=None) ->
  dict`. Task 2 consumes the CLI; Task 6 consumes the importable form.

- [ ] **Step 1: Write the script**

```python
"""Calibration probe (spec 2026-07-28 rung 1): flips-per-token under
a Q-lattice snap. Teacher-forced greedy argmax on a fixed 400-row
probe set; count positions where the snapped twin's argmax differs
from the unsnapped model's. Control arm: Q=None (no snap) must read
exactly 0 flips. Also reports the logit-margin distribution at flip
sites (the snap-anatomy read: flips should sit at tiny margins).
Usage: calib_probe.py <ckpt> <d> <layers> <ffn> <heads> [Q=16]
"""
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402

from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

PROBE_SEED = 99_000_001


def rat_snap(sd, Q):
    # same snap rat_deploy.py applies: s * best p/q, q <= Q,
    # s = per-tensor absmean, 2-D float tensors only
    out = {}
    for k, w in sd.items():
        if w.ndim != 2 or not w.is_floating_point():
            out[k] = w
            continue
        wf = w.float()
        s = wf.abs().mean().clamp(min=1e-8)
        v = wf / s
        best = torch.round(v)
        err = (v - best).abs()
        for q in range(2, Q + 1):
            cand = torch.round(v * q) / q
            e = (v - cand).abs()
            m = e < err
            best = torch.where(m, cand, best)
            err = torch.where(m, e, err)
        out[k] = (best * s).to(w.dtype)
    return out


@torch.no_grad()
def flips_per_token(ckpt, d, layers, ffn, heads, Q=16, dev=None):
    dev = dev or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = MathTokenizer()
    rows = load_rows(gen4=True)
    random.Random(PROBE_SEED).shuffle(rows)
    probe = rows[:400]
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)

    def run(state):
        m = build_model(len(tok.vocab), d=d, layers=layers,
                        heads=heads, ffn=ffn).to(dev)
        m.load_state_dict(state)
        m.eval()
        args, tops = [], []
        for r in probe:
            t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
            try:
                ids = torch.tensor([tok.encode(t) + [tok.eos_id]],
                                   device=dev)
            except ValueError:
                continue
            lg = m(ids[:, :-1])[0]
            top2 = lg.topk(2, dim=-1)
            args.append(top2.indices[:, 0].cpu())
            tops.append((top2.values[:, 0] - top2.values[:, 1]).cpu())
        del m
        return torch.cat(args), torch.cat(tops)

    a0, m0 = run(sd)
    a1, _ = run(rat_snap(sd, Q) if Q else sd)
    flips = (a0 != a1)
    n = len(a0)
    return {
        "flips_per_token": flips.sum().item() / n,
        "n_tokens": n,
        "margin_median": m0.median().item(),
        "margin_at_flips": (m0[flips].median().item()
                            if flips.any() else float("nan")),
    }


if __name__ == "__main__":
    ckpt = sys.argv[1]
    d, layers, ffn, heads = map(int, sys.argv[2:6])
    Q = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    r = flips_per_token(ckpt, d, layers, ffn, heads, Q)
    print(f"{ckpt} Q={Q}: flips_per_token={r['flips_per_token']:.5f} "
          f"n_tokens={r['n_tokens']} "
          f"margin_median={r['margin_median']:.3f} "
          f"margin_at_flips={r['margin_at_flips']:.2e}", flush=True)
```

- [ ] **Step 2: Control arm — no-snap must read exactly 0 flips**

Run:
```bash
.venv/bin/python scratch/calib_probe.py \
    checkpoints/mathnative_wfloor_d256.pt 256 8 1024 4 0
```
Expected: `flips_per_token=0.00000` (Q=0 -> falsy -> identical
weights). Any nonzero = nondeterminism bug; fix before proceeding
(check model.eval() and that both runs share one device).

- [ ] **Step 3: Real read sanity**

Run:
```bash
.venv/bin/python scratch/calib_probe.py \
    checkpoints/mathnative_wfloor_d256.pt 256 8 1024 4 16
```
Expected: nonzero flips_per_token; `margin_at_flips` orders of
magnitude below `margin_median` (the snap-anatomy signature). Record
the numbers in the commit message body.

- [ ] **Step 4: Commit**

```bash
git add scratch/calib_probe.py
git commit -m "calib rung 1: flips-per-token probe (control arm 0-flips verified)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Rung 1 validation — snap-robustness ground truth + verdict

**Files:**
- Create: `scratch/calib_snap_gates.sh`
- Modify: `docs/RESULTS.md` (pre-reg entry, then verdict entry)

**Interfaces:**
- Consumes: Task 1 CLI; `scratch/rat_deploy.py`; `scratch/gate_ckpt.py`
  (`gate_ckpt.py <ckpt> <d> <heads> <ffn> <?> <tag>` — copy the exact
  arg order from `scratch/night_rat_s2.sh` for 19M: `384 8 1536 6`;
  check an existing d256 gate invocation in `scratch/` for the d256
  arg order before writing, e.g. `grep -l wfloor_d256 scratch/*.sh`).
- Produces: measured (probe, gate-drop) pairs for >=5 crystals; the
  R1 Spearman verdict in RESULTS.

- [ ] **Step 1: Book the R1 pre-registration in RESULTS.md**

Append (before any run fires):

```markdown
## PRE-REG: calibration probe R1 — flips/token vs snap robustness (2026-07-28, before the runs)

Spec: specs/2026-07-28-calibration-program.md rung 1. Crystals:
d256 zoo {wfloor 65, s2 63, s3 64, stream4 57, muon 34} + Mac-19M
fp32 (rat16 crack 49->26 already measured). Instrument: probe =
flips/token under Q=16 rat snap (calib_probe.py, 400 rows, MPS);
robustness = gate drop under the SAME Q=16 snap (gate_ckpt, same
device). PREDICTION: probe rank-correlates with gate drop
(higher flips/token = larger drop), Spearman rho > 0, and the
Mac-19M (known cracker) reads highest. FAILURE = probe is noise;
the calibration program closes at the cost of one script.
```

Then: `.venv/bin/python scripts/gen_results_index.py`, commit both.

- [ ] **Step 2: Write `scratch/calib_snap_gates.sh`**

```bash
#!/bin/bash
# Rung-1 validation: probe + snapped-gate for each crystal (spec
# 2026-07-28). Serial on purpose (one MPS device, paired reads).
set -e
cd ~/code/llmopt
PY=.venv/bin/python

# d256 zoo: probe, snap, gate the snap (gate args: copy the exact
# d256 invocation found in scratch/ before running)
for name in wfloor_d256 wfloor_d256_s2 wfloor_d256_s3 \
            wfloor_d256_stream4 wfloor_d256_muon; do
  ck=checkpoints/mathnative_${name}.pt
  $PY scratch/calib_probe.py $ck 256 8 1024 4 16 \
      | tee -a logs/calib_r1_probe.log
  $PY scratch/rat_deploy.py $ck 16 checkpoints/calib_${name}_q16.pt
  $PY scratch/gate_ckpt.py checkpoints/calib_${name}_q16.pt \
      256 4 1024 6 calib_${name}_q16 \
      > logs/calib_${name}_q16_gate.log 2>&1
done

# Mac-19M: probe only (its Q=16 gate 26/120 is already booked)
$PY scratch/calib_probe.py checkpoints/mathnative_19m_mac_fp32.pt \
    384 8 1536 8 16 | tee -a logs/calib_r1_probe.log
touch logs/calib_r1_done.marker
```

NOTE before running: verify the d256 `gate_ckpt.py` positional args
against an existing d256 gate call in the repo (`grep -rn
"gate_ckpt.*256" scratch/ docs/`), and confirm each snapped d256
crystal's UNSNAPPED gate is on the books (they are: 65/63/64/57/34).
Fix the arg order in the script if it differs, then run.

- [ ] **Step 3: Run it**

```bash
chmod +x scratch/calib_snap_gates.sh && scratch/calib_snap_gates.sh
```
Expected wall: ~6 gates x ~2-10 min + 6 probe reads. Collect
(probe, gate_drop) pairs; compute Spearman by hand or
`scipy.stats.spearmanr` in a one-liner.

- [ ] **Step 4: Book the R1 verdict in RESULTS.md**

Append the table (crystal | probe flips/token | gate pre | gate post |
drop), the Spearman rho, and the verdict against the pre-reg (PASS =
program continues; FAIL = program CLOSED, book the null). Regenerate
the index. Delete the `checkpoints/calib_*_q16.pt` intermediates.

- [ ] **Step 5: Commit**

```bash
git add scratch/calib_snap_gates.sh docs/RESULTS.md docs/results-index.jsonl
git commit -m "calib rung 1 verdict: probe-vs-robustness across 6 crystals

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Rung 2 — `scratch/mass_on_valid.py` (the mass leg)

**Files:**
- Create: `scratch/mass_on_valid.py`
- Modify: `docs/RESULTS.md` (pre-reg, then verdict)

**Interfaces:**
- Consumes: `llmopt.search.derivation.successors` (returns
  `list[tuple[rule_name, State]]`; `State.expr` is the sympy expr —
  read `derivation.py:50-60` for the exact State fields before
  writing), `bench_verify_fast.verify_wave`, the four d256 specimen
  checkpoints from `scratch/ce_gate_study.py:24-29` (reuse its MODELS
  dict verbatim), `train_mathnative.load_rows` for farm picks.
- Produces: per-model `(mass_valid, mass_pick, entropy_valid)` table;
  rung 3 consumes `mass_valid - mass_pick` as its baseline number.

- [ ] **Step 1: Book the R2 pre-registration in RESULTS.md**

```markdown
## PRE-REG: mass-on-valid — the branching-floor MASS leg (2026-07-28, before the run)

Spec rung 2 (amended: the sampled-coverage form already failed
2026-07-26; no CE-anti-track claim). 40 held-out cur states
(seed space 99M, L3-L7); per state the engine enumerates
successors, verify_wave certifies, and each specimen model's
teacher-forced sequence probability is computed for every valid
nxt and for the farm's banked pick. Readouts per model: (a)
sum mass on valid set, (b) mass on pick, (c) entropy over the
valid set. PREDICTION: (a) tracks the gate at least as well as
(b); the (a)-(b) delta is the novel number and rung 3's
baseline. (a)~(b) everywhere = floor theory unmeasurable at
this scale (books as such).
```

- [ ] **Step 2: Write the script**

```python
"""Mass-on-valid (spec 2026-07-28 rung 2): teacher-forced sequence
probability mass over the engine-enumerated verified-valid next-step
set, vs mass on the farm's single pick. No sampling anywhere.
"""
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from llmopt.search.derivation import State, successors  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402
from bench_verify_fast import verify_wave  # noqa: E402

MODELS = {  # ce_gate_study specimens, verbatim
    "muon":    ("checkpoints/mathnative_wfloor_d256_muon.pt", 34),
    "stream3": ("checkpoints/mathnative_wfloor_d256_stream3.pt", 45),
    "stream4": ("checkpoints/mathnative_wfloor_d256_stream4.pt", 57),
    "control": ("checkpoints/mathnative_wfloor_d256.pt", 65),
}
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
SEED = 99_100_000
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"

# 40 states; per state: enumerate + verify the valid nxt set
cells = []
for lv in (3, 4, 5, 6, 7):
    for i in range(8):
        p = _gen_isolated(lv, SEED + 1000 * lv + i)
        if p is None:
            continue
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        kids = successors(State(expr=p._expr_integral()
                          if hasattr(p, "_expr_integral") else
                          sp.Integral(p._expr, sp.Symbol("x"))))
        cands = list(dict.fromkeys(
            sp.sstr(s.expr) for _, s in kids
            if sp.sstr(s.expr).replace(" ", "") != cur.replace(" ", "")))
        wv = verify_wave(cur, cands) if cands else {}
        valid = [c for c in cands if wv.get(c, (False, False))[0]]
        if len(valid) >= 2:
            cells.append((lv, cur, valid))
print(f"{len(cells)} states with >=2 valid moves", flush=True)


@torch.no_grad()
def seq_logprob(model, cur, nxt):
    pre = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
    full = tok.encode(f"Current: {cur}\nHints: none\nStep: {nxt}\n")
    ids = torch.tensor([full + [tok.eos_id]], device=dev)
    lg = model(ids[:, :-1])[0]
    lp = F.log_softmax(lg, dim=-1)
    tgt = ids[0, 1:]
    span = range(len(pre) - 1, len(tgt))
    return sum(lp[t, tgt[t]].item() for t in span)


import math  # noqa: E402
print(f"{'model':8s} {'gate':>4s} {'mass_valid':>10s} "
      f"{'mass_pick':>10s} {'H_valid':>8s}", flush=True)
for name, (path, gate) in MODELS.items():
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    mv = mp = hh = 0.0
    for lv, cur, valid in cells:
        ps = [math.exp(seq_logprob(model, cur, v)) for v in valid]
        tot = sum(ps)
        mv += tot
        mp += max(ps)  # farm pick proxy: the modal valid move
        if tot > 0:
            q = [p / tot for p in ps if p > 0]
            hh += -sum(x * math.log2(x) for x in q)
    n = len(cells)
    print(f"{name:8s} {gate:4d} {mv/n:10.4f} {mp/n:10.4f} "
          f"{hh/n:8.3f}", flush=True)
    del model
```

NOTE at write time: read `derivation.py:50-60` first and fix the
`State(...)` construction to the real signature (the sketch above
guesses; the State takes the expression the engine searches over —
mirror however `beam_search`/`solve` builds its root state). If the
farm's actual banked pick for these fresh states is unavailable
(fresh states have no farm row), the modal-valid proxy stands —
note it in the booking.

- [ ] **Step 3: Run**

```bash
.venv/bin/python scratch/mass_on_valid.py | tee logs/calib_r2.log
```
Sanity: control's mass_valid should exceed muon's (muon emits
27.8% valid samples — its teacher-forced valid mass should be low).

- [ ] **Step 4: Book the R2 verdict + index, commit**

```bash
git add scratch/mass_on_valid.py docs/RESULTS.md docs/results-index.jsonl
git commit -m "calib rung 2: mass-on-valid verdict (branching-floor mass leg)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Rung 3 farm — `scratch/farm_dist_rows.py`

**Files:**
- Create: `scratch/farm_dist_rows.py`
- Output (untracked): `data/dist_rows_d256.jsonl`

**Interfaces:**
- Consumes: `successors`/`verify_wave` as Task 3;
  `MarkovPrior.load()` from `llmopt.search.engine` (`.bigram`/
  `.unigram` count dicts — read `engine.py:108-131` for the score
  form); the gen4 diet via `load_rows(gen4=True)`.
- Produces: `data/dist_rows_d256.jsonl` with rows
  `{"cur": str, "nxt": str, "w": float, "src": "dist"}` — one row per
  (state, valid move), `w` = MarkovPrior-weighted share, normalized
  per state. Task 5's trainer arm consumes it by WEIGHTED REPLICATION
  (a move with w=0.5 appears ~2x more often than w=0.25 at matched
  total rows) — no trainer surgery, the distribution arrives through
  sampling frequency, which IS a gradient-level soft label at matched
  dose (fence: hints-as-text twice-nulled; this is not text).

- [ ] **Step 1: Write the farmer**

```python
"""Distribution rows (spec 2026-07-28 rung 3): for each diet cur,
enumerate verified-valid moves, weight by MarkovPrior, emit ALL of
them as weighted rows. STREAM rows out incrementally (the killed-
worker doctrine); fork-boxed enumeration via the engine's own
timeboxed successors.
"""
import json
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402

from llmopt.search.derivation import State, successors  # noqa: E402
from llmopt.search.engine import MarkovPrior  # noqa: E402
from train_mathnative import load_rows  # noqa: E402
from bench_verify_fast import verify_wave  # noqa: E402

N_STATES = 4000  # ~matches one d256 diet share; tune to diet size
prior = MarkovPrior.load()
rows = load_rows(gen4=True)
random.Random(99_200_000).shuffle(rows)

out = open("data/dist_rows_d256.jsonl", "a")
done = 0
for r in rows[:N_STATES]:
    cur = r["cur"]
    try:
        expr = sp.sympify(cur)
    except Exception:
        continue
    kids = successors(State(expr=expr))  # fix ctor per Task 3 note
    cands = list(dict.fromkeys(
        sp.sstr(s.expr) for _, s in kids
        if sp.sstr(s.expr).replace(" ", "") != cur.replace(" ", "")))
    wv = verify_wave(cur, cands) if cands else {}
    valid = [(n, c) for (n, c) in
             ((n, sp.sstr(s.expr)) for n, s in kids)
             if wv.get(c, (False, False))[0]]
    if not valid:
        continue
    # MarkovPrior weight by rule name; uniform fallback
    ws = [max(prior.unigram.get(n, 1), 1) for n, _ in valid]
    tot = sum(ws)
    seen = set()
    for (n, c), w in zip(valid, ws):
        if c in seen:
            continue
        seen.add(c)
        out.write(json.dumps(
            {"cur": cur, "nxt": c, "w": w / tot, "src": "dist"}) + "\n")
    out.flush()
    done += 1
    if done % 200 == 0:
        print(f"{done} states", flush=True)
print(f"done: {done} states", flush=True)
```

NOTE at write time: `sp.sympify` on DIET rows is oracle-on-corpus
(farm-emitted, already certified strings), not oracle-on-model-text
— allowed. Confirm `MarkovPrior.unigram` is the real attribute name
(read `engine.py:108-131`); if the prior keys differ, use whatever
`from_rows` builds.

- [ ] **Step 2: Run, check yield**

```bash
.venv/bin/python scratch/farm_dist_rows.py | tee logs/calib_r3_farm.log
wc -l data/dist_rows_d256.jsonl
```
Expected: thousands of rows; mean valid-moves/state >= 2 (else the
distribution collapses to picks and the arm is vacuous — book that
honestly and stop rung 3).

- [ ] **Step 3: Commit (script only; jsonl stays untracked)**

```bash
git add scratch/farm_dist_rows.py
git commit -m "calib rung 3: distribution-row farmer (streamed, prior-weighted)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Rung 3 paired birth + verdict

**Files:**
- Create: `scratch/calib_dist_birth.sh`
- Modify: `docs/RESULTS.md` (pre-reg, then verdict)

**Interfaces:**
- Consumes: `data/dist_rows_d256.jsonl` (Task 4);
  `scripts/train_mathnative.py` — check whether it accepts an extra
  rows file (`grep -n "add_rows\|extra\|jsonl" scripts/
  train_mathnative.py`); if not, the cheapest lawful mechanism is a
  MERGED diet file: replace each state's single farm pick with its
  weighted-replicated distribution rows at MATCHED TOTAL ROW COUNT
  (dist arm) vs the untouched diet (control arm) — one variable, the
  label distribution.
- Produces: `checkpoints/calib_d256_dist.pt` + gate + probe delta;
  the R3 verdict.

- [ ] **Step 1: Book the R3 pre-registration**

```markdown
## PRE-REG: distribution rows at d256 (2026-07-28, before the birth)

Spec rung 3. Paired d256 births, same seed/device/recipe as the
wfloor d256 lineage (control 65 on the books): control diet vs
dist diet (each state's pick replaced by its verified-valid
distribution, prior-weighted replication, matched total rows).
PRIMARY: gate at L4 (canary). SECONDARY: calib_probe delta
(prediction: dist arm reads FEWER flips/token — soft labels
sharpen margins). Distinct-and-verified fence on the label set
(no identity moves). Null reading pre-declared: calibration is
a diagnostic, not a lever; rung 4 unaffected.
```

- [ ] **Step 2: Write `scratch/calib_dist_birth.sh`**

```bash
#!/bin/bash
# Rung-3 paired birth (spec 2026-07-28). Diet prep is done by
# farm_dist_rows + a merge step; copy the EXACT training invocation
# (flags, env) from the run that produced mathnative_wfloor_d256.pt
# (grep docs/RESULTS.md + scratch/*.sh for "wfloor_d256") — same
# recipe, dist diet the only variable.
set -e
cd ~/code/llmopt
# 1) build merged diet (inline python: replace picks w/ dist rows,
#    replicate by weight, match total row count, write
#    data/diet_dist_d256.jsonl)
# 2) birth with the wfloor recipe + the dist diet
# 3) gate_ckpt + calib_probe on the result
```

Fill the three sections at execution time from the recorded wfloor
recipe — the recipe line is IN RESULTS (the d256 pilot-substrate
adoption entry) and/or the shell script that ran it; do not guess
flags.

- [ ] **Step 3: Run both arms if needed**

The control (65) is on the books; rerun a fresh control ONLY if the
recipe cannot be matched exactly (same-day-control doctrine,
2026-07-27: a drifted tree invalidates old pairs — check `git log
--oneline -- scripts/train_mathnative.py` since the wfloor birth;
any training-path change since = rerun the control arm same-day).

- [ ] **Step 4: Book the R3 verdict (gate table, L4 line, probe
  delta), regenerate index, commit; promote to one 19M birth only
  if d256 pays**

```bash
git add scratch/calib_dist_birth.sh docs/RESULTS.md docs/results-index.jsonl
git commit -m "calib rung 3 verdict: distribution rows at d256

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Rung 4 — `scratch/judge_decode.py` (judge-collapsed decoding)

**Files:**
- Create: `scratch/judge_decode.py`
- Modify: `docs/RESULTS.md` (pre-reg, then verdict)

**Interfaces:**
- Consumes: rung 1 PASS (gate condition: ties concentrate where
  capability fails); `calib_probe.flips_per_token` importable (Task
  1); `verify_wave` as the step-boundary judge; a d256 crystal +
  frontier probe states (L5-L7, seed space 99_300_000).
- Produces: three-arm comparison at EQUAL TOKEN BUDGET:
  greedy / judge-collapsed / best-of-N.

- [ ] **Step 1: Book the R4 pre-registration**

```markdown
## PRE-REG: judge-collapsed decoding at d256 (2026-07-28, before the run)

Spec rung 4, gated on R1 PASS. At decode steps with top-2 margin
< 0.02 (the measured near-tie class), branch BOTH continuations
to the step boundary; the oracle (verify_wave) judges; greedy
elsewhere. Arms at equal TOTAL token budget on L5-L7 frontier
states: (a) plain greedy, (b) judge-collapsed, (c) best-of-N at
the same budget. PREDICTION (house, skeptical-honest): (b) > (a)
at equal tokens on the frontier band; (b) vs (c) is the real
race (regret-round-2 economics). (b) <= (a) = null, banked with
the regret lineage; the probe survives as an instrument.
```

- [ ] **Step 2: Write the script**

Core loop (the rest is the standard load/build boilerplate from
`ce_gate_study.py`):

```python
@torch.no_grad()
def decode_judge(model, tok, cur, dev, margin=0.02, max_new=120):
    """Greedy decode; at near-tie steps branch top-2 to the step
    boundary (newline), oracle picks, charge BOTH branches' tokens
    to the budget. Returns (text, tokens_spent)."""
    ids = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
    spent = 0

    def greedy_until_nl(prefix):
        nonlocal spent
        out = list(prefix)
        for _ in range(max_new):
            lg = model(torch.tensor([out], device=dev))[0, -1]
            t = int(lg.argmax())
            spent += 1
            out.append(t)
            if t in (tok.eos_id, tok.encode("\n")[-1]):
                break
        return out

    out = list(ids)
    for _ in range(max_new):
        lg = model(torch.tensor([out], device=dev))[0, -1]
        top2 = lg.topk(2)
        spent += 1
        gap = float(top2.values[0] - top2.values[1])
        if gap < margin:
            a = greedy_until_nl(out + [int(top2.indices[0])])
            b = greedy_until_nl(out + [int(top2.indices[1])])
            ta = tok.decode(a[len(ids):])
            tb = tok.decode(b[len(ids):])
            wv = verify_wave(cur, [ta, tb])
            pick = a if wv.get(ta, (False,))[0] else b
            return tok.decode(pick[len(ids):]), spent
        t = int(top2.indices[0])
        out.append(t)
        if t in (tok.eos_id, tok.encode("\n")[-1]):
            break
    return tok.decode(out[len(ids):]), spent
```

Score all three arms per state; best-of-N's N is set per-state so
its token count matches (b)'s actual spend (the equal-budget fence,
enforced per state not per run). Solved = the chain gate's own
criterion for a valid step toward solution — reuse the gate's
per-step verify (read `gate_ckpt.py` for its solved test and reuse
it verbatim).

- [ ] **Step 3: Run on 30 frontier states, book the R4 verdict,
  regenerate index, commit**

```bash
git add scratch/judge_decode.py docs/RESULTS.md docs/results-index.jsonl
git commit -m "calib rung 4 verdict: judge-collapsed decoding at equal budget

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Consolidation

**Files:**
- Modify: `docs/BOARD.md` (new CALIBRATION PROGRAM row, LIVE),
  `docs/RIFF-LEDGER.md` (mark the judge-collapsed-decoding and
  distribution-rows banks as RUN with their verdicts),
  `docs/FINDINGS.md` (only if a rung produced a curated-grade
  finding), new handoff `docs/handoffs/2026-07-28-*.md`.

- [ ] **Step 1: BOARD row + ledger updates + handoff**
- [ ] **Step 2: If any scripts/ file changed:**
  `.venv/bin/python scripts/gen_index.py`
- [ ] **Step 3: Final commit**

```bash
git add docs/
git commit -m "calibration program: consolidation (BOARD row, ledger verdicts, handoff)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
