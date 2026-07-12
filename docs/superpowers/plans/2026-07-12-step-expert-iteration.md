# Step-Level Expert Iteration Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An autonomous evaluate→mine→train→gate loop that raises a 0.5B
step-model's frontier on oracle-verified chains (spec:
docs/superpowers/specs/2026-07-12-step-expert-iteration-design.md).

**Architecture:** One new module `scripts/expert_loop.py` (the loop
driver: frontier scan, on-policy mining, gate, tripwires, LOOP-LOG)
reusing `scripts/expert_iter_steps.py` (engine chains + training) and
`scripts/bench_step_tokens.py` (sampling, verification, model load).
State lives on disk (corpus jsonl, adapter checkpoints, LOOP-LOG.md);
every phase is fork-isolated and resumable.

**Tech Stack:** torch (CUDA on 3080 / MPS fallback), transformers,
sympy (fork-isolated only), llmopt.train.lora.

## Global Constraints

- Seeds: mining 8_000_000+, evaluation 8_200_000+, held-out gate
  8_400_000+ — disjoint BLOCKS (spec: contamination scar tissue).
- Engine chains capped at 50% of each round's corpus additions.
- Retention: mine at F and F-1 (F-1 at 25% of mining problems).
- Gate: PROMOTE iff no level ≤ F regresses > 2 solves AND (frontier
  solves improve OR validity improves ≥ 2 points).
- Tripwires: 2 consecutive failed gates, mining validity < 1%,
  round wall > 2×90 min → HALT.
- Training: LoRA from BASE weights, cumulative corpus, r=16 alpha=32
  all-proj, loss on step tokens, 3 epochs, lr 2e-4 (train_calculus
  recipe; expert_iter_steps.phase_train already implements it).
- Every sympy touch runs forked (pathologies #7/#8/#10).
- Corpus rows: {"cur", "nxt", "level", "round", "source", "gate"}.

---

### Task 1: Gate logic (pure function + tests)

**Files:**
- Create: `scripts/expert_loop.py`
- Test: `tests/test_expert_loop.py`

**Interfaces:**
- Produces: `gate_verdict(prev: dict, new: dict, frontier: int) ->
  tuple[bool, str]` — scoreboards are `{"solves": {level: int},
  "validity": float}`; returns (promote, reason).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_expert_loop.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from expert_loop import gate_verdict


def _sb(solves, validity):
    return {"solves": solves, "validity": validity}


def test_gate_promotes_on_frontier_gain():
    prev = _sb({2: 30, 3: 20}, 10.0)
    new = _sb({2: 30, 3: 24}, 10.0)
    ok, reason = gate_verdict(prev, new, frontier=3)
    assert ok


def test_gate_promotes_on_validity_gain_alone():
    prev = _sb({2: 30, 3: 20}, 10.0)
    new = _sb({2: 30, 3: 20}, 12.5)
    ok, _ = gate_verdict(prev, new, frontier=3)
    assert ok


def test_gate_rejects_regression_even_with_frontier_gain():
    prev = _sb({2: 30, 3: 20}, 10.0)
    new = _sb({2: 27, 3: 25}, 15.0)   # L2 lost 3 > 2
    ok, reason = gate_verdict(prev, new, frontier=3)
    assert not ok and "regress" in reason


def test_gate_rejects_no_improvement():
    prev = _sb({2: 30, 3: 20}, 10.0)
    new = _sb({2: 30, 3: 20}, 11.0)   # validity +1 < 2 points
    ok, _ = gate_verdict(prev, new, frontier=3)
    assert not ok
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_expert_loop.py -q`
Expected: ImportError / ModuleNotFoundError (expert_loop missing).

- [ ] **Step 3: Implement gate_verdict in scripts/expert_loop.py**

```python
"""Autonomous expert-iteration loop driver (spec:
docs/superpowers/specs/2026-07-12-step-expert-iteration-design.md).
Round = evaluate -> mine -> train -> gate; state on disk; tripwires
halt the loop. All sympy touches forked (pathologies #7/#8/#10)."""
from __future__ import annotations


def gate_verdict(prev: dict, new: dict, frontier: int) -> tuple[bool, str]:
    """PROMOTE iff no level <= frontier regresses by more than 2
    solves AND (frontier solves improve OR validity gains >= 2 pts)."""
    for lv, s in prev["solves"].items():
        if lv <= frontier and new["solves"].get(lv, 0) < s - 2:
            return False, f"L{lv} regressed {s}->{new['solves'].get(lv, 0)}"
    gain_frontier = (new["solves"].get(frontier, 0)
                     > prev["solves"].get(frontier, 0))
    gain_validity = new["validity"] >= prev["validity"] + 2.0
    if gain_frontier or gain_validity:
        return True, "frontier gain" if gain_frontier else "validity gain"
    return False, "no improvement"
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/bin/python3 -m pytest tests/test_expert_loop.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/expert_loop.py tests/test_expert_loop.py
git commit -m "feat: expert-loop gate logic (promote/rollback rules, tested)"
```

### Task 2: Evaluate phase (frontier scan, reusing bench machinery)

**Files:**
- Modify: `scripts/expert_loop.py`
- Modify: `scripts/bench_step_tokens.py` (extract a reusable
  `solve_chain(tok, model, integ, budget, seed0)` from main's arm 2)

**Interfaces:**
- Consumes: `bench_step_tokens.load(adapter)`, `sample()`,
  `verify_step()`, `_gen_isolated()`.
- Produces: `bench_step_tokens.solve_chain(tok, model, integ: str,
  budget: int, seed0: int) -> tuple[bool, list[tuple[str, str]], int, int]`
  — (solved, verified_pairs, valid_steps, tried_steps).
  `expert_loop.evaluate(tok, model, levels, n_per, seed_base, budget)
  -> dict` scoreboard `{"solves": {lv: int}, "validity": float,
  "chains": {lv: [pairs...]}}` and `frontier(scoreboard, n_per) -> int`.

- [ ] **Step 1: Extract solve_chain in bench_step_tokens.py** — move
arm-2's loop body into:

```python
def solve_chain(tok, model, integ: str, budget: int, seed0: int):
    """Oracle-gated chain; returns (solved, verified_pairs,
    valid_steps, tried_steps). verified_pairs only from this trace."""
    cur = f"Integral({integ}, x)"
    pairs, used, j, ok, valid, tried = [], 0, 0, False, 0, 0
    while used < budget and not ok and len(pairs) < 12:
        prompt = FEWSHOT + f"\nCurrent: {cur}\nStep:"
        text, spent = sample(tok, model, prompt, seed=seed0 + 7919 * j)
        used += max(spent, 1)
        j += 1
        tried += 1
        cand = text.splitlines()[0].strip() if text else ""
        if not cand:
            continue
        okp, solved = verify_step(cur, cand)
        if okp:
            valid += 1
            pairs.append((cur, cand))
            cur = cand
            ok = solved
    return ok, (pairs if ok else []), valid, tried
```

and have `main`'s arm 2 call it (identical behavior: same seeds
`seed0=500 + i`, same budget loop semantics — chain-length cap 12 is
NEW and comes from the spec).

- [ ] **Step 2: Implement evaluate + frontier in expert_loop.py**

```python
def evaluate(tok, model, levels, n_per, seed_base, budget=768):
    import sympy as sp
    from bench_step_tokens import _gen_isolated, solve_chain
    sb = {"solves": {}, "validity": 0.0, "chains": {}}
    valid = tried = 0
    for lv in levels:
        s = 0
        sb["chains"][lv] = []
        for i in range(n_per):
            p = _gen_isolated(lv, seed_base + 1000 * lv + i)
            if p is None:
                continue
            ok, pairs, v, t = solve_chain(
                tok, model, sp.sstr(p._expr), budget,
                seed0=seed_base + 1000 * lv + i)
            s += ok
            valid += v
            tried += t
            if ok:
                sb["chains"][lv].extend(pairs)
        sb["solves"][lv] = s
        if s < 0.2 * n_per:      # frontier scan stops below 20%
            break
    sb["validity"] = 100.0 * valid / max(tried, 1)
    return sb


def frontier(sb: dict, n_per: int) -> int:
    """Highest level in the 20-80% band; else highest evaluated."""
    band = [lv for lv, s in sb["solves"].items()
            if 0.2 * n_per <= s <= 0.8 * n_per]
    return max(band) if band else max(sb["solves"])
```

- [ ] **Step 3: Regression-test the bench refactor**

Run: `.venv/bin/python3 scripts/bench_step_tokens.py --n 2 --budget 256`
Expected: completes; output format unchanged (solves may differ from
the original run only via the new chain cap).

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_step_tokens.py scripts/expert_loop.py
git commit -m "feat: expert-loop evaluate phase + solve_chain extraction"
```

### Task 3: Mine + train + round driver + tripwires + LOOP-LOG

**Files:**
- Modify: `scripts/expert_loop.py`

**Interfaces:**
- Consumes: `expert_iter_steps._chain_worker` (engine chains),
  `expert_iter_steps.phase_train` (cumulative retrain; parameterize
  its adapter output path: add `out: Path = ADAPTER` argument).
- Produces: `run_round(round_no) -> str` verdict; `main()` loop with
  halts; docs/LOOP-LOG.md one line per round.

- [ ] **Step 1: Implement mine_round**

```python
def mine_round(round_no, F, sb, seed_base, n_mine=60):
    """On-policy chains from evaluation + extra mining at F (and F-1
    at 25%), engine chains capped at 50% of additions."""
    import json
    import multiprocessing as mp
    from pathlib import Path
    from expert_iter_steps import _chain_worker
    corpus = Path("data/step_chains.jsonl")
    seen = set()
    for line in corpus.read_text().splitlines():
        r = json.loads(line)
        seen.add((r["cur"], r["nxt"]))
    model_pairs = []
    for lv in (F, F - 1):
        for pair in sb["chains"].get(lv, []):
            if pair not in seen:
                seen.add(pair)
                model_pairs.append((lv, pair))
    # engine chains, capped at len(model_pairs) (i.e. <= 50% of adds)
    ctx = mp.get_context("fork")
    engine_pairs = []
    lv_plan = [F] * int(n_mine * 0.75) + [F - 1] * int(n_mine * 0.25)
    for k, lv in enumerate(lv_plan):
        if len(engine_pairs) >= max(len(model_pairs), 20):
            break
        q = ctx.Queue()
        pr = ctx.Process(target=_chain_worker,
                         args=(lv, seed_base + 500 * round_no + k, q))
        pr.start()
        pr.join(90)
        if pr.is_alive():
            pr.kill()
            pr.join()
            continue
        try:
            for pair in q.get(timeout=10):
                if tuple(pair) not in seen:
                    seen.add(tuple(pair))
                    engine_pairs.append((lv, tuple(pair)))
        except Exception:
            continue
    with corpus.open("a") as f:
        for src, items in (("model", model_pairs), ("engine", engine_pairs)):
            for lv, (cur, nxt) in items:
                f.write(json.dumps({"cur": cur, "nxt": nxt, "level": lv,
                                    "round": round_no, "source": src,
                                    "gate": "pending"}) + "\n")
    return len(model_pairs), len(engine_pairs)
```

- [ ] **Step 2: Parameterize phase_train output** — in
`scripts/expert_iter_steps.py` change the signature to
`def phase_train(epochs: int, lr: float, out: Path = ADAPTER)` and the
save line to `torch.save({...}, out)`.

- [ ] **Step 3: Implement run_round + main loop with tripwires**

```python
def run_round(round_no: int) -> str:
    import shutil
    import time
    from pathlib import Path
    from bench_step_tokens import load
    from expert_iter_steps import phase_train
    t0 = time.monotonic()
    promoted = Path("checkpoints/step_lora.pt")
    tok, model = load(str(promoted) if promoted.exists() else None)
    sb = evaluate(tok, model, levels=(2, 3, 4, 5), n_per=40,
                  seed_base=8_200_000 + 10_000 * round_no)
    F = frontier(sb, 40)
    n_model, n_engine = mine_round(round_no, F, sb,
                                   seed_base=8_000_000)
    if sb["validity"] < 1.0:
        return f"HALT validity {sb['validity']:.1f}%"
    cand = Path(f"checkpoints/step_lora_r{round_no}.pt")
    phase_train(epochs=3, lr=2e-4, out=cand)
    del model
    tok, model = load(str(cand))
    gate_sb = evaluate(tok, model, levels=tuple(range(2, F + 1)),
                       n_per=40, seed_base=8_400_000)
    ok, reason = gate_verdict(
        {"solves": sb["solves"], "validity": sb["validity"]},
        {"solves": gate_sb["solves"], "validity": gate_sb["validity"]}, F)
    if ok:
        shutil.copy(cand, promoted)
    mins = (time.monotonic() - t0) / 60
    line = (f"| {round_no} | F=L{F} | +{n_model}m/+{n_engine}e | "
            f"val {sb['validity']:.1f}->{gate_sb['validity']:.1f} | "
            f"{sb['solves']} -> {gate_sb['solves']} | "
            f"{'PROMOTE' if ok else 'ROLLBACK'}: {reason} | "
            f"{mins:.0f}m |")
    with open("docs/LOOP-LOG.md", "a") as f:
        f.write(line + "\n")
    if mins > 180:
        return f"HALT wall {mins:.0f}m"
    return "PROMOTE" if ok else "ROLLBACK"


def main(max_rounds: int) -> None:
    fails = 0
    for r in range(1, max_rounds + 1):
        verdict = run_round(r)
        print(f"round {r}: {verdict}", flush=True)
        if verdict.startswith("HALT"):
            break
        fails = 0 if verdict == "PROMOTE" else fails + 1
        if fails >= 2:
            print("HALT: two consecutive failed gates", flush=True)
            break


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    main(a.rounds)
```

- [ ] **Step 4: Create docs/LOOP-LOG.md header**

```markdown
# Expert-iteration loop log

| round | frontier | mined (model/engine) | validity | solves (pre -> gate) | verdict | wall |
|---|---|---|---|---|---|---|
```

- [ ] **Step 5: Run unit tests**

Run: `.venv/bin/python3 -m pytest tests/test_expert_loop.py -q`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/expert_loop.py scripts/expert_iter_steps.py docs/LOOP-LOG.md
git commit -m "feat: expert-iteration loop driver — mine/train/gate rounds, tripwires, LOOP-LOG"
```

### Task 4: Manual round 1 (spec requirement before arming autonomy)

**Files:** none new; runs the pipeline.

- [ ] **Step 1: Ensure round-1 corpus exists** (chains farm output,
`data/step_chains.jsonl` — farming as this plan is written; rows lack
the round/source/gate tags, which is fine: mine_round only APPENDS
tagged rows, and phase_train reads cur/nxt only).

- [ ] **Step 2: Train round-1 adapter (3080 preferred)**

Run: `.venv/bin/python3 scripts/expert_iter_steps.py --phase train`
Expected: per-epoch loss decreasing; saves checkpoints/step_lora.pt.

- [ ] **Step 3: Race adapter vs base on fresh seeds**

Run: `.venv/bin/python3 scripts/bench_step_tokens.py --n 30 --seed-base 8_600_000 --adapter checkpoints/step_lora.pt`
and the same command without --adapter.
Expected: adapter validity > 5% baseline; steps-solves >= base.
Record both in RESULTS.md.

- [ ] **Step 4: One supervised loop round**

Run: `.venv/bin/python3 scripts/expert_loop.py --rounds 1`
Expected: LOOP-LOG gains one line; verdict PROMOTE or ROLLBACK with
sane numbers. Review before arming --rounds 4.

- [ ] **Step 5: Commit results**

```bash
git add docs/LOOP-LOG.md docs/RESULTS.md checkpoints/step_lora.pt
git commit -m "feat: expert iteration round 1 — adapter verdict + first supervised loop round"
```
