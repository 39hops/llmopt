# Adaptive-K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entropy-gated per-state branching — prune deep when the ranker is confident, widen when it's uncertain — raced against fixed k, full enumeration, and k1×3 restarts.

**Architecture:** `propose_k` generalizes to accept a callable `(state, ranked_children, scores) -> int`; the proposer callable's contract grows an optional score channel; `entropy_k` policy lives in `llmopt/search/proposer.py`; the race extends `bench_proposer.py`-style protocol in a new script.

**Tech Stack:** sympy, torch (proposer inference), pytest.

**Spec:** `docs/superpowers/specs/2026-07-07-adaptive-k-design.md`

## Global Constraints

- Swap-not-blend discipline unchanged; adaptive-k only changes HOW MANY children survive, never re-scores states.
- Defaults k_min=1, k_max=6; normalized entropy H = entropy(softmax(scores/T)) / log(n); T calibrated on train-split states only, never race problems.
- Race on `proposer-race-*` seeds, budgets 25/50/100/200, both kinds; configs: full+hce, k3+prop, k1x3-restarts, adaptive+prop. Prediction pre-registered: adaptive ≈ k1x3 on diff L2-3 AND ≈ full on int L3.
- Honest null: if H doesn't localize (report the H histogram per kind/level), say the confidence signal is the gap, not the mechanism.

---

### Task 1: score-carrying proposer + entropy_k policy

**Files:**
- Modify: `llmopt/search/proposer.py`
- Test: `tests/test_proposer.py`

**Interfaces:**
- Produces: `make_scoring_proposer(score_fn)` returning `proposer(state, children) -> (ranked_children, scores_desc)` — a NEW callable form that also returns the sorted scores; `entropy_k(k_min=1, k_max=6, temperature=1.0)` returning `(state, ranked, scores) -> int`.
- Existing `make_proposer` unchanged (back-compat for bench_proposer.py).

- [ ] **Step 1: Failing tests** — append to `tests/test_proposer.py`:

```python
import math

from llmopt.search.proposer import entropy_k, make_scoring_proposer


def test_scoring_proposer_returns_scores():
    def score_fn(state_str, labels):
        return [float(len(l)) for l in labels]

    prop = make_scoring_proposer(score_fn)
    s = State(sp.Derivative(x**2, x))
    kids = [("aa", State(x)), ("dddd", State(2 * x)), ("c", State(x + 1))]
    ranked, scores = prop(s, kids)
    assert [n for n, _ in ranked] == ["dddd", "aa", "c"]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) == 3


def test_entropy_k_peaked_vs_flat():
    policy = entropy_k(k_min=1, k_max=6)
    peaked = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    flat = [1.0] * 6
    assert policy(None, [None] * 6, peaked) == 1
    assert policy(None, [None] * 6, flat) == 6
    mid = policy(None, [None] * 6, [3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    assert 1 <= mid <= 6


def test_entropy_k_single_child():
    policy = entropy_k()
    assert policy(None, [None], [5.0]) == 1
```

- [ ] **Step 2: Run, expect ImportError.**

- [ ] **Step 3: Implement** — append to `llmopt/search/proposer.py`:

```python
def make_scoring_proposer(score_fn: ScoreFn):
    """Like make_proposer, but returns (ranked_children, scores_desc)
    so an adaptive-k policy can read the ranker's confidence."""

    def proposer(state: State, children: list[tuple[str, State]]):
        if not children:
            return children, []
        labels = [name for name, _ in children]
        scores = score_fn(sp.sstr(state.expr), labels)
        order = sorted(range(len(children)), key=lambda i: -scores[i])
        return [children[i] for i in order], [scores[i] for i in order]

    return proposer


def entropy_k(k_min: int = 1, k_max: int = 6, temperature: float = 1.0):
    """Confidence-gated branching: peaked ranking -> deep (k_min);
    flat ranking -> wide (k_max). H is normalized entropy of the
    softmax over child scores (spec: 2026-07-07-adaptive-k-design.md)."""
    import math

    def policy(state, ranked, scores) -> int:
        n = len(scores)
        if n <= 1:
            return max(1, k_min)
        m = max(s / temperature for s in scores)
        exps = [math.exp(s / temperature - m) for s in scores]
        z = sum(exps)
        ps = [e / z for e in exps]
        h = -sum(p * math.log(p) for p in ps if p > 0) / math.log(n)
        return k_min + round(h * (k_max - k_min))

    return policy
```

- [ ] **Step 4: Tests pass. Step 5: Commit** `feat: search/proposer — scoring proposer + entropy_k policy`.

---

### Task 2: propose_k callable in beam_search

**Files:**
- Modify: `llmopt/search/derivation.py`
- Test: `tests/test_derivation_search.py`

**Interfaces:**
- `propose_k: int | Callable | None`; when callable AND the proposer returns `(ranked, scores)`, k = `propose_k(state, ranked, scores)`. Plain proposers (list return) keep int-only propose_k.

- [ ] **Step 1: Failing test** — append to `tests/test_derivation_search.py`:

```python
def test_adaptive_propose_k_callable():
    def scoring_proposer(state, children):
        n = len(children)
        return children, [float(n - i) for i in range(n)]

    ks_seen = []

    def policy(state, ranked, scores):
        ks_seen.append(len(scores))
        return 2

    r = beam_search(sp.Derivative(x**2 * sp.sin(x), x),
                    proposer=scoring_proposer, propose_k=policy)
    assert r.solved
    assert ks_seen, "policy never consulted"
```

- [ ] **Step 2: fails (tuple unpack / TypeError). Step 3: Implement** — in `beam_search`'s expansion block, replace the proposer/truncation lines with:

```python
            kids = list(successors(s, use_macros=use_macros))
            scores = None
            if proposer is not None:
                out = proposer(s, kids)
                if isinstance(out, tuple):
                    kids, scores = out
                else:
                    kids = out
            if propose_k is not None:
                k = (propose_k(s, kids, scores) if callable(propose_k)
                     else propose_k)
                kids = kids[:max(1, int(k))]
```

Type of `propose_k` in the signature becomes `int | Callable[..., int] | None`.

- [ ] **Step 4: full search suite passes. Step 5: Commit** `feat: search — propose_k accepts a policy callable (adaptive branching)`.

---

### Task 3: the adaptive race

**Files:**
- Create: `scripts/bench_adaptive.py` (loader code shared shape with bench_proposer.py)

Configs: `full+hce`, `k3+prop`, `k1x3` (restart_search from bench_ksweep pattern), `adaptive+prop` (scoring proposer + entropy_k(1, 6)). Same seeds/budgets/wall pattern as bench_proposer.py; also print per-cell mean-k and an H histogram summary (deciles) for the adaptive config — the null-check instrument. n=15.

- [ ] **Step 1: write script (assemble from bench_proposer.py parts + Task 1/2 interfaces; loader identical, race loop identical shape).**
- [ ] **Step 2: launch on the freed Mac in background; record table.**
- [ ] **Step 3: Commit with numbers + README paragraph; verdict against the pre-registered prediction.**

## Self-review

Spec coverage: mechanism (T1), engine hook (T2), race + H-histogram null instrument (T3). Temperature calibration deferred to T=1.0 default with the H histogram as the diagnostic — acceptable first pass, noted in spec terms.
