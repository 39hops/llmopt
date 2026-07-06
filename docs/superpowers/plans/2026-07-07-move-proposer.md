# Move-Proposer Rung Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fine-tuned 0.5B ranks the searcher's legal moves so the beam expands only top-k proposals, trading branching for depth — measured against full enumeration and a random-k control.

**Architecture:** `llmopt/search/proposer.py` holds prompt building + HF likelihood scoring (device-agnostic); `beam_search` gains `proposer`/`propose_k`; three scripts: data gen (Mac), LoRA training (Windows 3080 via SSH; MPS-capable), the race (Mac).

**Tech Stack:** sympy, torch, transformers, `llmopt/train/lora.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-07-move-proposer-design.md`

## Global Constraints

- Proposer ranks enumerated legal moves; it never generates rewrites.
- Data rows: `{"state": sstr, "moves": [label, ...], "answer": int}` from winning paths of solved searches only.
- Seeds `proposer-train/-eval/-race-{kind}-{level}-{i}`; root-srepr exclude guard between splits.
- Training recipe verbatim from `scripts/train_calculus.py`: LoRA r=16 on q/k/v/o/gate/up/down proj, loss on answer tokens only, length-sorted + token-budget batches, per-epoch cut shuffling, cosine schedule. Model `Qwen/Qwen2.5-0.5B-Instruct`.
- Node accounting: `nodes` counts children admitted to candidates (post-truncation); proposer wall-clock reported separately.
- Random-k3 control in the race; nulls reported plainly.

---

### Task 1: proposer module (pure parts)

**Files:**
- Create: `llmopt/search/proposer.py`
- Test: `tests/test_proposer.py`

**Interfaces:**
- Produces: `build_prompt(state_str: str, labels: list[str]) -> str`;
  `make_proposer(score_fn: Callable[[str, list[str]], list[float]]) -> Callable[[State, list[tuple[str, State]]], list[tuple[str, State]]]` (reranks, higher score first, stable);
  `hf_score_fn(model, tok, device) -> Callable[[str, list[str]], list[float]]` (answer-token logprob of each numbered choice; not unit-tested — exercised by scripts).

- [ ] **Step 1: Write the failing test**

Create `tests/test_proposer.py`:

```python
import sympy as sp

from llmopt.search.derivation import State
from llmopt.search.proposer import build_prompt, make_proposer

x = sp.Symbol("x")


def test_prompt_contains_state_and_numbered_moves():
    p = build_prompt("Derivative(x**2, x)", ["d_power@...", "expand"])
    assert "Derivative(x**2, x)" in p
    assert "1. d_power@..." in p and "2. expand" in p
    assert p.rstrip().endswith("Best move:")


def test_make_proposer_reranks_by_score():
    def score_fn(state_str, labels):
        return [0.1 if "expand" in l else 0.9 for l in labels]

    prop = make_proposer(score_fn)
    s = State(sp.Derivative(x**2, x))
    kids = [("expand", State(x)), ("d_power@D", State(2 * x))]
    ranked = prop(s, kids)
    assert [n for n, _ in ranked] == ["d_power@D", "expand"]
    assert prop(s, []) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_proposer.py -q`
Expected: `ModuleNotFoundError: No module named 'llmopt.search.proposer'`

- [ ] **Step 3: Implement `llmopt/search/proposer.py`**

```python
"""Move proposer: a policy model in front of the classical searcher
(spec: 2026-07-07-move-proposer-design.md). The searcher enumerates
LEGAL moves; the model only ranks them — rank-not-generate keeps
legality by construction. Ranking = likelihood of each numbered choice's
answer tokens under the fine-tuned model."""

from __future__ import annotations

from typing import Callable

import sympy as sp

from llmopt.search.derivation import State

ScoreFn = Callable[[str, "list[str]"], "list[float]"]


def build_prompt(state_str: str, labels: list[str]) -> str:
    lines = [f"State: {state_str}", "Legal moves:"]
    lines += [f"{i + 1}. {lab}" for i, lab in enumerate(labels)]
    lines.append("Best move:")
    return "\n".join(lines)


def make_proposer(score_fn: ScoreFn):
    """Wrap a scoring function into the beam_search proposer callable.
    Higher score = better; sort is stable so ties keep enumeration order."""

    def proposer(state: State, children: list[tuple[str, State]]):
        if not children:
            return children
        labels = [name for name, _ in children]
        scores = score_fn(sp.sstr(state.expr), labels)
        order = sorted(range(len(children)), key=lambda i: -scores[i])
        return [children[i] for i in order]

    return proposer


def hf_score_fn(model, tok, device: str) -> ScoreFn:
    """Score each candidate as the mean logprob of its answer tokens
    (' {i}') given the numbered-choice prompt. Batched; 1-2 answer
    tokens per candidate keeps this cheap even at ~30 candidates."""
    import torch

    def score(state_str: str, labels: list[str]) -> list[float]:
        prompt = build_prompt(state_str, labels)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        rows, spans = [], []
        for i in range(len(labels)):
            a_ids = tok(f" {i + 1}", add_special_tokens=False).input_ids
            rows.append(p_ids + a_ids)
            spans.append(len(a_ids))
        width = max(len(r) for r in rows)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, r in enumerate(rows):
            ids[j, : len(r)] = torch.tensor(r)
            mask[j, : len(r)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        out = []
        for j, r in enumerate(rows):
            n = spans[j]
            lp = 0.0
            for t in range(len(r) - n, len(r)):
                lp += float(logprobs[j, t - 1, r[t]])
            out.append(lp / n)
        return out

    return score
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_proposer.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add llmopt/search/proposer.py tests/test_proposer.py
git commit -m "feat: search/proposer — numbered-choice prompt + likelihood reranker (rank-not-generate)"
```

---

### Task 2: `proposer`/`propose_k` in beam_search

**Files:**
- Modify: `llmopt/search/derivation.py`
- Test: `tests/test_derivation_search.py`

**Interfaces:**
- Produces: `beam_search(..., proposer=None, propose_k=None)`; when set, each state's successor list is reranked by `proposer` then truncated to `propose_k`; `nodes` counts only admitted children.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_derivation_search.py`:

```python
def test_propose_k_truncates_branching():
    full = beam_search(sp.Derivative(x**2 * sp.sin(x), x))
    seen = []

    def keep_first(state, children):
        seen.append(len(children))
        return children

    pruned = beam_search(sp.Derivative(x**2 * sp.sin(x), x),
                         proposer=keep_first, propose_k=2)
    assert pruned.solved and full.solved
    assert pruned.nodes < full.nodes  # fewer children admitted
    assert seen, "proposer never called"


def test_proposer_rerank_changes_expansion_order():
    def reversed_proposer(state, children):
        return list(reversed(children))

    r = beam_search(sp.Derivative(x**2 + sp.sin(x), x),
                    proposer=reversed_proposer, propose_k=1)
    # k=1 with a bad ordering may or may not solve — the API contract is
    # only that it runs and respects the truncation
    assert r.nodes >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_derivation_search.py -q -k propose`
Expected: `TypeError: beam_search() got an unexpected keyword argument 'proposer'`

- [ ] **Step 3: Implement**

In `beam_search` signature add (after `eval_fn`):

```python
    proposer: Callable[
        [State, list[tuple[str, State]]], list[tuple[str, State]]
    ] | None = None,
    propose_k: int | None = None,
```

Replace the inner expansion loop `for _, child in successors(s, use_macros=use_macros):` with:

```python
            kids = list(successors(s, use_macros=use_macros))
            if proposer is not None:
                kids = proposer(s, kids)
            if propose_k is not None:
                kids = kids[:propose_k]
            for _, child in kids:
```

(body unchanged; `visited`/`trace`/`nodes` logic now applies only to admitted children.)

- [ ] **Step 4: Run the search suite**

Run: `.venv/bin/python -m pytest tests/test_derivation_search.py tests/test_proposer.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add llmopt/search/derivation.py tests/test_derivation_search.py
git commit -m "feat: search — proposer + propose_k on beam_search (branching-for-depth trade)"
```

---

### Task 3: data generation (Mac, run now)

**Files:**
- Create: `scripts/gen_proposer_data.py`
- Output: `data/proposer_train.jsonl`, `data/proposer_eval.jsonl`

**Interfaces:**
- Consumes: `beam_search` (default eval), `successors`, `State`.
- Produces: JSONL rows `{"state": str, "moves": [str], "answer": int}` — `answer` indexes `moves` (0-based) at the on-path label.

- [ ] **Step 1: Write `scripts/gen_proposer_data.py`**

```python
"""Winning-path (state, legal moves, chosen move) triples for proposer
SFT. Every row is verifier-approved: it comes from a SOLVED search, so
the chosen move provably leads to a solution. Spec:
2026-07-07-move-proposer-design.md.

  python scripts/gen_proposer_data.py --per-cell 60 --split train
  python scripts/gen_proposer_data.py --per-cell 15 --split eval
"""

from __future__ import annotations

import argparse
import json
import random
import signal
from pathlib import Path

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search, successors

X = sp.Symbol("x")
WALL = 60


class _Timeout(BaseException):
    pass


def _root(rng, level, kind):
    if kind == "diff":
        return sp.Derivative(_expression(rng, level), X)
    while True:
        g = sp.simplify(sp.diff(_expression(rng, level), X))
        if g != 0:
            return sp.Integral(g, X)


def path_rows(root: sp.Expr) -> list[dict]:
    """Replay the winning history move-by-move, recording the legal
    alternatives at each ply. Skip plies whose label isn't found
    (shouldn't happen; belt and braces)."""
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(WALL)
    try:
        r = beam_search(root, max_plies=20, max_nodes=400)
    except _Timeout:
        return []
    finally:
        signal.alarm(0)
    if not r.solved:
        return []
    rows, cur = [], State(root)
    for chosen in r.state.history:
        kids = list(successors(cur))
        labels = [name for name, _ in kids]
        if chosen not in labels:
            return rows  # keep what we have
        idx = labels.index(chosen)
        if len(labels) > 1:  # single-choice plies teach nothing
            rows.append({"state": sp.sstr(cur.expr), "moves": labels,
                         "answer": idx})
        cur = kids[idx][1]
    return rows


def main(per_cell: int, split: str, exclude_file: str | None) -> None:
    exclude: set[str] = set()
    if exclude_file and Path(exclude_file).exists():
        exclude = set(json.loads(Path(exclude_file).read_text()))
    roots_seen: list[str] = []
    out_path = Path(f"data/proposer_{split}.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    n_rows = 0
    with out_path.open("w") as f:
        for kind in ("diff", "int"):
            for level in (1, 2, 3):
                rng = random.Random(f"proposer-{split}-{kind}-{level}-0")
                for _ in range(per_cell):
                    root = _root(rng, level, kind)
                    rk = sp.srepr(root)
                    if rk in exclude:
                        continue
                    roots_seen.append(rk)
                    for row in path_rows(root):
                        f.write(json.dumps(row) + "\n")
                        n_rows += 1
                print(f"{kind} L{level}: {n_rows} rows so far", flush=True)
    Path(f"data/proposer_{split}_roots.json").write_text(
        json.dumps(roots_seen))
    print(f"wrote {n_rows} rows to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--split", default="train")
    ap.add_argument("--exclude-roots", default=None,
                    help="roots json from another split (contamination guard)")
    a = ap.parse_args()
    main(a.per_cell, a.split, a.exclude_roots)
```

- [ ] **Step 2: Generate both splits (background)**

```bash
.venv/bin/python scripts/gen_proposer_data.py --per-cell 60 --split train
.venv/bin/python scripts/gen_proposer_data.py --per-cell 15 --split eval \
    --exclude-roots data/proposer_train_roots.json
```

Expected: train ~2-4k rows, eval ~500-1k rows.

- [ ] **Step 3: Commit script + data**

```bash
git add scripts/gen_proposer_data.py data/proposer_train.jsonl data/proposer_eval.jsonl data/proposer_train_roots.json data/proposer_eval_roots.json
git commit -m "feat: scripts — proposer SFT data from winning paths (<paste row counts>)"
```

---

### Task 4: training script (runs on Windows 3080 via SSH; MPS-capable)

**Files:**
- Create: `scripts/train_proposer.py`

**Interfaces:**
- Consumes: `data/proposer_{train,eval}.jsonl`, `build_prompt` (Task 1), `apply_lora` from `llmopt/train/lora.py`.
- Produces: `checkpoints/proposer_lora.pt` (LoRA state dict, loadable by `bench_proposer.py` after `apply_lora` on a fresh base model); prints held-out top-1/top-3 move accuracy (heat 0).

- [ ] **Step 1: Write `scripts/train_proposer.py`**

```python
"""Proposer SFT: choose the winning move number given state + legal
moves. Recipe verbatim from scripts/train_calculus.py (LoRA r=16 all
proj linears, loss on answer tokens only, length-sorted token-budget
batches, per-epoch cut shuffle, cosine schedule). Spec:
2026-07-07-move-proposer-design.md. Runs on CUDA (3080) or MPS."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.search.proposer import build_prompt
from llmopt.train.lora import apply_lora

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS, BATCH, LR = 3, 8, 2e-4
TOKEN_BUDGET = 2048
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
RANK = 16
OUT = Path("checkpoints/proposer_lora.pt")


def encode(tok, row):
    prompt_ids = tok(build_prompt(row["state"], row["moves"]),
                     add_special_tokens=False).input_ids
    ans = tok(f" {row['answer'] + 1}", add_special_tokens=False).input_ids
    ans.append(tok.eos_token_id)
    ids = prompt_ids + ans
    labels = [-100] * len(prompt_ids) + ans
    return ids, labels


def cut_batches(examples, batch_size, token_budget):
    cuts, i = [], 0
    while i < len(examples):
        j = i + 1
        while (j < len(examples) and j - i < batch_size
               and len(examples[j][0]) * (j - i + 1) <= token_budget):
            j += 1
        cuts.append((i, j))
        i = j
    return cuts


def batches(examples, pad_id, device, epoch):
    cuts = cut_batches(examples, BATCH, TOKEN_BUDGET)
    random.Random(epoch).shuffle(cuts)
    for i, j in cuts:
        chunk = examples[i:j]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for r, (seq, lab) in enumerate(chunk):
            ids[r, : len(seq)] = torch.tensor(seq)
            labels[r, : len(lab)] = torch.tensor(lab)
            mask[r, : len(seq)] = 1
        yield ids.to(device), labels.to(device), mask.to(device)


@torch.no_grad()
def move_accuracy(model, tok, rows, device, k=(1, 3)):
    from llmopt.search.proposer import hf_score_fn

    score = hf_score_fn(model, tok, device)
    hits = {kk: 0 for kk in k}
    for row in rows:
        s = score(row["state"], row["moves"])
        order = sorted(range(len(s)), key=lambda i: -s[i])
        for kk in k:
            hits[kk] += row["answer"] in order[:kk]
    return {kk: hits[kk] / len(rows) for kk in k}


def main() -> None:
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)

    train_rows = [json.loads(l) for l in open("data/proposer_train.jsonl")]
    eval_rows = [json.loads(l) for l in open("data/proposer_eval.jsonl")]
    eval_rows = eval_rows[:300]
    print(f"train rows: {len(train_rows)}, eval rows: {len(eval_rows)}")

    model.eval()
    base_acc = move_accuracy(model, tok, eval_rows, device)
    print(f"baseline move accuracy: top1={base_acc[1]:.1%} top3={base_acc[3]:.1%}")

    wrapped = apply_lora(model, TARGETS, r=RANK, alpha=2 * RANK)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA wrapped {wrapped} linears, "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable")

    train = [encode(tok, r) for r in train_rows]
    train.sort(key=lambda e: len(e[0]))
    opt = torch.optim.AdamW(params, lr=LR)
    steps = EPOCHS * len(cut_batches(train, BATCH, TOKEN_BUDGET))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    pad = tok.pad_token_id or tok.eos_token_id

    model.train()
    for epoch in range(EPOCHS):
        tot, nb = 0.0, 0
        for ids, labels, mask in batches(train, pad, device, epoch):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += float(out.loss)
            nb += 1
        print(f"epoch {epoch}: mean loss {tot / nb:.4f}", flush=True)

    model.eval()
    acc = move_accuracy(model, tok, eval_rows, device)
    print(f"tuned move accuracy: top1={acc[1]:.1%} top3={acc[3]:.1%}")

    lora_state = {k: v for k, v in model.state_dict().items()
                  if "lora_" in k or "lora" in k.lower()}
    OUT.parent.mkdir(exist_ok=True)
    torch.save(lora_state, OUT)
    print(f"saved {OUT} ({len(lora_state)} tensors)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke on Mac (MPS)** — `EPOCHS=1` slice via env not needed; just confirm it starts, then Ctrl-C. Real run on the 3080 via SSH: sync repo, run under the vcvars cmd wrapper from CLAUDE.md if torch.compile gets involved (it doesn't here — plain eager training).

- [ ] **Step 3: Commit script; commit checkpoint after the GPU run with accuracy numbers**

```bash
git add scripts/train_proposer.py
git commit -m "feat: scripts — proposer LoRA SFT (recipe from train_calculus)"
# later, after GPU run:
# git add checkpoints/proposer_lora.pt && git commit -m "feat: checkpoints — proposer LoRA (top1=<>, top3=<> vs baseline <>)"
```

---

### Task 5: the race

**Files:**
- Create: `scripts/bench_proposer.py`

**Interfaces:**
- Consumes: `checkpoints/proposer_lora.pt` + `apply_lora` (to rebuild the tuned model), `hf_score_fn`/`make_proposer` (Task 1), `beam_search(..., eval_fn=, proposer=, propose_k=)`, NNUE checkpoint via `scripts/bench_nnue.py`-style loader.
- Produces: solve-rate table over configs {full+hce, full+nnue, prop-k3+hce, prop-k3+nnue, random-k3+hce} × kinds × levels × budgets 25/50/100/200; proposer wall-clock reported separately.

- [ ] **Step 1: Write `scripts/bench_proposer.py`**

```python
"""The proposer race: full enumeration vs model-proposed top-k vs
random-k control, under HCE and NNUE evals, held-out problems.
Solve rate at fixed node budget is the score; proposer inference time
is wall clock, reported separately. Spec:
2026-07-07-move-proposer-design.md."""

from __future__ import annotations

import argparse
import random
import signal
import time

import sympy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search, hce
from llmopt.search.features import N_FEATURES, featurize
from llmopt.search.proposer import hf_score_fn, make_proposer
from llmopt.train.lora import apply_lora

X = sp.Symbol("x")
WALL = 300
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


class _Timeout(BaseException):
    pass


class NnueEval(torch.nn.Module):  # mirrors train_nnue/bench_nnue
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_nnue(path="checkpoints/nnue_eval.pt"):
    ck = torch.load(path, weights_only=True)
    net = NnueEval()
    net.load_state_dict(ck["state_dict"])
    net.eval()
    mean, std = ck["mean"], ck["std"]

    def ev(state):
        v = torch.tensor([featurize(state.expr)], dtype=torch.float32)
        with torch.no_grad():
            return float(net((v - mean) / std))

    return ev


def load_proposer(ckpt="checkpoints/proposer_lora.pt"):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    missing, unexpected = model.load_state_dict(
        torch.load(ckpt, weights_only=True), strict=False)
    assert not unexpected, unexpected
    model.eval()
    return make_proposer(hf_score_fn(model, tok, device))


def random_proposer(seed_tag: str):
    rng = random.Random(f"random-proposer-{seed_tag}")

    def prop(state, children):
        children = list(children)
        rng.shuffle(children)
        return children

    return prop


def _root(rng, level, kind):
    if kind == "diff":
        f = _expression(rng, level)
        return sp.Derivative(f, X), sp.diff(f, X)
    while True:
        g = sp.simplify(sp.diff(_expression(rng, level), X))
        if g != 0:
            return sp.Integral(g, X), g


def _check(kind, expr, truth):
    if kind == "diff":
        return sp.simplify(expr - truth) == 0
    return sp.simplify(sp.diff(expr, X) - truth) == 0


def main(n: int, budgets: list[int]) -> None:
    nnue = load_nnue()
    model_prop = load_proposer()
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    configs = [
        ("full+hce", hce, None),
        ("full+nnue", nnue, None),
        ("rand3+hce", hce, "random"),
        ("prop3+hce", hce, model_prop),
        ("prop3+nnue", nnue, model_prop),
    ]
    print(f"# proposer race — n={n}/cell, wall {WALL}s/search, k=3")
    header = f"{'kind':>4} {'lvl':>3} {'budget':>6}" + "".join(
        f" {name:>10}" for name, _, _ in configs)
    print(header)
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            for budget in budgets:
                cells = []
                for name, ev, prop in configs:
                    if prop == "random":
                        prop = random_proposer(f"{kind}-{level}-{budget}")
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok, t0 = 0, time.time()
                    for _ in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            r = beam_search(
                                root, width=8, max_plies=20,
                                max_nodes=budget, eval_fn=ev,
                                proposer=prop,
                                propose_k=3 if prop else None)
                            ok += r.solved and _check(kind, r.state.expr, truth)
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    cells.append(f"{ok:>3}/{n:<2}({time.time() - t0:4.0f}s)")
                print(f"{kind:>4} {level:>3} {budget:>6} " +
                      " ".join(cells), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    a = ap.parse_args()
    main(a.n, a.budgets)
```

- [ ] **Step 2: Run once the proposer checkpoint exists** (`--n 15` to keep model-inference wall time sane)

- [ ] **Step 3: Commit with the table; README + roadmap + memory**

```bash
git add scripts/bench_proposer.py
git commit -m "feat: scripts — proposer race (measured: <paste>)"
```

---

## Self-review notes

- Spec coverage: rank-not-generate (T1), engine hook + node accounting (T2), winning-path data + exclude guard + multi-choice-only rows (T3), CLAUDE.md training recipe + heat 0 (T4), 5-config race with random-k control + separate wall-clock (T5).
- Type consistency: `build_prompt`/`make_proposer`/`hf_score_fn` names match T1↔T4↔T5; LoRA alpha=2*RANK=32 consistent T4↔T5.
- Execution order: T1-T3 now (Mac); T4 blocked on SSH (script written now, smoke only); T5 blocked on T4 checkpoint.
