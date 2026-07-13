"""Step-level expert iteration (the repo's founding long-term goal,
first concrete round; step-tokens measured 5/30 vs one-shot 0/30 at 5%
step validity — this trains the 5%).

Round 1 = imitation seeding: the ENGINE's winning derivations are
verified step chains by construction (replay_verify's walk yields the
intermediate expressions). Train the 0.5B to emit one rewrite per call
in the bench_step_tokens format; the eval is the same oracle-gated
race on fresh seeds. Round 2+ (later): mix in the model's OWN verified
chains at rising difficulty — generator as self-play opponent.

  --phase chains : fork-isolated engine solves (L2-L5), replay winning
                   histories, emit (cur, nxt) sympy-text pairs
  --phase train  : LoRA (train_calculus recipe: r=16 all-proj, loss on
                   step tokens only, length-bucketed, order-shuffled)
  --phase race   : bench_step_tokens arms with the adapter loaded,
                   fresh seeds — bar: validity% AND solves > base
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

CHAINS = Path("data/step_chains.jsonl")
ADAPTER = Path("checkpoints/step_lora.pt")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


def _chain_worker(level: int, seed: int, q: "mp.Queue") -> None:
    import sympy as sp

    from llmopt.mathgen.problems import make_integrate
    from llmopt.search import derivation as D
    from llmopt.search import rules as R
    from llmopt.search.derivation import State, _timeboxed, successors
    from llmopt.search.engine import solve
    from llmopt.search.rules import INT_RULES

    p = make_integrate(level, seed)
    root = sp.Integral(p._expr, sp.Symbol("x"))
    res = solve(root, budget=200)
    if not res.solved:
        q.put([])
        return
    # replay the winning history, collecting the expression chain
    # (replay_verify's backtracking walk, kept verbatim: labels are
    # not unique across siblings)
    saved, D.RULE_WALL = D.RULE_WALL, 60.0
    hist = res.state.history
    out: list[tuple] = []
    by_name = dict(INT_RULES)

    def annotate(cur_expr, rule_label):
        """Round-3 fields (Artin's GOs): hints = the rule-fire
        syndrome as names (the engine's sensory organs, in text);
        think = the ansatz rule's verbalized internal derivation."""
        node = max(cur_expr.atoms(sp.Integral), key=sp.count_ops,
                   default=None)
        hints = []
        if node is not None:
            for rn, rule in INT_RULES:
                if _timeboxed(rule, node, default=[]):
                    hints.append(rn)
        think = None
        rname = rule_label.split("@")[0]
        if rname in ("i_linear_basis", "i_sqrt_basis") and node is not None:
            R.DERIV_TRACE = []
            try:
                _timeboxed(by_name[rname], node, default=[])
                if R.DERIV_TRACE:
                    think = R.DERIV_TRACE[-1]
            finally:
                R.DERIV_TRACE = None
        return hints, think

    def walk(cur: State, i: int, acc: list) -> bool:
        if i == len(hist):
            out.extend(acc)
            return True
        for name, child in successors(cur, use_macros=True, verify_p=1.0):
            if name == hist[i]:
                hints, think = annotate(cur.expr, name)
                if walk(child, i + 1,
                        acc + [(sp.sstr(cur.expr), sp.sstr(child.expr),
                                hints, think)]):
                    return True
        return False

    walk(State(root), 0, [])
    D.RULE_WALL = saved
    q.put(out)


def phase_chains(n_per_level: int, seed_base: int,
                 levels=(2, 3, 4, 5), min_pairs: int = 1,
                 append: bool = False) -> None:
    """min_pairs: keep only chains with >= this many steps — round 1
    measured single-hop collapse (the one-ply-dominated corpus taught
    answers, not chaining); round 2 mines where the engine CHAINS."""
    ctx = mp.get_context("fork")
    seen: set = set()
    if append and CHAINS.exists():
        for line in CHAINS.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["cur"], r["nxt"]))
    n = 0
    with CHAINS.open("a" if append else "w") as f:
        for level in levels:
            for i in range(n_per_level):
                q = ctx.Queue()
                pr = ctx.Process(target=_chain_worker,
                                 args=(level, seed_base + i, q))
                pr.start()
                pr.join(90)
                if pr.is_alive():
                    pr.kill()
                    pr.join()
                    continue
                try:
                    pairs = q.get(timeout=10)
                except Exception:
                    continue
                if len(pairs) < min_pairs:
                    continue
                for cur, nxt, hints, think in pairs:
                    if (cur, nxt) in seen:
                        continue
                    seen.add((cur, nxt))
                    f.write(json.dumps({"cur": cur, "nxt": nxt,
                                        "level": level,
                                        "source": "engine",
                                        "hints": hints,
                                        "think": think}) + "\n")
                    n += 1
            print(f"L{level} done: {n} pairs total", flush=True)
    print(f"CHAINS done: {n} verified step pairs -> {CHAINS}")


def _reverse_worker(level: int, seed: int, q: "mp.Queue") -> None:
    """The REVERSE ENGINE (Artin, 2026-07-12): make_integrate draws
    the ANSWER F first — so mint the teaching chain from the answer
    side, no search at all. Additive peeling: F = F1+F2+... gives
    term-for-term integrand correspondence (fi = dFi), and the chain
    Integral(f) -> split -> solve one term at a time -> F is verified
    by construction at every step. Guaranteed multi-step chains at
    ANY level, including ones the forward engine one-plies."""
    import sympy as sp

    from llmopt.mathgen.problems import make_integrate
    p = make_integrate(level, seed)
    x = sp.Symbol("x")
    try:
        F = sp.sympify(p.answer)
    except Exception:
        q.put([])
        return
    terms = list(F.args) if isinstance(F, sp.Add) else [F]
    if len(terms) < 2:
        q.put([])
        return
    fis = [sp.diff(Fi, x) for Fi in terms]
    states = [sp.sstr(sp.Integral(p._expr, x))]
    # step 1: split into per-term integrals
    split = sp.Add(*(sp.Integral(fi, x) for fi in fis), evaluate=False)
    states.append(sp.sstr(split))
    # then peel one term at a time
    solved: list = []
    for k in range(len(terms)):
        solved.append(terms[k])
        rest = [sp.Integral(fi, x) for fi in fis[k + 1:]]
        states.append(sp.sstr(sp.Add(*(solved + rest), evaluate=False)))
    pairs = []
    for a, b in zip(states, states[1:]):
        think = ("split the integral term by term" if a == states[0]
                 else "solve the leading unsolved term")
        pairs.append((a, b, [], think))
    q.put(pairs)


def phase_reverse(n_per_level: int, seed_base: int,
                  levels=(4, 5, 6, 7, 8)) -> None:
    ctx = mp.get_context("fork")
    seen: set = set()
    if CHAINS.exists():
        for line in CHAINS.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["cur"], r["nxt"]))
    n = 0
    with CHAINS.open("a") as f:
        for level in levels:
            for i in range(n_per_level):
                q = ctx.Queue()
                pr = ctx.Process(target=_reverse_worker,
                                 args=(level, seed_base + i, q))
                pr.start()
                pr.join(60)
                if pr.is_alive():
                    pr.kill()
                    pr.join()
                    continue
                try:
                    pairs = q.get(timeout=10)
                except Exception:
                    continue
                for cur, nxt, hints, think in pairs:
                    if (cur, nxt) in seen:
                        continue
                    seen.add((cur, nxt))
                    f.write(json.dumps(
                        {"cur": cur, "nxt": nxt, "level": level,
                         "source": "reverse", "hints": hints,
                         "think": think}) + "\n")
                    n += 1
            print(f"L{level} reverse: {n} pairs total", flush=True)
    print(f"REVERSE done: {n} pairs -> {CHAINS}")


def _magic_buckets(states: list[str]) -> dict:
    """Fork-isolated IN CHUNKS (one wedging state must not poison the
    batch — pathology rules): estimator v7 cost head scores each
    state; buckets by corpus-relative terciles. The magic-adaptive
    granularity rung: unit size tracks local predictability."""
    costs: dict = {}
    CHUNK = 50
    for c0 in range(0, len(states), CHUNK):
        chunk = states[c0:c0 + CHUNK]
        got = _magic_chunk(chunk)
        costs.update(got)
    vals = sorted(v for v in costs.values() if v is not None)
    if not vals:
        print("MAGIC BUCKETS: nothing scored — skips will be unsized",
              flush=True)
        return {}
    lo, hi = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    return {s: ("easy" if c <= lo else "hard" if c >= hi else "mid")
            for s, c in costs.items() if c is not None}


def _magic_chunk(states: list[str]) -> dict:
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def _w():
        import sympy as sp
        import torch

        from llmopt.search.features import featurize
        sys_path = str(Path(__file__).resolve().parent)
        import sys as _s
        if sys_path not in _s.path:
            _s.path.insert(0, sys_path)
        from train_magic_estimator import Estimator
        pay = torch.load("checkpoints/magic_estimator_v7.pt",
                         weights_only=False)
        est = Estimator(d_in=len(pay["mu"]))
        est.load_state_dict(pay["state_dict"])
        est.eval()
        out = {}
        for s in states:
            try:
                e = sp.sympify(s, locals={"Integral": sp.Integral,
                                          "x": sp.Symbol("x")})
                f = torch.tensor([featurize(e)], dtype=torch.float32)
                with torch.no_grad():
                    _, cost = est((f - pay["mu"]) / pay["sd"])
                out[s] = float(cost.item())
            except Exception:
                out[s] = None
        q.put(out)

    p = ctx.Process(target=_w)
    p.start()
    p.join(90)
    if p.is_alive():
        p.kill()
        p.join()
        return {}   # this chunk unsized; others unaffected
    try:
        return q.get(timeout=15)
    except Exception:
        return {}


_MAGIC_SKIP = {"easy": 5, "mid": 3, "hard": 0}  # max jump per bucket


def phase_skips() -> None:
    """Macro-distillation (Artin's COCONUT riff, 2026-07-12): skip
    pairs (state_i -> state_{i+k}) are verified FOR FREE by
    transitivity of equivalence — rule COMPOSITION as data
    augmentation. Chains are reconstructed from corpus adjacency
    (row.nxt == next_row.cur runs); every skip is tagged
    source="skip" and deduped against the corpus."""
    rows = [json.loads(l) for l in CHAINS.read_text().splitlines()]
    seen = {(r["cur"], r["nxt"]) for r in rows}
    # reconstruct maximal runs: consecutive rows that link up
    runs: list[list[dict]] = []
    cur_run: list[dict] = []
    for r in rows:
        if r.get("source") == "skip":
            continue
        if cur_run and cur_run[-1]["nxt"] == r["cur"]:
            cur_run.append(r)
        else:
            if len(cur_run) >= 2:
                runs.append(cur_run)
            cur_run = [r]
    if len(cur_run) >= 2:
        runs.append(cur_run)
    all_states = sorted({s for run in runs
                         for s in [run[0]["cur"]] + [r["nxt"] for r in run]})
    buckets = _magic_buckets(all_states)
    n = 0
    with CHAINS.open("a") as f:
        for run in runs:
            states = [run[0]["cur"]] + [r["nxt"] for r in run]
            lv = max(r["level"] for r in run)
            for i in range(len(states)):
                # magic-adaptive jump cap: easy states may leap far,
                # hard states must single-step (no skips minted)
                cap = _MAGIC_SKIP.get(buckets.get(states[i], "mid"), 3)
                for k in range(i + 2, min(i + 1 + cap, len(states))):
                    pair = (states[i], states[k])
                    if pair in seen:
                        continue
                    seen.add(pair)
                    f.write(json.dumps(
                        {"cur": pair[0], "nxt": pair[1], "level": lv,
                         "source": "skip", "gate": "pending",
                         "magic": buckets.get(states[i], "mid")}) + "\n")
                    n += 1
    print(f"SKIPS done: {len(runs)} chains -> {n} magic-sized skip "
          f"pairs appended")


def phase_train(epochs: int, lr: float,
                out: Path = ADAPTER) -> None:
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_calculus import batches
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmopt.train.lora import apply_lora

    device = ("cuda" if torch.cuda.is_available() else "mps")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    model.train()

    rows = [json.loads(l) for l in CHAINS.read_text().splitlines()]
    examples = []
    for r in rows:
        # the bench_step_tokens plain-completion format, verbatim.
        # Round-3 fields: Hints line (rule-fire syndrome as text) in
        # the PROMPT; think rationale inline in the completion before
        # "=>" (only the expression after "=>" is oracle-verified)
        hints = ", ".join(r.get("hints") or []) or "none"
        prompt = tok(f"Current: {r['cur']}\nHints: {hints}\nStep:",
                     add_special_tokens=False).input_ids
        target = ((r["think"] + " => ") if r.get("think") else "") + r["nxt"]
        step = tok(" " + target + "\n",
                   add_special_tokens=False).input_ids
        ids = prompt + step
        if len(ids) > 512:
            continue
        labels = [-100] * len(prompt) + step
        examples.append((ids, labels))
    examples.sort(key=lambda e: len(e[0]))
    print(f"{len(examples)} training pairs")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for ep in range(epochs):
        tot = steps = 0
        for ids, labels, mask in batches(examples, tok.pad_token_id
                                         or tok.eos_token_id, 8, device,
                                         epoch=ep):
            mo = model(input_ids=ids, attention_mask=mask,
                       labels=labels)
            mo.loss.backward()
            opt.step()
            opt.zero_grad()
            tot += float(mo.loss)
            steps += 1
        print(f"epoch {ep}: loss {tot / max(steps, 1):.4f}", flush=True)
    # the train_calculus save convention: raw {**.a, **.b} adapter dict
    torch.save({k: v.cpu() for k, v in model.state_dict().items()
                if k.split(".")[-1] in ("a", "b")}, out)
    print(f"saved adapter -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["chains", "train", "skips", "reverse"])
    ap.add_argument("--n-per-level", type=int, default=150)
    ap.add_argument("--seed-base", type=int, default=8_000_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--levels", type=int, nargs="+",
                    default=[2, 3, 4, 5])
    ap.add_argument("--min-pairs", type=int, default=1)
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    if a.phase == "chains":
        phase_chains(a.n_per_level, a.seed_base, tuple(a.levels),
                     a.min_pairs, a.append)
    elif a.phase == "reverse":
        phase_reverse(a.n_per_level, a.seed_base, tuple(a.levels))
    elif a.phase == "skips":
        phase_skips()
    else:
        phase_train(a.epochs, a.lr)
