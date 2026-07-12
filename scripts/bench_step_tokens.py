"""Bigger tokens for the LLM (Artin, 2026-07-12): the unit of
generation becomes a whole derivation STEP — model emits a candidate
rewrite of the current integral as sympy text, the ORACLE verifies the
step before it stands (equivalence via differentiation, fork-safe),
invalid steps are resampled. Regret at step boundaries, where the unit
economics are verified-macro-token-sized, vs one-shot answers at equal
token budget.

This is the repo's banked long-term goal (step-level search, model
proposes rewrites) in its minimal form, and the LLM-side mirror of the
engine result (one ply = one verified macro-token -> 2.1x). Model:
base Qwen instruct + few-shot (the calculus LoRA is answer-only
trained — it never saw steps). Kill-switch: if step-parse rate is ~0,
record the null and stop.

Mac MPS. Arms at equal token budget per problem:
  one_shot : best-of-N answers, sympy-checked (budget-exhausting)
  steps    : chain of verified rewrites; invalid step = resample;
             solved when a step has no Integral left and diff-verifies
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 96
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")

FEWSHOT = """You rewrite integrals one step at a time. Each reply is ONE step: optionally a short reason, then "=>", then a single sympy expression equal to the current expression. Use Integral(f, x) for unevaluated integrals. Hints list which rewrite rules fire.

Current: Integral(2*x + 3, x)
Hints: i_sum, i_table
Step: split the sum => Integral(2*x, x) + Integral(3, x)

Current: Integral(2*x, x) + Integral(3, x)
Hints: i_table, i_const
Step: term-by-term antiderivatives => x**2 + 3*x

Current: Integral(x*cos(x), x)
Hints: i_parts, i_linear_basis
Step: integrate by parts with u = x => x*sin(x) - Integral(sin(x), x)

Current: x*sin(x) - Integral(sin(x), x)
Hints: i_table
Step: table antiderivative of sin => x*sin(x) + cos(x)
"""


def _hints_isolated(cur_s: str, wall: int = 15) -> list[str]:
    """Rule-fire syndrome of a (possibly MODEL-written) expression —
    fork-isolated: rules on model text obey the same pathology rules
    as the oracle."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def _w():
        import sympy as sp

        from llmopt.search.derivation import _timeboxed
        from llmopt.search.rules import INT_RULES
        try:
            env = {"Integral": sp.Integral, "x": sp.Symbol("x")}
            e = sp.sympify(cur_s, locals=env)
            node = max(e.atoms(sp.Integral), key=sp.count_ops,
                       default=None)
            names = []
            if node is not None:
                for rn, rule in INT_RULES:
                    if _timeboxed(rule, node, default=[]):
                        names.append(rn)
            q.put(names)
        except Exception:
            q.put([])

    p = ctx.Process(target=_w)
    p.start()
    p.join(wall)
    if p.is_alive():
        p.kill()
        p.join()
        return []
    try:
        return q.get(timeout=5)
    except Exception:
        return []


def _verify_step(prev_s: str, cand_s: str, q: "mp.Queue") -> None:
    """Child: parse + equivalence check (diff both, simplify difference
    of doit forms) — sympy pathology rules: this ONLY runs forked."""
    import sympy as sp
    x = sp.Symbol("x")
    try:
        env = {"Integral": sp.Integral, "x": x, "sqrt": sp.sqrt,
               "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "exp": sp.exp, "log": sp.log, "atan": sp.atan,
               "asin": sp.asin, "pi": sp.pi, "E": sp.E}
        prev = sp.sympify(prev_s, locals=env)
        cand = sp.sympify(cand_s, locals=env)
        d = (prev.doit(deep=True) - cand.doit(deep=True))
        ok = sp.simplify(sp.diff(d, x)) == 0 if d.has(x) else \
            sp.simplify(d) == 0
        solved = ok and not cand.atoms(sp.Integral)
        q.put((bool(ok), bool(solved), cand_s))
    except Exception:
        q.put((False, False, cand_s))


def verify_step(prev_s: str, cand_s: str, wall: int = 15):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_verify_step, args=(prev_s, cand_s, q))
    p.start()
    p.join(wall)
    if p.is_alive():
        p.kill()
        p.join()
        return False, False
    try:
        ok, solved, _ = q.get(timeout=5)
        return ok, solved
    except Exception:
        return False, False


def load(adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to(DEV)
    if adapter:  # step LoRA (train_calculus adapter-dict convention)
        from llmopt.train.lora import apply_lora
        apply_lora(model, ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"),
                   r=16, alpha=32)
        model.load_state_dict(
            torch.load(adapter, weights_only=False,
                       map_location="cpu"), strict=False)
        model.to(DEV)
    model.eval()
    return tok, model


_EXPR_MASK = None


def _expr_mask(tok):
    """Charset-constrained decoding v1 (Artin's GO after round 1:
    the adapter's 1742 invalid attempts were expression-space
    near-misses — make non-expression tokens UNSAMPLEABLE). Allows
    tokens whose text uses only sympy-expression characters; the
    newline stays allowed (it terminates the step)."""
    global _EXPR_MASK
    if _EXPR_MASK is None:
        allowed = set("0123456789+-*/(), .xIEabcdefghijklmnopqrstuvwxyz"
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ_^ \n=>'")  # => rationale
        size = model_vocab = len(tok)
        mask = torch.zeros(size, dtype=torch.bool)
        for tid in range(model_vocab):
            s = tok.decode([tid])
            if s and all(c in allowed for c in s):
                mask[tid] = True
        eos = tok.eos_token_id
        if eos is not None:
            mask[eos] = True
        _EXPR_MASK = mask
    return _EXPR_MASK


@torch.inference_mode()
def sample(tok, model, prompt: str, seed: int,
           constrain: bool = False) -> tuple[str, int]:
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    past = None
    cur = ids
    for _ in range(MAX_NEW):
        o = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[0, -1].float().cpu() / 0.7
        if constrain:
            m = _expr_mask(tok)
            if m.shape[0] < logits.shape[0]:
                # model logit width > tokenizer vocab (padded for
                # tensor cores); the padding ids are never valid
                pad = torch.zeros(logits.shape[0] - m.shape[0],
                                  dtype=torch.bool)
                m = torch.cat([m, pad])
            logits = logits.masked_fill(~m, float("-inf"))
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1,
                                    generator=gen))
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        if tok.decode(out).endswith("\n"):
            break
        cur = torch.tensor([[nxt]], device=DEV)
    return tok.decode(out).strip(), len(out)


@torch.inference_mode()
def sample_batch(tok, model, prompt: str, seeds: list[int],
                 constrain: bool = False):
    """B parallel sampled completions of the same prompt — the step
    harness's inference rung (2026-07-12): resamples at a given state
    are embarrassingly parallel, and a 0.5B at batch 1 leaves the GPU
    ~98% idle. One tiled prefill + joint decode; per-stream seeds kept
    (reproducibility). Returns (texts, spent_tokens) per stream; spent
    counts only each stream's own generated tokens (budget semantics
    unchanged)."""
    B = len(seeds)
    enc = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    cur = enc.repeat(B, 1)
    past = None
    gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
    out: list[list[int]] = [[] for _ in range(B)]
    done = [False] * B
    eos = tok.eos_token_id
    m = _expr_mask(tok) if constrain else None
    for _ in range(MAX_NEW):
        o = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[:, -1].float().cpu() / 0.7
        if m is not None:
            mm = m
            if mm.shape[0] < logits.shape[1]:
                mm = torch.cat([mm, torch.zeros(
                    logits.shape[1] - mm.shape[0], dtype=torch.bool)])
            logits = logits.masked_fill(~mm, float("-inf"))
        nxts = []
        for b in range(B):
            if done[b]:
                nxts.append(eos or 0)
                continue
            nxt = int(torch.multinomial(
                torch.softmax(logits[b], -1), 1, generator=gens[b]))
            if nxt == eos:
                done[b] = True
            else:
                out[b].append(nxt)
                if tok.decode(out[b]).endswith("\n"):
                    done[b] = True
            nxts.append(nxt)
        if all(done):
            break
        cur = torch.tensor(nxts, device=DEV).unsqueeze(1)
    return ([tok.decode(o_).strip() for o_ in out],
            [max(len(o_), 1) for o_ in out])


def _gen_isolated(level: int, seed: int, wall: int = 45):
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def _w():
        from llmopt.mathgen.problems import make_integrate
        q.put(make_integrate(level, seed))

    p = ctx.Process(target=_w)
    p.start()
    p.join(wall)
    if p.is_alive():
        p.kill()
        p.join()
        return None
    try:
        return q.get(timeout=10)
    except Exception:
        return None


def solve_chain(tok, model, integ: str, budget: int, seed0: int):
    """Oracle-gated chain; returns (solved, verified_pairs,
    valid_steps, tried_steps). verified_pairs only from SOLVED traces
    (spec: no-op steps must not enter the corpus via stalls); chain
    length capped at 12 (kills stall loops)."""
    cur = f"Integral({integ}, x)"
    pairs: list[tuple[str, str]] = []
    used = j = valid = tried = 0
    ok = False
    B = 8   # batched resampling: parallel draws at the same state
    while used < budget and not ok and len(pairs) < 12:
        hints = ", ".join(_hints_isolated(cur)) or "none"
        prompt = FEWSHOT + f"\nCurrent: {cur}\nHints: {hints}\nStep:"
        texts, spents = sample_batch(
            tok, model, prompt,
            seeds=[seed0 + 7919 * (j + b) for b in range(B)],
            constrain=True)
        j += B
        advanced = False
        for text, spent in zip(texts, spents):
            used += spent
            tried += 1
            if advanced or used > budget:
                continue   # spend still counted (honest budget)
            line = text.splitlines()[0].strip() if text else ""
            # rationale rides before "=>"; only the expression after
            # the last "=>" is submitted to the oracle
            cand = line.split("=>")[-1].strip()
            if not cand:
                continue
            okp, solved = verify_step(cur, cand)
            if okp:
                valid += 1
                pairs.append((cur, cand))
                cur = cand
                ok = solved
                advanced = True
    return ok, (pairs if ok else []), valid, tried


def main(n: int, seed_base: int, budget: int,
         adapter: str | None = None) -> None:
    import sympy as sp
    tok, model = load(adapter)
    res = {"one_shot": 0, "steps": 0}
    parse_ok = step_ok = step_tries = 0
    for i in range(n):
        lv = 2 + i % 2                      # L2/L3: format probe first
        p = _gen_isolated(lv, seed_base + i)
        if p is None:
            continue
        integ = sp.sstr(p._expr)
        # arm 1: one-shot best-of-N (budget-exhausting)
        msgs = [{"role": "user", "content":
                 f"Find an antiderivative of: {integ}\n"
                 "Reply with ONLY the sympy expression."}]
        prompt1 = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
        used = 0
        ok1 = False
        j = 0
        while used < budget:
            texts, spents = sample_batch(
                tok, model, prompt1,
                seeds=[100 + 7919 * (j + b) + i for b in range(8)])
            j += 8
            for text, spent in zip(texts, spents):
                used += spent
                if ok1 or used > budget:
                    continue
                okp, solved = verify_step(f"Integral({integ}, x)", text)
                ok1 = ok1 or (okp and solved)
        res["one_shot"] += ok1
        # arm 2: verified macro-token chain
        ok2, _, v, t = solve_chain(tok, model, integ, budget,
                                   seed0=500 + i)
        step_ok += v
        step_tries += t
        res["steps"] += ok2
        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{n}] one_shot={res['one_shot']} "
                  f"steps={res['steps']} (valid-step rate "
                  f"{step_ok}/{step_tries})", flush=True)
    print(f"STEP-TOKEN RACE n={n} budget={budget} tok/arm")
    print(f"  one_shot: {res['one_shot']}  steps: {res['steps']}")
    print(f"  step validity: {step_ok}/{step_tries} "
          f"({100*step_ok/max(step_tries,1):.0f}%)")
    print("kill-switch: validity ~0% = format null; "
          "bar: steps > one_shot at equal budget")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=7_000_000)
    ap.add_argument("--budget", type=int, default=768)
    ap.add_argument("--adapter", type=str, default=None)
    a = ap.parse_args()
    main(a.n, a.seed_base, a.budget, a.adapter)
