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

FEWSHOT = """You rewrite integrals one step at a time. Each reply is ONE step: a single sympy expression equal to the current expression. Use Integral(f, x) for unevaluated integrals. No prose.

Current: Integral(2*x + 3, x)
Step: Integral(2*x, x) + Integral(3, x)

Current: Integral(2*x, x) + Integral(3, x)
Step: x**2 + 3*x

Current: Integral(x*cos(x), x)
Step: x*sin(x) - Integral(sin(x), x)

Current: x*sin(x) - Integral(sin(x), x)
Step: x*sin(x) + cos(x)
"""


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


@torch.inference_mode()
def sample(tok, model, prompt: str, seed: int) -> tuple[str, int]:
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    past = None
    cur = ids
    for _ in range(MAX_NEW):
        o = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[0, -1].float().cpu() / 0.7
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1,
                                    generator=gen))
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        if tok.decode(out).endswith("\n"):
            break
        cur = torch.tensor([[nxt]], device=DEV)
    return tok.decode(out).strip(), len(out)


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
            text, spent = sample(tok, model, prompt1, seed=100 + 7919 * j + i)
            used += max(spent, 1)
            j += 1
            okp, solved = verify_step(f"Integral({integ}, x)", text)
            ok1 = ok1 or (okp and solved)
        res["one_shot"] += ok1
        # arm 2: verified macro-token chain
        cur = f"Integral({integ}, x)"
        used = 0
        ok2 = False
        j = 0
        while used < budget and not ok2:
            prompt2 = FEWSHOT + f"\nCurrent: {cur}\nStep:"
            text, spent = sample(tok, model, prompt2, seed=500 + 7919 * j + i)
            used += max(spent, 1)
            j += 1
            step_tries += 1
            cand = text.splitlines()[0].strip() if text else ""
            if not cand:
                continue
            parse_ok += 1
            okp, solved = verify_step(cur, cand)
            if okp:
                step_ok += 1
                cur = cand
                ok2 = solved
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
