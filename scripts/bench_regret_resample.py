"""Regret-gated resampling (2026-07-11, Artin's thesis: 'the best
skill is knowing when to regret/reconsider').

The engine's budget lesson ported to tokens: more budget doesn't buy
solves, better-SPENT budget does. A value probe reads a generation
trace mid-flight; when P(eventually correct) drops below threshold,
ABORT and respend the saved tokens on a fresh sample. Raced against
best-of-N at EQUAL token budget (pass@budget: solved iff any
completed sample is sympy-correct — the oracle never steers, only
scores).

Three phases (run on the 3080, overnight chain):
  --phase labels : calculus-LoRA 0.5B samples K solutions/problem at
                   T=0.7, hidden L20 state recorded at token
                   checkpoints, final correctness by Problem.check()
  --phase probe  : train MLP (checkpoint state -> eventual correct)
  --phase race   : best-of-N vs regret-gated vs greedy at equal
                   token budget, fresh problem band
"""

from __future__ import annotations

import argparse
import json
import signal as _signal
from pathlib import Path

import torch


def _checked(problem, text) -> bool:
    """FORK-ISOLATED oracle call (sympy pathology #10, 2026-07-12):
    p.check() on hallucinated model text can wedge in sympy loops
    that never deliver SIGALRM — the alarm box (pathology #8's fix)
    live-locked the pool farm 2.5h at 103% CPU. Same lesson as
    make_integrate (pathology #7): only fork + SIGKILL is a real
    timebox. Hang counts as wrong (conservative for every arm)."""
    if not text:
        return False
    import multiprocessing as _mp
    ctx = _mp.get_context("fork")
    q = ctx.Queue()

    def _w():
        try:
            q.put(bool(problem.check(text)))
        except Exception:
            q.put(False)

    p = ctx.Process(target=_w)
    p.start()
    p.join(10)
    if p.is_alive():
        p.kill()
        p.join()
        return False
    try:
        return bool(q.get(timeout=5))
    except Exception:
        return False

LAYER = 20
CKPTS = (8, 16, 32, 64)   # token positions where the probe looks
MAX_NEW = 160
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA = Path("checkpoints/calculus_lora.pt")
PROBE = Path("checkpoints/regret_probe.pt")


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmopt.train.lora import apply_lora
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda")
    # checkpoint = raw {**.a, **.b} adapter dict (train_calculus.py
    # convention); map_location: Mac-trained MPS storages crash CUDA
    # loads without it
    targets = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")
    apply_lora(model, targets, r=16, alpha=32)
    adapters = torch.load(LORA, weights_only=False, map_location="cpu")
    model.load_state_dict(adapters, strict=False)
    model.eval()
    return tok, model


def build_prompt(tok, problem) -> str:
    from llmopt.mathgen.evaluate import SYSTEM
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem.prompt}]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


@torch.inference_mode()
def sample_with_states(tok, model, prompt: str, seed: int,
                       abort_check=None):
    """One sampled completion; returns (text, states, aborted,
    tokens_spent). states: {ckpt: L20 last-token hidden (list)}.
    abort_check(ckpt, state_tensor) -> True aborts generation."""
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out_ids = []
    past = None
    cur = ids
    states = {}
    for step in range(MAX_NEW):
        o = model(input_ids=cur, past_key_values=past, use_cache=True,
                  output_hidden_states=(step + 1) in CKPTS)
        past = o.past_key_values
        if (step + 1) in CKPTS:
            h = o.hidden_states[LAYER][0, -1].float().cpu()
            states[step + 1] = h
            if abort_check is not None and abort_check(step + 1, h):
                return None, states, True, step + 1
        logits = o.logits[0, -1] / 0.7
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1,
                                    generator=gen))
        if nxt == tok.eos_token_id:
            break
        out_ids.append(nxt)
        cur = torch.tensor([[nxt]], device="cuda")
    # always record the FINAL state ("ckpt" 0 = end of trace): short
    # correct answers never reach token-24, and a probe trained only
    # on long traces sees an all-negative world (measured: base rate
    # 0.000, AUC nan on the first farm)
    o = model(input_ids=cur, past_key_values=past, use_cache=True,
              output_hidden_states=True)
    states[0] = o.hidden_states[LAYER][0, -1].float().cpu()
    return tok.decode(out_ids), states, False, len(out_ids)


def phase_labels(n_problems: int, k: int, seed_base: int,
                 out: Path) -> None:
    import sympy as sp  # noqa: F401  (Problem.check needs sympy env)

    from llmopt.mathgen.problems import make_integrate
    tok, model = load_model()
    n_rows = 0
    with out.open("w") as f:
        for i in range(n_problems):
            lv = 2 + i % 3
            p = make_integrate(lv, seed_base + i)
            prompt = build_prompt(tok, p)
            for j in range(k):
                text, states, _, spent = sample_with_states(
                    tok, model, prompt, seed=seed_base + 7919 * j + i)
                ok = _checked(p, text)
                for ck, h in states.items():
                    f.write(json.dumps(
                        {"level": lv, "seed": seed_base + i, "k": j,
                         "ckpt": ck, "state": h.tolist(),
                         "correct": ok}) + "\n")
                    n_rows += 1
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{n_problems}] rows={n_rows}", flush=True)
    print(f"wrote {n_rows} trace-state rows -> {out}")


def phase_probe(labels: Path, epochs: int) -> None:
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    train = [r for r in rows if r["seed"] % 2 == 0]
    test = [r for r in rows if r["seed"] % 2 == 1]

    def tensors(rs):
        x = torch.tensor([r["state"] for r in rs])
        y = torch.tensor([float(r["correct"]) for r in rs])
        return x, y

    xtr, ytr = tensors(train)
    xte, yte = tensors(test)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(xtr).squeeze(-1), ytr)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pr = torch.sigmoid(net(xte).squeeze(-1))
    # AUC
    pos, neg = pr[yte == 1], pr[yte == 0]
    auc = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item() \
        if len(pos) and len(neg) else float("nan")
    print(f"probe held-out ({len(test)}): AUC {auc:.3f} "
          f"(base rate {yte.mean().item():.3f})")
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd},
               PROBE)
    print(f"saved -> {PROBE}")


def phase_race(n_problems: int, k: int, seed_base: int,
               thresh: float) -> None:
    from llmopt.mathgen.problems import make_integrate
    tok, model = load_model()
    p_ = torch.load(PROBE, weights_only=False)
    net = torch.nn.Sequential(
        torch.nn.Linear(len(p_["mu"]), 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 1))
    net.load_state_dict(p_["state_dict"])
    net.eval()

    def prob_ok(h):
        with torch.no_grad():
            return torch.sigmoid(
                net((h - p_["mu"]) / p_["sd"])).item()

    budget = k * MAX_NEW
    res = {"greedy": 0, "best_of_n": 0, "regret": 0}
    spent = {"best_of_n": 0, "regret": 0}
    attempts_hist = []
    for i in range(n_problems):
        lv = 2 + i % 3
        p = make_integrate(lv, seed_base + i)
        prompt = build_prompt(tok, p)
        # greedy floor (1 sample worth of budget, T~0 via seed 0 draw)
        text, _, _, _ = sample_with_states(tok, model, prompt, seed=1)
        res["greedy"] += _checked(p, text)
        # best-of-N, budget-exhausting: keep sampling full attempts
        # until the token budget is spent (fixed-k stopped at ~14
        # tokens/sample and never used its budget — the first race
        # compared 16k vs 193k tokens and was VOID)
        ok = False
        used = 0
        j = 0
        while used < budget:
            text, _, _, sp_ = sample_with_states(
                tok, model, prompt, seed=1000 + 7919 * j + i)
            j += 1
            used += max(sp_, 1)
            ok = ok or _checked(p, text)
        res["best_of_n"] += ok
        spent["best_of_n"] += used
        # regret-gated: same TOKEN budget, abort when probe < thresh
        ok = False
        used = 0
        att = 0
        j = 0
        while used < budget:
            def check(ck, h, _used=used):
                return prob_ok(h) < thresh
            text, _, aborted, sp_ = sample_with_states(
                tok, model, prompt, seed=5000 + 7919 * j + i,
                abort_check=check)
            j += 1
            att += 1
            used += sp_
            if not aborted and text:
                ok = ok or _checked(p, text)
        res["regret"] += ok
        spent["regret"] += used
        attempts_hist.append(att)
        if (i + 1) % 15 == 0:
            print(f"[{i+1}] " + " ".join(f"{a}={v}"
                  for a, v in res.items()), flush=True)
    print(f"RESAMPLE RACE n={n_problems} budget={budget} tok/arm")
    for a, v in res.items():
        s = spent.get(a)
        print(f"  {a}: {v}/{n_problems}"
              + (f" (tokens {s})" if s else ""))
    print(f"regret attempts/problem: "
          f"{sum(attempts_hist)/len(attempts_hist):.1f}")
    print("bar: regret > best_of_n at equal token budget")


def phase_pool(n_problems: int, seed_base: int, pool: int,
               out: Path) -> None:
    """Round 2, farm half: FULL traces only (no aborts), logging every
    checkpoint's probe probability + the oracle verdict per attempt.
    The sweep phase replays abort policies offline against this pool —
    the router playbook (raw signal -> offline threshold sweep -> only
    then a live race). Round 1's naive 0.15@ckpt-8 policy lost 78 vs
    100 precisely because it skipped this step."""
    from llmopt.mathgen.problems import make_integrate
    tok, model = load_model()
    p_ = torch.load(PROBE, weights_only=False)
    net = torch.nn.Sequential(
        torch.nn.Linear(len(p_["mu"]), 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 1))
    net.load_state_dict(p_["state_dict"])
    net.eval()

    def prob_ok(h):
        with torch.no_grad():
            return torch.sigmoid(
                net((h - p_["mu"]) / p_["sd"])).item()

    # resume: attempts are seed-deterministic, so skip (i, j) pairs
    # already on disk (pathology-#10 live-lock cost one 2.5h farm)
    done: dict = {}
    if out.exists():
        with out.open() as f:
            for line in f:
                r = json.loads(line)
                done[r["i"]] = done.get(r["i"], 0) + 1
    with out.open("a" if done else "w") as f:
        for i in range(n_problems):
            if done.get(i, 0) >= pool:
                continue
            lv = 2 + i % 3
            p = make_integrate(lv, seed_base + i)
            prompt = build_prompt(tok, p)
            for j in range(done.get(i, 0), pool):
                text, states, _, sp_ = sample_with_states(
                    tok, model, prompt, seed=9000 + 7919 * j + i)
                probs = {str(ck): round(prob_ok(h), 4)
                         for ck, h in states.items()}
                f.write(json.dumps(
                    {"i": i, "level": lv, "j": j,
                     "tokens": max(sp_, 1), "probs": probs,
                     "correct": bool(_checked(p, text))}) + "\n")
            if (i + 1) % 10 == 0:
                print(f"POOL [{i+1}/{n_problems}]", flush=True)
    print(f"POOL done -> {out}")


def phase_sweep(problog: Path, k: int) -> None:
    """Round 2, judgment half: replay abort policies (threshold x
    min-checkpoint) against the farmed pool at the same token budget
    the live race used. Aborting at checkpoint c costs c tokens; kept
    attempts cost their full length and contribute their verdict.
    Purely offline — a full grid costs seconds, not a night."""
    import collections
    budget = k * MAX_NEW
    by: dict = collections.defaultdict(list)
    with problog.open() as f:
        for line in f:
            r = json.loads(line)
            by[r["i"]].append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r["j"])

    def run(thresh: float, min_ck: int):
        solved = exhausted = 0
        for rs in by.values():
            used = 0
            ok = False
            it = iter(rs)
            while used < budget:
                r = next(it, None)
                if r is None:          # pool ran dry before budget —
                    exhausted += 1     # pessimistic for that policy
                    break
                ab = None
                for ck in CKPTS:
                    if ck < min_ck:
                        continue
                    pr = r["probs"].get(str(ck))
                    if pr is not None and pr < thresh:
                        ab = ck
                        break
                if ab is not None:
                    used += ab
                else:
                    used += max(r["tokens"], 1)
                    ok = ok or r["correct"]
            solved += ok
        return solved, exhausted

    base, _ = run(-1.0, 0)  # never abort == best-of-N replay
    n = len(by)
    print(f"best-of-N replay: {base}/{n} (budget {budget} tok)")
    rows = []
    for min_ck in CKPTS:
        for thresh in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            s, ex = run(thresh, min_ck)
            rows.append((s, -ex, min_ck, thresh))
            print(f"  min_ck={min_ck:2d} thresh={thresh:.2f}: "
                  f"{s}/{n} (pool-exhausted {ex})")
    s, ex, mc, th = max(rows)[0], -max(rows)[1], max(rows)[2], max(rows)[3]
    print(f"SWEEP BEST: min_ck={mc} thresh={th:.2f} -> {s}/{n} "
          f"vs best-of-N {base}/{n}")
    print("bar: sweep best > best-of-N replay; if it clears, race the "
          "winning config LIVE on fresh problems (round 3)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["labels", "probe", "race", "pool", "sweep"])
    ap.add_argument("--n-problems", type=int, default=150)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=3_000_000)
    ap.add_argument("--labels", type=Path,
                    default=Path("data/regret_trace_labels.jsonl"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--pool", type=int, default=48)
    ap.add_argument("--problog", type=Path,
                    default=Path("data/regret_pool.jsonl"))
    a = ap.parse_args()
    if a.phase == "labels":
        phase_labels(a.n_problems, a.k, a.seed_base, a.labels)
    elif a.phase == "probe":
        phase_probe(a.labels, a.epochs)
    elif a.phase == "pool":
        # fresh problem block: probe was TRAINED on the 3M block and
        # round 1 raced on 3.5M — the sweep judges on unseen ground
        phase_pool(a.n_problems, a.seed_base + 1_000_000, a.pool,
                   a.problog)
    elif a.phase == "sweep":
        phase_sweep(a.problog, a.k)
    else:
        phase_race(a.n_problems, a.k, a.seed_base + 500_000, a.thresh)
