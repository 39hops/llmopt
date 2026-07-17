"""Phase 2: GRPO from birth — the math-native 19M climbs on the Mac.

Same algorithm as step_grpo (waves are groups, verifier is reward,
mixed groups only, identity + cycle guards, gate/checkpoint/rollback
ladder, per-gate snapshots for the formation movie) adapted to the
micro-model: full-parameter updates (no LoRA — the whole model is
ours), MPS-native, 45-token decode.

Phase-1 start: 65.6% unseen validity at L2-4 — no famine here, so
collection runs at L3-6 where mixed groups live.

    caffeinate -i .venv/bin/python -u scripts/step_grpo_micro.py --cycles 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.train.mathnative import MathTokenizer, build_model

B = 8
GROUPS = 64   # override with --groups (speed-first: small cycles,
GATE_EVERY = 2  # quick feedback — Artin 2026-07-17)
LR = 1e-5
CLIP = 0.2
# run 2 curriculum ascent (run 1 plateaued at {12,12,7,7}/~80%
# for 4 straight gates — L2/L3 maxed, deeper levels needed)
LEVELS = (4, 5, 6, 7)
GATE_LEVELS = (3, 4, 5, 6, 7)
GATE_N = 24  # 12/level left +-1 solve inside the gate's noise floor (run 2b)
SEED_BASE = 66_000_000  # 50.4M v2.1 GRPO run 1 (65M = L4 farm band)
GATE_BAND = 9_900_000
CKPT = Path("checkpoints/mathnative_grpo.pt")
CORPUS = Path("data/micromodel_grpo_mined.jsonl")  # untracked sidecar


def sample_wave_lp(model, tok, prompt_ids, seeds, dev, max_new=120):
    import torch
    Bn = len(seeds)
    ids = torch.tensor([prompt_ids] * Bn, device=dev)
    gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
    out = [[] for _ in range(Bn)]
    lps = [0.0] * Bn
    done = [False] * Bn
    nl = tok.id["\n"]
    for _ in range(max_new):
        logits = model(ids)[:, -1].float().cpu() / 0.7
        probs = torch.softmax(logits, -1)
        nxts = []
        for b in range(Bn):
            if done[b]:
                nxts.append(tok.pad_id)
                continue
            nxt = int(torch.multinomial(probs[b], 1, generator=gens[b]))
            lps[b] += float(torch.log(probs[b, nxt] + 1e-20))
            if nxt in (nl, tok.eos_id, tok.pad_id):
                done[b] = True
            else:
                out[b].append(nxt)
            nxts.append(nxt)
        if all(done):
            break
        import torch as t
        ids = t.cat([ids, t.tensor(nxts, device=dev)[:, None]], 1)
    return [tok.decode(o).strip() for o in out], out, lps


def collect(model, tok, dev, n_groups, seed0):
    import sympy as sp

    from bench_step_tokens import _gen_isolated
    from bench_verify_fast import verify_wave
    groups, mined = [], []
    stats = {"waves": 0, "mixed": 0, "allfail": 0, "allpass": 0}
    pi = 0
    import torch
    with torch.no_grad():
        while len(groups) < n_groups:
            lv = LEVELS[pi % len(LEVELS)]
            p = _gen_isolated(lv, seed0 + pi)
            pi += 1
            if p is None:
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            visited = {cur.replace(" ", "")}
            for ply in range(12):
                prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
                texts, tok_ids, lps = sample_wave_lp(
                    model, tok, prompt,
                    [seed0 * 13 + pi * 977 + ply * 101 + b
                     for b in range(B)], dev)
                stats["waves"] += 1
                distinct = [t_ for t_ in dict.fromkeys(texts) if t_]
                wv = verify_wave(cur, distinct) if distinct else {}

                def r_of(t_):
                    ok, _ = wv.get(t_, (False, False))
                    if not ok:
                        return 0.0
                    return 0.0 if t_.replace(" ", "") in visited else 1.0
                rewards = [r_of(t_) for t_ in texts]
                n_ok = sum(1 for r in rewards if r > 0)
                if 0 < n_ok < B:
                    stats["mixed"] += 1
                    groups.append({"prompt": prompt, "tok_ids": tok_ids,
                                   "logps": lps, "rewards": rewards,
                                   "level": lv})
                elif n_ok == 0:
                    stats["allfail"] += 1
                else:
                    stats["allpass"] += 1
                nxt = None
                for t_ in texts:
                    ok, so = wv.get(t_, (False, False))
                    if ok and t_.replace(" ", "") not in visited:
                        mined.append({"cur": cur, "nxt": t_, "level": lv,
                                      "source": "grpo-micro"})
                        if so:
                            nxt = "SOLVED"
                            break
                        nxt = t_
                if nxt in (None, "SOLVED"):
                    break
                cur = nxt
                visited.add(cur.replace(" ", ""))
                if len(groups) >= n_groups:
                    break
    return groups, mined, stats


def logp_new(model, tok, g, dev):
    import torch
    seqs = [list(g["prompt"]) + list(o) for o in g["tok_ids"]]
    L = max(len(s) for s in seqs)
    ids = torch.tensor([s + [tok.pad_id] * (L - len(s)) for s in seqs],
                       device=dev)
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                         for s in seqs], device=dev)
    logits = model(ids, mask)
    plen = len(g["prompt"])
    out = []
    for b, o in enumerate(g["tok_ids"]):
        if not o:
            out.append(torch.tensor(0.0, device=dev))
            continue
        lg = logits[b, plen - 1: plen - 1 + len(o)].float() / 0.7
        lp = torch.log_softmax(lg, -1)
        idx = torch.tensor(o, device=dev)
        out.append(lp.gather(1, idx[:, None]).sum())
    return torch.stack(out)


def gate_eval(model, tok, dev, n=None):
    """Honest chain gate. n<GATE_N = cheap proxy tier (same seeds,
    prefix subset — noisier per reading, never used for promotion)."""
    import sympy as sp
    import torch

    from bench_step_tokens import _gen_isolated
    from bench_verify_fast import verify_wave
    solves = {}
    valid = tried = 0
    with torch.no_grad():
        for lv in GATE_LEVELS:
            s = 0
            for i in range(n or GATE_N):
                p = _gen_isolated(lv, GATE_BAND + 1000 * lv + i)
                if p is None:
                    continue
                cur = f"Integral({sp.sstr(p._expr)}, x)"
                visited = {cur.replace(" ", "")}
                done = False
                for ply in range(12):
                    prompt = tok.encode(
                        f"Current: {cur}\nHints: none\nStep: ")
                    texts, _, _ = sample_wave_lp(
                        model, tok, prompt,
                        [GATE_BAND + i * 31 + ply * 7 + b
                         for b in range(B)], dev)
                    tried += len(texts)
                    distinct = [t_ for t_ in dict.fromkeys(texts)
                                if t_ and t_.replace(" ", "") not in visited]
                    wv = verify_wave(cur, distinct) if distinct else {}
                    nxt = None
                    for t_ in texts:
                        ok, so = wv.get(t_, (False, False))
                        if ok and t_.replace(" ", "") not in visited:
                            valid += 1
                            if nxt is None:
                                nxt = "SOLVED" if so else t_
                    if nxt == "SOLVED":
                        done = True
                        break
                    if nxt is None:
                        break
                    cur = nxt
                    visited.add(cur.replace(" ", ""))
                s += done
            solves[lv] = s
    return solves, 100 * valid / max(tried, 1)


def main(cycles: int, src_path: str | None = None,
         out_path: str | None = None, d: int = 384,
         layers: int = 8, ffn: int = 1536, heads: int = 6,
         groups_n: int = GROUPS) -> None:
    import shutil

    import torch

    from llmopt.train.preference import grpo_advantages, grpo_loss
    global CKPT
    if out_path:
        CKPT = Path(out_path)
    tok = MathTokenizer()
    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(tok.vocab), d=d, layers=layers,
                        heads=heads, ffn=ffn).to(dev)
    src = (Path(src_path) if src_path
           else CKPT if CKPT.exists()
           else Path("checkpoints/mathnative_19m.pt"))
    model.load_state_dict(torch.load(src, map_location="cpu"))
    model.to(dev)
    print(f"starting from {src}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    model.eval()
    solves, validity = gate_eval(model, tok, dev)
    print(f"baseline: {solves} validity {validity:.2f}%", flush=True)
    best_v, best_s = validity, sum(solves.values())
    torch.save(model.state_dict(), CKPT)
    retried = False

    for cyc in range(1, cycles + 1):
        t0 = time.time()
        model.eval()
        groups, mined, stats = collect(model, tok, dev, groups_n,
                                       SEED_BASE + 100_000 * cyc)
        with CORPUS.open("a") as f:
            for r in mined:
                f.write(json.dumps(r) + "\n")
        lv_mined = {}
        for r in mined:
            lv_mined[r["level"]] = lv_mined.get(r["level"], 0) + 1
        lv_grp = {}
        for g in groups:
            lv_grp[g["level"]] = lv_grp.get(g["level"], 0) + 1
        print(f"cycle {cyc}: {stats} mined +{len(mined)} "
              f"({time.time() - t0:.0f}s) | mined/lv "
              f"{dict(sorted(lv_mined.items()))} | groups/lv "
              f"{dict(sorted(lv_grp.items()))}", flush=True)
        model.train()
        order = list(range(len(groups)))
        random.Random(cyc).shuffle(order)
        tot = 0.0
        for gi in order:
            g = groups[gi]
            lp_n = logp_new(model, tok, g, dev)
            lp_o = torch.tensor(g["logps"], device=dev)
            rw = torch.tensor(g["rewards"], device=dev)
            adv = grpo_advantages(rw, torch.zeros(B, dtype=torch.long,
                                                  device=dev))
            loss = grpo_loss(lp_n, lp_o, adv, clip=CLIP)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            tot += float(loss.detach())
        print(f"cycle {cyc}: grpo loss {tot / max(len(groups), 1):.4f}",
              flush=True)
        if cyc % GATE_EVERY != 0:
            # two-tier: cheap proxy every off-cycle — visibility only,
            # never a verdict (GATE_N=8 is +-2 solves of noise)
            model.eval()
            psolves, pval = gate_eval(model, tok, dev, n=8)
            print(f"cycle {cyc} proxy: {psolves} "
                  f"= {sum(psolves.values())}/{8*len(GATE_LEVELS)} "
                  f"@ {pval:.2f}%", flush=True)
        if cyc % GATE_EVERY == 0:
            model.eval()
            solves, validity = gate_eval(model, tok, dev)
            s_now = sum(solves.values())
            print(f"cycle {cyc} gate: {solves} validity {validity:.2f}%"
                  f" (best {best_v:.2f}/{best_s})", flush=True)
            # snapshot-before-verdict: rollbacks used to DISCARD the
            # candidate (the 62-solve record lost at run-1b cycle 8)
            torch.save(model.state_dict(), CKPT.with_name(
                f"{CKPT.stem}_cand{cyc:03d}.pt"))
            # solves-primary: solves are the currency; validity is a
            # drift alarm (2.0-pt band), not a veto
            if s_now >= best_s and validity >= best_v - 2.0:
                best_v = max(best_v, validity)
                best_s = max(best_s, s_now)
                shutil.copy(CKPT, CKPT.with_name(
                    f"{CKPT.stem}_c{cyc:03d}.pt")) if CKPT.exists() else None
                torch.save(model.state_dict(), CKPT)
                print("  checkpointed", flush=True)
                retried = False
            else:
                print("  ROLLBACK, lr halved", flush=True)
                model.load_state_dict(torch.load(CKPT, map_location="cpu"))
                model.to(dev)
                for gp in opt.param_groups:
                    gp["lr"] /= 2
                if retried:
                    print("HALT: two consecutive rollbacks", flush=True)
                    break
                retried = True
    print("micro GRPO run complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--from", dest="src", default=None,
                    help="starting checkpoint (default: OUT if it "
                         "exists, else the phase-1 base)")
    ap.add_argument("--out", default=None,
                    help="gate-checkpoint path (default: "
                         "mathnative_grpo.pt)")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1536)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--groups", type=int, default=GROUPS,
                    help="mixed groups per cycle (speed-first: "
                         "smaller cycles, quicker gate feedback)")
    a = ap.parse_args()
    main(a.cycles, a.src, a.out, a.d, a.layers, a.ffn, a.heads,
         a.groups)
