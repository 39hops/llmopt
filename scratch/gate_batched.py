"""Batched gate v2 (2026-07-21): batch ACROSS problems, 8 seeds
each — one forward serves K*8 rows instead of 8. Right-padded
buffer + attn_mask (model supports it); per-row write positions
keep RoPE phases identical to the unbatched path. NOTE: float
reduction order changes => near-ties may resolve differently =>
this is a NEW GATE LINEAGE (re-baseline models of record once).
Usage: gate_batched.py <ckpt> <d> <layers> <ffn> <heads> <label> [K]"""
import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave


def batched_wave(model, tok, prompts, seed_lists, dev, max_new=120):
    """prompts: list of K token-lists; seed_lists: K lists of B seeds.
    Returns per-prompt (texts, lps)."""
    rows = []          # (prompt_idx, gen)
    for k, seeds in enumerate(seed_lists):
        for s in seeds:
            rows.append((k, torch.Generator(device="cpu")
                         .manual_seed(s)))
    R = len(rows)
    Lmax = max(len(p) for p in prompts)
    T = Lmax + max_new
    ids = torch.full((R, T), tok.pad_id, dtype=torch.long, device=dev)
    mask = torch.zeros((R, T), dtype=torch.bool, device=dev)
    lens = []
    for r, (k, _) in enumerate(rows):
        p = prompts[k]
        ids[r, :len(p)] = torch.tensor(p, device=dev)
        mask[r, :len(p)] = True
        lens.append(len(p))
    lens = torch.tensor(lens)
    out = [[] for _ in range(R)]
    lps = [0.0] * R
    done = [False] * R
    nl = tok.id["\n"]
    for step in range(max_new):
        cur_T = int(lens.max())
        logits = model(ids[:, :cur_T], attn_mask=mask[:, :cur_T])
        last = logits[torch.arange(R), lens - 1].float().cpu() / 0.7
        probs = torch.softmax(last, -1)
        for r, (k, gen) in enumerate(rows):
            if done[r]:
                continue
            nxt = int(torch.multinomial(probs[r], 1, generator=gen))
            lps[r] += float(torch.log(probs[r, nxt] + 1e-20))
            if nxt in (nl, tok.eos_id, tok.pad_id):
                done[r] = True
            else:
                ids[r, lens[r]] = nxt
                mask[r, lens[r]] = True
                out[r].append(nxt)
                lens[r] += 1
        if all(done):
            break
    res = []
    B = len(seed_lists[0])
    for k in range(len(prompts)):
        rr = [r for r, (kk, _) in enumerate(rows) if kk == k]
        res.append(([tok.decode(out[r]).strip() for r in rr],
                    [lps[r] for r in rr]))
    return res


def gate_eval_batched(model, tok, dev, K=12):
    solves = {}
    valid = tried = 0
    probs_all = []
    for lv in G.GATE_LEVELS:
        for i in range(G.GATE_N):
            p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
            if p is not None:
                probs_all.append((lv, i,
                                  f"Integral({sp.sstr(p._expr)}, x)"))
    state = {j: {"cur": c, "visited": {c.replace(" ", "")},
                 "done": False, "solved": False, "lv": lv, "i": i}
             for j, (lv, i, c) in enumerate(probs_all)}
    with torch.no_grad():
        for ply in range(12):
            live = [j for j, st in state.items() if not st["done"]]
            if not live:
                break
            for b0 in range(0, len(live), K):
                chunk = live[b0:b0 + K]
                prompts = [tok.encode(f"Current: {state[j]['cur']}\n"
                                      f"Hints: none\nStep: ")
                           for j in chunk]
                seeds = [[G.GATE_BAND + state[j]["i"] * 31 + ply * 7
                          + b for b in range(G.B)] for j in chunk]
                waves = batched_wave(model, tok, prompts, seeds, dev)
                for j, (texts, _) in zip(chunk, waves):
                    st = state[j]
                    distinct = [t_ for t_ in dict.fromkeys(texts)
                                if t_ and t_.replace(" ", "")
                                not in st["visited"]]
                    wv = (verify_wave(st["cur"], distinct)
                          if distinct else {})
                    tried += len(texts)
                    nxt = None
                    for t_ in texts:
                        ok, so = wv.get(t_, (False, False))
                        if ok and t_.replace(" ", "") not in st["visited"]:
                            valid += 1
                            if nxt is None:
                                nxt = "SOLVED" if so else t_
                    if nxt == "SOLVED":
                        st["done"] = st["solved"] = True
                    elif nxt is None:
                        st["done"] = True
                    else:
                        st["cur"] = nxt
                        st["visited"].add(nxt.replace(" ", ""))
    for lv in G.GATE_LEVELS:
        solves[lv] = sum(1 for st in state.values()
                         if st["lv"] == lv and st["solved"])
    return solves, 100 * valid / max(tried, 1)


if __name__ == "__main__":
    ckpt, d, layers, ffn, heads, label = (sys.argv[1],
        int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
        int(sys.argv[5]), sys.argv[6])
    K = int(sys.argv[7]) if len(sys.argv) > 7 else 12
    tok = MathTokenizer()
    if torch.cuda.is_available():
        dev = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(len(tok.vocab), d=d, layers=layers,
                        heads=heads, ffn=ffn).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    t0 = time.time()
    solves, valid = gate_eval_batched(model, tok, dev, K=K)
    print(f"{label} gate-v2(K={K}): {solves} = "
          f"{sum(solves.values())}/120 @ {valid:.2f}% "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
