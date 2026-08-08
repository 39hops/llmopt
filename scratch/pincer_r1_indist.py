"""Pincer R1a-INDIST (Phase-4 row 1 of the results-hardening
false-null hunt; pre-reg in RESULTS before fire): the R1a peeling
probe re-run on IN-DISTRIBUTION prompts.

Mechanism basis (AMENDMENT TENET-R0-REV-DIST + VERDICT
TENET-R0-REV-B): reversed pairs map later-state -> earlier-state,
so a reverse crystal's in-distribution INPUTS are post-step
states; R1a's 11% per-candidate validity was read on pooled
mid-chain states without conditioning on that asymmetry. Here the
probe states are constructed exactly as the amendment's diagnostic
did: one forward engine step applied to each gate-band problem,
prompt = the child (first encodable engine child; Subs-form
children skipped and counted).

Scoring loop = pincer_r1_probe.py verbatim frame (k=8 T-sampled
predecessors, VALID iff p != t and the FORWARD step p -> t
verifies — the bidirectional-cheat fence). Differences from the
frozen probe, exactly three: (1) state construction (above);
(2) engine-step child minting runs FORK-ISOLATED (solve_isolated
pattern — successors on model-free band text, but the fence is
free); (3) sidecar streams one row per state.

    python scratch/pincer_r1_indist.py <ckpt> <label>
Sidecar: logs/pincer/indist_<label>.jsonl
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
import step_grpo_micro as G  # noqa: E402
from bench_verify_fast import verify_wave  # noqa: E402

K = 8
DEADLINE = float(os.environ.get("DEADLINE", "20"))
ckpt, label = sys.argv[1], sys.argv[2]
norm = lambda s: s.replace(" ", "")  # noqa: E731


def _mint_child(prob, q):
    """One forward engine step; first encodable child (fork)."""
    import sympy as sp
    from llmopt.search.derivation import State, successors
    try:
        cur = sp.sympify(prob)
        for _name, ch in successors(State(cur), use_macros=True):
            s = sp.sstr(ch.expr)
            if "Subs(" in s:
                continue
            q.put(("ok", s))
            return
        q.put(("none", None))
    except Exception as e:
        q.put(("err", str(e)[:120]))


def main():
    out_p = Path(f"logs/pincer/indist_{label}.jsonl")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        raise SystemExit(f"REFUSING: {out_p} exists")

    tok = MathTokenizer()
    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                        ffn=1024).to(dev)
    model.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    # in-dist states: one engine step off each gate-band problem,
    # minted with the gate's own recipe (fork-isolated already)
    import sympy as sp
    from bench_step_tokens import _gen_isolated
    probs = []
    for lv in G.GATE_LEVELS:
        for i in range(G.GATE_N):
            p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
            if p is not None:
                probs.append((lv, f"Integral({sp.sstr(p._expr)}, x)"))
    ctx = mp.get_context("fork")
    states, skipped = [], {"none": 0, "err": 0, "timeout": 0,
                           "unencodable": 0}
    for lv, prob in probs:
        q = ctx.Queue()
        p = ctx.Process(target=_mint_child, args=(prob, q))
        p.start()
        p.join(DEADLINE)
        if p.is_alive():
            p.kill()
            p.join()
            skipped["timeout"] += 1
            continue
        try:
            status, child = q.get_nowait()
        except Exception:
            status, child = "err", None
        if status != "ok":
            skipped[status] += 1
            continue
        try:
            tok.encode(child)
        except ValueError:
            skipped["unencodable"] += 1
            continue
        states.append((lv, child))
    print(f"{len(states)} in-dist states | skipped {skipped}",
          flush=True)

    cand_valid = cand_tried = 0
    per_lv = {}
    with out_p.open("w") as out, torch.no_grad():
        for si, (lv, t) in enumerate(states):
            prompt = tok.encode(f"Current: {t}\nHints: none\nStep: ")
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [G.GATE_BAND + si * 31 + b for b in range(K)], dev)
            distinct = [p for p in dict.fromkeys(texts)
                        if p and norm(p) != norm(t)]
            cand_tried += len(texts)
            good = []
            for p in distinct:
                wv = verify_wave(p, [t])
                if wv.get(t, (False, False))[0]:
                    good.append(p)
            cand_valid += len(good)
            cov, n = per_lv.get(lv, (0, 0))
            per_lv[lv] = (cov + bool(good), n + 1)
            out.write(json.dumps({
                "level": lv, "state": t,
                "n_distinct": len(distinct),
                "n_valid": len(good), "preds": good[:3],
            }) + "\n")
            out.flush()
    cov = {lv: f"{c}/{n}" for lv, (c, n) in sorted(per_lv.items())}
    print(f"[{label}] INDIST per-cand validity "
          f"{cand_valid}/{cand_tried} = "
          f"{100 * cand_valid / max(cand_tried, 1):.1f}% | "
          f"coverage {cov}", flush=True)


if __name__ == "__main__":
    main()
