"""TENET D1: THE REVERSE GATE (spec 2026-08-05-tenet-battery.md).

Prior verdicts measured backward capability only through
forward-facing instruments (RESULTS 9737-9741 names the gap). This
is the missing instrument: a fixed 120-problem battery (same band
seeds as the forward gate: GATE_BAND + 1000*lv + i, levels 3-7) on
the REVERSED prompt frame. The model sees a band expression as
"Current:" and must emit a PREDECESSOR as "Step:"; scoring is
forward verification at the MINT: s must be a legal engine child of
the candidate (successors replay, the D2 mechanism, fork-boxed line
server). Never corpus match (bidirectional-cheat fence). Verified
AND distinct: identity emissions never score.

INSTRUMENT CORRECTION (2026-08-05, caught in smoke BEFORE any arm
fired): the first cut scored by verify_wave(cand, [s]) — but that
oracle checks derivative-EQUIVALENCE, which is SYMMETRIC and
direction-blind: a forward model's emitted successor of s verifies
as a "predecessor" (smoke: 4/10 on a forward checkpoint). The
replay criterion breaks the symmetry (a parent is not a child of
its child). Equivalence validity is still reported as a DIAGNOSTIC
(the R1a 11% emission-validity anchor), never as the solve
criterion. Booked as AMENDMENT TENET-R0-REV-D1.

Single-ply by design: backward emission validity is the R1a metric
(11% booked); chains come later rungs. Solved = any of B=8 samples
verified-distinct. Prints the weights sha first (provenance rule
2026-07-31). Sigma fence as the forward gate (~5 on 120).

Usage (standalone gate of an existing checkpoint):
    CKPT=checkpoints/sym_birth_dense_revcert.pt \
        .venv/bin/python scratch/tenet_d1_revgate.py
Importable: rev_gate_eval(model, tok, dev).
"""
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")


def rev_gate_eval(model, tok, dev, n=None):
    """Reverse gate: solves per level + per-candidate equivalence %.

    Same shape as G.gate_eval's return so fences transfer:
    ({level: solves}, valid%) — but valid% here is the DIAGNOSTIC
    equivalence rate (see docstring), solves are replay-scored.
    Timeouts/membombs/crashes in the replay server default to
    reject (conservative; the reason travels in the counters)."""
    import hashlib

    import sympy as sp
    import torch

    import step_grpo_micro as G
    from bench_step_tokens import _gen_isolated
    from bench_verify_fast import verify_wave
    from tenet_d2_revdiet import Replayer

    wh = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        v = v.detach().cpu().contiguous()
        if v.dtype == torch.bfloat16:
            v = v.view(torch.int16)
        wh.update(v.numpy().tobytes())
    print(f"[revgate] weights sha {wh.hexdigest()[:16]}", flush=True)
    solves = {}
    valid = tried = skipped = 0
    replayer = Replayer()
    counts = {s: 0 for s in ("unique", "ambig", "miss", "err",
                             "timeout", "membomb", "crash")}
    replay_cache = {}
    with torch.no_grad():
        for lv in G.GATE_LEVELS:
            s_lv = 0
            for i in range(n or G.GATE_N):
                p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
                if p is None:
                    skipped += 1
                    continue
                s = f"Integral({sp.sstr(p._expr)}, x)"
                s_norm = s.replace(" ", "")
                prompt = tok.encode(
                    f"Current: {s}\nHints: none\nStep: ")
                texts, _, _ = G.sample_wave_lp(
                    model, tok, prompt,
                    [G.GATE_BAND + 1000 * lv + i * 31 + b
                     for b in range(G.B)], dev)
                tried += len(texts)
                done = False
                for cand in dict.fromkeys(texts):
                    c_norm = cand.replace(" ", "") if cand else ""
                    if not c_norm or c_norm == s_norm:
                        continue  # verified AND distinct doctrine
                    # diagnostic only: symmetric equivalence oracle
                    ok, _ = verify_wave(cand, [s]).get(s, (False, False))
                    valid += ok
                    if done:
                        continue  # solve decided; skip replay cost
                    st = replay_cache.get((c_norm, s_norm))
                    if st is None:
                        st = replayer.check_sync(cand, s, counts)
                        replay_cache[(c_norm, s_norm)] = st
                    if st in ("unique", "ambig"):  # s is a legal
                        done = True                # child of cand
                s_lv += done
            solves[lv] = s_lv
            print(f"[revgate] L{lv}: {s_lv}/{n or G.GATE_N}",
                  flush=True)
    replayer.kill()
    if skipped:
        print(f"[revgate] SKIPPED {skipped} band problems "
              f"(generator returned None)", flush=True)
    anomalies = {k_: v for k_, v in counts.items()
                 if k_ in ("timeout", "membomb", "crash", "err") and v}
    print(f"[revgate] replay census {counts}"
          + (f" | ANOMALIES {anomalies}" if anomalies else ""),
          flush=True)
    return solves, 100 * valid / max(tried, 1)


if __name__ == "__main__":
    import torch

    from llmopt.train.mathnative import MathTokenizer, build_model

    CKPT = os.environ["CKPT"]
    D = int(os.environ.get("D", "64"))
    LAYERS = int(os.environ.get("LAYERS", "8"))
    FFN = int(os.environ.get("FFN", "256"))
    HEADS = int(os.environ.get("HEADS", "4"))
    tok = MathTokenizer()
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                     weights_only=True))
    model.eval()
    solves, valid = rev_gate_eval(model, tok, dev)
    print(f"[revgate] {CKPT}: {solves} = "
          f"{sum(solves.values())}/120 @ {valid:.2f}% per-cand valid",
          flush=True)
