"""Precompute for the expert-atlas animatic (storyboard 2026-08-13).

Emits data/anim/atlas.npz. Every field is measured; nothing here is
styled or invented.

  order        (48,128) per-layer PIOOLED demand ranking, IDENTICAL to
               the ordering scratch/ex3_build.py uses to define each
               carrier's +-8 control window (source: moe_gt1_arm0.json
               counts). Column index = demand rank within the layer.
  rate_prefill (48,128) selection RATE = picks / routing opportunities
  rate_decode  (48,128) same, decode rows only
               Both indexed by RANK COLUMN, not raw expert id, so the
               column ordering is fixed across phases by construction.
  carriers     (80,2)  [layer, rank_col] of the pooled-lens carriers
  controls     (80,2)  [layer, rank_col] of each carrier's matched
               control, from replaying the ex3_build sampler (verified
               byte-identical to checkpoints/ex3_del_rand0.json)
  trace        (48,8)  rank columns fired by ONE prefill token, chosen
               by a fixed rule (see TRACE_RULE in meta)
  trace_scores (48,8)  that token's gate scores

Usage: .venv/bin/python scratch/atlas_precompute.py
"""
from __future__ import annotations

import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TRAJ = "logs/opus/moe_gt1_traj_v2.jsonl"
L, E, K = 48, 128, 8
OUT = Path("data/anim/atlas.npz")


def main() -> None:
    arm0 = json.loads(Path("checkpoints/moe_gt1_arm0.json").read_text())
    pooled = {int(l): r for l, r in arm0["counts"].items()}
    # The sampler's ordering, verbatim (ex3_build.py:123-124).
    order = np.array([sorted(range(E), key=lambda x: (-pooled[l][x], x))
                      for l in range(L)], dtype=np.int32)
    col_of = np.zeros((L, E), dtype=np.int32)
    for l in range(L):
        for c, e in enumerate(order[l]):
            col_of[l, e] = c

    picks = {"prefill": np.zeros((L, E)), "decode": np.zeros((L, E))}
    events = {"prefill": Counter(), "decode": Counter()}
    plen: dict[int, set] = defaultdict(set)
    rows: dict[tuple, tuple] = {}
    # Accumulation snapshots: the prefill field as it stood after each
    # 1/NSNAP of prefill events IN FILE ORDER (prompt-major). Real
    # partial sums, so the "painting in" beat is measured, not eased.
    NSNAP = 8
    snaps: list = []
    nprefill = 0
    with open(TRAJ) as f:
        for line in f:
            r = json.loads(line)
            ph, l = r["phase"], r["layer"]
            events[ph][l] += 1
            for e in r["topk"]:
                picks[ph][l, e] += 1
            if ph == "prefill":
                nprefill += 1
                plen[r["prompt"]].add(r["pos"])
                rows[(r["prompt"], r["pos"], l)] = (r["topk"], r["scores"])
    total_prefill = nprefill
    step = total_prefill // NSNAP
    nprefill = 0
    acc = np.zeros((L, E))
    with open(TRAJ) as f:
        for line in f:
            r = json.loads(line)
            if r["phase"] != "prefill":
                continue
            nprefill += 1
            for e in r["topk"]:
                acc[r["layer"], e] += 1
            if nprefill % step == 0 and len(snaps) < NSNAP:
                snaps.append((acc / max(nprefill / L, 1)).copy())

    # Opportunities are per (layer, phase); assert layer-uniformity so a
    # single per-phase denominator is legitimate.
    for ph in events:
        v = set(events[ph].values())
        assert len(v) == 1, f"{ph} events vary by layer: {sorted(v)[:5]}"
    opp = {ph: next(iter(events[ph].values())) for ph in events}

    rate = {}
    for ph in ("prefill", "decode"):
        r = picks[ph] / opp[ph]
        # reindex raw expert id -> rank column
        rr = np.zeros((L, E))
        for l in range(L):
            rr[l, col_of[l]] = r[l]
        rate[ph] = rr

    # TRACE RULE (fixed, stated, not aesthetic): the lowest-indexed
    # prompt whose prefill length equals the battery MEDIAN prefill
    # length; token = that prompt's final prefill position.
    lens = {p: len(s) for p, s in plen.items()}
    med = int(np.median(list(lens.values())))
    cand = sorted(p for p, n in lens.items() if n == med)
    if not cand:                       # median falls between two lengths
        med = min(lens.values(), key=lambda n: (abs(n - med), n))
        cand = sorted(p for p, n in lens.items() if n == med)
    tp = cand[0]
    tpos = max(plen[tp])
    trace = np.zeros((L, K), dtype=np.int32)
    tscore = np.zeros((L, K))
    for l in range(L):
        tk, sc = rows[(tp, tpos, l)]
        trace[l] = [col_of[l, e] for e in tk]
        tscore[l] = sc

    carriers = json.loads(Path("checkpoints/ex3_inv_pooled.json").read_text())
    inv_p = {tuple(x) for x in carriers}
    # Replay the sampler (ex3_build.py:119-130) to recover the PAIRING.
    rng = random.Random("ex3-rand-0")
    deleted: set = set()
    pairs = []
    for (l, e) in sorted(inv_p):
        o = sorted(range(E), key=lambda x: (-pooled[l][x], x))
        rank = o.index(e)
        window = [x for x in o[max(0, rank - 8):rank + 9]
                  if (l, x) not in inv_p and (l, x) not in deleted and x != e]
        c = rng.choice(window)
        deleted.add((l, c))
        pairs.append((l, e, c))
    # Certify against the cited artifact.
    cited = json.loads(Path("checkpoints/ex3_del_rand0.json").read_text())
    rebuilt = {str(l): sorted(set(range(E)) - {e for (ll, e) in deleted
                                               if ll == l}) for l in range(L)}
    assert rebuilt == cited, "control replay does NOT match ex3_del_rand0"
    print("control pairing: replay MATCHES ex3_del_rand0.json")

    # reindex accumulation snapshots into rank columns too
    accum = np.zeros((len(snaps), L, E))
    for i, sn in enumerate(snaps):
        for l in range(L):
            accum[i, l, col_of[l]] = sn[l]

    car = np.array([[l, col_of[l, e]] for (l, e, _) in pairs], dtype=np.int32)
    ctl = np.array([[l, col_of[l, c]] for (l, _, c) in pairs], dtype=np.int32)

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    from llmopt.figures import figstyle
    from matplotlib.colors import to_hex
    ramp = {m: [to_hex(figstyle.continuous("magnitude", m)(i / 15))
                for i in range(16)] for m in ("light", "dark")}
    meta = {
        "head": head,
        "provenance": (f"{TRAJ} · ex3_inv_pooled/ex3_del_rand0 · "
                       f"VERDICT EX-FRESH RESULTS.md#L22454"),
        "opp_prefill": opp["prefill"], "opp_decode": opp["decode"],
        "trace_rule": (f"prompt {tp}, final prefill position {tpos} "
                       f"(lowest-indexed prompt of median prefill "
                       f"length {med})"),
        "arms": {"full": 189, "control": 217, "carriers": 244},
        "unit": "pooled solves over 3 paired seeds, of 360",
        "accum_note": ("prefill field after each 1/8 of prefill events, "
                       "file order (prompt-major); real partial sums"),
        "ramp": ramp, "chrome": figstyle.CHROME,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, meta=np.array(json.dumps(meta)),
        order=order, rate_prefill=rate["prefill"], rate_decode=rate["decode"],
        carriers=car, controls=ctl, trace=trace, trace_scores=tscore,
        accum=accum)
    print(f"wrote {OUT}")
    print(f"opportunities  prefill {opp['prefill']}  decode {opp['decode']}")
    for ph in ("prefill", "decode"):
        r = rate[ph]
        print(f"{ph:>8} rate  max {r.max():.4f}  mean {r.mean():.4f}  "
              f"p99 {np.percentile(r, 99):.4f}  zeros {(r == 0).sum()}")
    d = np.abs(car[:, 1] - ctl[:, 1])
    print(f"carrier->control column hop: max {d.max()}  mean {d.mean():.2f}")
    print(f"carrier rank columns: min {car[:,1].min()} median "
          f"{int(np.median(car[:,1]))} max {car[:,1].max()}")


if __name__ == "__main__":
    main()
