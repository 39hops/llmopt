"""ROUTE-TIME-0R (desk, frozen pre-run): isolated-stream rerun
closing the three seams the outside audit found in the frozen
scratch/routedb_time.py (results-cited, imported never edited):

 1. HOLDOUT ISOLATION: build_rows ran on the full alternating
    even/odd stream, so train-row features carried test-prompt
    history and train labels could resolve inside test prompts.
    Here train_events = even prompts only and test_events = odd
    prompts only, with build_rows called INDEPENDENTLY on each
    stream (this also closes the fit-full/serve-test index-scale
    mismatch already named in the verdict: fit and serve now both
    live on own-stream indices).
 2. CURRENT-TOKEN PROTECTION in the closed loop: the sequential
    per-expert loop could evict an expert still required later in
    the same token's top-8. All replayed policies here (LEARNED,
    LRU) protect the current token's top-8 from eviction, at
    K=32 and K=48, so the comparison is internally consistent
    (frozen replay2 numbers are NOT overwritten; this receipt has
    its own protected LRU baselines).
 3. Same arms/horizons as ROUTE-TIME-0 so the amendment can
    quote isolated-v-entangled deltas directly.

Receipt: logs/routetime/time0r_receipt.json (refuse-if-exists).

    .venv/bin/python scratch/routedb_time0r.py             (Mac desk)
"""
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

_spec = importlib.util.spec_from_file_location(
    "routedb_time", Path(__file__).parent / "routedb_time.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)
r1 = rt.r1

HS = rt.HS
N_LAYERS = rt.N_LAYERS
FEATS = rt.FEATS
OUT = Path("logs/routetime/time0r_receipt.json")


def closed_loop_protected(train_events, test_events, policy,
                          w, mu, sd, k, eb):
    """Replay test_events with warm start fit on train_events.
    policy in ("LEARNED", "LRU"). The current token's top-8 per
    layer is protected from eviction."""
    plain = defaultdict(Counter)
    for _, _, _, layers in train_events:
        for li, tk in layers.items():
            plain[li].update(tk)
    caches = [dict() for _ in range(N_LAYERS)]
    for li in range(N_LAYERS):
        for e, _ in plain[li].most_common(k):
            caches[li][e] = 0
    last = {}
    cnt = defaultdict(int)
    frq = defaultdict(float)
    gsum = defaultdict(int)
    lgap = {}
    prev_active = set()
    bytes_ = {"prefill": 0.0, "decode": 0.0}
    n_tok = {"prefill": 0, "decode": 0}
    for i, (p, _, ph, layers) in enumerate(test_events):
        phn = "decode" if rt.dec(ph) else "prefill"
        n_tok[phn] += 1
        cur_active = set()
        for li, tk in layers.items():
            c = caches[li]
            protect = set(tk)
            for e in tk:
                key = (li, e)
                if e in c:
                    c[e] = i
                else:
                    bytes_[phn] += eb
                    if len(c) >= k:
                        pool = [x for x in c if x not in protect]
                        if policy == "LRU":
                            victim = min(pool, key=lambda x: c[x])
                        else:
                            F = np.zeros((len(pool), len(FEATS)),
                                         dtype=np.float32)
                            for a, x in enumerate(pool):
                                kx = (li, x)
                                F[a] = (i - last[kx] if kx in last
                                        else i + 1,
                                        frq[kx], cnt[kx],
                                        lgap.get(kx, 0),
                                        gsum[kx] / max(cnt[kx], 1),
                                        float(kx in prev_active),
                                        float(rt.dec(ph)), li)
                            pr = rt.predict(w, mu, sd, F)
                            victim = pool[int(pr.argmin())]
                        del c[victim]
                    c[e] = i
                cur_active.add(key)
                if key in last:
                    g = i - last[key]
                    lgap[key] = g
                    gsum[key] += g
                last[key] = i
                cnt[key] += 1
                frq[key] = frq[key] * 0.99 + 1
        prev_active = cur_active
    return {ph: round(bytes_[ph] / 1e6 / max(n_tok[ph], 1), 2)
            for ph in ("prefill", "decode")}


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/routedb_time0r.py", "scratch/routedb_time.py",
         "scratch/routedb_replay.py", "logs/routetime/"
         "time0_receipt.json", "logs/routedb/replay2_receipt.json"])
    rng = np.random.default_rng(0)
    eb = r1.expert_bytes()
    results = {}
    deltas_h8 = []
    for tname, path in r1.TRACES.items():
        events = r1.load_events(path)
        tr_ev = [ev for ev in events if ev[0] % 2 == 0]
        te_ev = [ev for ev in events if ev[0] % 2 == 1]
        Xtr, Ttr, EItr, _, _, LEtr = rt.build_rows(tr_ev)
        Xte, Tte, EIte, _, DECte, LEte = rt.build_rows(te_ev)
        rt.qualify(tr_ev, Xtr, Ttr, EItr, LEtr, rng)
        rt.qualify(te_ev, Xte, Tte, EIte, LEte, rng)
        res = {"n_train_rows": int(len(Ttr)),
               "n_test_decode_rows": int(DECte.sum())}
        per_h = {}
        model32 = None
        prev_p = None
        for H in HS:
            ytr = Ttr <= H
            yte = Tte <= H
            frac = float(yte[DECte].mean())
            if prev_p is not None:
                assert frac >= prev_p - 1e-9, ("q1 FAIL", tname, H)
            prev_p = frac
            arms = {}
            cols = {"AGE-ONLY": [0], "AGE+FREQ": [0, 1],
                    "FULL": list(range(len(FEATS)))}
            for arm, ci in cols.items():
                w, mu, sd = rt.fit_logistic(Xtr[:, ci], ytr)
                s = rt.predict(w, mu, sd, Xte[DECte][:, ci])
                arms[arm] = round(rt.auc(s, yte[DECte]), 4)
                if arm == "FULL" and H == 32:
                    model32 = (w, mu, sd)
            arms["pos_frac"] = round(frac, 4)
            per_h[f"H{H}"] = arms
        res["auc"] = per_h
        d = per_h["H8"]["FULL"] - per_h["H8"]["AGE-ONLY"]
        res["delta_full_minus_age_H8"] = round(d, 4)
        deltas_h8.append(d)
        w, mu, sd = model32
        cl = {}
        cl["LEARNED_K32"] = closed_loop_protected(
            tr_ev, te_ev, "LEARNED", w, mu, sd, 32, eb)
        cl["LRU_K32"] = closed_loop_protected(
            tr_ev, te_ev, "LRU", None, None, None, 32, eb)
        cl["LRU_K48"] = closed_loop_protected(
            tr_ev, te_ev, "LRU", None, None, None, 48, eb)
        res["closed_loop_protected"] = cl
        results[tname] = res
        print(f"[rt0r] {tname}: H8 delta {d:+.4f} | learned32 "
              f"{cl['LEARNED_K32']['decode']} lru32 "
              f"{cl['LRU_K32']['decode']} lru48 "
              f"{cl['LRU_K48']['decode']}", flush=True)
    med = float(np.median(deltas_h8))
    print(f"[rt0r] median H8 delta (isolated) = {med:.4f}",
          flush=True)
    rcpt = {"note": "ROUTE-TIME-0R: isolated even-train/odd-test "
                    "streams (independent build_rows), current-"
                    "token-protected closed loop, own protected "
                    "LRU baselines; amendment evidence for "
                    "ROUTE-TIME-0",
            "start": START, "completion_commit": completion_commit(),
            "features": FEATS, "horizons": list(HS),
            "per_expert_bytes": eb,
            "qualification": {"q1": "pass",
                              "q2_n": 1000, "q3_n": 200,
                              "rng_seed": 0,
                              "streams": "train and test audited "
                                         "independently"},
            "median_delta_H8_isolated": round(med, 4),
            "results": results}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
