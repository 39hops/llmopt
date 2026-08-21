"""ROUTE-TIME-0 driver (desk, frozen pre-run): finite-horizon
next-use prediction over the six frozen route traces, per PRE-REG
ROUTE-TIME-0 (docs/RESULTS.md).

One row per (token event, layer, activated expert). Labels
y_H = [T <= H] with T = token events until the expert's next
activation on that layer within the trace stream, H in
{1,2,4,8,16,32,64}. Features strict past-only (age, decayed freq,
count, last gap, mean gap, previous-token co-activation, phase,
layer). Split = replay2's deterministic prompt holdout (even train,
odd test). Arms per H: AGE-ONLY / AGE+FREQ / FULL numpy logistic;
ranking baselines age, freq, prev-token. Metric: test-prompt
decode-row AUC per trace.

Qualification fail-closed before any bar: (q1) P(T<=H) monotone in
H per trace; (q2) next-use labels match a direct recompute on a
1000-row sample; (q3) features match a truncated-stream recompute
on a 200-row sample.

BAR 1: median over traces of FULL - AGE-ONLY AUC at H=8 >= 0.02.
BAR 2 (only if BAR 1 fires): closed-loop K=32 replay, evict lowest
predicted P(T<=32); fires if learned@K32 decode MB/token <= warm
LRU@K48 on >= 4/6 traces. Otherwise books NOT-RUN.

Receipt: logs/routetime/time0_receipt.json (refuse-if-exists).

    .venv/bin/python scratch/routedb_time.py               (Mac desk)
"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

_spec = importlib.util.spec_from_file_location(
    "routedb_replay", Path(__file__).parent / "routedb_replay.py")
r1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r1)

HS = (1, 2, 4, 8, 16, 32, 64)
N_LAYERS = 48
BIG = 1 << 30
OUT = Path("logs/routetime/time0_receipt.json")
FEATS = ("age", "freq", "count", "last_gap", "mean_gap",
         "prev_tok", "phase", "layer")


def dec(ph):
    return ph not in ("prefill", "prompt_tail")


def build_rows(events):
    """Single causal pass. Returns feature matrix X, next-use gap T,
    event index, prompt id, decode flag, (layer, expert) per row.
    Features computed BEFORE updating state with the current event."""
    n = len(events) * N_LAYERS * 8
    X = np.zeros((n, len(FEATS)), dtype=np.float32)
    T = np.zeros(n, dtype=np.int64)
    EI = np.zeros(n, dtype=np.int64)
    P = np.zeros(n, dtype=np.int64)
    DEC = np.zeros(n, dtype=bool)
    LE = np.zeros((n, 2), dtype=np.int32)
    last = {}
    cnt = defaultdict(int)
    frq = defaultdict(float)
    gsum = defaultdict(int)
    lgap = {}
    prev_active = set()
    uses = defaultdict(list)
    for i, (_, _, _, layers) in enumerate(events):
        for li, tk in layers.items():
            for e in tk:
                uses[(li, e)].append(i)
    ptr = defaultdict(int)
    r = 0
    for i, (p, _, ph, layers) in enumerate(events):
        cur_active = set()
        for li, tk in layers.items():
            for e in tk:
                key = (li, e)
                age = i - last[key] if key in last else i + 1
                X[r] = (age, frq[key], cnt[key],
                        lgap.get(key, 0), gsum[key] / max(cnt[key], 1),
                        float(key in prev_active), float(dec(ph)), li)
                u = uses[key]
                j = ptr[key]
                while j < len(u) and u[j] <= i:
                    j += 1
                ptr[key] = j
                T[r] = (u[j] - i) if j < len(u) else BIG
                EI[r], P[r], DEC[r] = i, p, dec(ph)
                LE[r] = (li, e)
                r += 1
                cur_active.add(key)
                if key in last:
                    g = i - last[key]
                    lgap[key] = g
                    gsum[key] += g
                last[key] = i
                cnt[key] += 1
                frq[key] = frq[key] * 0.99 + 1
        prev_active = cur_active
    return X[:r], T[:r], EI[:r], P[:r], DEC[:r], LE[:r]


def fit_logistic(X, y, iters=150, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1] + 1, dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    Z1 = np.hstack([Z, np.ones((len(Z), 1), dtype=np.float32)])
    yy = y.astype(np.float64)
    for t in range(1, iters + 1):
        p = 1 / (1 + np.exp(-(Z1 @ w)))
        g = Z1.T @ (p - yy) / len(yy)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (
            np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w, mu, sd


def predict(w, mu, sd, X):
    Z = (X - mu) / sd
    return 1 / (1 + np.exp(-(Z @ w[:-1] + w[-1])))


def auc(score, y):
    order = np.argsort(score, kind="mergesort")
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    # midranks for ties
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rank[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rank[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def qualify(events, X, T, EI, LE, rng):
    # q1 handled by caller (monotone P). q2: next-use recompute.
    idx = rng.choice(len(T), size=min(1000, len(T)), replace=False)
    uses = defaultdict(list)
    for i, (_, _, _, layers) in enumerate(events):
        for li, tk in layers.items():
            for e in tk:
                uses[(li, e)].append(i)
    for r in idx:
        li, e = int(LE[r][0]), int(LE[r][1])
        i = int(EI[r])
        later = [u for u in uses[(li, e)] if u > i]
        want = (later[0] - i) if later else BIG
        assert want == T[r], ("q2 FAIL", r, want, int(T[r]))
    # q3: truncated-stream feature recompute on 200 rows
    idx3 = rng.choice(len(T), size=min(200, len(T)), replace=False)
    for r in sorted(int(x) for x in idx3):
        i = int(EI[r])
        li, e = int(LE[r][0]), int(LE[r][1])
        last = None
        c = 0
        f = 0.0
        gs = 0
        lg = 0
        prev_active = False
        for i2 in range(i):
            _, _, _, layers = events[i2]
            if li in layers and e in layers[li]:
                if last is not None:
                    g = i2 - last
                    lg = g
                    gs += g
                last = i2
                c += 1
                f = f * 0.99 + 1
        if i > 0:
            _, _, _, lay_prev = events[i - 1]
            prev_active = li in lay_prev and e in lay_prev[li]
        age = (i - last) if last is not None else i + 1
        want = (age, f, c, lg, gs / max(c, 1), float(prev_active))
        got = tuple(float(x) for x in X[r][:6])
        assert np.allclose(want, got, atol=1e-4), \
            ("q3 FAIL", r, want, got)
    return True


def closed_loop(events, w, mu, sd, k, eb):
    """K=32 replay, learned eviction: evict cached expert with
    lowest predicted P(T<=32), features maintained online.
    Warm start = train-fit top-K table (as replay2 LRU)."""
    from collections import Counter
    plain = defaultdict(Counter)
    for p, _, ph, layers in events:
        if p % 2 == 0:
            for li, tk in layers.items():
                plain[li].update(tk)
    caches = [dict() for _ in range(N_LAYERS)]
    for li in range(N_LAYERS):
        for e, _ in plain[li].most_common(k):
            caches[li][e] = True
    test = [ev for ev in events if ev[0] % 2 == 1]
    last = {}
    cnt = defaultdict(int)
    frq = defaultdict(float)
    gsum = defaultdict(int)
    lgap = {}
    prev_active = set()
    bytes_ = {"prefill": 0.0, "decode": 0.0}
    n_tok = {"prefill": 0, "decode": 0}
    for i, (p, _, ph, layers) in enumerate(test):
        phn = "decode" if dec(ph) else "prefill"
        n_tok[phn] += 1
        cur_active = set()
        for li, tk in layers.items():
            c = caches[li]
            for e in tk:
                key = (li, e)
                if e not in c:
                    bytes_[phn] += eb
                    if len(c) >= k:
                        pool = list(c)
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
                                    float(dec(ph)), li)
                        pr = predict(w, mu, sd, F)
                        del c[pool[int(pr.argmin())]]
                    c[e] = True
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
    START = start_provenance(["scratch/routedb_time.py",
                              "scratch/routedb_replay.py",
                              "scratch/routedb_replay2.py"])
    rng = np.random.default_rng(0)
    eb = r1.expert_bytes()
    results = {}
    deltas_h8 = []
    models32 = {}
    for tname, path in r1.TRACES.items():
        events = r1.load_events(path)
        X, T, EI, P, DEC, LE = build_rows(events)
        qualify(events, X, T, EI, LE, rng)
        tr = P % 2 == 0
        te_dec = (P % 2 == 1) & DEC
        prev_p = None
        res = {"n_rows": int(len(T)),
               "n_train": int(tr.sum()),
               "n_test_decode": int(te_dec.sum())}
        per_h = {}
        for H in HS:
            y = T <= H
            frac = float(y[te_dec].mean())
            if prev_p is not None:
                assert frac >= prev_p - 1e-9, ("q1 FAIL", tname, H)
            prev_p = frac
            arms = {}
            cols = {"AGE-ONLY": [0], "AGE+FREQ": [0, 1],
                    "FULL": list(range(len(FEATS)))}
            for arm, ci in cols.items():
                w, mu, sd = fit_logistic(X[tr][:, ci], y[tr])
                s = predict(w, mu, sd, X[te_dec][:, ci])
                arms[arm] = round(auc(s, y[te_dec]), 4)
                if arm == "FULL" and H == 32:
                    models32[tname] = (w, mu, sd)
            arms["rank_age"] = round(auc(-X[te_dec][:, 0],
                                         y[te_dec]), 4)
            arms["rank_freq"] = round(auc(X[te_dec][:, 1],
                                          y[te_dec]), 4)
            arms["rank_prevtok"] = round(auc(X[te_dec][:, 5],
                                             y[te_dec]), 4)
            arms["pos_frac"] = round(frac, 4)
            per_h[f"H{H}"] = arms
        res["auc"] = per_h
        d = per_h["H8"]["FULL"] - per_h["H8"]["AGE-ONLY"]
        res["delta_full_minus_age_H8"] = round(d, 4)
        deltas_h8.append(d)
        results[tname] = res
        print(f"[rt0] {tname}: H8 FULL {per_h['H8']['FULL']} "
              f"AGE {per_h['H8']['AGE-ONLY']} delta {d:+.4f}",
              flush=True)
    bar1_median = float(np.median(deltas_h8))
    bar1 = bar1_median >= 0.02
    print(f"[rt0] BAR1 median delta H8 = {bar1_median:.4f} "
          f"fire={bar1}", flush=True)
    bar2 = None
    if bar1:
        wins = 0
        for tname, path in r1.TRACES.items():
            events = r1.load_events(path)
            w, mu, sd = models32[tname]
            learned = closed_loop(events, w, mu, sd, 32, eb)
            lru48 = None  # from frozen replay2 receipt
            r2 = json.load(open("logs/routedb/replay2_receipt.json"))
            lru48 = r2["results"][tname]["K48"]["LRU"]["decode"]
            win = learned["decode"] <= lru48
            wins += win
            results[tname]["closed_loop_K32"] = {
                "learned_decode_MB_per_tok": learned["decode"],
                "lru48_decode_MB_per_tok": lru48, "win": bool(win)}
            print(f"[rt0] {tname}: learned32 {learned['decode']} "
                  f"v lru48 {lru48} win={win}", flush=True)
        bar2 = wins >= 4
        print(f"[rt0] BAR2 wins {wins}/6 fire={bar2}", flush=True)
    rcpt = {"note": "ROUTE-TIME-0: finite-horizon next-use "
                    "prediction, strict past-only features, prompt "
                    "holdout; BAR2 conditional on BAR1",
            "start": START, "completion_commit": completion_commit(),
            "features": FEATS, "horizons": list(HS),
            "per_expert_bytes": eb,
            "bar1_median_delta_H8": round(bar1_median, 4),
            "bar1_fire": bool(bar1),
            "bar2_fire": bar2 if bar2 is None else bool(bar2),
            "results": results}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
