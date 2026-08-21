"""ROUTE-DB rung 2: within-domain prompt-holdout replay with the
three -SIM corrections (desk, observation-only; design frozen by
this commit before any value is read). routedb_replay.py is
results-cited/frozen — imported for load_events/expert_bytes, never
edited.

CORRECTIONS IMPLEMENTED:
  charged transitions   static tables pay exact per-layer table-
                        difference bytes at EVERY phase boundary
                        (prefill->decode and back at the next
                        prompt), charged into the phase being
                        ENTERED.
  BELADY-BYPASS         clairvoyant with optional insertion (on a
                        miss, if the incoming expert's next use is
                        farthest of cache+item, bypass) and
                        clairvoyant warm start. Asserted <= every
                        implementable policy per (trace, K).
  equal warm starts     every policy starts preloaded: LRU/LFU
                        with the train-fit top-K table
                        (implementable), STATIC/PHASE-STATIC with
                        their own tables, BELADY-BYPASS with the
                        soonest-first-use K.

HOLDOUT: deterministic split — even prompt ids fit the tables,
odd prompt ids replay (no randomness; stated, not seeded). All
fitted quantities (tables, warm starts) see TRAIN prompts only;
every reported number is TEST prompts only.

SHARP TARGET (frozen): per trace, does transported+charged
PHASE-STATIC@K32 hold decode MB/token at or under plain warm
LRU@K48? Reported as the ratio phase32/lru48 per trace; <= 1.0 is
the "phase knowledge buys a budget tier" read.

Receipt: logs/routedb/replay2_receipt.json (refuse-if-exists).

    .venv/bin/python scratch/routedb_replay2.py            (Mac desk)
"""
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

_spec = importlib.util.spec_from_file_location(
    "routedb_replay", Path(__file__).parent / "routedb_replay.py")
r1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r1)

KS = [32, 48]
N_LAYERS, N_EXP = 48, 128
OUT = Path("logs/routedb/replay2_receipt.json")


def phase2(ph):
    return "prefill" if ph in ("prefill", "prompt_tail") else "decode"


def fit_tables(events, k):
    """Train-prompt top-K tables: plain and per-phase."""
    plain = defaultdict(Counter)
    phased = defaultdict(Counter)
    first_use = defaultdict(dict)
    for i, (p, _, ph, layers) in enumerate(events):
        if p % 2 != 0:
            continue
        for li, tk in layers.items():
            plain[li].update(tk)
            phased[(li, phase2(ph))].update(tk)
            for e in tk:
                first_use[li].setdefault(e, i)
    t_plain = {li: set(e for e, _ in c.most_common(k))
               for li, c in plain.items()}
    t_phase = {key: set(e for e, _ in c.most_common(k))
               for key, c in phased.items()}
    return t_plain, t_phase


def replay(events, policy, k, t_plain, t_phase, eb):
    test = [(p, pos, ph, layers) for (p, pos, ph, layers) in events
            if p % 2 == 1]
    n_tok = {"prefill": 0, "decode": 0}
    for _, _, ph, _ in test:
        n_tok[phase2(ph)] += 1
    bytes_ = {"prefill": 0.0, "decode": 0.0}
    if policy in ("STATIC", "PHASE-STATIC"):
        cur_phase = None
        for p, _, ph, layers in test:
            ph_ = phase2(ph)
            if policy == "PHASE-STATIC" and ph_ != cur_phase:
                # charge the table difference entering this phase
                if cur_phase is not None:
                    for li in range(N_LAYERS):
                        a = t_phase.get((li, cur_phase), set())
                        b = t_phase.get((li, ph_), set())
                        bytes_[ph_] += len(b - a) * eb
                cur_phase = ph_
            for li, tk in layers.items():
                table = (t_phase.get((li, ph_), set())
                         if policy == "PHASE-STATIC"
                         else t_plain.get(li, set()))
                for e in tk:
                    if e not in table:
                        bytes_[ph_] += eb
        return bytes_, n_tok
    # dynamic policies, warm-started
    if policy == "BELADY-BYPASS":
        nxt = defaultdict(list)
        for i, (_, _, _, layers) in enumerate(test):
            for li, tk in layers.items():
                for e in tk:
                    nxt[(li, e)].append(i)
        ptr = defaultdict(int)

        def next_use(li, x, i):
            uses = nxt[(li, x)]
            j = ptr[(li, x)]
            while j < len(uses) and uses[j] <= i:
                j += 1
            ptr[(li, x)] = j
            return uses[j] if j < len(uses) else 1 << 60
        caches = []
        for li in range(N_LAYERS):
            firsts = sorted(nxt, key=lambda le: nxt[le][0]
                            if nxt[le] else 1 << 60)
            mine = [e for (l2, e) in firsts if l2 == li][:k]
            caches.append(set(mine))
    else:
        caches = [set(t_plain.get(li, set())) for li in range(N_LAYERS)]
    stamp = [dict.fromkeys(c, 0) for c in caches]
    freq = [Counter({e: 1.0 for e in c}) for c in caches]
    t = 0
    for i, (p, _, ph, layers) in enumerate(test):
        t += 1
        ph_ = phase2(ph)
        for li, tk in layers.items():
            c = stamp[li]
            for e in tk:
                if e in c:
                    c[e] = t
                    freq[li][e] = freq[li][e] * 0.99 + 1
                    continue
                bytes_[ph_] += eb
                if policy == "BELADY-BYPASS":
                    pool = list(c) + [e]
                    victim = max(pool, key=lambda x: next_use(li, x, i))
                    if victim == e:
                        continue        # bypass: load, don't insert
                    del c[victim]
                    c[e] = t
                    continue
                if len(c) >= k:
                    victim = (min(c, key=c.get) if policy == "LRU"
                              else min(c, key=lambda x: freq[li][x]))
                    del c[victim]
                c[e] = t
                freq[li][e] = freq[li][e] * 0.99 + 1
    return bytes_, n_tok


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/routedb_replay2.py",
                              "scratch/routedb_replay.py"])
    eb = r1.expert_bytes()
    results = {}
    for tname, path in r1.TRACES.items():
        events = r1.load_events(path)
        results[tname] = {}
        for k in KS:
            t_plain, t_phase = fit_tables(events, k)
            per_pol = {}
            for policy in ("LRU", "LFU", "STATIC", "PHASE-STATIC",
                           "BELADY-BYPASS"):
                bytes_, n_tok = replay(events, policy, k,
                                       t_plain, t_phase, eb)
                per_pol[policy] = {
                    ph: round(bytes_[ph] / 1e6 / max(n_tok[ph], 1), 2)
                    for ph in ("prefill", "decode")}
            # optimality assertion: belady-bypass <= implementables
            for pol in ("LRU", "LFU", "STATIC", "PHASE-STATIC"):
                for ph in ("prefill", "decode"):
                    assert (per_pol["BELADY-BYPASS"][ph]
                            <= per_pol[pol][ph] + 0.01), \
                        (tname, k, pol, ph, per_pol)
            results[tname][f"K{k}"] = per_pol
        r32 = results[tname]["K32"]["PHASE-STATIC"]["decode"]
        r48 = results[tname]["K48"]["LRU"]["decode"]
        results[tname]["sharp_target_phase32_over_lru48"] = round(
            r32 / max(r48, 0.01), 3)
        print(f"[rdb2] {tname}: phase32 {r32} v lru48 {r48} "
              f"ratio {results[tname]['sharp_target_phase32_over_lru48']}",
              flush=True)
    rcpt = {"note": "ROUTE-DB rung 2: within-domain prompt holdout "
                    "(even fit / odd test), charged phase-boundary "
                    "table churn, BELADY-BYPASS warm-start optimum "
                    "asserted <= implementables",
            "start": START, "completion_commit": completion_commit(),
            "per_expert_bytes": eb, "budgets_K": KS,
            "holdout": "prompt_id_even_train_odd_test",
            "results": results}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
