"""ROUTE-DB rung 1: frozen-trace expert-cache replay (desk,
observation-only, zero model cost — the EXPERTDB bank's pricing
rung). POLICIES AND PRICING CONSTANTS ARE FROZEN HERE, before any
replay value is read (the commit carrying this file predates the
receipt).

Traces: the six frozen routing trajectories (math
moe_gt1_traj_v2 + code/phys/proofs/prose/dialog), per
(prompt, layer, pos) top-8 expert selections with phase labels.
Replay: per layer an expert cache of capacity K; a token's
selections not resident are MISSES and are loaded (evicting per
policy) before the token can complete. Per-expert bytes derived
from the pinned snapshot's safetensors headers (fused switch_mlp
tensors / 128 experts).

POLICIES (all per-layer caches):
  LRU          least-recently-selected eviction
  LFU          decayed frequency (count *= 0.99 per token step)
  STATIC-DEMAND  offline per-layer top-K by whole-trace frequency,
               never evicts (implementable: demand table is a
               frozen artifact) — the EXPERTDB static baseline
  PHASE-STATIC same, but separate prefill/decode top-K tables,
               swapped at the phase boundary (EX6-motivated)
  PROMPT-PIN   during each prompt's prefill, pin every expert the
               prompt routed (LRU over the remainder); at decode,
               evict only non-pinned (predicts decode residency
               from own prefill)
  BELADY       clairvoyant farthest-next-use eviction (upper
               bound, not implementable)

BUDGETS: K in {16, 32, 48, 64, 96} of 128 experts/layer.

PREFETCH CENSUS (policy-independent): predictor "experts used by
the PREVIOUS token at layer l+d" scored against the current
token's actual selections at l+d, d in {1, 2, 4} — the temporal-
locality ceiling for lookahead prefetch; precision/recall over
decode rows.

PRICING CONSTANTS (frozen): SSD 4.2 GB/s single-stream / 8.3 GB/s
QD4 (measured 2026-08-21, EXACT-BF16 bank); stall = miss bytes /
bandwidth. PROMOTION READ (named before running): the cache
concept promotes if an IMPLEMENTABLE policy at K <= 32 (~4.3GB
expert-resident at the derived per-expert bytes — 10GB-3080
plausible) holds DECODE-phase miss traffic under 110 MB/token
(<= ~26 ms/token stall at QD1); kill if every implementable
policy at every K <= 48 exceeds twice that. Numbers book either
way.

Receipt: logs/routedb/replay_receipt.json (refuse-if-exists).

    .venv/bin/python scratch/routedb_replay.py             (Mac desk)
"""
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

TRACES = {"math": "logs/opus/moe_gt1_traj_v2.jsonl",
          "code": "logs/opus/gt2_code_traj.jsonl",
          "phys": "logs/opus/gt2_phys_traj.jsonl",
          "proofs": "logs/opus/gt3_proofs_traj.jsonl",
          "prose": "logs/opus/gt3_prose_traj.jsonl",
          "dialog": "logs/opus/gt4_dialog_traj.jsonl"}
KS = [16, 32, 48, 64, 96]
N_LAYERS, N_EXP, TOPK = 48, 128, 8
SSD = {"qd1_gbs": 4.2, "qd4_gbs": 8.3}
OUT = Path("logs/routedb/replay_receipt.json")
SNAP = (Path.home() / ".cache/huggingface/hub/"
        "models--mlx-community--Qwen3-30B-A3B-4bit/snapshots/"
        "d388dead1515f5e085ef7a0431dd8fadf0886c57")


def expert_bytes():
    """Per-expert bytes from safetensors headers (fused switch_mlp
    tensors summed over one layer, / 128)."""
    per_layer = defaultdict(int)
    for shard in sorted(SNAP.glob("model-*.safetensors")):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, meta in hdr.items():
            if "switch_mlp" in name:
                li = int(name.split("layers.")[1].split(".")[0])
                a, b = meta["data_offsets"]
                per_layer[li] += b - a
    assert len(per_layer) == N_LAYERS
    sizes = set(per_layer.values())
    assert len(sizes) == 1, sizes
    return per_layer[0] / N_EXP


def load_events(path):
    """[(prompt, pos, phase, [per-layer topk])] token events."""
    tok = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not isinstance(r["prompt"], int):
                continue
            key = (r["prompt"], r["pos"], r["phase"])
            tok.setdefault(key, {})[r["layer"]] = r["topk"]
    events = []
    for (p, pos, ph), layers in sorted(tok.items()):
        events.append((p, pos, ph, layers))
    return events


def replay(events, policy, k):
    caches = [dict() for _ in range(N_LAYERS)]   # expert -> meta
    freq = [Counter() for _ in range(N_LAYERS)]
    pinned = [set() for _ in range(N_LAYERS)]
    misses = {"prefill": 0, "decode": 0}
    sels = {"prefill": 0, "decode": 0}
    # static tables computed from the whole trace (frozen artifact
    # analogue); phase-static gets per-phase tables
    if policy in ("STATIC-DEMAND", "PHASE-STATIC"):
        counts = defaultdict(Counter)
        for _, _, ph, layers in events:
            for li, tk in layers.items():
                key = (li, ph) if policy == "PHASE-STATIC" else li
                counts[key].update(tk)
        static = {key: set(e for e, _ in c.most_common(k))
                  for key, c in counts.items()}
    if policy == "BELADY":
        nxt = defaultdict(list)   # (layer, expert) -> event indices
        for i, (_, _, _, layers) in enumerate(events):
            for li, tk in layers.items():
                for e in tk:
                    nxt[(li, e)].append(i)
        ptr = defaultdict(int)
    t = 0
    cur_prompt = None
    for i, (p, pos, ph, layers) in enumerate(events):
        t += 1
        ph2 = "prefill" if ph in ("prefill", "prompt_tail") else "decode"
        if policy == "PROMPT-PIN" and p != cur_prompt:
            cur_prompt = p
            pinned = [set() for _ in range(N_LAYERS)]
        for li, tk in layers.items():
            if policy in ("STATIC-DEMAND", "PHASE-STATIC"):
                key = (li, ph2) if policy == "PHASE-STATIC" else li
                table = static.get(key, set())
                for e in tk:
                    sels[ph2] += 1
                    if e not in table:
                        misses[ph2] += 1
                continue
            c = caches[li]
            for e in tk:
                sels[ph2] += 1
                if policy == "PROMPT-PIN" and ph2 == "prefill":
                    pinned[li].add(e)
                if e in c:
                    c[e] = t
                    freq[li][e] = freq[li][e] * 0.99 + 1
                    continue
                misses[ph2] += 1
                if len(c) >= k:
                    if policy == "LRU":
                        victim = min(c, key=c.get)
                    elif policy == "LFU":
                        victim = min(c, key=lambda x: freq[li][x])
                    elif policy == "PROMPT-PIN":
                        cand = [x for x in c if x not in pinned[li]]
                        victim = (min(cand, key=c.get) if cand
                                  else min(c, key=c.get))
                    elif policy == "BELADY":
                        def next_use(x):
                            uses = nxt[(li, x)]
                            j = ptr[(li, x)]
                            while j < len(uses) and uses[j] <= i:
                                j += 1
                            ptr[(li, x)] = j
                            return uses[j] if j < len(uses) else 1 << 60
                        victim = max(c, key=next_use)
                    del c[victim]
                c[e] = t
                freq[li][e] = freq[li][e] * 0.99 + 1
    return misses, sels


def prefetch_census(events):
    """Previous-token same-layer predictor at lookahead d (the
    temporal-locality ceiling), decode rows only."""
    out = {}
    prev = {}
    stats = {d: [0, 0, 0] for d in (1, 2, 4)}   # tp, pred, actual
    for p, pos, ph, layers in events:
        ph2 = "prefill" if ph in ("prefill", "prompt_tail") else "decode"
        if ph2 == "decode":
            for d in (1, 2, 4):
                for li, tk in layers.items():
                    if li < d:
                        continue
                    pred = prev.get((p, li))
                    if pred is None:
                        continue
                    tp = len(set(tk) & pred)
                    stats[d][0] += tp
                    stats[d][1] += len(pred)
                    stats[d][2] += len(tk)
        for li, tk in layers.items():
            prev[(p, li)] = set(tk)
    for d, (tp, npred, nact) in stats.items():
        out[f"d{d}"] = {"precision": round(tp / max(npred, 1), 4),
                        "recall": round(tp / max(nact, 1), 4)}
    return out


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/routedb_replay.py"])
    eb = expert_bytes()
    print(f"[rdb] per-expert bytes {eb:,.0f} "
          f"({eb * N_EXP * N_LAYERS / 1e9:.2f} GB all experts)",
          flush=True)
    results = {}
    prefetch = {}
    for tname, path in TRACES.items():
        events = load_events(path)
        n_tok = {"prefill": 0, "decode": 0}
        for _, _, ph, _ in events:
            n_tok["prefill" if ph in ("prefill", "prompt_tail")
                  else "decode"] += 1
        prefetch[tname] = prefetch_census(events)
        results[tname] = {"n_token_events": n_tok}
        for policy in ("LRU", "LFU", "STATIC-DEMAND", "PHASE-STATIC",
                       "PROMPT-PIN", "BELADY"):
            for k in KS:
                misses, sels = replay(events, policy, k)
                row = {}
                for ph in ("prefill", "decode"):
                    mb_tok = (misses[ph] * eb / 1e6
                              / max(n_tok[ph], 1))
                    row[ph] = {
                        "miss_rate": round(
                            misses[ph] / max(sels[ph], 1), 4),
                        "mb_per_token": round(mb_tok, 2),
                        "stall_ms_qd1": round(
                            mb_tok / SSD["qd1_gbs"], 2),
                        "stall_ms_qd4": round(
                            mb_tok / SSD["qd4_gbs"], 2)}
                results[tname][f"{policy}@K{k}"] = row
            print(f"[rdb] {tname} {policy} done", flush=True)
    rcpt = {"note": "ROUTE-DB rung 1: frozen-trace expert-cache "
                    "replay (desk pricing; policies + constants "
                    "frozen pre-run in the committed script)",
            "start": START, "completion_commit": completion_commit(),
            "per_expert_bytes": eb, "budgets_K": KS,
            "ssd_gbs": SSD, "results": results,
            "prefetch_prev_token_census": prefetch}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    print(f"receipt -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
