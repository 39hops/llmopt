"""Independent recomputation of the E0 measurements from the raw rows
(logs/qwen_e0/<arm>/positions.jsonl), for comparison with summary.json
before adjudication. Reads rows only; writes nothing.

Definitions (kept apart):
  top-1 agreement   = mean over ALL live positions of [draft argmax == recorded token]
  replay yield      = emitted tokens per verify block in the accepted-prefix replay
                      (blocks of k drafts in stream order; emitted = accepted + 1 bonus
                      teacher token, except when the stream ends inside the block)
  accepted/drafted  = sum of accepted over all blocks / sum of drafted over all blocks
                      (drafted counts every proposed token, including the ones past the
                      first rejection that are never evaluated)
  evaluated         = accepted + one rejected position per rejecting block
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, ".")
from llmopt.lab import spec_reference as sr  # noqa: E402

EDGES = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, float("inf")]


def main(path="logs/qwen_e0/C"):
    rows = [json.loads(line) for line in open(f"{path}/positions.jsonl")]
    s = json.load(open(f"{path}/summary.json"))
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    order = [p["prompt_id"] for p in s["per_prompt"]]
    assert set(order) == set(by_prompt), "prompt coverage"
    for pid in order:
        steps = [r["step"] for r in by_prompt[pid]]
        assert steps == list(range(len(steps))), f"step gap in {pid}"
    agree = np.array([r["agree"] for r in rows], dtype=float)
    print(f"positions {len(rows)}  prompts {len(order)}  pooled agreement {agree.mean():.4f} "
          f"(summary {s['agree_rate_pooled']:.4f})")
    cat = defaultdict(list)
    for r in rows:
        cat[r["category"]].append(r["agree"])
    for c, v in sorted(cat.items()):
        print(f"  category {c:>6}: n={len(v):4d} agreement {np.mean(v):.4f}")
    t_all, d_all = [], []
    for pid in order:
        t_all += [r["teacher_token"] for r in by_prompt[pid]]
        d_all += [r["draft_argmax"] for r in by_prompt[pid]]
    print("replay (pooled, prompt order):")
    for k in range(1, 9):
        rep = sr.replay_blocks(t_all, d_all, k)
        acc = Counter(rep["accepted_per_block"])
        emi = Counter(rep["emitted_per_block"])
        tot_acc = sum(rep["accepted_per_block"])
        tot_dr = sum(rep["drafted_per_block"])
        ev = tot_acc + rep["rejected_blocks"]
        sm = s["replay_pooled"][str(k)]["mean_emitted"]
        print(f"  k={k}: blocks {rep['blocks']:4d} mean emitted {rep['mean_emitted']:.3f} (summary {sm:.3f}) "
              f"mean accepted {rep['mean_accepted']:.3f}  accepted/drafted {tot_acc/tot_dr:.3f}  "
              f"agreement over evaluated {tot_acc/ev:.3f}  geometric@{agree.mean():.3f} "
              f"{sr.geometric_emitted(agree.mean(), k):.3f}")
        print(f"        accepted dist {dict(sorted(acc.items()))}")
        print(f"        emitted  dist {dict(sorted(emi.items()))}")
    amb = [r for r in rows if r["ambiguous"]]
    nonamb = [r for r in rows if not r["ambiguous"]]
    cert = np.mean([r["margin_certified"] for r in nonamb]) if nonamb else float("nan")
    print(f"ambiguous {len(amb)} (recorded != fp16 argmax: "
          f"{sum(not r['recorded_equals_fp16_argmax'] for r in rows)}, margin <= ulp: "
          f"{sum(r['margin_fp16'] <= r['fp16_ulp_at_top'] for r in rows)}); "
          f"certified fraction over non-ambiguous {cert:.4f} (summary {s['certified_fraction_nonambiguous']})")
    print("shift-aligned sup error and agreement by teacher margin bin (logit, nats):")
    for i in range(len(EDGES) - 1):
        b = [r for r in rows if EDGES[i] <= r["teacher_margin_logit"] < EDGES[i + 1]]
        if not b:
            print(f"  [{EDGES[i]},{EDGES[i+1]}): n=0")
            continue
        e = np.array([r["sup_err_shift"] for r in b])
        print(f"  [{EDGES[i]},{EDGES[i+1]}): n={len(b):4d} agree {np.mean([r['agree'] for r in b]):.3f} "
              f"sup_err median {np.median(e):.2f} p10 {np.percentile(e,10):.2f} p90 {np.percentile(e,90):.2f} "
              f"certified {sum(r['margin_certified'] for r in b)}")
    print(f"min_bar12_ratio recomputed {min(agree.mean()/0.75, sr.replay_blocks(t_all, d_all, 4)['mean_emitted']/2.5):.4f} "
          f"(summary {s['min_bar12_ratio']:.4f}); wall {s['wall_s']:.0f}s peak rss {s['peak_rss_bytes']/1e9:.2f} GB")


if __name__ == "__main__":
    main(*sys.argv[1:])
