"""GRF-0 audit rider (desk, frozen pre-read): the six checks the
external audit requested before any GRF successor rung.

 1. THINKING-MODE CENSUS: grf_capture omitted enable_thinking=False
    (the frozen MoE gate machinery disables it; this capture used
    the bare chat template) — count completions containing
    </think>, generated-length distribution, MAX_TOKENS ceiling
    hits, and MCQ rows whose answer letter appears after </think>
    (answer reach).
 2. MATCHED-PAIR SPLIT: 4 of 5 forms embed the question verbatim;
    split same-proposition pair cosine into q-form<->q-form pairs
    (share literal text) v completion<->q-form pairs (do not).
 3. TOPIC CONTRAST EXCLUDING same-proposition pairs (which inflate
    within-topic similarity).
 4. PER-TOPIC 8x8 nearest-centroid confusion (leave-one-
    proposition-out) + per-topic recall, prefill and decode — the
    evidence the "bio/chem behaved ordinarily" claim needs.
 5. PREFILL-COMPARABLE NN: ROUTE-BASIS-0 signatures were
    PREFILL-only, so its comparable GRF number is the prefill NN
    topic accuracy, not the full-signature one.
 6. DURABLE PROVENANCE: sha256 + row/phase/coverage counts for the
    untracked traj.jsonl and rows.jsonl, and MAX_TOKENS.

Receipt: logs/grf/rider.json (refuse-if-exists).

    .venv/bin/python scratch/grf_rider.py                  (Mac desk)
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

D = 48 * 128
OUT = Path("logs/grf/rider.json")
MAXTOK = 96
LETTERS = ("A", "B", "C", "D")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/grf_rider.py",
                              "scratch/grf_analyze.py",
                              "scratch/grf_corpus.py"])
    meta, comp = {}, {}
    for line in open("logs/grf/rows.jsonl"):
        r = json.loads(line)
        if "pid" in r:
            meta[r["pid"]] = r
            comp[r["pid"]] = r["completion"]
    # 1. thinking census
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "mlx-community/Qwen3-30B-A3B-4bit")
    n_think = sum("</think>" in c for c in comp.values())
    lens = {p: len(tok(c)["input_ids"]) for p, c in comp.items()}
    n_ceiling = sum(v >= MAXTOK for v in lens.values())
    mcq = [p for p in meta if meta[p]["form"] == "mcq"]
    mcq_reach = 0
    for p in mcq:
        c = comp[p]
        tail = c.split("</think>", 1)[1] if "</think>" in c else ""
        mcq_reach += any(ch in tail for ch in LETTERS)
    think = {"n_completions": len(comp),
             "n_with_close_think": n_think,
             "n_at_token_ceiling": n_ceiling,
             "gen_len_median": int(np.median(list(lens.values()))),
             "mcq_n": len(mcq),
             "mcq_answer_after_think": mcq_reach}

    vecs = {v: defaultdict(lambda: np.zeros(D))
            for v in ("full", "prefill", "decode")}
    phase_counts = Counter()
    n_rows = 0
    layer_cov, prompt_cov = set(), set()
    for line in open("logs/grf/traj.jsonl"):
        t = json.loads(line)
        n_rows += 1
        phase_counts[t["phase"]] += 1
        layer_cov.add(t["layer"])
        prompt_cov.add(t["prompt"])
        ph = ("prefill" if t["phase"] in ("prefill", "prompt_tail")
              else "decode")
        base = t["layer"] * 128
        for e in t["topk"]:
            vecs["full"][t["prompt"]][base + e] += 1
            vecs[ph][t["prompt"]][base + e] += 1
    pids = sorted(meta)
    X = {}
    for v in vecs:
        M = np.stack([vecs[v][p] for p in pids])
        M = M / (M.sum(1, keepdims=True) + 1e-12)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        X[v] = M
    topic = np.array([meta[p]["topic"] for p in pids])
    form = np.array([meta[p]["form"] for p in pids])
    prop_key = np.array([f"{meta[p]['topic']}:{meta[p]['prop_idx']}"
                         for p in pids])

    out = {}
    for v, M in X.items():
        S = M @ M.T
        qq, cq = [], []
        within_t, between_t = [], []
        for i, j in combinations(range(len(pids)), 2):
            same_prop = prop_key[i] == prop_key[j]
            if same_prop:
                if "completion" in (form[i], form[j]):
                    cq.append(S[i, j])
                else:
                    qq.append(S[i, j])
            else:
                (within_t if topic[i] == topic[j]
                 else between_t).append(S[i, j])
        out[v] = {
            "same_prop_qform_pairs": round(float(np.mean(qq)), 4),
            "same_prop_completion_pairs":
                round(float(np.mean(cq)), 4),
            "topic_contrast_excl_same_prop": round(
                float(np.mean(within_t)) - float(np.mean(between_t)),
                4),
        }
        # per-topic confusion, leave-one-proposition-out centroids
        names = sorted(set(topic))
        conf = {a: Counter() for a in names}
        for i in range(len(pids)):
            mask = prop_key != prop_key[i]
            cents = []
            for cl in names:
                sel = mask & (topic == cl)
                cents.append(M[sel].mean(0))
            C = np.stack(cents)
            C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            pred = names[int((C @ M[i]).argmax())]
            conf[topic[i]][pred] += 1
        out[v]["confusion"] = {a: dict(conf[a]) for a in names}
        out[v]["per_topic_recall"] = {
            a: round(conf[a][a] / max(sum(conf[a].values()), 1), 3)
            for a in names}

    rcpt = {"note": "GRF-0 audit rider: thinking census, matched-"
                    "pair split, same-prop-excluded contrasts, "
                    "per-topic confusion, durable provenance",
            "start": START, "completion_commit": completion_commit(),
            "thinking_census": think,
            "signature_checks": out,
            "provenance": {
                "traj_sha256": sha("logs/grf/traj.jsonl"),
                "rows_sha256": sha("logs/grf/rows.jsonl"),
                "traj_rows": n_rows,
                "phase_counts": dict(phase_counts),
                "n_layers_covered": len(layer_cov),
                "n_prompts_covered": len(prompt_cov),
                "max_tokens": MAXTOK}}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    print(json.dumps(think, indent=1))
    for v in X:
        o = out[v]
        print(f"{v:8s} qq {o['same_prop_qform_pairs']} cq "
              f"{o['same_prop_completion_pairs']} topicC-excl "
              f"{o['topic_contrast_excl_same_prop']}")
        print(f"  recall {o['per_topic_recall']}")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
