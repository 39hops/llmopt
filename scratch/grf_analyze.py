"""GENERAL-ROUTING-FACTORIAL analysis (desk, observation-only;
metrics FROZEN here before any value is read — the commit carrying
this file predates the receipt).

Signatures: per prompt, the L1-normalized 6144-dim (48x128)
selection-count vector, computed three ways — full, prefill-only
(prefill + prompt_tail), decode-only.

FROZEN QUESTIONS:
 1. SIMILARITY STRUCTURE: mean pairwise cosine within-topic v
    between-topic; within-form v between-form; within-operation v
    between-operation; each on all three signature variants. The
    contrast ratio (within - between) says which factor organizes
    routing space.
 2. MATCHED-PAIR DISSOCIATION (the design's point): mean cosine of
    SAME-PROPOSITION pairs (identical content, different form) v
    SAME-FORM-DIFFERENT-TOPIC pairs (identical wrapper, different
    content). Which is routing tracking?
 3. HELD-OUT PREDICTION, leakage-controlled: nearest-centroid
    accuracy for topic / form / operation with centroids built
    LEAVE-ONE-PROPOSITION-OUT (a prompt's own proposition — all
    its forms — never contributes to any centroid it is scored
    against). Chance: topic 1/8, form 1/5, operation from label
    frequencies (report empirical chance = max class share).
 4. TEMPLATE CONTROL for ROUTE-BASIS-0: NN topic accuracy under
    the same leave-one-proposition-out exclusion (was the 0.0
    cross-domain NN rate semantic or template-driven? Here the
    wrappers are SHARED across topics by construction).

Receipt: logs/grf/analysis.json (refuse-if-exists).

    .venv/bin/python scratch/grf_analyze.py                (Mac desk)
"""
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

D = 48 * 128
OUT = Path("logs/grf/analysis.json")


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/grf_analyze.py",
                              "scratch/grf_capture.py",
                              "scratch/grf_corpus.py"])
    meta = {}
    for line in open("logs/grf/rows.jsonl"):
        r = json.loads(line)
        if "pid" in r:
            meta[r["pid"]] = r
    vecs = {v: defaultdict(lambda: np.zeros(D))
            for v in ("full", "prefill", "decode")}
    for line in open("logs/grf/traj.jsonl"):
        t = json.loads(line)
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
    lab = {f: np.array([meta[p][f] for p in pids])
           for f in ("topic", "form", "operation", "prop_idx")}
    prop_key = np.array([f"{meta[p]['topic']}:{meta[p]['prop_idx']}"
                         for p in pids])

    def sim_contrast(M, labels):
        S = M @ M.T
        n = len(pids)
        iu = np.triu_indices(n, 1)
        same = labels[iu[0]] == labels[iu[1]]
        return {"within": round(float(S[iu][same].mean()), 4),
                "between": round(float(S[iu][~same].mean()), 4),
                "contrast": round(float(S[iu][same].mean()
                                        - S[iu][~same].mean()), 4)}

    out = {"similarity": {}, "matched_pairs": {},
           "heldout_centroid_acc": {}, "nn_topic_acc": {}}
    for v, M in X.items():
        out["similarity"][v] = {
            f: sim_contrast(M, lab[f])
            for f in ("topic", "form", "operation")}
        # matched pairs
        S = M @ M.T
        same_prop, same_form_diff_topic = [], []
        for i, j in combinations(range(len(pids)), 2):
            if prop_key[i] == prop_key[j]:
                same_prop.append(S[i, j])
            elif (lab["form"][i] == lab["form"][j]
                  and lab["topic"][i] != lab["topic"][j]):
                same_form_diff_topic.append(S[i, j])
        out["matched_pairs"][v] = {
            "same_content_diff_form":
                round(float(np.mean(same_prop)), 4),
            "same_form_diff_content":
                round(float(np.mean(same_form_diff_topic)), 4)}
        # leave-one-proposition-out centroid + NN
        accs = {}
        for f in ("topic", "form", "operation"):
            hit = 0
            for i in range(len(pids)):
                mask = prop_key != prop_key[i]
                cents, names = [], []
                for cl in sorted(set(lab[f])):
                    rowsel = mask & (lab[f] == cl)
                    if rowsel.any():
                        cents.append(M[rowsel].mean(0))
                        names.append(cl)
                C = np.stack(cents)
                C = C / (np.linalg.norm(C, axis=1, keepdims=True)
                         + 1e-12)
                hit += names[int((C @ M[i]).argmax())] == lab[f][i]
            accs[f] = round(hit / len(pids), 4)
        out["heldout_centroid_acc"][v] = accs
        hit = 0
        for i in range(len(pids)):
            mask = prop_key != prop_key[i]
            sims = (M @ M[i])
            sims[~mask] = -np.inf
            hit += lab["topic"][int(sims.argmax())] == lab["topic"][i]
        out["nn_topic_acc"][v] = round(hit / len(pids), 4)

    chance = {"topic": 1 / 8, "form": 1 / 5,
              "operation": round(max(np.mean(lab["operation"] == c)
                                     for c in set(lab["operation"])), 3)}
    rcpt = {"note": "GRF analysis: similarity structure, matched-"
                    "pair dissociation, leakage-controlled held-out "
                    "prediction (metrics frozen pre-read)",
            "start": START, "completion_commit": completion_commit(),
            "n_prompts": len(pids), "chance": chance, **out}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    for v in X:
        s = out["similarity"][v]
        mp = out["matched_pairs"][v]
        a = out["heldout_centroid_acc"][v]
        print(f"{v:8s} contrast T/F/O "
              f"{s['topic']['contrast']}/{s['form']['contrast']}/"
              f"{s['operation']['contrast']}  "
              f"pairs content {mp['same_content_diff_form']} v form "
              f"{mp['same_form_diff_content']}  "
              f"acc T/F/O {a['topic']}/{a['form']}/{a['operation']}  "
              f"nnT {out['nn_topic_acc'][v]}")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
