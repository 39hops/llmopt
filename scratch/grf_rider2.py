"""GRF-NOTHINK audit rider (desk, frozen pre-read): the receipt
seams from the 4530e49d audit.

 1. DURABLE PROVENANCE for the NOTHINK capture, fingerprinting the
    ACTUAL producer (grf_capture2.py — the analyzer's start block
    inherited grf_capture.py from the copied original, a wrong-
    producer fingerprint disclosed by the paired amendment):
    sha256 + row/phase/prompt/layer counts for traj_nothink and
    rows_nothink, generation-length/ceiling census.
 2. STRICT MCQ EXTRACTION (frozen here, before any scoring): the
    answer is the FIRST standalone letter A-D in the completion,
    matched as ^\\s*\\**\\s*([A-D])\\b or after 'answer is/: '
    (case-insensitive); rows where no anchored letter exists book
    as NO-ANSWER. This censuses answer-identity READINESS; scoring
    against the key stays a separate registration.
 3. MATCHED-HORIZON CONTRASTS: decode-only topic/form contrasts
    recomputed from ONLY the first N decode tokens per prompt
    (N = 8/16/24/32), both captures — does the thinking-v-answering
    topic-contrast difference exist early, or does it emerge as
    long greedy thinking adds generic routing traffic?

Receipt: logs/grf/rider2.json (refuse-if-exists).

    .venv/bin/python scratch/grf_rider2.py                 (Mac desk)
"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

D = 48 * 128
OUT = Path("logs/grf/rider2.json")
HORIZONS = (8, 16, 24, 32)
ANS_RE = re.compile(r"(?:^|answer\s+is\s+|answer\s*[:\-]\s*)"
                    r"\s*\**\s*([A-D])\b",
                    re.IGNORECASE | re.MULTILINE)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(path):
    meta = {}
    for line in open(path):
        r = json.loads(line)
        if "pid" in r:
            meta[r["pid"]] = r
    return meta


def horizon_contrasts(traj_path, meta, tok_budgets):
    """decode-only signatures truncated to the first N decode
    tokens per prompt; contrast = within minus between mean cosine
    for topic and form."""
    dec_pos = defaultdict(set)      # prompt -> ordered decode pos
    rows_by = defaultdict(list)
    for line in open(traj_path):
        t = json.loads(line)
        if t["phase"] == "decode":
            dec_pos[t["prompt"]].add(t["pos"])
            rows_by[t["prompt"]].append(t)
    order = {p: {pos: i for i, pos in enumerate(sorted(s))}
             for p, s in dec_pos.items()}
    out = {}
    for N in tok_budgets:
        vec = defaultdict(lambda: np.zeros(D))
        for p, rows in rows_by.items():
            for t in rows:
                if order[p][t["pos"]] < N:
                    base = t["layer"] * 128
                    for e in t["topk"]:
                        vec[p][base + e] += 1
        pids = sorted(meta)
        M = np.stack([vec[p] for p in pids])
        M = M / (M.sum(1, keepdims=True) + 1e-12)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        S = M @ M.T
        iu = np.triu_indices(len(pids), 1)
        res = {}
        for f in ("topic", "form"):
            lab = np.array([meta[p][f] for p in pids])
            same = lab[iu[0]] == lab[iu[1]]
            res[f] = round(float(S[iu][same].mean()
                                 - S[iu][~same].mean()), 4)
        out[f"N{N}"] = res
    return out


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/grf_rider2.py", "scratch/grf_capture2.py",
         "scratch/grf_analyze2.py", "scratch/grf_corpus.py"])
    meta = load_meta("logs/grf/rows_nothink.jsonl")
    meta0 = load_meta("logs/grf/rows.jsonl")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "mlx-community/Qwen3-30B-A3B-4bit")
    lens = {p: len(tok(meta[p]["completion"])["input_ids"])
            for p in meta}
    mcq = [p for p in meta if meta[p]["form"] == "mcq"]
    extracted = {}
    for p in mcq:
        m = ANS_RE.search(meta[p]["completion"])
        extracted[p] = m.group(1).upper() if m else None
    n_answered = sum(v is not None for v in extracted.values())

    phase_counts = Counter()
    n_rows = 0
    layers, prompts = set(), set()
    for line in open("logs/grf/traj_nothink.jsonl"):
        t = json.loads(line)
        n_rows += 1
        phase_counts[t["phase"]] += 1
        layers.add(t["layer"])
        prompts.add(t["prompt"])

    hz_no = horizon_contrasts("logs/grf/traj_nothink.jsonl", meta,
                              HORIZONS)
    hz_th = horizon_contrasts("logs/grf/traj.jsonl", meta0,
                              HORIZONS)

    rcpt = {"note": "GRF-NOTHINK audit rider: correct-producer "
                    "provenance, strict MCQ extraction census, "
                    "matched-horizon thinking-v-answering "
                    "contrasts",
            "start": START, "completion_commit": completion_commit(),
            "provenance": {
                "traj_nothink_sha256":
                    sha("logs/grf/traj_nothink.jsonl"),
                "rows_nothink_sha256":
                    sha("logs/grf/rows_nothink.jsonl"),
                "traj_rows": n_rows,
                "phase_counts": dict(phase_counts),
                "n_layers": len(layers), "n_prompts": len(prompts),
                "gen_len_median": int(np.median(list(lens.values()))),
                "n_at_ceiling": sum(v >= 96 for v in lens.values())},
            "mcq_strict_extraction": {
                "regex": ANS_RE.pattern,
                "n_mcq": len(mcq), "n_anchored_answer": n_answered,
                "letters": dict(Counter(v for v in extracted.values()
                                        if v))},
            "matched_horizon_contrasts": {
                "nothink": hz_no, "thinking": hz_th}}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    print(f"mcq anchored {n_answered}/{len(mcq)}")
    for N in HORIZONS:
        print(f"N{N}: think T/F {hz_th[f'N{N}']['topic']}/"
              f"{hz_th[f'N{N}']['form']}  nothink T/F "
              f"{hz_no[f'N{N}']['topic']}/{hz_no[f'N{N}']['form']}")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
