"""GRF horizon-window rider (desk, frozen pre-run): windowed
adjudication of the matched-horizon reading in rider2.json.

rider2's horizon contrasts were CUMULATIVE first-N signatures, so a
rising contrast with N does not separate temporal structure from
routing-frequency noise reduction (more samples per signature), and
variable nothink decode lengths mean prompts shorter than N
contribute fewer than N samples. This rider computes:

 1. ELIGIBILITY census: prompts with decode_len >= 8/16/24/32, by
    topic and by form, both captures (decode_len = distinct decode
    positions in the traj).
 2. EQUAL-WIDTH WINDOWS 1:8 / 9:16 / 17:24 / 25:32 on a FIXED
    COHORT: prompts with >= 32 decode steps in BOTH captures, cohort
    composition disclosed by topic and form. Each window signature
    uses exactly 8 decode tokens per prompt, so windows are
    sample-matched and later-window changes are temporal, not
    noise-floor.
 3. Topic/form contrast per window EXCLUDING same-proposition pairs
    (which share literal content and inflate within-topic
    similarity), plus leave-one-proposition-out centroid accuracy
    per window (topic and form).
 4. STRICT MCQ extraction census on the nothink completions: an
    answer counts only as a letter-only line or an explicit
    "answer is / answer:" anchor; NO-ANSWER books against the all-40
    denominator. (Census only; scoring against the key stays a
    separate registration.)

Receipt: logs/grf/rider3.json (refuse-if-exists).

    .venv/bin/python scratch/grf_rider3.py                 (Mac desk)
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
OUT = Path("logs/grf/rider3.json")
HORIZONS = (8, 16, 24, 32)
WINDOWS = ((0, 8), (8, 16), (16, 24), (24, 32))

STRICT_LINE = re.compile(r"^\s*\**\s*([A-D])\**\s*[.):]?\s*$")
STRICT_ANCH = re.compile(
    r"(?:answer\s+is|answer\s*[:\-])\s*\**\s*([A-D])\b", re.IGNORECASE)


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


def decode_index(traj_path):
    """prompt -> list of decode rows, each tagged with the rank of
    its position among that prompt's sorted distinct decode
    positions."""
    rows_by = defaultdict(list)
    pos_by = defaultdict(set)
    for line in open(traj_path):
        t = json.loads(line)
        if t["phase"] == "decode":
            rows_by[t["prompt"]].append(t)
            pos_by[t["prompt"]].add(t["pos"])
    order = {p: {pos: i for i, pos in enumerate(sorted(s))}
             for p, s in pos_by.items()}
    return rows_by, order, {p: len(s) for p, s in pos_by.items()}


def window_table(rows_by, order, cohort, meta):
    out = {}
    for lo, hi in WINDOWS:
        vec = {p: np.zeros(D) for p in cohort}
        for p in cohort:
            for t in rows_by[p]:
                if lo <= order[p][t["pos"]] < hi:
                    base = t["layer"] * 128
                    for e in t["topk"]:
                        vec[p][base + e] += 1
        pids = sorted(cohort)
        M = np.stack([vec[p] for p in pids])
        M = M / (M.sum(1, keepdims=True) + 1e-12)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        S = M @ M.T
        topic = np.array([meta[p]["topic"] for p in pids])
        form = np.array([meta[p]["form"] for p in pids])
        prop = np.array([f"{meta[p]['topic']}:{meta[p]['prop_idx']}"
                         for p in pids])
        iu = np.triu_indices(len(pids), 1)
        same_prop = prop[iu[0]] == prop[iu[1]]
        res = {}
        for f, lab in (("topic", topic), ("form", form)):
            same = (lab[iu[0]] == lab[iu[1]]) & ~same_prop
            diff = (lab[iu[0]] != lab[iu[1]]) & ~same_prop
            res[f"{f}_contrast_excl_same_prop"] = round(
                float(S[iu][same].mean() - S[iu][diff].mean()), 4)
        for f, lab in (("topic", topic), ("form", form)):
            names = sorted(set(lab))
            hit = 0
            for i in range(len(pids)):
                mask = prop != prop[i]
                C = np.stack([M[mask & (lab == c)].mean(0)
                              for c in names])
                C = C / (np.linalg.norm(C, axis=1, keepdims=True)
                         + 1e-12)
                hit += names[int((C @ M[i]).argmax())] == lab[i]
            res[f"{f}_lopo_centroid_acc"] = round(hit / len(pids), 3)
        out[f"w{lo + 1}_{hi}"] = res
    return out


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/grf_rider3.py", "scratch/grf_rider2.py",
         "scratch/grf_capture2.py", "scratch/grf_corpus.py"])
    meta = load_meta("logs/grf/rows_nothink.jsonl")
    meta0 = load_meta("logs/grf/rows.jsonl")
    r_no, o_no, len_no = decode_index("logs/grf/traj_nothink.jsonl")
    r_th, o_th, len_th = decode_index("logs/grf/traj.jsonl")

    elig = {}
    for name, lens, mm in (("nothink", len_no, meta),
                           ("thinking", len_th, meta0)):
        e = {}
        for N in HORIZONS:
            ok = [p for p in lens if lens[p] >= N]
            e[f"N{N}"] = {
                "n": len(ok),
                "by_topic": dict(Counter(mm[p]["topic"] for p in ok)),
                "by_form": dict(Counter(mm[p]["form"] for p in ok))}
        elig[name] = e

    cohort = sorted(p for p in len_no
                    if len_no[p] >= 32 and len_th.get(p, 0) >= 32)
    comp = {"n": len(cohort),
            "by_topic": dict(Counter(meta[p]["topic"]
                                     for p in cohort)),
            "by_form": dict(Counter(meta[p]["form"]
                                    for p in cohort))}

    win_no = window_table(r_no, o_no, cohort, meta)
    win_th = window_table(r_th, o_th, cohort, meta0)

    mcq = [p for p in meta if meta[p]["form"] == "mcq"]
    strict = {}
    for p in mcq:
        c = meta[p]["completion"]
        m = STRICT_ANCH.search(c)
        if not m:
            for ln in c.splitlines():
                m = STRICT_LINE.match(ln)
                if m:
                    break
        strict[p] = m.group(1).upper() if m else None
    n_strict = sum(v is not None for v in strict.values())

    rcpt = {"note": "GRF horizon-window rider: eligibility census, "
                    "equal-width sample-matched windows on a fixed "
                    ">=32-step cohort, same-prop-excluded contrasts "
                    "+ LOPO accuracy per window, strict MCQ census",
            "start": START,
            "completion_commit": completion_commit(),
            "provenance": {
                "traj_sha256": sha("logs/grf/traj.jsonl"),
                "traj_nothink_sha256":
                    sha("logs/grf/traj_nothink.jsonl")},
            "eligibility": elig,
            "cohort_ge32_both": comp,
            "windows_nothink": win_no,
            "windows_thinking": win_th,
            "mcq_strict": {
                "line_regex": STRICT_LINE.pattern,
                "anchor_regex": STRICT_ANCH.pattern,
                "n_mcq": len(mcq), "n_strict_answer": n_strict,
                "n_no_answer": len(mcq) - n_strict,
                "letters": dict(Counter(v for v in strict.values()
                                        if v))}}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    print(f"cohort {comp['n']}  by_form {comp['by_form']}")
    for w in win_no:
        a, b = win_th[w], win_no[w]
        print(f"{w}: think T {a['topic_contrast_excl_same_prop']}"
              f" acc {a['topic_lopo_centroid_acc']} | nothink T "
              f"{b['topic_contrast_excl_same_prop']} acc "
              f"{b['topic_lopo_centroid_acc']}")
    print(f"mcq strict {n_strict}/{len(mcq)}")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
