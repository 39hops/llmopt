"""LOSS-FLOOR-1: empirical conditional entropy of the sat_s2 warm diet
vs the measured 0.348 train-loss floor (RESULTS pre-reg LOSS-FLOOR-1).

Replicates train_mathnative's exact token stream (template, strict
encode with whole-row skip, SEQ_CAP, eos) and measures, in nats:
  H_k    — next-token entropy given the last k tokens
  H_full — next-token entropy given the entire prefix (rolling hash)

Positions weighted uniformly, matching the training CE (full-sequence
loss, all non-pad next-token positions). H_full uses per-sequence
prefix contexts hashed with blake2b-8B; collisions merge contexts and
bias H_full DOWN (fenced in the pre-reg — already downward-biased by
unique prefixes).
"""
import hashlib
import math
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.train_mathnative as TM  # noqa: E402
from llmopt.train.mathnative import MathTokenizer  # noqa: E402

FLOOR = 0.3480  # sat_s2 ep3 mean train loss, nats (RESULTS 26259)
KS = (1, 2, 4, 8, 16, 32, 64)


def encode_corpus():
    tok = MathTokenizer(extra=None)
    rows = TM.load_rows(True, True, True, True, True, False, None)
    cap = int(os.environ.get("SEQ_CAP", "512"))
    enc, skipped = [], 0
    for r in rows:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try:
            ids = tok.encode(t) + [tok.eos_id]
        except ValueError:
            skipped += 1
            continue
        if len(ids) <= cap:
            enc.append(ids)
    print(f"[census] {len(rows)} rows -> {len(enc)} sequences "
          f"({skipped} skipped out-of-language), vocab {len(tok.vocab)}",
          flush=True)
    return enc


def entropy_of_groups(groups):
    """Mean -log p_hat(next | context) over all positions, nats."""
    total_positions = 0
    total_nats = 0.0
    for ctr in groups.values():
        n = sum(ctr.values())
        total_positions += n
        for c in ctr.values():
            total_nats += -c * math.log(c / n)
    return total_nats / max(total_positions, 1), total_positions


def h_kgram(enc, k):
    groups = defaultdict(Counter)
    for seq in enc:
        for i in range(1, len(seq)):
            ctx = tuple(seq[max(0, i - k):i])
            groups[ctx][seq[i]] += 1
    h, n = entropy_of_groups(groups)
    print(f"[census] H_{k:<3d} = {h:.4f} nats  "
          f"({len(groups)} contexts, {n} positions)", flush=True)
    return h


def h_full(enc):
    groups = defaultdict(Counter)
    for seq in enc:
        hsh = hashlib.blake2b(digest_size=8)
        for i in range(1, len(seq)):
            hsh.update(bytes([seq[i - 1] & 0xFF, (seq[i - 1] >> 8) & 0xFF]))
            groups[hsh.digest()][seq[i]] += 1
    h, n = entropy_of_groups(groups)
    print(f"[census] H_full = {h:.4f} nats  "
          f"({len(groups)} contexts, {n} positions)", flush=True)
    return h


def main():
    t0 = time.time()
    enc = encode_corpus()
    for k in KS:
        h_kgram(enc, k)
    hf = h_full(enc)
    frac = hf / FLOOR
    print(f"[census] floor {FLOOR:.4f} | H_full {hf:.4f} | "
          f"H_full/floor = {frac:.3f}", flush=True)
    if hf >= 0.7 * FLOOR:
        print("[census] BAR: P-FLOOR-IS-ENTROPY fires (>= 0.70)", flush=True)
    elif hf <= 0.3 * FLOOR:
        print("[census] BAR: P-HEADROOM fires (<= 0.30)", flush=True)
    else:
        print("[census] BAR: MIXED — book the split", flush=True)
    print(f"[census] wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
