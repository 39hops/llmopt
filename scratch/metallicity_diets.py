"""METALLICITY-1 diet grades — the same cloud at four refinements.

Base pool: the standard v22+gen4 verified rows (what the micro-star
recipe trains on). Grades, refinement ascending:

  z0  vacuum      — every row's cur/nxt/think char-shuffled in place:
                    identical charset, identical length distribution,
                    zero syntax. The no-structure control.
  z1  pop3        — z2's pollution PLUS duplication: half the unique
                    rows, each twice (same row count, half the unique
                    mass — the poorly-cooling primordial cloud).
  z2  polluted    — 35% of rows get another row's `nxt` (answer
                    shuffle: format-valid, wrong — the underdetermined
                    /wrong-step class measured to train hallucination).
  z3  verified    — the standard rows, untouched.

All grades are the SAME row count. String-seeded (doctrine). Writes
data/metallicity/z{0..3}.jsonl + a manifest with sha256 per file.

Usage: .venv/bin/python scratch/metallicity_diets.py
"""
import glob
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

OUT = "data/metallicity"
POLLUTE = 0.35


def base_rows() -> list[dict]:
    rows = []
    for f in sorted(glob.glob("data/micromodel_chains_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    rows += [json.loads(l) for l in open("data/step_chains.jsonl")]
    for f in sorted(glob.glob("data/micromodel_v22_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    rows += [json.loads(l)
             for l in open("data/micromodel_gen4_sidecar.jsonl")]
    # mirror the trainer's identity guard so grade row counts are
    # comparable after ITS filter runs (train_mathnative load_rows)
    return [r for r in rows
            if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]


def char_shuffle(s: str, rng: random.Random) -> str:
    cs = list(s)
    rng.shuffle(cs)
    return "".join(cs)


def write(name: str, rows: list[dict]) -> str:
    p = f"{OUT}/{name}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    print(f"{name}: {len(rows)} rows sha {h}")
    return h


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    z3 = base_rows()
    n = len(z3)

    # z2: answer-shuffle 35% — donor nxt drawn from the whole pool
    rng = random.Random("metallicity-z2-1")
    z2 = [dict(r) for r in z3]
    hit = rng.sample(range(n), int(n * POLLUTE))
    donors = rng.sample(range(n), len(hit))
    for i, j in zip(hit, donors):
        z2[i]["nxt"] = z3[j]["nxt"]

    # z1: z2's pollution + duplication (half unique, each twice)
    rng = random.Random("metallicity-z1-1")
    keep = rng.sample(range(n), n // 2)
    z1 = [dict(z2[i]) for i in keep for _ in (0, 1)][:n]

    # z0: char-shuffle every text field of z3 in place
    rng = random.Random("metallicity-z0-1")
    z0 = []
    for r in z3:
        r = dict(r)
        for k in ("cur", "nxt", "think"):
            if isinstance(r.get(k), str):
                r[k] = char_shuffle(r[k], rng)
        z0.append(r)

    man = {name: write(name, rows) for name, rows in
           [("z0", z0), ("z1", z1), ("z2", z2), ("z3", z3)]}
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
