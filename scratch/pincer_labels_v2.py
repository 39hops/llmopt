"""Pincer label prep v2 — the MIGRATED replacement for
pincer_r1b_labels.py (migration mandated by the results-hardening
spec: the v1 driver calls successors() IN-PROCESS under
derivation.py's SIGALRM box — the one true checkpoint-selection
carrier left in the program. SIGALRM cannot box sympy; rows whose
replay hangs vanish silently and the MISS class is undercounted).

v2 changes EXACTLY two things (classification logic is verbatim v1
— tests/test_pincer_labels_v2.py holds the behavior guard):
1. FORK-PER-STATE: each row's replay runs in a forked worker with
   a hard join deadline + SIGKILL (the gen_magic_labels
   solve_isolated pattern). Hung rows become a VISIBLE 'timeout'
   class instead of vanishing.
2. STREAMED ROWS: one jsonl line per row, written as processed
   (an outer wall leaves a complete partial record, never a
   selection effect).

    python scratch/pincer_labels_v2.py [n_sample=300] \
        [OUT=logs/pincer/labels_v2.jsonl] [DEADLINE=20]
"""
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = Path(os.environ.get("OUT", "logs/pincer/labels_v2.jsonl"))
DEADLINE = float(os.environ.get("DEADLINE", "20"))
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    raise SystemExit(f"REFUSING: {OUT} exists (streamed record; "
                     "move it or pick a new OUT)")

norm = lambda s: s.replace(" ", "")  # noqa: E731  (verbatim v1)


def _worker(cur_s, nxt_s, q):
    """Replay one row; runs in a fork, killed from outside."""
    import sympy as sp
    from llmopt.search.derivation import State, successors
    try:
        cur = sp.sympify(cur_s)
        target = norm(sp.sstr(sp.sympify(nxt_s)))
        names = [name for name, ch in
                 successors(State(cur), use_macros=True)
                 if norm(sp.sstr(ch.expr)) == target]
        q.put(("ok", names))
    except Exception as e:
        q.put(("err", str(e)[:120]))


def classify(names):
    """Verbatim v1 decision structure (behavior-guarded)."""
    if len(names) == 1:
        return "unique"
    if names:
        return "ambig"
    return "miss"


def main():
    from train_mathnative import load_rows
    rows = load_rows(gen4=True)
    rows = [r for r in rows if norm(r["cur"]) != norm(r["nxt"])]
    rnd = random.Random("r1b-labels-1")  # v1's exact sample
    sample = rnd.sample(rows, N)

    counts = {"unique": 0, "ambig": 0, "miss": 0, "err": 0,
              "timeout": 0}
    rule_hist = {}
    ctx = mp.get_context("fork")
    with OUT.open("w") as f:
        for i, r in enumerate(sample):
            q = ctx.Queue()
            p = ctx.Process(target=_worker,
                            args=(r["cur"], r["nxt"], q))
            p.start()
            p.join(DEADLINE)
            if p.is_alive():
                p.kill()
                p.join()
                cls, names = "timeout", []
            else:
                try:
                    status, payload = q.get_nowait()
                except Exception:
                    status, payload = "err", "no-result"
                if status == "err":
                    cls, names = "err", []
                else:
                    names = payload
                    cls = classify(names)
            counts[cls] += 1
            for n_ in names:
                rule_hist[n_] = rule_hist.get(n_, 0) + 1
            f.write(json.dumps({"i": i, "cur": r["cur"],
                                "nxt": r["nxt"], "cls": cls,
                                "names": names}) + "\n")
            f.flush()
    top = sorted(rule_hist.items(), key=lambda kv: -kv[1])[:8]
    rec = counts["unique"] + counts["ambig"]
    print(f"R1b-v2 label recovery on {N} rows: {counts} -> "
          f"recoverable {100 * rec / max(N, 1):.1f}%")
    print("top rules:", top)


if __name__ == "__main__":
    main()
