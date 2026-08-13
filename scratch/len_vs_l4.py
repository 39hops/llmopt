"""LENGTH-VS-L4-1 desk analysis (pre-reg RESULTS 2026-08-13): read
the gate_pp sidecar, compute per-problem prompt token length and
root char length, then the pre-registered bars: pooled within-level
Spearman rho(token length, solved), per-level length medians, and
the within-L4 rho named by REFUTED-IF. Deterministic from the
sidecar; stdlib + tokenizer only.

Usage: .venv/bin/python scratch/len_vs_l4.py logs/pp_phase19m_final.jsonl
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.train.mathnative import MathTokenizer  # noqa: E402


def spearman(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[s[k]] = (i + j) / 2
            i = j + 1
        return rk
    ra, rb = rank(xs), rank(ys)
    n = len(xs)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((a - ma) * (b - mb) for a, b in zip(ra, rb))
    den = (sum((a - ma) ** 2 for a in ra)
           * sum((b - mb) ** 2 for b in rb)) ** 0.5
    return num / den if den else 0.0


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    tok = MathTokenizer()
    for r in rows:
        prompt = f"Current: {r['root']}\nHints: none\nStep: "
        r["tok_len"] = len(tok.encode(prompt))
        r["char_len"] = len(r["root"])

    by_level = {}
    for r in rows:
        by_level.setdefault(r["level"], []).append(r)

    print("per-level: n, solves, median tok_len, median char_len")
    for lv in sorted(by_level):
        g = by_level[lv]
        print(f"  L{lv}: n={len(g)} solves={sum(r['solved'] for r in g)}"
              f" med_tok={statistics.median(r['tok_len'] for r in g)}"
              f" med_char={statistics.median(r['char_len'] for r in g)}")

    # Bar 1: pooled within-level rho — z-score length per level.
    xs, ys = [], []
    for lv, g in by_level.items():
        m = statistics.mean(r["tok_len"] for r in g)
        sd = statistics.pstdev(r["tok_len"] for r in g) or 1.0
        for r in g:
            xs.append((r["tok_len"] - m) / sd)
            ys.append(1 if r["solved"] else 0)
    rho = spearman(xs, ys)
    print(f"BAR1 pooled within-level rho(tok_len, solved) = {rho:.4f}"
          f" -> {'FIRES' if rho <= -0.30 else 'NO-FIRE'} (bar <= -0.30)")

    meds = {lv: statistics.median(r["tok_len"] for r in by_level[lv])
            for lv in by_level}
    fires2 = all(meds[4] > meds[lv] for lv in (5, 6, 7))
    print(f"BAR2 medians tok_len {meds} -> "
          f"{'FIRES' if fires2 else 'NO-FIRE'} (L4 > each of L5,L6,L7)")

    g4 = by_level[4]
    rho4 = spearman([r["tok_len"] for r in g4],
                    [1 if r["solved"] else 0 for r in g4])
    print(f"within-L4 rho = {rho4:.4f} (REFUTED-IF leg needs <= -0.40)")

    tot = {lv: sum(r["solved"] for r in by_level[lv])
           for lv in sorted(by_level)}
    print(f"precondition dict: {tot} = {sum(tot.values())}/120")


if __name__ == "__main__":
    main()
