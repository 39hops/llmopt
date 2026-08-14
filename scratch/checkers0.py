"""CHECKERS-0 instrument (pre-reg RESULTS 2026-08-14): bounded
search v greedy material on an exact win-in-K oracle. English
draughts, forced captures, mandatory multi-jump chains, promotion
ends the move. Pure CPU, no deps beyond stdlib.

200 positions drawn by random legal playout from the start until
<= 6 total pieces (string seeds "checkers0-{i}", deduped). Oracle:
depth-limited minimax with memo — WIN iff side to move forces a
win within K=20 plies (loss = no legal move). Arms: material sign
(man=1, king=2) v the same solver under a hard 2,000-node budget.
Receipts stream to logs/checkers0/probe.jsonl, one row per
position, so a wall-kill still leaves bookable cells.

Usage: .venv/bin/python scratch/checkers0.py  [SMOKE=1: 5 pos, K=10]
"""
import json
import os
import random
from pathlib import Path

SMOKE = os.environ.get("SMOKE") == "1"
N_POS = 5 if SMOKE else 200
K = 10 if SMOKE else 20
BUDGET = 2_000
OUT = Path("logs/checkers0/probe.jsonl")

# Board: dict {(r, c): piece} on dark squares only ((r+c) odd),
# r=0 is White's back rank. 'w'/'b' men, 'W'/'B' kings.
# White men move toward higher r, black men toward lower r.


def start_board():
    bd = {}
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                if r < 3:
                    bd[(r, c)] = "w"
                elif r > 4:
                    bd[(r, c)] = "b"
    return bd


def dirs(piece):
    if piece == "w":
        return [(1, -1), (1, 1)]
    if piece == "b":
        return [(-1, -1), (-1, 1)]
    return [(1, -1), (1, 1), (-1, -1), (-1, 1)]


def on(r, c):
    return 0 <= r < 8 and 0 <= c < 8


def _promote(piece, r):
    if piece == "w" and r == 7:
        return "W"
    if piece == "b" and r == 0:
        return "B"
    return piece


def _capture_chains(bd, pos, piece, side):
    """All maximal-continuation capture chains from pos. Each chain
    is (final_board,); promotion ends the chain (standard rule)."""
    out = []
    r, c = pos
    found = False
    for dr, dc in dirs(piece):
        mr, mc, lr, lc = r + dr, c + dc, r + 2 * dr, c + 2 * dc
        if (on(lr, lc) and (mr, mc) in bd and (lr, lc) not in bd
                and bd[(mr, mc)].lower() != side):
            found = True
            nb = dict(bd)
            del nb[(r, c)]
            del nb[(mr, mc)]
            np = _promote(piece, lr)
            nb[(lr, lc)] = np
            if np != piece:            # promoted: chain ends
                out.append(nb)
            else:
                cont = _capture_chains(nb, (lr, lc), np, side)
                out.extend(cont if cont else [nb])
    return out if found else []


def legal_moves(bd, side):
    """List of successor boards. Captures mandatory."""
    caps, quiets = [], []
    for (r, c), piece in bd.items():
        if piece.lower() != side:
            continue
        caps.extend(_capture_chains(bd, (r, c), piece, side))
        if not caps:
            for dr, dc in dirs(piece):
                nr, nc = r + dr, c + dc
                if on(nr, nc) and (nr, nc) not in bd:
                    nb = dict(bd)
                    del nb[(r, c)]
                    nb[(nr, nc)] = _promote(piece, nr)
                    quiets.append(nb)
    return caps if caps else quiets


def key(bd, side):
    return (side, tuple(sorted((r, c, p) for (r, c), p in bd.items())))


def solve(bd, side, depth, memo, counter, budget):
    """1 iff side to move forces a win within depth plies.
    counter[0] = nodes; budget None = oracle (uncapped)."""
    counter[0] += 1
    if budget is not None and counter[0] > budget:
        raise _Budget
    if depth == 0:
        return 0
    k = (key(bd, side), depth)
    if k in memo:
        return memo[k]
    moves = legal_moves(bd, side)
    if not moves:
        memo[k] = 0            # side to move has no move: LOSS, not win
        return 0
    opp = "b" if side == "w" else "w"
    res = 0
    for nb in moves:
        if not legal_moves(nb, opp):
            res = 1            # opponent left with no move: win now
            break
        # win iff opponent cannot force a win AND we can force from
        # every opponent reply... full minimax on the win predicate:
        if all(solve(nb2, side, depth - 2, memo, counter, budget)
               for nb2 in legal_moves(nb, opp)):
            res = 1
            break
    memo[k] = res
    return res


class _Budget(Exception):
    pass


def material(bd, side):
    v = {"w": 1, "W": 2, "b": -1, "B": -2}
    s = sum(v[p] for p in bd.values())
    return s if side == "w" else -s


def draw_positions():
    seen, out = set(), []
    i = 0
    while len(out) < N_POS:
        rng = random.Random(f"checkers0-{i}")
        i += 1
        bd, side = start_board(), "b"   # black moves first? standard:
        side = "b"                       # black opens in English draughts
        for _ in range(200):
            if len(bd) <= 6:
                k = key(bd, side)
                if k not in seen:
                    seen.add(k)
                    out.append((bd, side))
                break
            moves = legal_moves(bd, side)
            if not moves:
                break
            bd = rng.choice(moves)
            side = "b" if side == "w" else "w"
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["i"] for l in OUT.read_text().splitlines() if l}
    poss = draw_positions()
    n = 0
    for i, (bd, side) in enumerate(poss):
        if i in done:
            continue
        label = solve(bd, side, K, {}, [0], None)
        base = 1 if material(bd, side) > 0 else 0
        cnt = [0]
        try:
            pred = solve(bd, side, K, {}, cnt, BUDGET)
        except _Budget:
            pred = 0
        row = {"i": i, "pieces": len(bd), "side": side, "label": label,
               "baseline": base, "search": pred, "nodes": cnt[0]}
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l]
    n = len(rows)
    ab = sum(r["label"] == r["baseline"] for r in rows)
    asr = sum(r["label"] == r["search"] for r in rows)
    wins = sum(r["label"] for r in rows)
    print(f"[checkers0] n={n} win-in-{K} rate {wins}/{n} | "
          f"baseline acc {ab}/{n} = {100*ab/n:.1f}% | "
          f"search acc {asr}/{n} = {100*asr/n:.1f}%", flush=True)


if __name__ == "__main__":
    main()
