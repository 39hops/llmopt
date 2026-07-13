"""Ansatz-STRUCTURE search (VGE rung 2, Artin's GO 2026-07-12):
greedy beam over layer-token sequences, energy oracle judging each
optimized candidate — the engine move transplanted to circuit design.

Vocabulary spans BOTH hand families ({ry, rx, cz} hardware-efficient
blocks; {zz, xm} HVA blocks), so the search can discover either or
mix them. The scientific bet (pre-registered): the WINNING STRUCTURE
changes with the phase — HVA-blocks at/above criticality, rotation/
entangle blocks in the ordered phase ("the judge reads off which
phase of matter it's standing in").

Bar (pre-registered): at each h, the searched structure (<= 6 tokens)
beats the best HAND-designed arm from rung 1 at that h, at equal or
fewer parameters. Honest loss recorded otherwise.
"""
import time

from llmopt.quantum.ground import (build_tfim, exact_ground,
                                   struct_nparams, struct_optimize)

N = 10
VOCAB = ("ry", "rx", "cz", "zz", "xm")
MAX_TOKENS = 6
BEAM = 3
# best hand arms from rung 1 (RESULTS 2026-07-12), per h:
HAND_BEST = {0.5: ("l4-HE", 0.00031, 40), 1.0: ("hva3", 0.00694, 6),
             2.0: ("hva3", 0.00030, 6)}


def search(H, e0):
    beam: list[tuple[float, list[str]]] = [(1e9, [])]
    best_seen: tuple[float, list[str]] = (1e9, [])
    for _ in range(MAX_TOKENS):
        cands: list[tuple[float, list[str]]] = []
        for _, toks in beam:
            for t in VOCAB:
                if toks and toks[-1] == t and t == "cz":
                    continue          # cz twice = identity
                nt = toks + [t]
                e = struct_optimize(H, nt, N, iters=120)
                err = (e - e0) / abs(e0)
                cands.append((err, nt))
        cands.sort(key=lambda c: c[0])
        beam = cands[:BEAM]
        if beam[0][0] < best_seen[0]:
            best_seen = beam[0]
        print(f"    depth {len(beam[0][1])}: best {beam[0][0]*100:.3f}% "
              f"{beam[0][1]}", flush=True)
    return best_seen


def main() -> None:
    ok = True
    for h in (0.5, 1.0, 2.0):
        H = build_tfim(N, h)
        e0 = exact_ground(H)
        t0 = time.time()
        err, toks = search(H, e0)
        npar = struct_nparams(toks, N)
        hname, herr, hpar = HAND_BEST[h]
        verdict = (err < herr and npar <= hpar)
        ok = ok and verdict
        print(f"h={h}: searched {toks} ({npar} params) err "
              f"{err*100:.3f}% vs hand {hname} {herr*100:.3f}% "
              f"({hpar}p) -> {'BEAT' if verdict else 'no'} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"BAR: searched beats best hand arm at every h at <= params "
          f"-> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
