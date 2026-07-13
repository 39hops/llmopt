"""Ansatz-structure search 2b: evolutionary (rung 2's greedy FAILED —
first-token prefix lock; mutation can rewrite any position).
Population over token sequences, mutate (replace/insert/delete),
param-count penalty, elites refined with bigger budgets + restarts.

Recalibrated bar (rediscovery): per phase, find a structure with
err <= 1.1x the best HAND arm at <= its params — the search must
RECOVER expert design without expert knowledge. Phase-signature
check rides along (rotation blocks vs Hamiltonian blocks per phase).
"""
import random
import time

from llmopt.quantum.ground import (build_tfim, exact_ground,
                                   struct_nparams, struct_optimize)

N = 10
VOCAB = ("ry", "rx", "cz", "zz", "xm")
HAND_BEST = {0.5: ("l4-HE", 0.00031, 40), 1.0: ("hva3", 0.00694, 6),
             2.0: ("hva3", 0.00030, 6)}
POP, GENS = 12, 14


def mutate(rng, toks):
    toks = list(toks)
    op = rng.choice(["replace", "insert", "delete"])
    if op == "delete" and len(toks) > 2:
        toks.pop(rng.randrange(len(toks)))
    elif op == "insert" and len(toks) < 6:
        toks.insert(rng.randrange(len(toks) + 1), rng.choice(VOCAB))
    else:
        toks[rng.randrange(len(toks))] = rng.choice(VOCAB)
    return toks


def evolve(H, e0, rng):
    def score(toks, iters=150, seeds=(0,)):
        e = min(struct_optimize(H, toks, N, iters=iters, seed=s)
                for s in seeds)
        err = (e - e0) / abs(e0)
        return err + 0.0004 * struct_nparams(toks, N), err

    pop = [[rng.choice(VOCAB) for _ in range(rng.randint(3, 6))]
           for _ in range(POP)]
    scored = [(score(t)[0], t) for t in pop]
    for g in range(GENS):
        scored.sort(key=lambda x: x[0])
        elites = scored[: POP // 3]
        children = [mutate(rng, t) for _, t in elites * 2]
        scored = elites + [(score(t)[0], t) for t in children]
        if g % 4 == 0:
            print(f"    gen {g}: best {scored[0][0]:.5f} "
                  f"{scored[0][1]}", flush=True)
    scored.sort(key=lambda x: x[0])
    # refine top-3 with bigger budget + restarts, report true err
    best = (1e9, None)
    for _, toks in scored[:3]:
        _, err = score(toks, iters=350, seeds=(0, 1))
        if err < best[0]:
            best = (err, toks)
    return best


def main() -> None:
    ok = True
    for h in (0.5, 1.0, 2.0):
        H = build_tfim(N, h)
        e0 = exact_ground(H)
        rng = random.Random(int(h * 10))
        t0 = time.time()
        err, toks = evolve(H, e0, rng)
        npar = struct_nparams(toks, N)
        hname, herr, hpar = HAND_BEST[h]
        verdict = err <= herr * 1.1 and npar <= hpar
        ok = ok and verdict
        print(f"h={h}: {toks} ({npar}p) err {err*100:.3f}% vs hand "
              f"{hname} {herr*100:.3f}% ({hpar}p) -> "
              f"{'REDISCOVERED' if verdict else 'no'} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"BAR (rediscovery): match hand best within 10% at <= params "
          f"at every h -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
