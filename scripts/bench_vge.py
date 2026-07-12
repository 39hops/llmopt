"""Variational ground-state engine, rung-1 race (spec 2026-07-12).

TFIM n=10 across the phase diagram: h=0.5 (ordered), h=1.0 (the
CRITICAL point — entanglement peaks, mean-field is worst), h=2.0
(paramagnetic). Arms: product state (mean-field baseline, layers=0),
shallow (2 layers), deeper (4 layers). Pre-registered bar: deeper
reaches <1% relative error AT CRITICALITY and beats product at
every h.
"""
import time

from llmopt.quantum.ground import build_tfim, exact_ground, optimize

N = 10


def main() -> None:
    print(f"{'h':>4s} {'exact E0':>10s} {'product':>9s} {'2-layer':>9s} "
          f"{'4-layer':>9s}   (relative errors)")
    ok_bar = True
    for h in (0.5, 1.0, 2.0):
        H = build_tfim(N, h)
        e0 = exact_ground(H)
        errs = {}
        for name, layers, iters in (("product", 0, 200),
                                    ("l2", 2, 300), ("l4", 4, 300)):
            t0 = time.time()
            e, _ = optimize(H, N, layers, iters=iters)
            errs[name] = (e - e0) / abs(e0)
            print(f"    {name} L{layers}: E={e:.4f} "
                  f"err={errs[name]*100:.3f}% ({time.time()-t0:.0f}s)",
                  flush=True)
        print(f"{h:4.1f} {e0:10.4f} {errs['product']*100:8.3f}% "
              f"{errs['l2']*100:8.3f}% {errs['l4']*100:8.3f}%")
        if not (errs["l4"] < errs["product"]):
            ok_bar = False
        if h == 1.0 and errs["l4"] > 0.01:
            ok_bar = False
    print("BAR (pre-registered): l4 < 1% at h=1.0 AND l4 < product at "
          f"every h -> {'PASS' if ok_bar else 'FAIL'}")


if __name__ == "__main__":
    main()
