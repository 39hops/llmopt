"""2b re-check with an EXACT verifier (Fraction arithmetic — the
first checker itself rounded: c*2^74 > 2^53)."""
from fractions import Fraction
import numpy as np
exec(open("scratch/ozaki_rung2bc.py").read().split("# 2b:")[0])
worst = None
for i in range(0, N, 13):
    for j in range(0, N, 13):
        e = [0.0]
        for p in aligned_partials(A, B):
            e = exp_add(e, p[i, j])
        tot = sum(Fraction(c) for c in e) * (1 << SH)
        dev = tot - int(P[i, j])
        if worst is None or abs(dev) > abs(worst): worst = dev
print(f"[2b exact-check] max deviation = {worst}  (0 = perfectly exact)")
