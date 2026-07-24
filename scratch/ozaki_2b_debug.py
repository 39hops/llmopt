from fractions import Fraction
import numpy as np
exec(open("scratch/ozaki_rung2bc.py").read().split("# 2b:")[0])
i = j = 0
parts = [p[i, j] for p in aligned_partials(A, B)]
frac_sum = sum(Fraction(x) for x in parts) * (1 << SH)
print("exact Fraction sum of partials == P ?",
      frac_sum == int(P[i, j]), "dev", frac_sum - int(P[i, j]))
e = [0.0]
for x in parts:
    e = exp_add(e, x)
exp_sum = sum(Fraction(c) for c in e) * (1 << SH)
print("expansion sum == P ?", exp_sum == int(P[i, j]),
      "dev", exp_sum - int(P[i, j]), "| comps", len(e))
