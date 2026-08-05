"""MOE-GT-5c keep-sets: symbolic core + RANDOM non-core fill,
matched per-layer to the union mask's exact sizes.

The three-way discriminator for the GT-5 resurrection: per layer,
keep = frozen D3 core + a uniform random draw from the 128-minus-core
complement, sized so |keep| equals the union mask's |core | vcore|
at that layer (65-86). String-seeded per draw (house convention).
Note the expectation math registered in the pre-reg: a uniform
random fill captures ~41/91 of the non-core demand MASS in
expectation (~0.75 open recall on the arm0 axis), i.e. HIGHER than
the verbal fill's measured 0.729 — so R-SHOULDER predicts random
does at least as well as the union, not merely "something".

Writes checkpoints/gt5c_keep_r{0,1,2}.json.
"""

import json
import random

core = {int(li): set(v) for li, v in
        json.load(open("checkpoints/gt3_core_keep.json")).items()}
union = {int(li): set(v) for li, v in
         json.load(open("checkpoints/gt5_union_keep.json")).items()}

for draw in range(3):
    rng = random.Random(f"gt5c-randfill-{draw}")
    keep = {}
    for li in core:
        pool = sorted(set(range(128)) - core[li])
        n_fill = len(union[li]) - len(core[li])
        keep[str(li)] = sorted(core[li] | set(rng.sample(pool, n_fill)))
        assert len(keep[str(li)]) == len(union[li])
    out = f"checkpoints/gt5c_keep_r{draw}.json"
    json.dump(keep, open(out, "w"))
    sizes = [len(v) for v in keep.values()]
    print(f"wrote {out} | mean {sum(sizes)/len(sizes):.1f}/128")
