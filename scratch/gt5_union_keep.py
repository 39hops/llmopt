"""MOE-GT-5 keep-set: per-layer UNION of the two branch cores.

symbolic core = checkpoints/gt3_core_keep.json (the frozen D3 core,
3-way math&phys&code at matched 80-prompt size); verbal core =
prose&dialog keep-sets recomputed from the trajs (the gt4_verbal_core
pipeline). Writes checkpoints/gt5_union_keep.json in the KEEPSET
format ({layer: [experts]}) for scratch/moe_gt1_arm2.py.

The point (pre-reg MOE-GT-5): the union is 78.2/128 = 61.1% per
layer — numerically at MOE-GT-1's 61.8% tie-with-full keep — but by
construction it contains BOTH bases and few of math's ~21 extension
experts. Identity-vs-fraction gets its sharpest test yet.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gt2_jaccard import decode_counts, keep

core_sym = {int(li): set(v) for li, v in
            json.load(open("checkpoints/gt3_core_keep.json")).items()}
kp = keep(decode_counts("logs/opus/gt3_prose_traj.jsonl"))
kd = keep(decode_counts("logs/opus/gt4_dialog_traj.jsonl"))
union = {li: core_sym[li] | (kp[li] & kd[li]) for li in core_sym}
sizes = [len(v) for v in union.values()]
out = "checkpoints/gt5_union_keep.json"
json.dump({str(li): sorted(v) for li, v in union.items()}, open(out, "w"))
print(f"wrote {out} | mean {sum(sizes)/len(sizes):.1f}/128 "
      f"(min {min(sizes)} max {max(sizes)})")
