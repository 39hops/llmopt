"""V4-Flash offline anatomy: streamed experts -> instruments -> lake.

The first drive of llmopt.lab.shards (2026-08-11): sample cached
routed-expert w1 matrices from the local V4-Flash shard cache (44 GB,
built by the frozen v4flash_f1* drivers), one expert at a time —
dequant exactly, weigh (capacity meter M, kurtosis, row-norm stats),
land rows in the lake, and render the dot-view triptych over the
pooled sample.

EXPLORATION grade — no pre-reg, nothing books from this run; it
exists to prove the streaming path end-to-end and to give the
big-MoE dot view its first real render. Heavy IO: never run beside a
live Mac training run.

Usage: .venv/bin/python scratch/v4flash_anatomy.py [N_EXPERTS]
Outputs: logs/v4flash_anatomy/neurons-v4flash-{dark,light}.png,
rows appended to data/lake/weights.parquet.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import torch  # noqa: E402

from llmopt.lab import anatomy, lake, shards  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OUT = "logs/v4flash_anatomy"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows, mats = [], []
    for label, W in shards.iter_v4flash_experts(sample=N, seed=1):
        rows.append(shards.weigh(W, source=label,
                                 model="deepseek-v4-flash", proj="w1"))
        mats.append(W)
        if len(rows) % 16 == 0:
            print(f"[v4anat] {len(rows)}/{N} weighed", flush=True)
    lake.append_weights(rows)
    print(f"[v4anat] {len(rows)} rows -> data/lake/weights.parquet",
          flush=True)

    pooled = torch.cat(mats)
    outs = anatomy.render_dot_views(
        pooled, f"{OUT}/neurons-v4flash",
        title="DeepSeek V4-Flash — weight-space anatomy of "
              f"{len(rows)} routed experts",
        source_label=f"w1 rows of {len(rows)} sampled experts, "
                     "exact MXFP4 dequant",
        provenance="checkpoints/v4flash_f1 (V4-F1d cache) · "
                   "v4flash_anatomy.py")
    for o in outs:
        print(f"[v4anat] saved {o}", flush=True)
    ms = sorted(r["meter_m_bits"] for r in rows)
    print(f"[v4anat] meter M bits: min {ms[0]:.2f} "
          f"median {ms[len(ms) // 2]:.2f} max {ms[-1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
