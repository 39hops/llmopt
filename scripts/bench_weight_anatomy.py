"""Weight anatomy: do closed-system (RL-climbed) weights LOOK
different from imitation (SFT) weights?

The bet (2026-07-14 night, pre-registered in the post-climb
strategy spec): Artin — elegant/simple OR interwoven-complex;
Claude — CONCENTRATED (mid-network mass, lower effective rank than
SFT at equal norm).

Specimens (all LoRA adapter dicts, {name.a, name.b}):
  RL move:   BA(step_lora_grpo run-2b final) - BA(pre_grpo promoted)
             — what ~23 GRPO cycles WROTE onto the promoted model.
  SFT moves: BA(adapter) for fresh-from-base SFT runs (control,
             dietB, syn0, syn3, pre_grpo itself = rounds-2/3) —
             what each SFT run wrote onto the base.
Both are "what the training process wrote"; norms are normalized
before shape comparisons.

Metrics per (layer, module): Frobenius mass of the written delta
-> depth profile. Per module: stable rank ||D||_F^2 / ||D||_2^2 ->
how many directions the write actually used. Iron-rule reminder:
weights can mislead — the FUNCTION verdicts (probes) ride alongside
in part 2 (layer sweep on climbed vs pre model).

    .venv/bin/python scripts/bench_weight_anatomy.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPECIMENS = {
    "pre_grpo(SFT r2/3)": "checkpoints/step_lora_pre_grpo_backup.pt",
    "SFT control": "checkpoints/step_lora_control.pt",
    "SFT dietB": "checkpoints/step_lora_dietB.pt",
    "SFT syn0": "checkpoints/step_lora_syn0.pt",
    "SFT syn3": "checkpoints/step_lora_syn3.pt",
}
RL_FINAL = "checkpoints/step_lora_grpo.pt"
RL_BASE = "checkpoints/step_lora_pre_grpo_backup.pt"
SCALING = 2.0  # alpha/r = 32/16


def composed(sd):
    """{layer_idx, module: BA weight-space delta}"""
    import torch
    out = {}
    pairs = defaultdict(dict)
    for k, v in sd.items():
        m = re.search(r"layers\.(\d+)\..*?(\w+_proj)\.(a|b)$", k)
        if not m:
            continue
        pairs[(int(m.group(1)), m.group(2))][m.group(3)] = v.float()
    for key, ab in pairs.items():
        if "a" in ab and "b" in ab:
            out[key] = SCALING * (ab["b"] @ ab["a"])
    return out


def depth_profile(deltas):
    import torch
    by_layer = defaultdict(float)
    for (l, _mod), d in deltas.items():
        by_layer[l] += float(d.norm() ** 2)
    tot = sum(by_layer.values()) or 1.0
    return {l: v / tot for l, v in sorted(by_layer.items())}


def stable_rank(deltas):
    import torch
    srs = []
    for d in deltas.values():
        f2 = float(d.norm() ** 2)
        s2 = float(torch.linalg.matrix_norm(d, 2) ** 2)
        if s2 > 1e-12:
            srs.append(f2 / s2)
    return sum(srs) / max(len(srs), 1)


def main() -> None:
    import torch
    rl_final = composed(torch.load(RL_FINAL, weights_only=False,
                                   map_location="cpu"))
    rl_base = composed(torch.load(RL_BASE, weights_only=False,
                                  map_location="cpu"))
    rl_move = {k: rl_final[k] - rl_base.get(k, torch.zeros_like(v))
               for k, v in rl_final.items()}
    rows = [("RL move (grpo - pre)", rl_move)]
    for name, path in SPECIMENS.items():
        sd = torch.load(path, weights_only=False, map_location="cpu")
        rows.append((name, composed(sd)))

    print(f"{'specimen':>22s} {'|D|_F':>8s} {'stable-rank':>11s} "
          f"{'mass@L0-7':>9s} {'L8-15':>6s} {'L16-23':>7s}")
    profiles = {}
    for name, deltas in rows:
        tot = sum(float(d.norm() ** 2) for d in deltas.values()) ** 0.5
        prof = depth_profile(deltas)
        profiles[name] = prof
        early = sum(v for l, v in prof.items() if l <= 7)
        mid = sum(v for l, v in prof.items() if 8 <= l <= 15)
        late = sum(v for l, v in prof.items() if l >= 16)
        print(f"{name:>22s} {tot:8.2f} {stable_rank(deltas):11.2f} "
              f"{100*early:8.0f}% {100*mid:5.0f}% {100*late:6.0f}%")
    print("\nper-layer mass, RL move vs mean SFT (percent):")
    sft_names = [n for n, _ in rows if n != "RL move (grpo - pre)"]
    for l in sorted(profiles["RL move (grpo - pre)"]):
        rl = 100 * profiles["RL move (grpo - pre)"][l]
        sf = 100 * sum(profiles[n].get(l, 0) for n in sft_names) / len(sft_names)
        bar_rl = "#" * int(rl * 2)
        print(f"  L{l:2d}  RL {rl:5.1f} {bar_rl:<20s} SFT {sf:5.1f}")


if __name__ == "__main__":
    main()
