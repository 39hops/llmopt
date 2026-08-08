"""Extract one gate_proj matrix from HF-cached Qwen2.5-0.5B base
and Instruct into plot_neurons-compatible .pt files, so the
--displace (central-lattice whisper-zoom) view can render an
INTERNET-trained model's post-training displacement next to the
closed-system natives' (the chaos-vs-structure tell, Artin's ask
2026-08-08; generator-loss lesson: this adapter is COMMITTED).

Layer choice: L14 gate_proj [4864, 896] — the same matrix class
the original crystal drew (weight-anatomy era).

Usage: .venv/bin/python scratch/qwen_displace_extract.py
Writes: checkpoints/qwen05b_base_l14gate.pt / _instruct_l14gate.pt
        (tiny single-matrix files, untracked per data convention)
"""
import glob
import os

import torch
from safetensors.torch import load_file

KEY = "model.layers.14.mlp.gate_proj.weight"
OUT_KEY = "blocks.14.gate.weight"  # plot_neurons key_sub-friendly

for tag, repo in [("base", "models--Qwen--Qwen2.5-0.5B"),
                  ("instruct", "models--Qwen--Qwen2.5-0.5B-Instruct")]:
    snaps = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/{repo}/snapshots/*/*.safetensors"))
    assert snaps, f"no safetensors for {repo}"
    sd = load_file(snaps[0])
    W = sd[KEY].float()
    out = f"checkpoints/qwen05b_{tag}_l14gate.pt"
    torch.save({OUT_KEY: W}, out)
    print(f"{out}: {tuple(W.shape)}")
