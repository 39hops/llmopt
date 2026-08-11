"""Tensor-class ternary sensitivity profile (PRE-REG STAR-PROFILE-1,
2026-08-10). Star frame: precision belongs to interfaces/core;
traversal tolerates {-1,0,1}.

Post-hoc (PTQ) absmean ternarization of ONE tensor class at a time
on the fp-trained d256 crystal; standard 120 chain gate per arm.
Seven arms, one device, one seed, same gate seeds. Rows STREAM.

Usage (3080): python scratch/star_profile.py [ARM ...]
  arms default: base emb head norms attn ffn body
"""
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
import step_grpo_micro as G  # noqa: E402

CKPT = os.environ.get("SP_CKPT", "checkpoints/mathnative_wfloor_d256.pt")
D = int(os.environ.get("SP_D", 256))
LAYERS = int(os.environ.get("SP_L", 8))
FFN = int(os.environ.get("SP_FFN", 1024))
HEADS = int(os.environ.get("SP_H", 4))
OUT = os.environ.get("SP_OUT",
                     "logs/star_profile/star_profile_d256.jsonl")

CLASSES = {
    "emb": lambda k: k == "emb.weight",
    "head": lambda k: k.startswith("head."),
    "norms": lambda k: k.endswith(".g"),
    "attn": lambda k: (".qkv." in k or ".o." in k),
    "ffn": lambda k: (".gate." in k or ".up." in k or ".down." in k),
}
CLASSES["body"] = lambda k: CLASSES["attn"](k) or CLASSES["ffn"](k)


def ternary(w):
    """absmean ternary — row-scale for 2D (train_ternary.py's form),
    global absmean for 1D gain vectors."""
    if w.dim() >= 2:
        s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    else:
        s = w.abs().mean().clamp(min=1e-8)
    return torch.where(w.abs() < 0.5 * s,
                       torch.zeros_like(w), torch.sign(w) * s)


def main():
    arms = sys.argv[1:] or ["base", "emb", "head", "norms",
                            "attn", "ffn", "body"]
    tok = MathTokenizer()
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    sd0 = torch.load(CKPT, map_location="cpu", weights_only=True)
    os.makedirs("logs/star_profile", exist_ok=True)
    out = open(OUT, "a")
    for arm in arms:
        sd = {k: v.clone() for k, v in sd0.items()}
        n_q, n_tot = 0, sum(v.numel() for v in sd.values())
        if arm != "base":
            pick = CLASSES[arm]
            for k in sd:
                if pick(k):
                    sd[k] = ternary(sd[k])
                    n_q += sd[k].numel()
        model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                            heads=HEADS, ffn=FFN).to(dev)
        model.load_state_dict(sd)
        model.eval()
        solves, valid = G.gate_eval(model, tok, dev)
        tot = sum(solves.values())
        row = {"ckpt": os.path.basename(CKPT), "d": D, "layers": LAYERS,
               "arm": arm, "solves": solves, "total": tot,
               "valid_pct": round(valid, 2), "dev": dev,
               "quantized_params": n_q, "param_share": round(
                   n_q / n_tot, 4)}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"[{arm}] {solves} = {tot}/120 (valid {valid:.2f}%, "
              f"share {row['param_share']})", flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
