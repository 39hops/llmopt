"""SOFT-PROMPT-1: is there capability behind the tokenizer?

Optimal-input-vector rung (Artin, 2026-08-11 night): the model's
input does not have to be text. Learn P continuous prefix vectors
(as VIRTUAL TOKENS — the embedding gains P trainable rows, ids
V..V+P-1; every original weight frozen) by CE on the model's own
diet, then run the standard 120 gate with the prefix prepended to
every prompt. Controls: the plain gate (no prefix) and an untrained
random prefix (mechanism perturbation control).

If the trained prefix clears the plain gate by >= 8 solves, there is
capability the tokenizer's discrete alphabet cannot elicit but an
optimal vector can. Virtual ids are masked out of the output logits
(-1e9), so decode can never emit them; the oracle path is untouched.

Usage (3080): .venv/bin/python scratch/softprompt1.py <ckpt> <d>
  env: P (prefix len, default 8), STEPS (default 600), SMOKE=1
Receipts: logs/softprompt1/.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, "scripts")

import torch  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

P = int(os.environ.get("P", "8"))
STEPS = int(os.environ.get("STEPS", "600"))
SMOKE = os.environ.get("SMOKE") == "1"
OUT = "logs/softprompt1"


def with_virtual_tokens(ckpt, d, layers, ffn, heads, dev):
    """Model whose vocab is V+P: rows [:V] are the frozen checkpoint,
    rows [V:] are the trainable prefix. Head logits for virtual ids
    are clamped so decode cannot emit them."""
    tok = MathTokenizer()
    V = len(tok.vocab)
    sd = torch.load(ckpt, map_location="cpu") if ckpt else None
    model = build_model(V + P, d=d, layers=layers, heads=heads,
                        ffn=ffn).to(dev)
    if sd is not None:
        with torch.no_grad():
            # copy every tensor; emb/head first V rows only
            msd = model.state_dict()
            for k, w in sd.items():
                if k in ("emb.weight", "head.weight"):
                    msd[k][:V] = w
                else:
                    msd[k].copy_(w)
    for p_ in model.parameters():
        p_.requires_grad = False
    model.emb.weight.requires_grad = True

    def _mask_virtual(grad):  # only the P new rows learn
        grad[:V] = 0
        return grad
    model.emb.weight.register_hook(_mask_virtual)

    head_fwd = model.head.forward

    def head_masked(x):
        out = head_fwd(x)
        out[..., V:] = -1e9
        return out
    model.head.forward = head_masked
    prefix_ids = list(range(V, V + P))
    return model, tok, V, prefix_ids


def diet_batches(tok, n_steps, bs=24, seq_cap=192):
    """CE batches from the standard diet (same rows the ckpt was
    born on — the prefix learns to help on the model's own world)."""
    import random
    rows = [json.loads(l) for l in open("data/metallicity/z3.jsonl")]
    rng = random.Random("softprompt-1")
    rng.shuffle(rows)
    texts = []
    for r in rows:
        s = f"{r['cur']}>{r['nxt']}"
        ids = tok.encode(s, strict=False)
        if ids and len(ids) <= seq_cap:
            texts.append(ids)
        if len(texts) >= n_steps * bs:
            break
    for i in range(0, n_steps * bs, bs):
        yield texts[i:i + bs]


def train_prefix(model, tok, prefix_ids, dev, steps):
    opt = torch.optim.Adam([model.emb.weight], lr=3e-3)
    pad = tok.pad_id
    step = 0
    for batch in diet_batches(tok, steps):
        L = max(len(s) for s in batch)
        ids = torch.tensor([prefix_ids + s + [pad] * (L - len(s))
                            for s in batch], device=dev)
        mask = torch.tensor([[1] * P + [1] * len(s) + [0] * (L - len(s))
                             for s in batch], device=dev)
        logits = model(ids[:, :-1], mask[:, :-1])
        labels = ids[:, 1:].clone()
        labels[:, :P - 1] = -100          # no loss on prefix positions
        labels[mask[:, 1:] == 0] = -100
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
            ignore_index=-100)
        loss.backward()
        opt.step()
        opt.zero_grad()
        step += 1
        if step % 100 == 0 or step == 1:
            print(f"  prefix step {step}/{steps} ce {loss:.4f}",
                  flush=True)
    return model


def gate_with_prefix(model, tok, dev, prefix_ids):
    """The standard 120 gate, prompts prefixed with the virtual ids.
    tok.encode is wrapped for the duration; everything else is the
    stock gate_eval path."""
    import step_grpo_micro as G
    orig = tok.encode

    def enc(s, strict=True):
        return prefix_ids + orig(s, strict)
    tok.encode = enc
    try:
        solves, valid = G.gate_eval(model, tok, dev)
    finally:
        tok.encode = orig
    return solves, valid


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" \
        else None
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    layers, ffn, heads = 8, d * 4, 4
    dev = ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    import step_grpo_micro as G

    cells = {}
    # plain gate (no prefix, stock model width V)
    model, tok, V, prefix_ids = with_virtual_tokens(
        ckpt, d, layers, ffn, heads, dev)
    if SMOKE:
        ids = torch.tensor([prefix_ids + tok.encode("Integral(x, x)",
                                                    False)], device=dev)
        out = model(ids, torch.ones_like(ids))
        assert out.shape[-1] == V + P and float(out[..., V:].max()) < -1e8
        model = train_prefix(model, tok, prefix_ids, dev, steps=3)
        print("SMOKE OK: forward, virtual-logit mask, prefix grads",
              flush=True)
        return
    model.eval()
    solves, valid = G.gate_eval(model, tok, dev)
    cells["plain"] = (solves, sum(solves.values()), valid)
    print(f"plain gate: {solves} = {cells['plain'][1]}/120", flush=True)

    # random-prefix control (untrained rows)
    solves, valid = gate_with_prefix(model, tok, dev, prefix_ids)
    cells["random_prefix"] = (solves, sum(solves.values()), valid)
    print(f"random-prefix gate: {solves} = "
          f"{cells['random_prefix'][1]}/120", flush=True)

    # trained prefix
    model.train()
    model = train_prefix(model, tok, prefix_ids, dev, STEPS)
    model.eval()
    solves, valid = gate_with_prefix(model, tok, dev, prefix_ids)
    cells["trained_prefix"] = (solves, sum(solves.values()), valid)
    print(f"trained-prefix gate: {solves} = "
          f"{cells['trained_prefix'][1]}/120", flush=True)

    json.dump({k: {"solves": v[0], "total": v[1], "valid": v[2]}
               for k, v in cells.items()},
              open(f"{OUT}/cells.json", "w"), indent=1)
    print("SOFT-PROMPT-1 done", flush=True)


if __name__ == "__main__":
    main()
