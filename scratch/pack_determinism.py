"""PACKED CRYSTAL C4 (pre-reg 2026-07-29 night): cross-device
determinism. Hash A: integer-GEMM outputs of every block Linear's
sigma-law codes x a fixed integer activation battery, accumulated
via fp64 matmul (all partials integers < 2^53 -> EXACT, reduction-
order-invariant) — must match across devices. Hash B: fp32 full
forward logits on a fixed prompt battery — expected to differ.
Greedy token streams reported alongside. Run on each machine at the
same commit; compare printed hashes. __main__-guarded.
"""
import hashlib
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

from bench_step_tokens import _gen_isolated  # noqa: E402
import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CKPT = "checkpoints/sym_birth_dense_mps_h8_ema.pt"
CFG = dict(d=64, layers=8, heads=8, ffn=256)


def main():
    tok = MathTokenizer()
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    base = torch.load(CKPT, map_location="cpu", weights_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False  # exactness fence

    # --- hash A: exact integer GEMM on-device (fp32 carrier: MPS has
    # no fp64; |code|<2^6, |x|<2^6, sum over <=256 -> partials < 2^24,
    # exact in fp32 and reduction-order-invariant) ---
    h = hashlib.sha256()
    g = torch.Generator().manual_seed(4)
    for k in sorted(base):
        v = base[k]
        if not (v.ndim == 2 and k.startswith("blocks.")):
            continue
        s = float(v.float().std())
        q = math.ceil(2.0 / s)
        codes = torch.round(v.float() * q).to(dev)
        x = torch.randint(-(2 ** 6), 2 ** 6,
                          (v.shape[1], 8), generator=g,
                          dtype=torch.int64).float().to(dev)
        y = codes @ x  # integer values in fp32: exact, order-invariant
        yi = y.to(torch.int64).cpu().numpy()
        assert float((y - y.round()).abs().max()) == 0.0
        h.update(yi.tobytes())
    print(f"C4 hash A (integer GEMM, {dev}): {h.hexdigest()}",
          flush=True)

    # --- hash B: fp32 full forward + greedy streams ---
    m = build_model(len(tok.vocab), **CFG).to(dev)
    m.load_state_dict({k: v.to(dev) for k, v in base.items()})
    m.eval()
    hb = hashlib.sha256()
    streams = []
    with torch.no_grad():
        for lv in G.GATE_LEVELS:
            p = _gen_isolated(lv, G.GATE_BAND + 700_000 + lv)
            if p is None:
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            ids = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
            t = torch.tensor([ids], device=dev)
            hb.update(m(t).float().cpu().numpy().tobytes())
            toks = []
            for _ in range(40):
                nxt = int(m(t)[0, -1].argmax())
                toks.append(nxt)
                t = torch.cat([t, torch.tensor([[nxt]], device=dev)], 1)
            streams.append(toks)
    print(f"C4 hash B (fp32 logits, {dev}): {hb.hexdigest()}",
          flush=True)
    hs = hashlib.sha256(repr(streams).encode()).hexdigest()
    print(f"C4 greedy streams ({dev}): {hs} "
          f"({len(streams)} prompts x 40 toks)", flush=True)


if __name__ == "__main__":
    main()
