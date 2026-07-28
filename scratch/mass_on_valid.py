"""Mass-on-valid (spec 2026-07-28 rung 2): teacher-forced sequence
probability mass over the engine-enumerated verified-valid next-step
set, vs the modal valid move (farm-pick proxy: fresh states have no
banked row). No sampling anywhere. successors() output is already
sympy-verified and non-identity (derivation.py docstring), so the
valid set is the enumeration itself.
"""
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from llmopt.search.derivation import State, successors  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402

MODELS = {  # ce_gate_study specimens, verbatim
    "muon":    ("checkpoints/mathnative_wfloor_d256_muon.pt", 34),
    "stream3": ("checkpoints/mathnative_wfloor_d256_stream3.pt", 45),
    "stream4": ("checkpoints/mathnative_wfloor_d256_stream4.pt", 57),
    "control": ("checkpoints/mathnative_wfloor_d256.pt", 65),
}
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
SEED = 99_100_000
X = sp.Symbol("x")
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"

cells = []
for lv in (3, 4, 5, 6, 7):
    for i in range(8):
        p = _gen_isolated(lv, SEED + 1000 * lv + i)
        if p is None:
            continue
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        kids = list(successors(State(sp.Integral(p._expr, X))))
        valid = list(dict.fromkeys(sp.sstr(s.expr) for _, s in kids))
        if len(valid) >= 2:
            cells.append((lv, cur, valid))
print(f"{len(cells)} states with >=2 valid moves "
      f"(mean {sum(len(v) for _, _, v in cells)/max(len(cells),1):.1f} "
      f"moves/state)", flush=True)


@torch.no_grad()
def seq_logprob(model, cur, nxt):
    pre = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
    try:
        full = tok.encode(f"Current: {cur}\nHints: none\nStep: {nxt}\n")
    except ValueError:
        return None  # move not spellable in the vocab; skip
    ids = torch.tensor([full + [tok.eos_id]], device=dev)
    lg = model(ids[:, :-1])[0]
    lp = F.log_softmax(lg, dim=-1)
    tgt = ids[0, 1:]
    return sum(lp[t, tgt[t]].item()
               for t in range(len(pre) - 1, len(tgt)))


print(f"{'model':8s} {'gate':>4s} {'mass_valid':>10s} "
      f"{'mass_modal':>10s} {'delta':>8s} {'H_valid':>8s}", flush=True)
for name, (path, gate) in MODELS.items():
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    model.load_state_dict(torch.load(path, map_location="cpu",
                                     weights_only=True))
    model.eval()
    mv = mp = hh = 0.0
    n_h = 0
    for lv, cur, valid in cells:
        lps = [seq_logprob(model, cur, v) for v in valid]
        ps = [math.exp(l) for l in lps if l is not None]
        if not ps:
            continue
        tot = sum(ps)
        mv += tot
        mp += max(ps)
        if tot > 0:
            q = [p / tot for p in ps if p > 0]
            hh += -sum(x * math.log2(x) for x in q)
            n_h += 1
    n = len(cells)
    print(f"{name:8s} {gate:4d} {mv/n:10.4f} {mp/n:10.4f} "
          f"{(mv-mp)/n:8.4f} {hh/max(n_h,1):8.3f}", flush=True)
    del model
