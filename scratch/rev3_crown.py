"""REVIVE track item 2 wrapper: CROWN-TIE BIRTHS — the production
crown tie (gen6_grown fp32 76 v merged_grown ternary 75, booked
2026-07-23; draw-noise leg closed by HARDENING-P2 R8 "tiebreak
still needs births") gets n=3 SAME-DEVICE fresh birth pairs.

ARMS (one pair per seed, both on the Mac):
  c = the champion LINE fresh: fp32 birth d512/L12/h8/ffn2048 on
      the gen-4 rebirth diet (v22+gen4, 3ep), grow +256 FFN/layer
      (grow_mathnative spray, fp32 = exactly function-preserving),
      identity pre-check (gate(grown) MUST equal gate(birth) —
      exact, zero-tolerance: zeroed down-cols change nothing),
      then 3 warm epochs on the gen-6 corpus (v22+l8+gen4).
      Growth-inheritance IS the champion's mechanism — the full
      lineage replicates, not just the endpoint arch.
  m = the merged LINE fresh: ternary tournament birth DIRECTLY at
      d768/L8/h12/ffn3840 on data/merged_diet.jsonl (3ep).
      No growth stage: ternary growth is not function-preserving
      (RESULTS 2026-07-23 ~12AM, absmean coupling) — the fresh
      birth tests the LINE, not that one-off pathology.

Hard gates (house wrapper clauses): D2 excision on load_rows
(patched before any driver binds it — covers the diet= path too,
the wrap is on the function not the flag), refuse-if-exists on
every stage-1/grown OUT, BIRTH_SEED env, GRAD_CKPT default on.

Usage: ARM=c|m SEED=<n> .venv/bin/python scratch/rev3_crown.py
Device: Mac mps (Artin's call — same-device pairing, never read
against the original cross-era cells).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

ARM = os.environ["ARM"]
SEED = os.environ["SEED"]
assert ARM in ("c", "m"), ARM
os.environ["BIRTH_SEED"] = SEED
os.environ.setdefault("GRAD_CKPT", "1")

if ARM == "m":
    targets = [Path(f"checkpoints/tourn_T_crown_s{SEED}.pt"),
               Path(f"checkpoints/tourn_T_crown_s{SEED}_latent.pt")]
else:
    BIRTH = Path(f"checkpoints/crown_c_birth_s{SEED}.pt")
    GROWN = Path(f"checkpoints/crown_c_grown_s{SEED}.pt")
    targets = [BIRTH, GROWN]
for t in targets:
    if t.exists():
        raise SystemExit(f"REFUSING: {t} exists (use unspent SEED)")

import train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

_orig_rows = TM.load_rows


def excised_load_rows(*a, **kw):
    rows = _orig_rows(*a, **kw)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[rev3-crown] D2 excision: {len(rows)} -> {len(kept)} "
          f"rows ({len(rows) - len(kept)} excised)", flush=True)
    return kept


TM.load_rows = excised_load_rows


def _gate(ckpt, d, layers, heads, ffn):
    import torch
    import step_grpo_micro as G
    from llmopt.train.mathnative import MathTokenizer, build_model
    tok = MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(len(tok.vocab), d=d, layers=layers,
                       heads=heads, ffn=ffn).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                     weights_only=True))
    model.eval()
    solves, valid = G.gate_eval(model, tok, dev)
    print(f"[rev3-crown] gate {ckpt}: {solves} = "
          f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)
    return solves


if ARM == "m":
    sys.argv = ["tournament_birth.py", "--alpha", "T",
                "--diet", "data/merged_diet.jsonl",
                "--epochs", "3", "--d", "768", "--layers", "8",
                "--ffn", "3840", "--heads", "12",
                "--tag", f"_crown_s{SEED}"]
    from tournament_birth import main as tourn_main  # noqa: E402
    tourn_main()
else:
    # stage 1: gen-4-recipe fp32 birth
    TM.main(v2=False, d=512, layers=12, ffn=2048, heads=8,
            out=str(BIRTH), v21=False, fast=False,
            v22=True, gen4=True, epochs=3)
    # stage 2: function-preserving growth (+256 FFN/layer)
    sys.argv = ["grow_mathnative.py", "--src", str(BIRTH),
                "--out", str(GROWN), "--grow", "256"]
    from grow_mathnative import main as grow_main  # noqa: E402
    grow_main()
    # identity pre-check: fp32 growth must be gate-EXACT
    g_birth = _gate(BIRTH, 512, 12, 8, 2048)
    g_grown = _gate(GROWN, 512, 12, 8, 2304)
    if g_birth != g_grown:
        raise SystemExit(f"IDENTITY PRE-CHECK FAILED: {g_birth} "
                         f"!= {g_grown} — growth not preserving")
    print("[rev3-crown] identity pre-check PASSED (exact)",
          flush=True)
    # stage 3: warm epochs on the gen-6 corpus (.ep=-1 resume)
    TM.main(v2=False, d=512, layers=12, ffn=2304, heads=8,
            out=str(GROWN), v21=False, fast=False,
            v22=True, gen4=True, l8=True, epochs=3)
