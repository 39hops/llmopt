"""GRAD-MAP-0 RD2 completion pass — ctrl3218 v noheur signatures at
stock_s3 against the F4/F6 failure columns (the pre-declared L6
falsifier rider, spec 2026-08-15 amended 475a6f3). Companion to
scratch/gradmap0_probe.py; imports its functions, writes a fully
provenanced row to the same receipt file."""
import json
import os
import sys
import time

os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "3")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import gradmap0_probe as P  # noqa: E402

t0 = time.time()
tok = P.TM.MathTokenizer()
model = P.load_model(P.CKPTS["stock"], tok)
fams = {
    "ctrl3218": [json.loads(l)
                 for l in open("data/micromodel_atoms_ctrl3218.jsonl")],
    "noheur": [json.loads(l)
               for l in open("data/micromodel_atoms_noheur.jsonl")],
}
pp = [json.loads(l) for l in open(P.PP_PATH)]
cols = {}
for lv in (4, 6):
    rows = []
    for r in pp:
        if r["level"] == lv and not r["solved"]:
            p = P._gen_isolated(lv, P.GATE_BAND + 1000 * lv + r["i"])
            assert p is not None, (lv, r["i"])
            rows.append({"cur": r["root"], "nxt": p.answer})
    cols[f"F{lv}"] = P.encode_rows(rows, tok)
sigs = {}
for name, rows in fams.items():
    sigs[name], nb = P.mean_grad(model, tok, P.encode_rows(rows, tok), name)
    print(f"[sig] {name}: {nb} batches |g|={float(sigs[name].norm()):.4f}",
          flush=True)
for c, enc in cols.items():
    sigs[c], _ = P.mean_grad(model, tok, enc, c)
row = {
    "probe": "gradmap0-rd2", "smoke": False, "device": P.DEV,
    "dtype": str(next(model.parameters()).dtype),
    "code_commit": P.git_sha(short=True),
    "ckpt": {"stock": {"path": P.CKPTS["stock"],
                       "sha16": P.sha16(P.CKPTS["stock"])}},
    "pp_receipt": P.PP_PATH, "pp_sha16": P.sha16(P.PP_PATH),
    "cos": {f"{n}:{c}": P.cos(sigs[n], sigs[c])
            for n in fams for c in cols},
    "norms": {k: float(v.norm()) for k, v in sigs.items()},
    "wall_s": round(time.time() - t0, 1),
}
P.append_jsonl("logs/gradmap0/signatures.jsonl", row)
print(json.dumps(row["cos"], indent=1))
