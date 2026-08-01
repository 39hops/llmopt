"""House-side acceptance of axiom's intbirth PRIMITIVE layer
(relay 2026-08-01-3): rebuild the R2b training loop from
intbirth.Block / AdamW / rdiv alone, house-authored composition,
and check all 8 r2b_ref.json milestone digests + losses. This is
also the shape the multi-block reference will take (dx0 chaining,
one AdamW over the concatenated param list).

Usage: .venv/bin/python scratch/verify_intbirth_prims.py \
           [axiom_build_dir] [tables.bin]
"""
import hashlib
import json
import sys

import numpy as np

BUILD = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/artin/code/axiom/build-rel"
TABLES = sys.argv[2] if len(sys.argv) > 2 else \
    "/Users/artin/code/axiom/tools/int_adamw/r2b_tables.bin"
sys.path.insert(0, BUILD)
import intbirth  # noqa: E402

REF = "scratch/detbwd_r2b_ref"
ref = json.load(open(f"{REF}/r2b_ref.json"))
ct = ref["contract"]
Q, SHIFT, GBOOST = 512, ct["SHIFT"], ct["GBOOST"]
KEYS = ("wq", "wk", "wv", "wo", "wg", "wu", "wd", "wh",
        "g1", "g2", "g3")
T, D, DH, F, V = 32, 64, 16, 128, 64
SHAPES = {"wq": (DH, D), "wk": (DH, D), "wv": (DH, D),
          "wo": (D, DH), "wg": (F, D), "wu": (F, D), "wd": (D, F),
          "wh": (V, D), "g1": (D,), "g2": (D,), "g3": (D,)}

init = open(f"{REF}/r2b_init.bin", "rb").read()
assert hashlib.sha256(init).hexdigest() == ref["init_sha"], \
    "init artifact does not match ref"
raw = np.frombuffer(init, dtype="<i8")
off, wide = 0, {}
for k in KEYS:
    n = int(np.prod(SHAPES[k]))
    wide[k] = (raw[off:off + n].reshape(SHAPES[k]) << SHIFT).copy()
    off += n
x = raw[off:off + T * D].reshape(T, D).copy()
off += T * D
tgt = raw[off:off + T].copy()

blk = intbirth.Block(open(TABLES, "rb").read(), ct)
opt = intbirth.AdamW(SHIFT, 1, 1000)
onehot = np.zeros((T, V), dtype=np.int64)
onehot[np.arange(T), tgt] = 1

th = hashlib.sha256()
prev, ok = 0, True
for ms in ref["milestones"]:
    for _ in range(ms["step"] - prev):
        w = {k: intbirth.rdiv(wide[k], 1 << SHIFT) for k in KEYS}
        logits, cache = blk.fwd(w, x)
        pp = blk.softmax_rows(logits, Q)
        loss = int((Q - pp[np.arange(T), tgt]).sum())
        G, dx0 = blk.bwd(w, (pp - Q * onehot) * GBOOST, cache)
        opt.step([wide[k] for k in KEYS],
                 [intbirth.rdiv(G[k], Q * GBOOST) for k in KEYS])
    prev = ms["step"]
    for k in KEYS:
        th.update(wide[k].tobytes())
    good = th.hexdigest() == ms["traj_sha"] and loss == ms["loss"]
    ok &= good
    print(f"step {ms['step']:4d} loss {loss} nz {opt.nz:.3f} "
          f"{'PASS' if good else 'FAIL'}", flush=True)
print("HOUSE-SIDE PRIMITIVES ACCEPTANCE: "
      + ("ALL 8 MILESTONES PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
