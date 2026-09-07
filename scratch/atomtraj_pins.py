"""ATOM-DIET-TRAJECTORY-1 pin derivation (pre-reg RESULTS 2026-09-07).
Zero training. Reproduces every constant the pre-reg freezes from the
frozen curric functions, the frozen shard, the pinned torch scheduler
and two existing seed-3 stock checkpoints:

  * the six actual token-id batch-stream digests (stock/atoms x epoch),
    with the secondary counts (enc sizes, dropped batches, atom rows);
  * the OneCycle peak optimizer step under the pinned arguments;
  * the counterbalanced arm order (2:1 within-pair, seed chosen by a
    fixed hash of the rung name);
  * the canonical state-dict digest law (state_digest);
  * the literal 59-key P8 / C8 law and the fixed-seed mps run-noise
    anchor (n = 1 pair: booked stock s3 v the L29985 rerun control).

Usage: .venv/bin/python scratch/atomtraj_pins.py   (prints a JSON block)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ["ARM"] = "off"          # frozen module import side-effects only
os.environ.setdefault("BIRTH_SEED", "3")

import torch  # noqa: E402

import birth19m_curric as C  # noqa: E402
import train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

RUNG = "ATOM-DIET-TRAJECTORY-1"
SEEDS = (5, 6, 7)
STEPS_TOTAL = 15_420
GRID = 1_028
SHARD = Path("data/micromodel_atoms_shard0.jsonl")
BLOCK2D = ("qkv.weight", "o.weight", "gate.weight", "up.weight", "down.weight")
CLASSES = {
    "qkv": [f"blocks.{l}.qkv.weight" for l in range(8)],
    "o": [f"blocks.{l}.o.weight" for l in range(8)],
    "gate": [f"blocks.{l}.gate.weight" for l in range(8)],
    "up": [f"blocks.{l}.up.weight" for l in range(8)],
    "down": [f"blocks.{l}.down.weight" for l in range(8)],
    "norms": [f"blocks.{l}.{n}.g" for l in range(8) for n in ("n1", "n2")] + ["norm.g"],
    "emb": ["emb.weight"],
    "head": ["head.weight"],
}


def state_digest(sd):
    """Canonical tensor digest: keys sorted, each tensor as float32
    contiguous little-endian bytes preceded by its name and shape."""
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().to("cpu", torch.float32).contiguous()
        h.update(f"{k}:{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def stream_digest(enc, stream):
    h = hashlib.sha256()
    for a, b in stream:
        for j in range(a, b):
            h.update(json.dumps(enc[j]).encode())
        h.update(b"|")
    return h.hexdigest()


def encode_flagged(rows, tok):
    triples = []
    for r in rows:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try:
            ids = tok.encode(t) + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= int(os.environ.get("SEQ_CAP", "512")):
            triples.append((ids, int(r["level"]), r.get("source") == "atom-oneply"))
    triples.sort(key=lambda p: len(p[0]))
    return [p[0] for p in triples], [p[2] for p in triples]


def p8_c8(sd, sd0):
    e = [sum(float(((sd[f"blocks.{l}.{t}"] - sd0[f"blocks.{l}.{t}"]).double() ** 2).sum()) for t in BLOCK2D) for l in range(8)]
    tot = sum(e)
    p8 = [x / tot for x in e]
    ce = {c: sum(float(((sd[k] - sd0[k]).double() ** 2).sum()) for k in ks) for c, ks in CLASSES.items()}
    ct = sum(ce.values())
    return p8, {c: v / ct for c, v in ce.items()}, sum(i * x for i, x in enumerate(p8)), tot


def arm_order():
    h = hashlib.sha256(f"{RUNG}-order".encode()).hexdigest()
    atoms_first = SEEDS[int(h, 16) % 3]
    order = []
    for s in SEEDS:
        pair = ["atoms", "stock"] if s == atoms_first else ["stock", "atoms"]
        order += [f"{a}_s{s}" for a in pair]
    return h[:16], atoms_first, order


def main():
    out = {"rung": RUNG, "torch": torch.__version__}
    tok = TM.MathTokenizer()
    torch.manual_seed(3)
    m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    sd0 = {k: v.detach().clone() for k, v in m.state_dict().items()}
    assert sorted(sum(CLASSES.values(), [])) == sorted(sd0), "C8 classes must partition the state dict"
    assert m.head.weight.data_ptr() != m.emb.weight.data_ptr(), "head must be untied from emb"
    out["n_keys"] = len(sd0)
    out["n_params"] = sum(v.numel() for v in sd0.values())
    # OneCycle peak step: lr applied at optimizer step k (1-indexed)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=STEPS_TOTAL, pct_start=0.03)
    lrs = []
    for _ in range(STEPS_TOTAL):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        if sched.last_epoch < STEPS_TOTAL - 1:
            sched.step()
    mx = max(lrs)
    out["onecycle_peak_steps"] = [i + 1 for i, v in enumerate(lrs) if v == mx]
    out["snapshot_steps"] = [0] + out["onecycle_peak_steps"] + list(range(GRID, STEPS_TOTAL + 1, GRID))
    out["order_hash"], out["atoms_first_seed"], out["execution_order"] = arm_order()
    # streams
    stock_rows = C.load_excised_rows()
    enc_stock, _ = C.encode_with_levels(stock_rows, tok)
    atoms = [json.loads(l) for l in SHARD.open()]
    band = set(gate_band_exprs())
    atoms = [r for r in atoms if norm(str(r["cur"])) not in band and norm(str(r["nxt"])) not in band]
    enc_atoms, is_atom = encode_flagged(stock_rows + atoms, tok)
    spe = len(C.stock_epoch_stream(len(enc_stock), 0))
    out["shard_sha256"] = hashlib.sha256(SHARD.read_bytes()).hexdigest()
    out["shard_rows_after_excision"] = len(atoms)
    out["enc_stock"], out["enc_atoms"], out["steps_per_epoch"] = len(enc_stock), len(enc_atoms), spe
    out["steps_total"] = C.EPOCHS * (len(enc_stock) // C.BS)
    assert out["steps_total"] == STEPS_TOTAL
    out["streams"] = {}
    for arm, enc, fl in (("stock", enc_stock, [False] * len(enc_stock)), ("atoms", enc_atoms, is_atom)):
        for ep in range(C.EPOCHS):
            st = C.stock_epoch_stream(len(enc), ep)
            dropped = len(st) - spe
            st = st[:spe]
            out["streams"][f"{arm}_e{ep}"] = {"sha256": stream_digest(enc, st), "dropped": dropped,
                                              "atom_rows": sum(1 for a, b in st for j in range(a, b) if fl[j])}
    # noise anchor, n = 1 pair
    anchor = {}
    for name in ("gallery19m_stock_s3.pt", "gallery19m_softspeed_control_s3.pt"):
        p = Path("checkpoints") / name
        if not p.exists():
            anchor[name] = "ABSENT"
            continue
        sd = torch.load(p, map_location="cpu")
        p8, c8, cen, tot = p8_c8(sd, sd0)
        anchor[name] = {"file_sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16], "state_digest": state_digest(sd)[:16],
                        "P8": [round(x, 5) for x in p8], "C8": {k: round(v, 5) for k, v in c8.items()}, "centroid": round(cen, 4), "E2d": tot}
    vals = [v for v in anchor.values() if v != "ABSENT"]
    if len(vals) == 2:
        a, b = vals
        anchor["P8_L2"] = round(sum((x - y) ** 2 for x, y in zip(a["P8"], b["P8"])) ** 0.5, 6)
        anchor["C8_L2"] = round(sum((a["C8"][k] - b["C8"][k]) ** 2 for k in a["C8"]) ** 0.5, 6)
        anchor["centroid_diff"] = round(a["centroid"] - b["centroid"], 4)
        anchor["E2d_rel"] = round(abs(a["E2d"] - b["E2d"]) / a["E2d"], 5)
    out["noise_anchor_n1_pair"] = anchor
    out["init_state_digest_seed3"] = state_digest(sd0)[:16]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
