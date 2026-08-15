"""GRAD-MAP-0 desk probe — gradient/data-worth atlas signatures.

Spec: docs/superpowers/specs/2026-08-15-grad-map-0.md (amended
2026-08-15, pre-look). New driver, not a member of the frozen
birth19m family; imports birth19m_curric (library class) and
copies the frozen loss path verbatim (birth19m_atoms_rule.py:126-139
shape, backward only, no optimizer step, no clip — raw gradients
per the amended spec).

Phases (all Mac mps fp32, same process):
  1. load families + failure columns (from logs/pp_gradmap0_stock_s3.jsonl)
  2. v_proxy: running mean of g^2 over 50 batches of stock stream
  3. per-family mean gradients at stock_s3, TWO passes (repeatability)
  4. per-row sampled gradients (metric 5, streaming identity)
  5. dose-rider family-gradient norms at dose1/dose3p5/dose7
  6. metrics + receipts -> logs/gradmap0/

SMOKE=1 shrinks every population and writes receipts to
logs/gradmap0/smoke.jsonl; real receipts never see smoke rows.
Signatures are device-local (Mac); no cross-device comparison.
"""
import hashlib
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("ARM", "off")        # frozen module import side-effects only
os.environ.setdefault("BIRTH_SEED", "3")

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import torch  # noqa: E402

import birth19m_curric as C          # noqa: E402  (frozen, import-only)
import train_mathnative as TM        # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402
from llmopt.lab.gen import _gen_isolated            # noqa: E402
from llmopt.lab.jsonl import append_jsonl           # noqa: E402
from llmopt.lab.hash import git_sha                 # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUTDIR = "logs/gradmap0"
RECEIPT = f"{OUTDIR}/{'smoke' if SMOKE else 'signatures'}.jsonl"
PP_PATH = "logs/pp_gradmap0_stock_s3.jsonl"
GATE_BAND = 9_900_000
SEQ_CAP = int(os.environ.get("SEQ_CAP", "512"))  # = C.encode_with_levels default
BS = 32
M5_SAMPLE = 8 if SMOKE else 512      # per-row gradient sample per family
VPROXY_BATCHES = 3 if SMOKE else 50
FAM_CAP = 64 if SMOKE else None      # SMOKE row cap per family
STOCK_SAMPLE = 6000                  # stock-chain family sample (matches atom scale)
LEVEL_SAMPLE = 2000                  # per-level slice sample
PROJ_DIM = 64                        # metric 7 low-rank projection (stored dims)
CKPTS = {
    "stock": "checkpoints/gallery19m_stock_s3.pt",
    "dose1": "checkpoints/gallery19m_dose1_s3.pt",
    "dose3p5": "checkpoints/gallery19m_dose3p5_s3.pt",
    "dose7": "checkpoints/gallery19m_dose7_s3.pt",
}


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_model(ckpt_path, tok):
    model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6,
                           ffn=1536).to(DEV)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.train()                    # gradient mode; no dropout in this arch
    return model


def encode_rows(rows, tok):
    """The trainer's text/encode/filter path, verbatim format."""
    enc = []
    for r in rows:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try:
            ids = tok.encode(t) + [tok.eos_id]
        except ValueError:           # unencodable row: skip, as the library does
            continue
        if len(ids) <= SEQ_CAP:
            enc.append(ids)
    enc.sort(key=len)                # stable length sort, as stock
    return enc


def batches(enc, bs=BS):
    for i in range(0, len(enc) - bs + 1, bs):
        yield enc[i:i + bs]
    tail = enc[len(enc) - (len(enc) % bs):]
    if tail and len(enc) >= bs:
        pass                         # drop ragged tail, matches stream shape
    elif tail:
        yield tail                   # tiny family: keep the only batch


def backward_on(model, tok, batch):
    """Loss path copied from the frozen sibling; backward only."""
    L = max(len(s) for s in batch)
    ids = torch.tensor([s + [tok.pad_id] * (L - len(s)) for s in batch],
                       device=DEV)
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in batch],
                        device=DEV)
    logits = model(ids[:, :-1], mask[:, :-1])
    labels = ids[:, 1:].clone()
    labels[mask[:, 1:] == 0] = -100
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=-100)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return float(loss.detach())


def flat_grad(model):
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for _, p in sorted(model.named_parameters())]).to("cpu", torch.float32)


def group_slices(model):
    """Parameter-group index slices over the sorted-name flat vector."""
    slices, off = {}, 0
    for name, p in sorted(model.named_parameters()):
        n = p.numel()
        if name.startswith("emb"):
            g = "emb"
        else:
            try:
                blk = int(name.split(".")[1])
                g = "early" if blk < 4 else "late"
            except (IndexError, ValueError):
                g = "other"
        slices.setdefault(g, []).append((off, off + n))
        off += n
    return slices, off


def group_vec(vec, spans):
    return torch.cat([vec[a:b] for a, b in spans])


def cos(a, b):
    a, b = a.double(), b.double()    # fp32 dot over 19M dims drifts ~0.5%
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


def mean_grad(model, tok, enc, label, max_batches=None):
    acc, nb = None, 0
    for batch in batches(enc):
        backward_on(model, tok, batch)
        g = flat_grad(model)
        acc = g if acc is None else acc + g
        nb += 1
        if max_batches and nb >= max_batches:
            break
    assert nb > 0, f"empty family {label}"
    return acc / nb, nb


def per_row_stats(model, tok, enc, sample_key):
    """Metric 5 via the streaming identity: within-family mean pairwise
    dot = (|S|^2 - sum|g_i|^2) / (n(n-1)); returns (S, sum_sq, n)."""
    rows = list(enc)
    random.Random(sample_key).shuffle(rows)
    rows = rows[:M5_SAMPLE]
    S, sum_sq, n = None, 0.0, 0
    for r in rows:
        backward_on(model, tok, [r])
        g = flat_grad(model)
        gd = g.double()
        S = gd if S is None else S + gd
        sum_sq += float(gd @ gd)
        n += 1
    return S, sum_sq, n


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    tok = TM.MathTokenizer()         # default vocab: gallery family born with VOCAB_EXTRA unset

    # ---- families -------------------------------------------------
    atoms = [json.loads(l) for l in open("data/micromodel_atoms_shard0.jsonl")]
    heur = [r for r in atoms if r["rule"] == "i_heurisch"]
    noheur = [r for r in atoms if r["rule"] != "i_heurisch"]
    assert (len(atoms), len(heur), len(noheur)) == (6000, 2782, 3218)

    ax_all = [json.loads(l)
              for l in open("data/micromodel_atoms_axiom_shard0.jsonl")]
    random.Random("dose-ladder-1").shuffle(ax_all)   # the dose driver's exact draw
    ax = ax_all[:6000]
    band = set(gate_band_exprs())
    ax = [r for r in ax if norm(str(r["cur"])) not in band
          and norm(str(r["nxt"])) not in band]

    stock_rows = C.load_excised_rows()
    stock_sample = list(stock_rows)
    random.Random("gradmap0-stock").shuffle(stock_sample)
    stock_sample = stock_sample[:STOCK_SAMPLE]
    lv_slices = {}
    for lv in (3, 4, 5, 6, 7):
        rows_lv = [r for r in stock_rows if int(r.get("level", 0)) == lv]
        random.Random(f"gradmap0-L{lv}").shuffle(rows_lv)
        lv_slices[lv] = rows_lv[:LEVEL_SAMPLE]

    families = {
        "heur": heur, "noheur": noheur, "sympy6k": atoms,
        "axiom6k": ax, "stock": stock_sample,
    }
    families.update({f"L{lv}": rs for lv, rs in lv_slices.items() if rs})
    enc_fam = {}
    for name, rows in families.items():
        rows = rows[:FAM_CAP] if FAM_CAP else rows
        enc_fam[name] = encode_rows(rows, tok)
        print(f"[fam] {name}: {len(rows)} rows -> {len(enc_fam[name])} seq",
              flush=True)

    # ---- failure columns (stock_s3's own gate) --------------------
    pp = [json.loads(l) for l in open(PP_PATH)]
    cols = {}
    for lv in (3, 4, 5, 6, 7):
        fails = [r for r in pp if r["level"] == lv and not r["solved"]]
        col_rows = []
        for r in fails:
            p = _gen_isolated(lv, GATE_BAND + 1000 * lv + r["i"])
            assert p is not None, (lv, r["i"])
            col_rows.append({"cur": r["root"], "nxt": p.answer})
        if col_rows:
            cols[f"F{lv}"] = encode_rows(col_rows, tok)
        print(f"[col] F{lv}: {len(fails)} failures", flush=True)
    if SMOKE:
        cols = {k: v[:4] for k, v in cols.items()}

    # ---- signature checkpoint: stock_s3 ---------------------------
    model = load_model(CKPTS["stock"], tok)
    spans, total_params = group_slices(model)
    print(f"[model] {total_params} params, groups "
          f"{ {g: sum(b-a for a, b in s) for g, s in spans.items()} }",
          flush=True)

    # metric 2 proxy: v = running mean of g^2 over stock stream
    v_acc, nb = None, 0
    for batch in batches(enc_fam["stock"]):
        backward_on(model, tok, batch)
        g2 = flat_grad(model) ** 2
        v_acc = g2 if v_acc is None else v_acc + g2
        nb += 1
        if nb >= VPROXY_BATCHES:
            break
    v_proxy = v_acc / nb
    precond = 1.0 / torch.sqrt(v_proxy + 1e-8)
    print(f"[vproxy] {nb} batches, {time.time()-t0:.0f}s", flush=True)

    # mean gradients, two same-process passes (repeatability fence)
    sig, sig2, repeat = {}, {}, {}
    gen = torch.Generator().manual_seed(
        int.from_bytes(hashlib.sha256(b"gradmap0-proj").digest()[:8], "big"))
    proj_chunks = []                 # fp16 chunks, ~2.4GB total at PROJ_DIM=64
    CH = 1_000_000
    for a in range(0, total_params, CH):
        b = min(a + CH, total_params)
        proj_chunks.append(
            (torch.randn(PROJ_DIM, b - a, generator=gen)
             / math.sqrt(PROJ_DIM)).to(torch.float16))

    def project(vec):
        acc = torch.zeros(PROJ_DIM)
        for k, a in enumerate(range(0, total_params, CH)):
            b = min(a + CH, total_params)
            acc += proj_chunks[k].float() @ vec[a:b]
        return acc
    for name, enc in {**enc_fam, **cols}.items():
        sig[name], nb1 = mean_grad(model, tok, enc, name)
        sig2[name], _ = mean_grad(model, tok, enc, name)
        repeat[name] = {
            "global": cos(sig[name], sig2[name]),
            **{g: cos(group_vec(sig[name], s), group_vec(sig2[name], s))
               for g, s in spans.items()},
        }
        print(f"[sig] {name}: {nb1} batches, repeat_cos "
              f"{repeat[name]['global']:.6f} (emb {repeat[name].get('emb', 0):.6f})"
              f", {time.time()-t0:.0f}s", flush=True)

    # metric 5 per-row streaming stats (heur/noheur cross for R3)
    m5 = {}
    for name in ("heur", "noheur", "sympy6k", "ctrl3218"):
        if name == "ctrl3218":
            rows = [json.loads(l)
                    for l in open("data/micromodel_atoms_ctrl3218.jsonl")]
            enc = encode_rows(rows[:FAM_CAP] if FAM_CAP else rows, tok)
        else:
            enc = enc_fam[name]
        S, sum_sq, n = per_row_stats(model, tok, enc, f"gradmap0-m5-{name}")
        m5[name] = {"S": S, "sum_sq": sum_sq, "n": n}
        within = ((float(S @ S) - sum_sq) / (n * (n - 1))) if n > 1 else 0.0
        m5[name]["within"] = within
        print(f"[m5] {name}: n={n} within={within:.6e}, "
              f"{time.time()-t0:.0f}s", flush=True)
    m5_cross = float(m5["heur"]["S"] @ m5["noheur"]["S"]) / (
        m5["heur"]["n"] * m5["noheur"]["n"])

    # ---- dose rider: family-gradient norm at each dose checkpoint --
    dose_norms = {}
    for tag in ("dose1", "dose3p5", "dose7"):
        m2 = load_model(CKPTS[tag], tok)
        g, nb2 = mean_grad(m2, tok, enc_fam["axiom6k"], f"axnorm-{tag}",
                           max_batches=10 if SMOKE else None)
        dose_norms[tag] = {"norm": float(g.norm()), "batches": nb2}
        del m2
        print(f"[dose] {tag}: |g|={dose_norms[tag]['norm']:.6e}", flush=True)

    # ---- RD4 surface baseline (no gradients) ----------------------
    def toks_of(enc):
        return [set(s) for s in enc]
    surface = {}
    col_toks = {c: toks_of(v) for c, v in cols.items()}
    for name, enc in enc_fam.items():
        fam_t = toks_of(enc[:512])
        surface[name] = {}
        for cname, cts in col_toks.items():
            js = [len(a & b) / len(a | b) for a in fam_t for b in cts]
            surface[name][cname] = sum(js) / len(js) if js else 0.0

    # ---- matrix ----------------------------------------------------
    fam_names = list(enc_fam)
    col_names = list(cols)
    matrix = {}
    for f in fam_names:
        matrix[f] = {}
        for c in col_names:
            gf, gc = sig[f], sig[c]
            matrix[f][c] = {
                "cos": cos(gf, gc),
                "cos_precond": cos(gf * precond, gc * precond),
                "proj": (float((gf.double() @ gc.double()) / gc.double().norm())
                         if float(gc.norm()) else 0.0),
                "cos_late": cos(group_vec(gf, spans["late"]),
                                group_vec(gc, spans["late"])),
                "cos_early": cos(group_vec(gf, spans["early"]),
                                 group_vec(gc, spans["early"])),
                "proj_late": float(
                    (group_vec(gf, spans["late"]).double()
                     @ group_vec(gc, spans["late"]).double())
                    / group_vec(gc, spans["late"]).double().norm()),
                "surface_knn": surface[f][c],
            }

    row = {
        "probe": "gradmap0", "smoke": SMOKE, "device": DEV,
        "dtype": "fp32", "code_commit": git_sha(short=True),
        "ckpt": {k: {"path": v, "sha16": sha16(v)} for k, v in CKPTS.items()},
        "pp_receipt": PP_PATH,
        "vproxy_batches": nb, "m5_sample": M5_SAMPLE,
        "stock_sample": STOCK_SAMPLE, "level_sample": LEVEL_SAMPLE,
        "proj_dim": PROJ_DIM, "seq_cap": SEQ_CAP, "bs": BS,
        "fam_norms": {f: float(sig[f].norm()) for f in fam_names},
        "col_norms": {c: float(sig[c].norm()) for c in col_names},
        "repeat_cos": repeat,
        "matrix": matrix,
        "m5_within": {k: v["within"] for k, v in m5.items()},
        "m5_cross_heur_noheur": m5_cross,
        "m5_n": {k: v["n"] for k, v in m5.items()},
        "dose_norms": dose_norms,
        "wall_s": round(time.time() - t0, 1),
    }
    # low-rank projected signatures, storable
    row["proj_sig"] = {f: [round(x, 6) for x in project(sig[f]).tolist()]
                       for f in fam_names + col_names}
    append_jsonl(RECEIPT, row)
    print(f"[done] receipt -> {RECEIPT}, wall {row['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
