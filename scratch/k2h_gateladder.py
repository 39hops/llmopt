"""K2-HORIZON-GATE-LADDER-0: the frozen multi-difficulty oracle gate
ladder for the K2 Horizon lifecycle program (0.9B qualification, 3.7B
and 7B transport). Raw completion, one prompt format, string-seeded
items, sympy truth, fork-boxed sympy equivalence (never string match).
Six tiers of 40 items (two families each, 20 per family):

  T1 polynomial derivative / expand two factors
  T2 product-rule derivative with sin / polynomial antiderivative
  T3 chain-rule derivative exp(poly) / expand three factors
  T4 second derivative of x^k sin(bx) / antiderivative of x exp(bx)
  T5 definite integral of a polynomial on [0, c] (exact rational) /
     2x2 determinant with symbolic entries
  T6 antiderivative of x^2 exp(bx) (double parts) / derivative of a
     quotient (ax+b)/(cx+d) simplified

Items and few-shot blocks are frozen by string seed; the item digest is
printed and pinned in the prereg. A tier is READ on a model only if the
tier below is not floored (<= 2 of 40 at every seed) or the model is
the qualification vehicle. Calibration mode (LADDER_CAL=1) scores the
ladder on 0.9B tags only.

Usage:
    .venv-k2/bin/python scratch/k2h_gateladder.py --digest
    LADDER_CAL=1 .venv-k2/bin/python scratch/k2h_gateladder.py <tag> [<tag> ...]
"""
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location("k2h", os.path.join(ROOT, "scratch/k2h_stagecensus.py"))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

N_PER_FAMILY = 20
N_SHOTS = 3
TIERS = {1: ("poly_diff", "expand2"), 2: ("prod_diff", "poly_int"), 3: ("chain_diff", "expand3"),
         4: ("second_diff", "xexp_int"), 5: ("definite_int", "det2"), 6: ("x2exp_int", "quot_diff")}
OUT = os.path.join(ROOT, "logs/k2h/gateladder")


def _item(fam, rng):
    import sympy as sp
    X = sp.Symbol("x")
    a, b, c, d = (rng.randint(1, 9) for _ in range(4))
    k = rng.randint(2, 4)
    if fam == "poly_diff":
        f = a * X**k + b * X; truth = sp.diff(f, X); q = f"Differentiate with respect to x: {sp.sstr(f)}"
    elif fam == "expand2":
        f = (a * X + b) * (c * X + d); truth = sp.expand(f); q = f"Expand fully: {sp.sstr(f)}"
    elif fam == "prod_diff":
        f = a * X**k * sp.sin(b * X) + c * X; truth = sp.diff(f, X); q = f"Differentiate with respect to x: {sp.sstr(f)}"
    elif fam == "poly_int":
        f = a * X**k + b; truth = sp.integrate(f, X); q = f"Find an antiderivative with respect to x (no constant): {sp.sstr(f)}"
    elif fam == "chain_diff":
        f = sp.exp(a * X**2 + b * X); truth = sp.diff(f, X); q = f"Differentiate with respect to x: {sp.sstr(f)}"
    elif fam == "expand3":
        f = (a * X + b) * (c * X + d) * (X - k); truth = sp.expand(f); q = f"Expand fully: {sp.sstr(f)}"
    elif fam == "second_diff":
        f = a * X**k * sp.sin(b * X); truth = sp.diff(f, X, 2); q = f"Compute the second derivative with respect to x: {sp.sstr(f)}"
    elif fam == "xexp_int":
        f = a * X * sp.exp(b * X); truth = sp.integrate(f, X); q = f"Find an antiderivative with respect to x (no constant): {sp.sstr(f)}"
    elif fam == "definite_int":
        f = a * X**k + b * X + d; truth = sp.integrate(f, (X, 0, c)); q = f"Compute the definite integral of {sp.sstr(f)} with respect to x from 0 to {c}"
    elif fam == "det2":
        M = sp.Matrix([[a * X + b, c], [d, X - k]]); truth = sp.expand(M.det())
        q = f"Compute the determinant of the 2x2 matrix with rows [{sp.sstr(M[0, 0])}, {sp.sstr(M[0, 1])}] and [{sp.sstr(M[1, 0])}, {sp.sstr(M[1, 1])}], expanded"
    elif fam == "x2exp_int":
        f = a * X**2 * sp.exp(b * X); truth = sp.integrate(f, X); q = f"Find an antiderivative with respect to x (no constant): {sp.sstr(f)}"
    elif fam == "quot_diff":
        f = (a * X + b) / (c * X + d); truth = sp.simplify(sp.diff(f, X)); q = f"Differentiate with respect to x and simplify: {sp.sstr(f)}"
    else:
        raise ValueError(fam)
    return {"family": fam, "prompt": q, "truth": sp.sstr(truth)}


def make_items():
    items = []
    for tier, fams in TIERS.items():
        for fam in fams:
            seen, i, got = set(), 0, 0
            while got < N_PER_FAMILY:
                it = _item(fam, random.Random(f"k2h-ladder-{tier}-{fam}-{i}"))
                i += 1
                if it["prompt"] in seen:
                    continue
                seen.add(it["prompt"])
                it.update({"tier": tier, "id": len(items)})
                items.append(it)
                got += 1
    return items


def make_shots(seed, tier, exclude):
    shots = []
    for fam in TIERS[tier]:
        j, got = 0, 0
        while got < N_SHOTS:
            it = _item(fam, random.Random(f"k2h-ladder-shot-{seed}-{tier}-{fam}-{j}"))
            j += 1
            if it["prompt"] in exclude:
                continue
            shots.append(it)
            got += 1
    random.Random(f"k2h-ladder-shot-order-{seed}-{tier}").shuffle(shots)
    return shots


def digest(items):
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def run_tier(model, tok, items, tier, seed, tag, rows_f, counters):
    """Per-tier gate: tier-specific shot block, same render / parse /
    boxed-check path as the frozen census gate (imported)."""
    import torch
    sub = [it for it in items if it["tier"] == tier]
    shots = make_shots(seed, tier, {it["prompt"] for it in items})
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    correct, by_fam, ids_all = 0, {}, []
    for lo in range(0, len(sub), K.BATCH):
        batch = sub[lo:lo + K.BATCH]
        enc = tok([K.render(shots, it) for it in batch], return_tensors="pt", padding=True)
        enc.pop("token_type_ids", None)
        enc = {k: v.to("mps") for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=K.MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=nl[-1])
        n_in = enc["input_ids"].shape[1]
        for it, seq in zip(batch, out):
            gen = seq[n_in:].tolist()
            ans = K.parse_answer(tok.decode(gen, skip_special_tokens=True))
            ok = K.check_boxed(ans, it["truth"], counters)
            correct += ok
            by_fam[it["family"]] = by_fam.get(it["family"], 0) + int(ok)
            ids_all.append(gen)
            rows_f.write(json.dumps({"tag": tag, "seed": seed, "tier": tier, "item": it["id"], "family": it["family"], "answer": ans,
                                     "truth": it["truth"], "correct": bool(ok), "gen_ids": gen}) + "\n")
        rows_f.flush()
    return {"correct": correct, "n": len(sub), "by_family": by_fam, "gen_sha256": hashlib.sha256(json.dumps(ids_all).encode()).hexdigest()}


def main():
    items = make_items()
    if "--digest" in sys.argv:
        print(json.dumps({"n_items": len(items), "items_sha256": digest(items), "tiers": {t: [f for f in fs] for t, fs in TIERS.items()},
                          "example": {t: next(it["prompt"] for it in items if it["tier"] == t) for t in TIERS}}, indent=1))
        return
    if os.environ.get("LADDER_CAL") != "1":
        raise SystemExit("set LADDER_CAL=1 for the 0.9B calibration, or pass --digest")
    from llmopt.lab.provenance import completion_commit, start_provenance
    tags = [a for a in sys.argv[1:] if not a.startswith("--")]
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    # tags are MUTABLE on the hub (re-pointed 2026-09-05 19:02 EDT while the
    # residues run was live), so revisions are the FULL commits recorded in
    # the locked L66337 receipt, never tag names
    booked = json.load(open(os.path.join(ROOT, "logs/k2h/stagecensus/receipt.json")))
    lock = json.load(open(os.path.join(ROOT, "docs/receipts.lock.json")))["receipts"]
    if K.sha_file(os.path.join(ROOT, "logs/k2h/stagecensus/receipt.json")) != lock["logs/k2h/stagecensus/receipt.json"]["sha256"]:
        raise SystemExit("REFUSING: booked receipt sha v lock")
    START = start_provenance(["scratch/k2h_gateladder.py", "scratch/k2h_stagecensus.py", "logs/k2h/stagecensus/receipt.json"])
    os.makedirs(OUT)
    rec = {"prereg": "K2-HORIZON-GATE-LADDER-0", "smoke": False, "start": START, "items_sha256": digest(items), "n_items": len(items), "tags": {}, "gate": {}}
    counters = {"timeout": 0, "empty": 0}
    rows_f = open(os.path.join(OUT, "ladder_rows.jsonl"), "w")
    t0 = time.time()
    for tag in tags:
        from huggingface_hub import snapshot_download
        commit = booked["tags"][tag]["commit"]
        path = snapshot_download(K.MODEL, revision=commit, allow_patterns=K.ALLOW)
        if os.path.basename(path) != commit:
            raise SystemExit(f"REFUSING: {tag} resolved to {os.path.basename(path)}, booked {commit}")
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
        shards = {sh: K.sha_file(os.path.join(path, sh)) for sh in sorted(set(idx["weight_map"].values()))}
        if shards != booked["tags"][tag]["shard_sha256"]:
            raise SystemExit(f"REFUSING: {tag} shard sha256 v booked receipt")
        model, tok, dev, rope = K.load_model(path)
        rec["tags"][tag] = {"commit": commit, "shard_sha256": shards, "rope_parameters": rope, "device": dev}
        rec["gate"][tag] = {}
        for tier in TIERS:
            g = run_tier(model, tok, items, tier, 0, tag, rows_f, counters)
            rec["gate"][tag][tier] = g
            print(f"[ladder] {tag} T{tier} {g['correct']}/{g['n']} {g['by_family']} {round(time.time() - t0)}s", flush=True)
        del model
    rows_f.close()
    rec["oracle_counters"] = counters
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["completion_commit"] = completion_commit()
    json.dump(rec, open(os.path.join(OUT, "receipt.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
