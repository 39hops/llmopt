"""EX6-MED-0 run-2 cells driver (frozen pre-launch): the
registered stepwise-decode fallback after run 1's qualification
failure (AMENDMENT EX6-MED-0-QUALFAIL — the appended-position
batch seam broke token identity on 12/240 diagonals).

Native call shapes preserved exactly: the prompt is processed as
ONE batch through a prompt cache, the forced token is fed as a
true T=1 call, and every subsequent token is a greedy argmax T=1
step (EOS-terminated, MAX_TOKENS budget). The instrument therefore
sees the same phase sequence as ordinary generation — batch
'prefill', first T=1 'prompt_tail' (= the forced token's own
routing), then 'decode' — so the ORIGINAL EX6 PROMPT predicate
(prefill + prompt_tail) applies verbatim to S=PROMPT cells; no
BATCH predicate exists in this driver.

Qualification (fail-closed, per prereg + standing review items):
z-table sha verified before use; seed-7001 diagonals compared as
LITERAL token-ID sequences INCLUDING length against native ids
captured via stream_generate under the frozen instrument; per-seed
cell-exact solve-count gates (NONE 64/61/66, PROMPT 79/80/79)
invalidate their own seed's cells.

Reuses scratch/ex6med.py for instrument/PREDS/prompt_text/paths
(imported, never edited). Receipts: logs/ex6med/{cells2.jsonl,
qual2.jsonl,summary2.json} (refuse-if-exists; SMOKE=1 -> smoke2_*).

    .venv/bin/python scratch/ex6med2.py                       (Mac)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scratch.ex6med as x  # noqa: E402
import scratch.moe_gt1_arm2 as m  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = [7001] if SMOKE else [7001, 8002, 9003]
N_EVAL = 4 if SMOKE else 120
PRE = "smoke2_" if SMOKE else ""
DIR = Path("logs/ex6med")
CELLS = DIR / f"{PRE}cells2.jsonl"
QUAL = DIR / f"{PRE}qual2.jsonl"
SUMM = DIR / f"{PRE}summary2.json"


def eos_set(tok):
    ids = getattr(tok, "eos_token_ids", None)
    if ids:
        return set(ids)
    return {tok.eos_token_id}


def stepwise(model, tok, ids, forced, budget):
    """Batch prompt through a prompt cache, feed `forced` as a true
    T=1 call, then greedy argmax T=1 steps. Returns generated ids
    starting with `forced` (EOS kept, as tokenizers stream it)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    eos = eos_set(tok)
    cache = make_prompt_cache(model)
    model(mx.array(ids)[None], cache=cache)
    out = [forced]
    cur = forced
    while len(out) < budget:
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
        if cur in eos:
            break
    return out


def native_ids(model, tok, keep, text, S, budget):
    from mlx_lm import stream_generate
    _, restore = x.instrument(model, keep, x.PREDS[S])
    try:
        out = []
        for r in stream_generate(model, tok, prompt=text,
                                 max_tokens=budget):
            out.append(int(r.token))
        return out
    finally:
        restore()


def main():
    for pth in (CELLS, QUAL, SUMM):
        if pth.exists():
            raise SystemExit(f"REFUSING: {pth} exists")
    START = start_provenance(
        ["scratch/ex6med2.py", "scratch/ex6med.py",
         "scratch/moe_gt1_arm2.py", x.KEEPSET])
    zsha = hashlib.sha256(x.ZCAP.read_bytes()).hexdigest()
    recorded = json.loads(x.ZSHA.read_text())["zcap_sha256"]
    assert zsha == recorded, ("z table hash mismatch", zsha, recorded)
    ztab = {}
    for line in x.ZCAP.open():
        r = json.loads(line)
        ztab[(r["seed"], r["state"], r["idx"])] = r["z1"]

    from mlx_lm import load

    from llmopt.mathgen.problems import make_dataset
    model, tok = load(m.MODEL)
    keep = {int(li): set(v) for li, v in
            json.loads(Path(x.KEEPSET).read_text()).items()}

    eos = eos_set(tok)
    cf, qf = CELLS.open("w"), QUAL.open("w")
    ident_ok = True
    invalid_seeds = set()
    counts = {}
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        native = {}
        if seed == 7001:
            for S in ("NONE", "PROMPT"):
                for i, p in enumerate(problems):
                    native[(S, i)] = native_ids(
                        model, tok, keep, x.prompt_text(tok, p),
                        S, m.MAX_TOKENS)
                print(f"[qual2] native {S} ids captured", flush=True)
        for S in ("NONE", "PROMPT"):
            for zsrc in ("NONE", "PROMPT"):
                pred = x.PREDS[S]
                n_ok = 0
                t0 = time.time()
                for i, p in enumerate(problems):
                    text = x.prompt_text(tok, p)
                    z1 = ztab[(seed, zsrc, i)]
                    _, restore = x.instrument(model, keep, pred)
                    try:
                        gen = stepwise(model, tok,
                                       tok.encode(text), z1,
                                       m.MAX_TOKENS)
                    finally:
                        restore()
                    # generate() strips special tokens from text;
                    # decode() keeps them and breaks the oracle
                    # extraction — strip EOS ids before decoding
                    comp = tok.decode([t for t in gen
                                       if t not in eos])
                    expr = m.extract_expression(comp)
                    (ok, parsed), t_out = m.check_isolated(p, expr)
                    n_ok += ok
                    cf.write(json.dumps({
                        "seed": seed, "state": S, "zsrc": zsrc,
                        "idx": i, "ok": bool(ok), "parsed": parsed,
                        "timeout": bool(t_out),
                        "gen_ids": len(gen)}) + "\n")
                    cf.flush()
                    if seed == 7001 and S == zsrc:
                        nat = native[(S, i)]
                        ident = gen == nat
                        row = {"seed": seed, "diag": S, "idx": i,
                               "token_id_identical": bool(ident)}
                        if not ident:
                            first = next(
                                (j for j in range(
                                    min(len(gen), len(nat)))
                                 if gen[j] != nat[j]),
                                min(len(gen), len(nat)))
                            row.update(len_native=len(nat),
                                       len_forced=len(gen),
                                       first_mismatch_pos=first)
                            ident_ok = False
                        qf.write(json.dumps(row) + "\n")
                        qf.flush()
                booked = (x.BOOKED[S].get(seed)
                          if S == zsrc else None)
                if S == zsrc and not SMOKE and n_ok != booked:
                    invalid_seeds.add(seed)
                    print(f"[qual2] DIAG COUNT MISMATCH {seed}/{S}:"
                          f" {n_ok} v booked {booked}", flush=True)
                counts[f"{seed}:{S}:{zsrc}"] = n_ok
                print(f"[cells2] seed {seed} S={S} z={zsrc}: "
                      f"{n_ok}/{len(problems)} "
                      f"{time.time() - t0:.0f}s", flush=True)
    cf.close()
    qf.close()
    SUMM.write_text(json.dumps({
        "note": "EX6-MED-0 run-2 (stepwise fallback) cell counts; "
                "interpretation gated by diagonals per prereg",
        "start": START, "completion_commit": completion_commit(),
        "zcap_sha256_verified": zsha,
        "token_id_identity_seed7001_pass": bool(ident_ok),
        "count_invalid_seeds": sorted(invalid_seeds),
        "cell_counts": counts}, indent=1) + "\n")
    if not ident_ok:
        print("[cells2] QUALIFICATION-FAIL: seed-7001 diagonals "
              "not token-ID-identical; no treatment read",
              flush=True)
        sys.exit(3)
    print(f"[cells2] done; invalid_seeds={sorted(invalid_seeds)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
