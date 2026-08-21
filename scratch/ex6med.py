"""EX6-MED-0 driver (frozen pre-launch): 2x2 mediation factorial
per PRE-REG EX6-MED-0 (docs/RESULTS.md) — prompt-state {NONE,
PROMPT-masked} x forced first token {z_NONE, z_PROMPT} on the
frozen EX6 stack (scratch/moe_gt1_arm2.py gate, seeds
7001/8002/9003, named-80 keepset).

The router wrapper below is the frozen ex6_phase wrapped-gate math
with the mask decision parametrized (every call runs the same
graph with a phase-selected mask, zero vector when unmasked, and
the always-forced want/.tolist() step — the v1 lazy-fusion
lesson). Mask predicates: NONE (never), PROMPT (prefill +
prompt_tail, the EX6 arm), BATCH (batch calls only — used for
forced-launch cells where the forced ids sit at the end of the
prompt' batch, so the masked position set equals EX6-PROMPT's).

PASS=cap    capture pass, OUTCOME-BLIND: native first-4 token ids
            per (seed, state, problem) via stream_generate under
            the state's mask; NO oracle call, no crossed cell.
            Writes zcap.jsonl + zcap_sha.json (sha256 of the z
            table) and a call-shape census for problem 0.
PASS=cells  refuses to run without zcap_sha.json; verifies the z
            table hash; runs seed-7001 native full completions
            (frozen-path generate, modes NONE/PROMPT) for the
            token-identity qualification, then all 4 cells x 3
            seeds with K=1 forced launch. Streams every row.
            Diagonal mismatch prints QUALIFICATION-FAIL and books
            no summary.

Receipts under logs/ex6med/ (refuse-if-exists; SMOKE=1 -> smoke_*
paths, N=4 problems, seed 7001 only).

    TRAJ off. PASS=cap  .venv/bin/python scratch/ex6med.py    (Mac)
              PASS=cells .venv/bin/python scratch/ex6med.py   (Mac)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scratch.moe_gt1_arm2 as m  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SMOKE = os.environ.get("SMOKE") == "1"
PASS = os.environ.get("PASS", "cap")
SEEDS = [7001] if SMOKE else [7001, 8002, 9003]
N_EVAL = 4 if SMOKE else 120
KEEPSET = "checkpoints/ex3_del_invp.json"
PRE = "smoke_" if SMOKE else ""
DIR = Path("logs/ex6med")
ZCAP = DIR / f"{PRE}zcap.jsonl"
ZSHA = DIR / f"{PRE}zcap_sha.json"
CELLS = DIR / f"{PRE}cells.jsonl"
QUAL = DIR / f"{PRE}qual.jsonl"
BOOKED = {"NONE": {7001: 64, 8002: 61, 9003: 66},
          "PROMPT": {7001: 79, 8002: 80, 9003: 79}}


def instrument(model, keep, pred):
    """Frozen ex6_phase wrapped-gate math, mask decision = pred(phase)."""
    import mlx.core as mx
    state = {"hits": 0, "slots": 0}
    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    masks, zeros, keepsets, tail_done = {}, {}, {}, {}
    for li, block in moe_layers:
        kept = keep[li]
        n_exp = block.gate.weight.shape[0]
        assert len(kept) >= block.top_k
        masks[id(block)] = mx.array(
            [0.0 if e in kept else float("-inf") for e in range(n_exp)])
        zeros[id(block)] = mx.array([0.0] * n_exp)
        keepsets[id(block)] = kept
    cls = type(moe_layers[0][1])
    original = cls.__call__
    shapes = state["shapes"] = []

    def phase_of(self, n_tokens):
        if n_tokens > 1:
            tail_done[id(self)] = False
            return "prefill"
        if not tail_done.get(id(self), False):
            tail_done[id(self)] = True
            return "prompt_tail"
        return "decode"

    def wrapped(self, x):
        logits = self.gate(x)
        k = self.top_k
        n_tokens = 1
        for d in logits.shape[:-1]:
            n_tokens *= d
        phase = phase_of(self, n_tokens)
        masked = pred(phase)
        if state.get("log_shapes"):
            shapes.append((n_tokens, phase, bool(masked)))
        want = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
        kept = keepsets[id(self)]
        for picks in want.reshape(-1, k).tolist():
            if masked:
                state["slots"] += k
                state["hits"] += sum(1 for e in picks if e in kept)
        gates = mx.softmax(
            logits + (masks[id(self)] if masked else zeros[id(self)]),
            axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)

    cls.__call__ = wrapped

    def restore():
        cls.__call__ = original

    return state, restore


PREDS = {"NONE": lambda ph: False,
         "PROMPT": lambda ph: ph in ("prefill", "prompt_tail"),
         "BATCH": lambda ph: ph == "prefill"}


def prompt_text(tok, p):
    msgs = [{"role": "system", "content": m.SYSTEM},
            {"role": "user", "content": p.prompt}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                   tokenize=False,
                                   enable_thinking=False)


def cap_pass(model, tok, keep, START):
    from mlx_lm import stream_generate
    for pth in (ZCAP, ZSHA):
        if pth.exists():
            raise SystemExit(f"REFUSING: {pth} exists")
    from llmopt.mathgen.problems import make_dataset
    zf = ZCAP.open("w")
    ztab = {}
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        for S in ("NONE", "PROMPT"):
            state, restore = instrument(model, keep, PREDS[S])
            try:
                for i, p in enumerate(problems):
                    state["log_shapes"] = (i == 0)
                    text = prompt_text(tok, p)
                    ids = []
                    for r in stream_generate(model, tok, prompt=text,
                                             max_tokens=4):
                        ids.append(int(r.token))
                    if i == 0:
                        seq = state["shapes"][:8]
                        print(f"[cap] {seed}/{S} shapes {seq}",
                              flush=True)
                        state["shapes"].clear()
                        state["log_shapes"] = False
                    row = {"seed": seed, "state": S, "idx": i,
                           "z4": ids[:4], "z1": ids[0]}
                    ztab[(seed, S, i)] = ids[:4]
                    zf.write(json.dumps(row) + "\n")
                    zf.flush()
            finally:
                restore()
            print(f"[cap] seed {seed} state {S} done", flush=True)
    zf.close()
    sha = hashlib.sha256(ZCAP.read_bytes()).hexdigest()
    d1 = d4 = 0
    for seed in SEEDS:
        for i in range(N_EVAL):
            a, b = ztab[(seed, "NONE", i)], ztab[(seed, "PROMPT", i)]
            d1 += a[0] != b[0]
            d4 += a != b
    ZSHA.write_text(json.dumps({
        "note": "EX6-MED z table frozen outcome-blind (no oracle "
                "call has run)",
        "start": START, "completion_commit": completion_commit(),
        "zcap_sha256": sha, "n_rows": len(ztab),
        "disagree_first_token_D1": d1,
        "disagree_prefix4_D4": d4}, indent=1) + "\n")
    print(f"[cap] D1={d1} D4={d4} of {len(ztab) // 2} pairs "
          f"sha {sha[:16]}", flush=True)


def gen_forced(model, tok, text, forced):
    """Forced launch: prompt ids + forced ids as one prompt, decode
    the remaining budget. Returns completion text incl. forced."""
    from mlx_lm import generate
    ids = tok.encode(text)
    out = generate(model, tok, prompt=ids + forced,
                   max_tokens=m.MAX_TOKENS - len(forced))
    return tok.decode(forced) + out


def cells_pass(model, tok, keep, START):
    from mlx_lm import generate
    if not ZSHA.exists():
        raise SystemExit("REFUSING: run PASS=cap first")
    zmeta = json.loads(ZSHA.read_text())
    assert (hashlib.sha256(ZCAP.read_bytes()).hexdigest()
            == zmeta["zcap_sha256"]), "z table hash mismatch"
    ztab = {}
    for line in ZCAP.open():
        r = json.loads(line)
        ztab[(r["seed"], r["state"], r["idx"])] = r["z4"]
    for pth in (CELLS, QUAL):
        if pth.exists():
            raise SystemExit(f"REFUSING: {pth} exists")
    cf, qf = CELLS.open("w"), QUAL.open("w")
    from llmopt.mathgen.problems import make_dataset
    ident_ok = True
    invalid_seeds = set()
    counts = {}
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        native = {}
        if seed == 7001:
            for S in ("NONE", "PROMPT"):
                state, restore = instrument(model, keep, PREDS[S])
                try:
                    for i, p in enumerate(problems):
                        c = generate(model, tok,
                                     prompt=prompt_text(tok, p),
                                     max_tokens=m.MAX_TOKENS)
                        native[(S, i)] = c
                finally:
                    restore()
                print(f"[qual] native {S} regenerated", flush=True)
        for S in ("NONE", "PROMPT"):
            for zsrc in ("NONE", "PROMPT"):
                pred = PREDS["BATCH" if S == "PROMPT"
                             else "NONE"]
                state, restore = instrument(model, keep, pred)
                n_ok = 0
                try:
                    t0 = time.time()
                    for i, p in enumerate(problems):
                        forced = [ztab[(seed, zsrc, i)][0]]
                        comp = gen_forced(model, tok,
                                          prompt_text(tok, p),
                                          forced)
                        expr = m.extract_expression(comp)
                        (ok, parsed), t_out = m.check_isolated(p, expr)
                        n_ok += ok
                        cf.write(json.dumps({
                            "seed": seed, "state": S, "zsrc": zsrc,
                            "idx": i, "ok": bool(ok),
                            "parsed": parsed, "timeout": bool(t_out),
                            "gen_len": len(comp)}) + "\n")
                        cf.flush()
                        if seed == 7001 and S == zsrc:
                            ident = comp == native[(S, i)]
                            qf.write(json.dumps({
                                "seed": seed, "diag": S, "idx": i,
                                "token_identical": bool(ident)})
                                + "\n")
                            qf.flush()
                            if not ident:
                                ident_ok = False
                except Exception as e:  # stream what we have
                    print(f"[cells] ABORT {seed}/{S}/{zsrc}: {e}",
                          flush=True)
                    raise
                finally:
                    restore()
                booked = BOOKED[S].get(seed) if S == zsrc else None
                if S == zsrc and not SMOKE and n_ok != booked:
                    invalid_seeds.add(seed)
                    print(f"[qual] DIAG MISMATCH {seed}/{S}: "
                          f"{n_ok} v booked {booked}", flush=True)
                counts[f"{seed}:{S}:{zsrc}"] = n_ok
                print(f"[cells] seed {seed} S={S} z={zsrc}: "
                      f"{n_ok}/{len(problems)} "
                      f"{time.time() - t0:.0f}s", flush=True)
    cf.close()
    qf.close()
    summary = {"note": "EX6-MED-0 cell counts; interpretation "
                       "gated by diagonals per prereg",
               "start": START,
               "completion_commit": completion_commit(),
               "zcap_sha256": zmeta["zcap_sha256"],
               "token_identity_seed7001_pass": bool(ident_ok),
               "count_invalid_seeds": sorted(invalid_seeds),
               "cell_counts": counts}
    (DIR / f"{PRE}summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    if not ident_ok:
        print("[cells] QUALIFICATION-FAIL: seed-7001 diagonals not "
              "token-identical; instrument invalid, no treatment "
              "read", flush=True)
        sys.exit(3)
    print(f"[cells] done; invalid_seeds={sorted(invalid_seeds)}",
          flush=True)


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/ex6med.py", "scratch/ex6_phase.py",
         "scratch/moe_gt1_arm2.py", KEEPSET])
    from mlx_lm import load
    model, tok = load(m.MODEL)
    keep = {int(li): set(v) for li, v in
            json.loads(Path(KEEPSET).read_text()).items()}
    if PASS == "cap":
        cap_pass(model, tok, keep, START)
    else:
        cells_pass(model, tok, keep, START)
    return 0


if __name__ == "__main__":
    sys.exit(main())
