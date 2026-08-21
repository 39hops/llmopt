"""EX6-MED-0 exact-token-ID diagonal audit (frozen pre-treatment-
read; run AFTER the cells run completes and BEFORE any treatment
value is opened — it regenerates, it reads no cells.jsonl rows).

The cells-pass qualification compared decoded STRINGS (comp ==
native). This audit hardens seed-7001 diagonal identity to exact
token-ID sequences: for each problem and each state S in
(NONE, PROMPT), capture native ids via stream_generate under the
frozen instrument (mode S), and forced-harness ids as
[z1] + stream_generate(prompt_ids + [z1]) under the cell mask
(BATCH for PROMPT, NONE for NONE), then require sequence equality
native_ids == [z1] + continuation_ids (compared over the shorter
of the two budgets; the harness generates MAX_TOKENS-1
continuation ids by construction).

Receipt: logs/ex6med/idaudit.json (refuse-if-exists).

    .venv/bin/python scratch/ex6med_idaudit.py               (Mac)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scratch.ex6med as x  # noqa: E402
import scratch.moe_gt1_arm2 as m  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

OUT = Path("logs/ex6med/idaudit.json")
SEED = 7001


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/ex6med_idaudit.py",
                              "scratch/ex6med.py",
                              x.KEEPSET])
    ztab = {}
    for line in x.ZCAP.open():
        r = json.loads(line)
        if r["seed"] == SEED:
            ztab[(r["state"], r["idx"])] = r["z1"]

    from mlx_lm import load, stream_generate

    from llmopt.mathgen.problems import make_dataset
    model, tok = load(m.MODEL)
    keep = {int(li): set(v) for li, v in
            json.loads(Path(x.KEEPSET).read_text()).items()}
    problems = make_dataset(120, seed=SEED)

    def ids_of(prompt, pred, n):
        state, restore = x.instrument(model, keep, pred)
        try:
            out = []
            for r in stream_generate(model, tok, prompt=prompt,
                                     max_tokens=n):
                out.append(int(r.token))
            return out
        finally:
            restore()

    mism = {"NONE": [], "PROMPT": []}
    for S in ("NONE", "PROMPT"):
        cell_pred = x.PREDS["BATCH" if S == "PROMPT" else "NONE"]
        for i, p in enumerate(problems):
            text = x.prompt_text(tok, p)
            native = ids_of(text, x.PREDS[S], m.MAX_TOKENS)
            z1 = ztab[(S, i)]
            cont = ids_of(tok.encode(text) + [z1], cell_pred,
                          m.MAX_TOKENS - 1)
            forced = [z1] + cont
            n = min(len(native), len(forced))
            if native[:n] != forced[:n] or native[0] != z1:
                mism[S].append(i)
        print(f"[idaudit] {S}: {len(mism[S])} mismatches",
              flush=True)
    ok = not mism["NONE"] and not mism["PROMPT"]
    OUT.write_text(json.dumps({
        "note": "seed-7001 exact-token-ID diagonal audit "
                "(hardens the string-identity qualification)",
        "start": START, "completion_commit": completion_commit(),
        "seed": SEED, "n_problems": len(problems),
        "mismatch_idx": mism,
        "token_id_identical_all": bool(ok)}, indent=1) + "\n")
    print(f"[idaudit] token_id_identical_all={ok} -> {OUT}",
          flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
