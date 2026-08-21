"""EX6-MED-0 exact-token-ID diagonal audit v2 (frozen pre-
treatment-read; supersedes the unrun v1, scratch/ex6med_idaudit.py,
which compared only the min-length prefix and would pass an
equal-prefix / different-length pair; v1 stays frozen unedited).

Differences from v1, both from outside review:
 1. LITERAL equality including length: native_ids == [z1] +
    continuation_ids exactly. Failures receipt both lengths and
    the first mismatching position.
 2. MEDIATOR PROVENANCE: before any use, verify
    sha256(zcap.jsonl) == zcap_sha.json["zcap_sha256"] and record
    that sha in this receipt (v1 fingerprinted code/keepset but
    not the untracked z table that supplies every forced
    mediator).

Receipt: logs/ex6med/idaudit2.json (refuse-if-exists).

    .venv/bin/python scratch/ex6med_idaudit2.py              (Mac)
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scratch.ex6med as x  # noqa: E402
import scratch.moe_gt1_arm2 as m  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

OUT = Path("logs/ex6med/idaudit2.json")
SEED = 7001


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/ex6med_idaudit2.py",
                              "scratch/ex6med.py", x.KEEPSET])
    zsha = hashlib.sha256(x.ZCAP.read_bytes()).hexdigest()
    recorded = json.loads(x.ZSHA.read_text())["zcap_sha256"]
    assert zsha == recorded, ("z table hash mismatch", zsha, recorded)
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
        _, restore = x.instrument(model, keep, pred)
        try:
            out = []
            for r in stream_generate(model, tok, prompt=prompt,
                                     max_tokens=n):
                out.append(int(r.token))
            return out
        finally:
            restore()

    fails = {"NONE": [], "PROMPT": []}
    for S in ("NONE", "PROMPT"):
        cell_pred = x.PREDS["BATCH" if S == "PROMPT" else "NONE"]
        for i, p in enumerate(problems):
            text = x.prompt_text(tok, p)
            native = ids_of(text, x.PREDS[S], m.MAX_TOKENS)
            z1 = ztab[(S, i)]
            cont = ids_of(tok.encode(text) + [z1], cell_pred,
                          m.MAX_TOKENS - 1)
            forced = [z1] + cont
            if native != forced:
                first = next((j for j in range(
                    min(len(native), len(forced)))
                    if native[j] != forced[j]),
                    min(len(native), len(forced)))
                fails[S].append({"idx": i, "len_native": len(native),
                                 "len_forced": len(forced),
                                 "first_mismatch_pos": first})
        print(f"[idaudit2] {S}: {len(fails[S])} failures",
              flush=True)
    ok = not fails["NONE"] and not fails["PROMPT"]
    OUT.write_text(json.dumps({
        "note": "seed-7001 exact-token-ID diagonal audit v2: "
                "literal equality incl length; z-table sha "
                "verified before use",
        "start": START, "completion_commit": completion_commit(),
        "zcap_sha256_verified": zsha,
        "seed": SEED, "n_problems": len(problems),
        "failures": fails,
        "token_id_identical_all": bool(ok)}, indent=1) + "\n")
    print(f"[idaudit2] token_id_identical_all={ok} -> {OUT}",
          flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
