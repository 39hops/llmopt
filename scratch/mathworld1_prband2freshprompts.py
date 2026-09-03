"""MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-FRESH-SEED-0
model-blind validation prompt extraction (prereg-time instrument).

Reads the sha-pinned, untracked atlas render stream and extracts
ONLY the two persisted 96-state prompt matrices that Stage A of the
fresh-seed validation may consume: RAW (atlas index 12) and R488
(atlas index 488, the ATLAS-MAXIMIN DISCOVERY RENDER booked at
RESULTS L65657). Writes a SMALL tracked artifact of exactly 192
rows (96 states x 2 views) with view name, atlas_index, render_id,
pair_id, theta, state index, the exact persisted cur, its sha256
and the prompt token count, plus a receipt carrying the RAW matrix
sha, the R488 matrix sha and the artifact sha. Gates: renders.jsonl
/ manifest / policies / primary pins; RAW cur byte-equal to the
frozen primary cur in 96/96; R488 render_id and role permutation
equal to the booked manifest entry; 48 pair_ids each with exactly
one SIN_LOW and one COS_LOW state. No checkpoint is opened, no
model is built, torch.load is trapped.

Usage:
    .venv/bin/python scratch/mathworld1_prband2freshprompts.py
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpchal import CTX, fsha  # noqa: E402
import torch  # noqa: E402


def _no_load(*a, **k):
    raise SystemExit("GATE FAILED: torch.load called in a model-blind step")


torch.load = _no_load

ATLAS = "logs/mathworld1/prband2atlas"
RENDERS = f"{ATLAS}/renders.jsonl"
MANIFEST = f"{ATLAS}/atlas_manifest.jsonl"
POLICIES = f"{ATLAS}/atlas_policies.jsonl"
PRIMARY = "logs/mathworld1/prband2prod/primary.jsonl"
PINS = {
    RENDERS: "2cac5570bc8eb6143a0a35797dafe1ea78147e6871ef93aceb87951e88419d8b",
    MANIFEST: "687b5e54e0da19bf057431eb4d44b755302c1963d18e13fb6d316fa99dd2f4b2",
    POLICIES: "b4a1c08308ca429d4bd7eb01210dfe469cd11eb9c9c527daf42997cec6d86c71",
    PRIMARY: "209391ef3b2e5c87308571d6ef309bb5724a214160caa1b7857f4a31f9112c34",
}
VIEWS = {"RAW": (12, "8f1479a9402429ac18d1d1e55f803d02bd47313f8c69dfbfae6abd0a4f5f26f2",
                 ["HI_D", "HI_L", "K", "LO_D", "LO_L", "W"]),
         "R488": (488, "91e4098b48717d6b611445824a6c987594c38df1b9c5f40e815ff95b36a5f9d9",
                  ["K", "HI_D", "LO_D", "LO_L", "HI_L", "W"])}
OUTDIR = Path("logs/mathworld1/prband2fresh")
OUT = OUTDIR / "validation_prompts.jsonl"
RECEIPT = OUTDIR / "validation_prompts_receipt.json"
PROMPT = "Current: {cur}\nHints: none\nStep: "
TOK = ActionGCTok()


def main():
    START = start_provenance(
        ["scratch/mathworld1_prband2freshprompts.py",
         "scratch/mathworld1_actiontok.py", "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpchal.py", "llmopt/lab/provenance.py"])
    gate(torch.load is _no_load, "TORCH LOAD TRAP")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN DRIFT {p}")
    for p in (OUT, RECEIPT):
        gate(not p.exists(), f"REFUSE OVERWRITE {p}")
    P = [json.loads(l) for l in open(PRIMARY)]
    gate(len(P) == 96, "N=96")
    order = [(r["pair_id"], r["theta"]) for r in P]
    by_pair = defaultdict(list)
    for r in P:
        by_pair[r["pair_id"]].append(r["theta"])
    gate(len(by_pair) == 48 and all(sorted(v) == ["COS_LOW", "SIN_LOW"]
                                    for v in by_pair.values()), "48 PAIRS x 2 THETA")
    man = {m["atlas_index"]: m for m in map(json.loads, open(MANIFEST))}
    pol = {p["atlas_index"]: p for p in map(json.loads, open(POLICIES))}
    for vn, (idx, rid, roles) in VIEWS.items():
        gate(man[idx]["render_id"] == rid and man[idx]["roles"] == roles
             and pol[idx]["render_id"] == rid and pol[idx]["roles"] == roles
             and pol[idx]["eligible"], f"{vn} IDENTITY")
    want = {v[0] for v in VIEWS.values()}
    cur = {}
    for l in open(RENDERS):
        r = json.loads(l)
        if r["atlas_index"] in want:
            k = (r["atlas_index"], r["pair_id"], r["theta"])
            gate(k not in cur, "DUP")
            cur[k] = r["cur"]
    gate(len(cur) == 192, "192 RENDER ROWS")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    mats = {}
    for vn, (idx, rid, roles) in VIEWS.items():
        strings = []
        for i, (pid, th) in enumerate(order):
            c = cur[(idx, pid, th)]
            if vn == "RAW":
                gate(c == P[i]["cur"] and hashlib.sha256(c.encode()).hexdigest()
                     == P[i]["cur_sha"], f"RAW BYTES {i}")
            prompt = PROMPT.format(cur=c)
            ids = TOK.encode(prompt)
            gate(TOK.decode(ids) == prompt, "TOK RT")
            gate(len(ids) + 9 <= CTX, "CTX")
            rows.append({"view": vn, "atlas_index": idx, "render_id": rid,
                         "roles": roles, "state": i, "pair_id": pid, "theta": th,
                         "cur": c, "cur_sha": hashlib.sha256(c.encode()).hexdigest(),
                         "prompt_tokens": len(ids)})
            strings.append(c)
        mats[vn] = hashlib.sha256(json.dumps(strings).encode()).hexdigest()
        gate(mats[vn] == pol[idx]["matrix_sha"], f"{vn} MATRIX SHA")
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = {"prereg": "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-FRESH-SEED-0",
               "pins": {p: fsha(p) for p in PINS}, "views": {
                   vn: {"atlas_index": v[0], "render_id": v[1], "roles": v[2],
                        "matrix_sha": mats[vn]} for vn, v in VIEWS.items()},
               "rows": len(rows), "prompt_tokens": {
                   vn: dict(Counter(r["prompt_tokens"] for r in rows if r["view"] == vn))
                   for vn in VIEWS},
               "artifact_sha256": fsha(str(OUT)), "prompt_law": PROMPT,
               "provenance": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("views", "rows", "prompt_tokens",
                                              "artifact_sha256", "completion_commit")},
                     indent=1))


if __name__ == "__main__":
    main()
