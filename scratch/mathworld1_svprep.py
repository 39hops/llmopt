"""MATH-CYBER-1 SVP-REPLICATION-DESIGN-0 — freeze the two
additional paired-birth inits (seed law 9001 + 1000k, k=1,2 ->
10001, 11001) BEFORE any replication outcome exists. Zero
training.

Outputs: checkpoints/svp_init_s10001.pt, svp_init_s11001.pt,
logs/mathworld1/svprep_design.json (all refuse-if-exists).

    .venv/bin/python scratch/mathworld1_svprep.py             (Mac)
"""
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402

SEED0_INIT = "checkpoints/svp_init.pt"
SEED0_SHA = ("18597944400e061f797755175d31a06690e378d8141c68760"
             "6724b95a7d0a86c")
SEEDS = [9001 + 1000 * k for k in (1, 2)]
RECEIPT = Path("logs/mathworld1/svprep_design.json")


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def state_bytes(seed, vocab):
    torch.manual_seed(seed)
    m = build_model(vocab, ctx=4096)
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in
                sorted(m.state_dict().items())}, buf)
    return buf.getvalue()


def main():
    outs = {s: Path(f"checkpoints/svp_init_s{s}.pt") for s in SEEDS}
    for p in list(outs.values()) + [RECEIPT]:
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    gate(SEEDS == [10001, 11001], "SEED LAW")
    gate(fsha(SEED0_INIT) == SEED0_SHA, "SEED0 INIT MUTATED")
    START = start_provenance(
        ["scratch/mathworld1_svprep.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    vocab = ActionGCTok().vocab_size
    gate(vocab == 332, "VOCAB")
    shas = {9001: SEED0_SHA}
    for s in SEEDS:
        b1 = state_bytes(s, vocab)
        b2 = state_bytes(s, vocab)
        gate(b1 == b2, f"INIT NONDETERMINISTIC seed {s}")
        outs[s].write_bytes(b1)
        shas[s] = fsha(outs[s])
    vals = list(shas.values())
    gate(len(set(vals)) == 3, "INITS NOT DISTINCT")
    gate(fsha(SEED0_INIT) == SEED0_SHA, "SEED0 INIT MUTATED POST")
    for p in ("checkpoints/svp_state.pt",
              "checkpoints/svp_program.pt"):
        gate(Path(p).exists(), f"BOOKED CHECKPOINT MISSING {p}")
    receipt = {
        "seed_law": "9001 + 1000k, k in (1, 2)",
        "seeds": SEEDS,
        "init_sha256": {str(k): v for k, v in shas.items()},
        "bars": {
            "SEED_LAW": SEEDS == [10001, 11001],
            "INIT_DETERMINISM": True,   # gate-enforced above
            "INIT_DISTINCT": len(set(vals)) == 3,
            "NO_PRODUCTION_TOUCH": fsha(SEED0_INIT) == SEED0_SHA,
        },
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k != "start"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
