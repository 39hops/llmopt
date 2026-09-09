"""WRITER-DFA-1 frozen probe batch (AMENDMENT -SEAL S8, -PRECISION P5e):
256 rows drawn by random.Random("writerdfa1-probe") from the
length-sorted encoded D2-excised stock diet
(C.encode_with_levels(C.load_excised_rows(), tok)); row ids = indices
into that list. logs/writerdfa1/probe.json records the row ids, the
sha256 of the JSON token lists, the diet size and the commit; it is
committed before any qualification birth and every consumer
(leakage smoke, ACT, alignment) re-derives the rows and asserts the
digest.

Usage: .venv/bin/python scratch/dfa_probe.py          (writes probe.json, refuses to overwrite)
       from dfa_probe import probe_rows, probe_tensors  (consumers)
"""
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")          # frozen curric module import side-effects only
os.environ.setdefault("BIRTH_SEED", "0")

import birth19m_curric as C  # noqa: E402  (frozen, import-only)
import train_mathnative as TM  # noqa: E402

PROBE = Path("logs/writerdfa1/probe.json")
N_ROWS = 256
KEY = "writerdfa1-probe"


def probe_rows(tok=None, assert_digest=True):
    """Returns (row_ids, token_lists, digest). Asserts against probe.json when it exists."""
    tok = tok or TM.MathTokenizer()
    enc_stock, _ = C.encode_with_levels(C.load_excised_rows(), tok)
    row_ids = sorted(random.Random(KEY).sample(range(len(enc_stock)), N_ROWS))
    rows = [enc_stock[i] for i in row_ids]
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    if assert_digest and PROBE.exists():
        rec = json.loads(PROBE.read_text())
        assert rec["row_ids"] == row_ids and rec["token_digest"] == digest and rec["enc_stock"] == len(enc_stock), "probe drift v probe.json"
    return row_ids, rows, digest, len(enc_stock)


def probe_tensors(rows, tok, device="cpu"):
    """Padded ids / mask exactly as the training loop builds them."""
    import torch
    L = max(len(s) for s in rows)
    ids = torch.tensor([s + [tok.pad_id] * (L - len(s)) for s in rows], device=device)
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in rows], device=device)
    labels = ids[:, 1:].clone()
    labels[mask[:, 1:] == 0] = -100
    return ids[:, :-1], mask[:, :-1], labels


def main():
    if PROBE.exists():
        raise SystemExit(f"REFUSING: {PROBE} exists")
    tok = TM.MathTokenizer()
    row_ids, rows, digest, n = probe_rows(tok, assert_digest=False)
    rec = {"prereg": "WRITER-DFA-1", "key": KEY, "n_rows": len(rows), "row_ids": row_ids, "token_digest": digest,
           "enc_stock": n, "max_len": max(len(r) for r in rows), "min_len": min(len(r) for r in rows),
           "n_tokens": sum(len(r) for r in rows), "vocab_len": len(tok.vocab),
           "code_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}
    PROBE.parent.mkdir(parents=True, exist_ok=True)
    PROBE.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "row_ids"}))


if __name__ == "__main__":
    main()
