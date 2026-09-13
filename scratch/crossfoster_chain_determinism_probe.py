"""CHAIN-DESK determinism probe (receipted form of the 2026-09-13 /tmp probe):
samples 8 texts from donor A on 30 probe roots (band 8,950,000; sample seeds
7,950,000 + 1000 * i + b) with llmopt.lab.gate.sample_wave_lp on the default
device, twice within the process, and appends one row {digest, within_process_
mismatch, pid, device} to logs/crossfoster1/chain_determinism_probe.jsonl.
Run it twice: two rows with equal digests = cross-process bit-exactness of the
sampler's texts. Usage: .venv/bin/python scratch/crossfoster_chain_determinism_probe.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

import crossfoster_donor as CD  # noqa: E402
import train_mathnative as TM  # noqa: E402
from llmopt.lab.gate import sample_wave_lp  # noqa: E402
from llmopt.lab.gen import _gen_isolated  # noqa: E402

OUT = Path("logs/crossfoster1/chain_determinism_probe.jsonl")


def main():
    tok = TM.MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m, rec = CD.load_donor("A", tok, dev)
    allt, mismatch, tot = [], 0, 0
    for lv in (3, 4, 5, 6, 7):
        for i in range(6):
            p = _gen_isolated(lv, 8_950_000 + 1000 * lv + i)
            if p is None:
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
            seeds = [7_950_000 + 1000 * i + b for b in range(8)]
            with torch.no_grad():
                a, _, _ = sample_wave_lp(m, tok, prompt, seeds, dev)
                b, _, _ = sample_wave_lp(m, tok, prompt, seeds, dev)
            tot += len(a)
            mismatch += sum(x != y for x, y in zip(a, b))
            allt += a
    row = {"kind": "chain_determinism_probe", "donor": rec, "device": dev, "pid": os.getpid(), "samples": tot, "within_process_mismatch": mismatch,
           "digest": hashlib.sha256("\n".join(allt).encode()).hexdigest()[:16], "torch_version": torch.__version__}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in ("device", "pid", "samples", "within_process_mismatch", "digest")}))


if __name__ == "__main__":
    main()
