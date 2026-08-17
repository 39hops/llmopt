"""CUDA leg rung 0: toolchain + VRAM budget receipt (3080/WSL).

Proves, as a receipt, the two unknowns every later rung stands on:
  1. Triton compiles AND runs a kernel in the WSL venv (which has no
     host C compiler; torch.compile is a separate MSVC-gated path).
  2. The actual free-VRAM budget from mem_get_info — the residency
     ceiling for artifact selection (10 GiB total is not the budget).

Receipt: logs/qwencuda/rung0.json (refuse-if-exists; new run = new
path). All provenance fields derived, never literals.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import torch
import triton
import triton.language as tl

OUT = "logs/qwencuda/rung0.json"


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x + y, mask=m)


def main() -> int:
    if os.path.exists(OUT):
        raise SystemExit(f"refuse: {OUT} exists — new run, new path")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    free, total = torch.cuda.mem_get_info()
    n = 1 << 20
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    o = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
    torch.cuda.synchronize()
    exact = bool(torch.equal(o, x + y))

    rec = {
        "rung": 0,
        "code_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "device_actual": torch.cuda.get_device_name(0),
        "driver_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "wsl": "microsoft" in os.uname().release.lower(),
        "vram_total_bytes": total,
        "vram_free_at_start_bytes": free,
        "triton_add_bit_exact": exact,
        "env": {k: os.environ.get(k) for k in
                ("TORCH_DISABLE_NATIVE_JIT", "PYTORCH_CUDA_ALLOC_CONF")},
    }
    with open(OUT, "x") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(json.dumps(rec, indent=1))
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
