"""The default kernel generation of llmopt.lab.qmetal, checked in fresh
subprocesses so the test sees what a serving process sees at import:
with QMETAL_VERSION unset the module and every new resident default to
generation 5 (VERDICT QWEN-METAL-S16-V5-0, promoted by VERDICT
QWEN-METAL-S16-V5-DEFAULT-0), and QMETAL_VERSION=4 selects the qualified
generation 4 control, which dispatches the shipped S16 kernel and
returns the same bits on the golden payload."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FX = os.path.join(HERE, "fixtures")

PROBE = r"""
import hashlib, json, os, sys
import numpy as np
from llmopt.lab import qmetal
out = {"version": qmetal.VERSION, "resident_class_version": qmetal._Resident.version}
if os.environ.get("PROBE_GEMM") == "1":
    import mlx.core as mx
    W = np.load(os.path.join(sys.argv[1], "qcodec_src.npy"))
    buf = open(os.path.join(sys.argv[1], "qcodec_s16.bin"), "rb").read()
    res = qmetal.S16Metal(buf, W.shape)
    x = mx.array(np.random.default_rng(0).normal(size=(1, W.shape[1])).astype(np.float32))
    y = np.asarray(res.gemm(x), dtype=np.float32)
    out["resident_version"] = res.version
    out["output_sha256"] = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
    out["compiled_kernels"] = sorted(f"{k[0]}:{k[1]}" for k in qmetal._kernels)
print(json.dumps(out))
"""


def _probe(env_version, gemm=False):
    env = {k: v for k, v in os.environ.items() if k != "QMETAL_VERSION"}
    if env_version is not None:
        env["QMETAL_VERSION"] = str(env_version)
    if gemm:
        env["PROBE_GEMM"] = "1"
    env["PYTHONPATH"] = ROOT
    p = subprocess.run([sys.executable, "-c", PROBE, FX], capture_output=True, text=True, env=env, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_fresh_import_without_env_defaults_to_generation_5():
    out = _probe(None)
    assert out["version"] == 5
    assert out["resident_class_version"] == 5


def test_env_version_4_selects_the_qualified_control():
    out = _probe(4)
    assert out["version"] == 4
    assert out["resident_class_version"] == 4


@pytest.mark.skipif(not __import__("llmopt.lab.qmetal", fromlist=["HAVE_MLX"]).HAVE_MLX, reason="needs mlx (Mac)")
def test_new_resident_dispatches_v5_by_default_and_v4_under_env_with_identical_bits():
    """A resident built in a fresh process with no env takes the S16 v5
    kernel at tile 1 (the compiled-kernel set is the dispatch witness);
    the same resident under QMETAL_VERSION=4 takes the shipped kernel;
    the two outputs are bit-identical on the golden payload."""
    d5 = _probe(None, gemm=True)
    d4 = _probe(4, gemm=True)
    assert d5["resident_version"] == 5 and d4["resident_version"] == 4
    assert "s16:5" in d5["compiled_kernels"] and "s16:5" not in d4["compiled_kernels"]
    assert "s16:3" in d4["compiled_kernels"] and "s16:3" not in d5["compiled_kernels"]
    assert d5["output_sha256"] == d4["output_sha256"]
