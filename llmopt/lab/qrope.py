"""RoPE value oracle — exact expectations, no thresholds.

Origin: the zeroed-inv_freq incident (AMENDMENT QWEN-TEACHER-0-
ROPE) plus two failed statistical guards: rope_calls>=1 passes the
incident it exists for, and min-positional-std refuses HEALTHY
models (RoPE's slowest band at theta=5e6, d=128 has positional std
~2.5e-9 at 512 positions — any threshold either misses the bug or
kills the healthy case). The oracle compares VALUES:

  check_inv_freq   rebuild 1/theta^(2i/d) from the pinned config,
                   compare element-wise
  check_cos_sin    compute expected cos/sin at fixed positions
                   from the rebuilt inv_freq, compare emitted

Fixtures that must fail (tests/test_qwen_qualify.py): all-zero,
half-zero, one wrong frequency, wrong ordering.
"""
from __future__ import annotations

import numpy as np

POSITIONS = np.array([0, 1, 2, 17], np.int64)


def expected_inv_freq(theta: float, dim: int) -> np.ndarray:
    return 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float64)
                           / dim)


def check_inv_freq(actual, theta: float, dim: int,
                   rtol: float = 1e-5) -> None:
    exp = expected_inv_freq(theta, dim)
    act = np.asarray(actual, np.float64).reshape(-1)
    if act.shape != exp.shape:
        raise SystemExit(f"ROPE ORACLE: inv_freq shape {act.shape} "
                         f"!= expected {exp.shape}")
    if not np.allclose(act, exp, rtol=rtol, atol=0):
        bad = int((~np.isclose(act, exp, rtol=rtol, atol=0)).sum())
        raise SystemExit(f"ROPE ORACLE: inv_freq differs from the "
                         f"pinned config at {bad}/{len(exp)} bands")


def check_cos_sin(cos, sin, theta: float, dim: int,
                  positions=POSITIONS, rtol: float = 1e-4) -> None:
    """cos/sin: [..., n_pos, dim] as emitted by the rotary module
    (frequencies duplicated across halves, HF convention)."""
    inv = expected_inv_freq(theta, dim)
    ang = positions[:, None].astype(np.float64) * inv[None, :]
    exp_cos = np.concatenate([np.cos(ang), np.cos(ang)], -1)
    exp_sin = np.concatenate([np.sin(ang), np.sin(ang)], -1)
    got_cos = np.asarray(cos, np.float64).reshape(-1, exp_cos.shape[-1])
    got_sin = np.asarray(sin, np.float64).reshape(-1, exp_sin.shape[-1])
    if got_cos.shape[0] < len(positions):
        raise SystemExit("ROPE ORACLE: fewer emitted positions than "
                         "the fixed check vector")
    sel_cos = got_cos[positions] if got_cos.shape[0] > positions.max() \
        else got_cos[:len(positions)]
    sel_sin = got_sin[positions] if got_sin.shape[0] > positions.max() \
        else got_sin[:len(positions)]
    for name, got, exp in (("cos", sel_cos, exp_cos),
                           ("sin", sel_sin, exp_sin)):
        if not np.allclose(got, exp, rtol=rtol, atol=1e-6):
            raise SystemExit(f"ROPE ORACLE: emitted {name} differs "
                             f"from expectation at the fixed positions")
