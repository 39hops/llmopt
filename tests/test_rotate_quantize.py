"""Rotation constructions and the incoherence-processing mechanism."""

import torch

from llmopt.quantize.rotate import hadamard, random_orthogonal, rotation_error
from llmopt.quantize.methods import rtn


def test_hadamard_orthonormal():
    h = hadamard(64)
    assert torch.allclose(h @ h.T, torch.eye(64), atol=1e-5)


def test_random_orthogonal_orthonormal_and_seeded():
    q1 = random_orthogonal(32, seed=3)
    q2 = random_orthogonal(32, seed=3)
    assert torch.equal(q1, q2)
    assert torch.allclose(q1 @ q1.T, torch.eye(32), atol=1e-5)


def test_rotation_is_lossless_before_quantization():
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    q = hadamard(64)
    assert torch.allclose(q.T @ (q @ w), w, atol=1e-5)


def test_no_rotation_matches_direct_rtn():
    torch.manual_seed(1)
    w = torch.randn(32, 32)
    direct = float((w - rtn(w, 4)).norm() / w.norm())
    assert abs(rotation_error(w, 4, None) - direct) < 1e-6


def test_hadamard_helps_on_outlier_matrix():
    # planted outlier columns: exactly the structure rotation smears out
    torch.manual_seed(2)
    w = torch.randn(64, 64)
    w[:, :3] *= 50.0
    err_none = rotation_error(w, 4, None)
    err_had = rotation_error(w, 4, hadamard(64))
    assert err_had < err_none
