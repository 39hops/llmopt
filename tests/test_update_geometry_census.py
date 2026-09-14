"""UPDATE-GEOMETRY-CENSUS-0 guards: the Gram-algebra geometry reproduces
direct dense linear algebra on synthetic matrices (held capture, centered v
uncentered, top-k projector on known-rank data, subspace overlap on
identical / orthogonal / partially overlapping subspaces, velocity
projection), the ladder is pure and inclusive at the literal thresholds,
the probe/tensor laws and thresholds are literal in the source, and the
isotropic reference reads the k / d floor."""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    prev = os.environ.get("SMOKE")
    os.environ["SMOKE"] = "1"
    for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scratch")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        m = importlib.import_module("update_geometry_census")
    finally:
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


def _dense_capture(G, fit, held, k, center=False):
    X = G[fit].astype(np.float64); Hh = G[held].astype(np.float64)
    if center:
        m = X.mean(axis=0); X = X - m; Hh = Hh - m
    X = X / np.linalg.norm(X, axis=1, keepdims=True); Hh = Hh / np.linalg.norm(Hh, axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = Vt[:k].T @ Vt[:k]
    return [float(np.linalg.norm(P @ h) ** 2) for h in Hh]


def test_held_capture_matches_dense_svd_raw_and_centered(mod):
    rng = np.random.default_rng(0)
    G = rng.standard_normal((12, 300)) + 3 * rng.standard_normal((1, 300))     # shared mean direction + noise
    fit, held = list(range(6)), list(range(6, 12)); K = G @ G.T
    geo = mod.geometry_from_grams(K, fit, held, [1, 2, 4])
    for k in (1, 2, 4):
        assert np.allclose(geo["raw"]["held_capture_values"][k], _dense_capture(G, fit, held, k), atol=1e-9)
        assert np.allclose(geo["centered"]["held_capture_values"][k], _dense_capture(G, fit, held, k, center=True), atol=1e-9)
    # raw capture at k=1 is dominated by the shared mean; centered is not
    assert geo["raw"]["held_capture"][1]["median"] > 0.8 > geo["centered"]["held_capture"][1]["median"]
    assert 0 < geo["Q"]["fit"] < 1


def test_topk_projector_on_known_rank(mod):
    rng = np.random.default_rng(1)
    basis = np.linalg.qr(rng.standard_normal((200, 3)))[0]                      # rank-3 subspace
    G = rng.standard_normal((16, 3)) @ basis.T
    fit, held = list(range(8)), list(range(8, 16)); K = G @ G.T
    geo = mod.geometry_from_grams(K, fit, held, [1, 2, 4])
    assert np.allclose(geo["raw"]["held_capture_values"][4], 1.0, atol=1e-9)      # k >= rank captures everything
    assert geo["raw"]["fit_spectrum"]["topk_energy"][4] > 0.999999
    assert 1 < geo["raw"]["fit_spectrum"]["participation_ratio"] <= 3 + 1e-9


def test_subspace_overlap_identical_orthogonal_partial(mod):
    rng = np.random.default_rng(2)
    Q = np.linalg.qr(rng.standard_normal((100, 8)))[0]
    X = rng.standard_normal((6, 4)) @ Q[:, :4].T                                  # span = first 4 basis vectors
    Y_same = rng.standard_normal((6, 4)) @ Q[:, :4].T
    Y_orth = rng.standard_normal((6, 4)) @ Q[:, 4:].T
    Y_half = rng.standard_normal((6, 4)) @ Q[:, 2:6].T                             # shares 2 of 4
    for Y, expect in ((Y_same, 1.0), (Y_orth, 0.0), (Y_half, 0.5)):
        s = mod.subspace_overlap(X @ X.T, Y @ Y.T, X @ Y.T, [4])[4]
        assert abs(s - expect) < 1e-9, (s, expect)
    # dense check on a generic pair at k = 2
    Y = rng.standard_normal((6, 100))
    s = mod.subspace_overlap(X @ X.T, Y @ Y.T, X @ Y.T, [2])[2]
    Vx = np.linalg.svd(X, full_matrices=False)[2][:2]; Vy = np.linalg.svd(Y, full_matrices=False)[2][:2]
    assert abs(s - np.linalg.norm(Vx @ Vy.T) ** 2 / 2) < 1e-9


def test_velocity_projection_matches_dense(mod):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((5, 50)); v = rng.standard_normal(50)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    K = Xn @ Xn.T; c = Xn @ v
    got = mod.projection_fraction(K, c, float(v @ v), [1, 3])
    Vt = np.linalg.svd(Xn, full_matrices=False)[2]
    for k in (1, 3):
        assert abs(got[k] - np.linalg.norm(Vt[:k] @ v) ** 2 / (v @ v)) < 1e-9


def test_centered_cross_gram_matches_dense(mod):
    rng = np.random.default_rng(4)
    X = rng.standard_normal((8, 40)) + 2; Y = rng.standard_normal((8, 40)) - 1
    fx, fy = list(range(4)), list(range(4))
    got = mod.cross_centered(X @ X.T, Y @ Y.T, X @ Y.T, fx, fy)
    Xc = X - X[fx].mean(axis=0); Yc = Y - Y[fy].mean(axis=0)
    assert np.allclose(got, Xc @ Yc.T)
    assert np.allclose(mod.centered_gram(X @ X.T, fx), Xc @ Xc.T)


def test_ladder_pure_and_inclusive(mod):
    mod.K_LADDER = 8
    def geom(c8, c16, cc8, rel):
        return {"raw": {"held_capture": {8: {"median": c8}, 16: {"median": c16}}, "reliability_fit_held": {8: rel}}, "centered": {"held_capture": {8: {"median": cc8}}}}
    fin = {"A": geom(0.5, 0.6, 0.1, 0.5), "B": geom(0.55, 0.7, 0.2, 0.6)}
    w0 = geom(0.3, 0.4, 0.1, 0.5)
    ax = mod.adjudicate(fin, w0, {"raw": {8: 0.4}}, [1, 2, 4, 8, 16])
    assert ax["thinness"] == "STABLE-THIN" and ax["residual"] == "MEAN-ONLY" and ax["writer"] == "WRITER-SHARED" and ax["learned_v_init"] == "LEARNED-DIFFERS-SHARPENED"
    ax = mod.adjudicate({"A": geom(0.2, 0.24, 0.1, 0.5), "B": geom(0.6, 0.7, 0.6, 0.6)}, w0, {"raw": {8: 0.2}}, [1, 2, 4, 8, 16])
    assert ax["thinness"] == "NO-THIN-GEOMETRY" and ax["writer"] == "WRITER-ROTATED" and ax["residual"] == "RESIDUAL-INDETERMINATE"
    ax = mod.adjudicate({"A": geom(0.3, 0.3, 0.5, 0.5), "B": geom(0.4, 0.4, 0.5, 0.5)}, w0, {"raw": {8: 0.3}}, [1, 2, 4, 8, 16])
    assert ax["thinness"] == "PARTIAL-THIN" and ax["residual"] == "RESIDUAL-THIN" and ax["writer"] == "WRITER-INDETERMINATE" and ax["learned_v_init"] == "INIT-PRESENT"
    ax = mod.adjudicate({"A": geom(0.9, 0.9, 0.9, 0.24), "B": geom(0.9, 0.9, 0.9, 0.9)}, w0, {"raw": {8: 0.9}}, [1, 2, 4, 8, 16])
    assert ax["resolution"] == "GEOMETRY-NOT-RESOLVED" and "thinness" not in ax


def test_registered_constants_literal():
    src = (ROOT / "scratch" / "update_geometry_census.py").read_text()
    for s in ("GRID = [0, 900, 3600, 7200, 10800, 13500, 15420]", "N_FIT = 4 if SMOKE else 32", "N_HELD = 4 if SMOKE else 32", 'PROBE_SEED = "ugc0-probe-v1"',
              "K_LIST = [1, 2, 4] if SMOKE else [1, 2, 4, 8, 16]", "K_THIN, C_THIN, K_FLOOR, C_FLOOR = 8, 0.5, 16, 0.25", "REL_MIN, SHARED_FRAC, ROTATED_FRAC, INIT_DELTA = 0.25, 0.8, 0.5, 0.15",
              'DIGEST_PREFIX = {"A_final": "4633efe5d376f911", "B_final": "4beeedec5f9f5e91", "W0": "eb4b0bb427f86972"}', "assert d == 18_911_616", "model.load_state_dict(sd); model.eval()", 'not os.environ.get("SEQ_CAP") and not os.environ.get("BIRTH_BS") and TM.BS == 32', 'census["artifacts"]["A_paths"][str(s_)] == A_PATHS[s_]',
              "ignore_index=-100", "random.Random(PROBE_SEED).sample(starts, N_PROBE)", "starts = [(i, i + TM.BS) for i in range(0, len(enc) - TM.BS, TM.BS)]", 'assert dg_after == dg_before',
              "PARITY_TOL = 1e-4", "torch.nn.utils.clip_grad_norm_" ):
        assert (s in src) != (s == "torch.nn.utils.clip_grad_norm_"), s     # every law literal present; clipping absent


def test_isotropic_reference_reads_the_floor(mod):
    rng = np.random.default_rng(5)
    d = 20000; G = rng.standard_normal((16, d))
    fit, held = list(range(8)), list(range(8, 16))
    geo = mod.geometry_from_grams(G @ G.T, fit, held, [1, 4])
    for k in (1, 4):
        assert geo["raw"]["held_capture"][k]["median"] < 3 * k / d
    assert geo["raw"]["reliability_fit_held"][4] < 0.01


def test_rank_deficient_overlap_is_finite_and_exact(mod):
    rng = np.random.default_rng(6)
    X = rng.standard_normal((4, 60)); Y = rng.standard_normal((4, 60))
    fx = [0, 1, 2, 3]
    Kx, Ky, Kxy = mod.centered_gram(X @ X.T, fx), mod.centered_gram(Y @ Y.T, fx), mod.cross_centered(X @ X.T, Y @ Y.T, X @ Y.T, fx, fx)
    s = mod.subspace_overlap(Kx, Ky, Kxy, [4])[4]            # centered 4-row panels have rank 3
    Xc = X - X.mean(axis=0); Yc = Y - Y.mean(axis=0)
    Vx = np.linalg.svd(Xc, full_matrices=False)[2][:3]; Vy = np.linalg.svd(Yc, full_matrices=False)[2][:3]
    assert np.isfinite(s) and abs(s - np.linalg.norm(Vx @ Vy.T) ** 2 / 4) < 1e-9
