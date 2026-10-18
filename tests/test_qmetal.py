"""Parity of the fused Metal decode-and-multiply kernels (llmopt.lab.qmetal)
against the canonical decoder llmopt.lab.qcodec, multiplied in float64.
Skips where mlx is absent. Bit exactness is not expected (different
accumulation order); the bar is |got - ref| <= 1e-5 |ref| + 2e-6 times
the row's absolute dot, plus one subnormal spacing per oracle-subnormal
weight, with non-finite outputs matched exactly (equal_nan)."""
import hashlib
import json
import os

import numpy as np
import pytest

from llmopt.lab import qcodec

mx = pytest.importorskip("mlx.core")
from llmopt.lab import qmetal  # noqa: E402

FX = os.path.join(os.path.dirname(__file__), "fixtures")
RTOL = 1e-5


@pytest.fixture(autouse=True, params=[1, 2, 3, 4], ids=["v1", "v2", "v3", "v4"])
def _kernel_version(request, monkeypatch):
    """Every parity case runs under all four kernel generations."""
    monkeypatch.setattr(qmetal, "VERSION", request.param)
    monkeypatch.setattr(qmetal._Resident, "version", request.param)
    yield request.param


def _w4_or_skip(buf, shape):
    if qmetal.VERSION == 2 and shape[1] % 256:
        with pytest.raises(ValueError):
            qmetal.W4Metal(buf, shape)
        pytest.skip("w4 v2 needs C % 256 == 0 (refusal verified)")
    return qmetal.W4Metal(buf, shape)
ATOL_COND = 2e-6


def _pack_w4(exps, cb, idx):
    return exps.astype(np.uint8).tobytes() + cb.astype(np.float16).tobytes() + idx.astype(np.uint8).tobytes()


def _pack_s16(exps, lv, codes):
    return exps.astype(np.uint8).tobytes() + lv.astype(np.float16).tobytes() + codes.astype(np.uint8).tobytes()


def _random_w4(rng, R, C, exp_pool=None, cb=None):
    n = R * C
    nb = n // 128
    exps = rng.choice(exp_pool if exp_pool is not None else np.arange(100, 140), nb)
    if cb is None:
        cb = rng.normal(size=(256, 4)).astype(np.float16)
    idx = rng.integers(0, 256, n // 4)
    return _pack_w4(exps, cb, idx)


def _random_s16(rng, R, C, exp_pool=None, lv=None):
    n = R * C
    nb = n // 128
    exps = rng.choice(exp_pool if exp_pool is not None else np.arange(100, 140), nb)
    if lv is None:
        lv = rng.normal(size=16).astype(np.float16)
    codes = rng.integers(0, 256, n // 2)
    return _pack_s16(exps, lv, codes)


def _exact64(buf, shape, codec):
    """The codec law evaluated in float64 (no float32 overflow or
    subnormal rounding): the mathematical value the payload encodes."""
    R, C = shape
    n = R * C
    nb = n // 128
    exps = np.frombuffer(buf, np.uint8, nb, 0).astype(np.int64)
    scale = np.exp2(exps - 127.0)
    scale[exps == 255] = np.inf          # the frozen law: exponent 255 is an infinite scale
    if codec == "w4":
        cb = np.frombuffer(buf, np.float16, 1024, nb).reshape(256, 4).astype(np.float64)
        idx = np.frombuffer(buf, np.uint8, n // 4, nb + 2048)
        Wn = cb[idx].reshape(nb, 128)
    else:
        lv = np.frombuffer(buf, np.float16, 16, nb).astype(np.float64)
        codes = np.frombuffer(buf, np.uint8, n // 2, nb + 32)
        c = np.empty(n, np.uint8)
        c[0::2] = codes >> 4
        c[1::2] = codes & 0xF
        Wn = lv[c].reshape(nb, 128)
    return (Wn * scale[:, None]).reshape(shape)


def _check(res, buf, shape, dec, x):
    W = dec(buf, shape).astype(np.float64)
    ref = W @ np.asarray(x, dtype=np.float64)
    got = np.asarray(res.gemv(mx.array(np.asarray(x, dtype=np.float32))), dtype=np.float64)
    assert got.shape == ref.shape
    # rows where the float32 oracle OVERFLOWED a weight (inf from a finite
    # codebook entry at a large exponent): the fused path scales the
    # word's partial dot, so it may stay finite where the oracle did not.
    # Such rows are judged against the exact float64 law instead: the
    # kernel must be inf exactly when the exact row value overflows
    # float32, and close to the exact value otherwise.
    codec = "w4" if dec is qcodec.dec_w4 else "s16"
    Wx = _exact64(buf, shape, codec)
    # rows holding an infinite-scale block (exponent 255): both the oracle
    # and the kernel produce a non-finite value; which non-finite value
    # (inf or nan) depends on summation order and is not a parity claim
    R, C = shape
    exps_row = np.frombuffer(buf, np.uint8, R * C // 128, 0).reshape(R, C // 128)
    inf_rows = (exps_row == 255).any(axis=1)
    if inf_rows.any():
        assert not np.isfinite(got[inf_rows]).any() and not np.isfinite(ref[inf_rows]).any()
        keep = ~inf_rows
        W, Wx, ref, got = W[keep], Wx[keep], ref[keep], got[keep]
    ovf_rows = (~np.isfinite(W) & np.isfinite(Wx)).any(axis=1)
    if ovf_rows.any():
        ex = Wx[ovf_rows] @ np.asarray(x, dtype=np.float64)
        g = got[ovf_rows]
        big = np.abs(ex) > np.finfo(np.float32).max
        # overflowing rows are non-finite in both (inf, or nan when partial
        # sums of both signs overflow, exactly as the oracle's inf - inf)
        assert not np.isfinite(g[big]).any()
        if (~big).any():
            assert np.allclose(g[~big], ex[~big], rtol=1e-4, atol=np.abs(ex[~big]).max() * 1e-5 + 1e-30)
        keep = ~ovf_rows
        W, ref, got = W[keep], ref[keep], got[keep]
    # a float64 reference beyond float32's range is not representable by a
    # float32 kernel: such outputs must be non-finite on the device
    over = np.isfinite(ref) & (np.abs(ref) > np.finfo(np.float32).max)
    if over.any():
        assert not np.isfinite(got[over]).any()
        keep = ~over
        W, ref, got = W[keep], ref[keep], got[keep]
    finite = np.isfinite(ref)
    assert np.array_equal(np.isfinite(got), finite)
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    assert np.array_equal(got[~finite & ~np.isnan(ref)], ref[~finite & ~np.isnan(ref)])  # infs with sign
    # outputs whose true value lies below the float32 normal range may be
    # flushed to zero on Metal (documented deviation); they are checked
    # separately in test_subnormal_outputs_flush_or_match
    tiny = np.abs(ref) < 1.2e-38 * 4
    assert np.all((got[tiny] == 0) | (np.abs(got[tiny] - ref[tiny]) <= 1e-37))
    finite = finite & ~tiny
    # float32 accumulation against a float64 oracle: |got - ref| must stay
    # within RTOL of |ref| plus ATOL_COND of the row's absolute dot
    # sum_c |W_rc x_c| (the cancellation-aware term). The reduction is
    # sequential per lane (about C/512 word partials, each of 16 or 64
    # products) followed by a five-level simd_sum, so the worst-case
    # rounding factor is about 3.3e-6 of the absolute dot at C = 17408
    # (4.7e-6 for v2), above ATOL_COND; adversarial cancellation probes
    # measure 1.8e-8, and the bound is checked empirically, not derived.
    absdot = (np.abs(W[finite]) @ np.abs(np.asarray(x, dtype=np.float64)))
    # weights the oracle decodes BELOW the float32 normal range are rounded
    # to subnormal precision (spacing 2^-149) before the oracle multiplies;
    # the fused path multiplies the full-precision codebook value first, so
    # it is allowed one subnormal spacing per such weight, times |x_c|.
    sub = (np.abs(W[finite]) < 1.1754943508222875e-38) & (W[finite] != 0)
    sub_term = (sub * np.abs(np.asarray(x, dtype=np.float64))).sum(axis=1) * 2.0 ** -149
    bound = RTOL * np.abs(ref[finite]) + ATOL_COND * absdot + sub_term
    err = np.abs(got[finite] - ref[finite])
    assert (err <= bound).all(), (err / np.maximum(bound, 1e-300)).max()


# ---------------- golden payloads from the frozen compiler ----------------

def test_golden_w4_and_s16_gemv_match_canonical_decode():
    W = np.load(os.path.join(FX, "qcodec_src.npy"))
    exp = json.load(open(os.path.join(FX, "qcodec_expected.json")))
    rng = np.random.default_rng(0)
    x = rng.normal(size=W.shape[1]).astype(np.float32)
    for name, dec, cls in (("w4", qcodec.dec_w4, qmetal.W4Metal), ("s16", qcodec.dec_s16, qmetal.S16Metal)):
        buf = open(os.path.join(FX, f"qcodec_{name}.bin"), "rb").read()
        assert hashlib.sha256(buf).hexdigest() == exp[name]["sha256"]
        res = _w4_or_skip(buf, W.shape) if name == "w4" else cls(buf, W.shape)
        _check(res, buf, W.shape, dec, x)


# ---------------- random payloads, several shapes ----------------

@pytest.mark.parametrize("R,C", [(1, 128), (3, 256), (7, 1024), (64, 5120), (5, 6144), (2, 17408)])
def test_w4_shapes(R, C):
    rng = np.random.default_rng(R * 1000 + C)
    buf = _random_w4(rng, R, C)
    x = rng.normal(size=C).astype(np.float32)
    _check(_w4_or_skip(buf, (R, C)), buf, (R, C), qcodec.dec_w4, x)


@pytest.mark.parametrize("R,C", [(1, 128), (3, 256), (7, 1024), (64, 5120), (5, 6144), (2, 17408)])
def test_s16_shapes(R, C):
    rng = np.random.default_rng(R * 1000 + C + 7)
    buf = _random_s16(rng, R, C)
    x = rng.normal(size=C).astype(np.float32)
    _check(qmetal.S16Metal(buf, (R, C)), buf, (R, C), qcodec.dec_s16, x)


# ---------------- edge exponents and codebook boundaries ----------------

@pytest.mark.parametrize("pool", [[0], [1], [127], [254], [0, 1, 127, 254], [60, 200]])
def test_w4_exponent_edges(pool):
    rng = np.random.default_rng(len(pool))
    R, C = 4, 512
    buf = _random_w4(rng, R, C, exp_pool=np.array(pool))
    x = rng.normal(size=C).astype(np.float32)
    _check(_w4_or_skip(buf, (R, C)), buf, (R, C), qcodec.dec_w4, x)


@pytest.mark.parametrize("pool", [[0], [1], [127], [254], [0, 1, 127, 254]])
def test_s16_exponent_edges(pool):
    rng = np.random.default_rng(len(pool) + 50)
    R, C = 4, 512
    buf = _random_s16(rng, R, C, exp_pool=np.array(pool))
    x = rng.normal(size=C).astype(np.float32)
    _check(qmetal.S16Metal(buf, (R, C)), buf, (R, C), qcodec.dec_s16, x)


def test_exponent_255_gives_inf_scale_like_the_oracle():
    rng = np.random.default_rng(255)
    R, C = 2, 256
    buf = _random_w4(rng, R, C, exp_pool=np.array([255, 127]))
    x = rng.normal(size=C).astype(np.float32)
    W = qcodec.dec_w4(buf, (R, C))
    assert not np.isfinite(W).all()          # the oracle carries inf through
    _check(_w4_or_skip(buf, (R, C)), buf, (R, C), qcodec.dec_w4, x)


def test_w4_codebook_boundaries_and_nonfinite_entries():
    rng = np.random.default_rng(9)
    R, C = 3, 1024
    cb = rng.normal(size=(256, 4)).astype(np.float16)
    cb[0] = [0, 0, 0, 0]
    cb[255] = [65504, -65504, 6e-5, -6e-5]   # fp16 max / min normal
    cb[17, 2] = np.inf
    cb[200, 0] = np.nan
    n = R * C
    idx = rng.integers(0, 256, n // 4)
    idx[0] = 255
    idx[-1] = 0
    buf = _pack_w4(rng.choice(np.arange(120, 134), n // 128), cb, idx)
    x = rng.normal(size=C).astype(np.float32)
    _check(_w4_or_skip(buf, (R, C)), buf, (R, C), qcodec.dec_w4, x)


def test_s16_nibble_order_is_high_first():
    """A payload whose levels make the high and low nibble distinguishable:
    swapping the convention changes every output."""
    R, C = 2, 256
    n = R * C
    lv = np.arange(16, dtype=np.float16)
    codes = np.full(n // 2, 0x1F, np.uint8)        # even = level 1, odd = level 15
    buf = _pack_s16(np.full(n // 128, 127), lv, codes)
    x = np.zeros(C, np.float32)
    x[0::2] = 1.0                                    # picks the even elements only
    got = np.asarray(qmetal.S16Metal(buf, (R, C)).gemv(mx.array(x)))
    assert np.allclose(got, (C // 2) * 1.0)          # level 1 per even element
    x2 = np.zeros(C, np.float32)
    x2[1::2] = 1.0
    got2 = np.asarray(qmetal.S16Metal(buf, (R, C)).gemv(mx.array(x2)))
    assert np.allclose(got2, (C // 2) * 15.0)


# ---------------- partial rows, multi-token, refusals ----------------

def test_partial_row_selection_matches_row_decoder():
    from llmopt.lab.qcodec_fast import S16Rows, W4Rows
    rng = np.random.default_rng(11)
    R, C = 40, 1024
    bw = _random_w4(rng, R, C)
    bs = _random_s16(rng, R, C)
    x = rng.normal(size=C).astype(np.float32)
    yw = np.asarray(qmetal.W4Metal(bw, (R, C)).gemv(mx.array(x)), dtype=np.float64)
    ys = np.asarray(qmetal.S16Metal(bs, (R, C)).gemv(mx.array(x)), dtype=np.float64)
    for lo, hi in ((0, 1), (13, 29), (39, 40)):
        rw = W4Rows(bw, (R, C)).rows(lo, hi).astype(np.float64) @ x.astype(np.float64)
        rs = S16Rows(bs, (R, C)).rows(lo, hi).astype(np.float64) @ x.astype(np.float64)
        assert np.allclose(yw[lo:hi], rw, rtol=RTOL, atol=1e-4)
        assert np.allclose(ys[lo:hi], rs, rtol=RTOL, atol=1e-4)


@pytest.mark.parametrize("T", [1, 2, 8, 9, 20])
def test_multi_token_gemm_matches_gemv_per_token(T):
    rng = np.random.default_rng(100 + T)
    R, C = 33, 768
    bw = _random_w4(rng, R, C)
    bs = _random_s16(rng, R, C)
    X = rng.normal(size=(T, C)).astype(np.float32)
    for buf, cls, dec in ((bw, qmetal.W4Metal, qcodec.dec_w4), (bs, qmetal.S16Metal, qcodec.dec_s16)):
        res = cls(buf, (R, C))
        W64 = dec(buf, (R, C)).astype(np.float64)
        Y = np.asarray(res.gemm(mx.array(X)), dtype=np.float64)
        ref = X.astype(np.float64) @ W64.T
        assert Y.shape == (T, R)
        absdot = np.abs(X.astype(np.float64)) @ np.abs(W64).T
        assert (np.abs(Y - ref) <= RTOL * np.abs(ref) + ATOL_COND * absdot).all()


def test_refuses_wrong_length_and_bad_shapes():
    rng = np.random.default_rng(3)
    buf = _random_w4(rng, 2, 256)
    with pytest.raises(ValueError):
        qmetal.W4Metal(buf[:-1], (2, 256))
    with pytest.raises(ValueError):
        qmetal.W4Metal(buf, (2, 255))
    with pytest.raises(ValueError):
        qmetal.S16Metal(buf, (2, 256))     # wrong codec length
    with pytest.raises(ValueError):
        qmetal.resident(buf, {"codec": "raw", "shape": [2, 256]})
    with pytest.raises(ValueError):
        qmetal.W4Metal(buf, (2, 256)).gemv(mx.zeros((128,)))


def test_no_decoded_matrix_is_held():
    """The resident object keeps only the payload parts (bytes close to
    the compressed size), never a float matrix."""
    rng = np.random.default_rng(4)
    R, C = 64, 5120
    buf = _random_w4(rng, R, C)
    res = qmetal.W4Metal(buf, (R, C))
    assert res.payload_bytes() == len(buf)
    assert not any(isinstance(v, mx.array) and v.dtype == mx.float32 for v in vars(res).values())


def test_subnormal_outputs_flush_or_match():
    """Exponent 0 blocks give weights near 2^-127: the oracle keeps the
    subnormal products, Metal may flush them. Normal-range outputs built
    from such blocks must still match (the scale is applied after the
    partial dot), and any output the oracle puts below the float32 normal
    range may read as zero on the GPU but never as anything else."""
    rng = np.random.default_rng(77)
    R, C = 8, 1024
    buf = _random_w4(rng, R, C, exp_pool=np.array([0]))
    x = (rng.normal(size=C) * 1e6).astype(np.float32)      # lifts outputs into the normal range
    W = qcodec.dec_w4(buf, (R, C)).astype(np.float64)
    ref = W @ x.astype(np.float64)
    assert np.abs(ref).min() > 1.2e-38
    _check(_w4_or_skip(buf, (R, C)), buf, (R, C), qcodec.dec_w4, x)


@pytest.mark.parametrize("R", [1, 2, 3, 5, 6, 7, 9, 13])
def test_w4_rows_not_multiple_of_four(R):
    """v4 groups four rows per simdgroup; every remainder class must be
    masked correctly (no read past the payload, no write past the output)
    and agree with the oracle. Runs under every generation."""
    rng = np.random.default_rng(900 + R)
    C = 1280
    buf = _random_w4(rng, R, C, exp_pool=np.array([120, 127, 130]))
    x = rng.normal(size=C).astype(np.float32)
    res = _w4_or_skip(buf, (R, C))
    _check(res, buf, (R, C), qcodec.dec_w4, x)
    # multi-token tiles on the same odd row count
    X = rng.normal(size=(5, C)).astype(np.float32)
    W = qcodec.dec_w4(buf, (R, C)).astype(np.float64)
    ref = X.astype(np.float64) @ W.T
    got = np.asarray(res.gemm(mx.array(X)), dtype=np.float64)
    assert got.shape == (5, R)
    absdot = np.abs(X.astype(np.float64)) @ np.abs(W).T
    assert np.all(np.abs(got - ref) <= RTOL * np.abs(ref) + ATOL_COND * absdot)


def test_w4_v4_rows_bit_identical_to_v3():
    """v4 computes each row by the same float32 operation sequence as v3,
    so on real-range payloads the outputs are bit-identical, not merely
    within the parity bound."""
    if qmetal.VERSION != 4:
        pytest.skip("checked once, under the v4 parameter")
    rng = np.random.default_rng(4242)
    R, C = 37, 5120
    buf = _random_w4(rng, R, C, exp_pool=np.arange(111, 129))
    X = rng.normal(size=(8, C)).astype(np.float32)
    res4 = qmetal.W4Metal(buf, (R, C))
    res3 = qmetal.W4Metal(buf, (R, C))
    res3.version = 3
    g4 = np.asarray(res4.gemm(mx.array(X)))
    g3 = np.asarray(res3.gemm(mx.array(X)))
    assert np.array_equal(g4, g3)
