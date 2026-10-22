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


@pytest.fixture(autouse=True, params=[1, 2, 3, 4, 5], ids=["v1", "v2", "v3", "v4", "v5"])
def _kernel_version(request, monkeypatch):
    """Every parity case runs under all five kernel generations."""
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
    res4 = qmetal.W4Metal(buf, (R, C))
    res3 = qmetal.W4Metal(buf, (R, C))
    res3.version = 3
    # tiles 1 and 2 run the v4 kernel, 8 dispatches to v3 under version 4;
    # all three must be bit-identical to the v3 control
    for T in (1, 2, 8):
        X = rng.normal(size=(T, C)).astype(np.float32)
        g4 = np.asarray(res4.gemm(mx.array(X)))
        g3 = np.asarray(res3.gemm(mx.array(X)))
        assert np.array_equal(g4, g3), T


def test_s16_v5_rows_bit_identical_to_shipped():
    """v5 reads each level from the device table instead of a per-lane
    array; the value and the per-row float32 operation sequence are the
    shipped kernel's, so outputs are bit-identical across the exponent
    classes (normal, the split accumulators, the infinite scale) and at
    every tile, including rows the oracle sends through the split
    accumulators."""
    if qmetal.VERSION != 5:
        pytest.skip("checked once, under the v5 parameter")
    rng = np.random.default_rng(5151)
    R, C = 37, 5120
    # pools: the real range; FINITE rows through the second accumulator
    # (exponents 20 to 30, scale 2^-107 to 2^-97) and the third (191 to 200,
    # scale 2^64 to 2^73) mixed with normal blocks; the overflow and
    # infinite-scale edges, whose rows are non-finite in both kernels
    pools = (np.arange(111, 129), np.r_[np.arange(20, 31), np.arange(111, 129)],
             np.r_[np.arange(191, 201), np.arange(111, 129)],
             np.array([0, 1, 30, 31, 127, 190, 191, 254]), np.array([120, 255]))
    for pool in pools:
        buf = _random_s16(rng, R, C, exp_pool=pool)
        res5 = qmetal.S16Metal(buf, (R, C))
        res3 = qmetal.S16Metal(buf, (R, C))
        res3.version = 3
        assert res5.version == 5
        for T in (1, 2, 8, 9):
            X = rng.normal(size=(T, C)).astype(np.float32)
            g5 = np.asarray(res5.gemm(mx.array(X)))
            g3 = np.asarray(res3.gemm(mx.array(X)))
            # bit-level equality (NaN payloads and signed zeros included)
            assert np.array_equal(g5.view(np.uint32), g3.view(np.uint32)), (pool[:3], T)
            if pool[0] in (20, 191):
                assert np.isfinite(g3).all(), "the split-accumulator pool must give finite rows"


def test_generation_5_dispatches_w4_as_generation_4():
    """Generation 5 changes the S16 kernel only: a W4 resident under
    version 5 returns the bits of version 4 at tiles 1, 2 and 8."""
    if qmetal.VERSION != 5:
        pytest.skip("checked once, under the v5 parameter")
    rng = np.random.default_rng(4545)
    R, C = 37, 5120
    buf = _random_w4(rng, R, C, exp_pool=np.arange(111, 129))
    res5 = qmetal.W4Metal(buf, (R, C))
    res4 = qmetal.W4Metal(buf, (R, C))
    res4.version = 4
    for T in (1, 2, 8):
        X = rng.normal(size=(T, C)).astype(np.float32)
        assert np.array_equal(np.asarray(res5.gemm(mx.array(X))), np.asarray(res4.gemm(mx.array(X)))), T


# ---------------- decode-once float32 GEMM prefill path ----------------
# Off by default (QMETAL_PREFILL_GEMM); the tests switch it on per case.

def _gemm_on(monkeypatch, min_t=1):
    monkeypatch.setattr(qmetal._Resident, "prefill_gemm", True)
    monkeypatch.setattr(qmetal._Resident, "gemm_min_t", min_t)


def _gemm_off(monkeypatch):
    monkeypatch.setattr(qmetal._Resident, "prefill_gemm", False)


RTOL_GEMM = 1e-5
ATOL_COND_GEMM = 1e-5     # the framework GEMM's blocked float32 reduction, bound checked empirically


@pytest.mark.parametrize("codec", ["w4", "s16"])
@pytest.mark.parametrize("pool", [None, [31, 100, 190], [40, 127, 150]])
def test_decode_f32_matches_canonical_decoder_bitwise(codec, pool):
    """The on-device float32 decode equals qcodec's decode cast to float32,
    bit for bit including the sign of zero, over the whole normal exponent
    class [E_LO, E_HI), whole matrix and row slabs alike."""
    rng = np.random.default_rng(11 if codec == "w4" else 12)
    R, C = 37, 1024
    ep = None if pool is None else np.array(pool)
    buf = _random_w4(rng, R, C, exp_pool=ep) if codec == "w4" else _random_s16(rng, R, C, exp_pool=ep)
    res = _w4_or_skip(buf, (R, C)) if codec == "w4" else qmetal.S16Metal(buf, (R, C))
    assert res.gemm_ok
    dec = qcodec.dec_w4 if codec == "w4" else qcodec.dec_s16
    ref = dec(buf, (R, C)).astype(np.float32)
    got = np.asarray(res.decode_f32(0, R))
    assert got.dtype == np.float32 and got.shape == (R, C)
    assert np.array_equal(got, ref)
    assert np.array_equal(np.signbit(got), np.signbit(ref))
    slab = np.asarray(res.decode_f32(5, 20))
    assert np.array_equal(slab, ref[5:20]) and np.array_equal(np.signbit(slab), np.signbit(ref[5:20]))


def test_decode_f32_keeps_signed_zero_and_routes_nonfinite_tables_to_fused(monkeypatch):
    rng = np.random.default_rng(13)
    R, C = 4, 512
    cb = rng.normal(size=(256, 4)).astype(np.float16)
    cb[0] = [0.0, -0.0, 1.0, -1.0]
    buf = _random_w4(rng, R, C, cb=cb)
    res = _w4_or_skip(buf, (R, C))
    assert res.gemm_ok
    got = np.asarray(res.decode_f32(0, R))
    ref = qcodec.dec_w4(buf, (R, C)).astype(np.float32)
    assert np.array_equal(got, ref) and np.array_equal(np.signbit(got), np.signbit(ref))
    assert (np.signbit(got) & (got == 0)).any()        # a negative zero survived the decode
    cb2 = cb.copy()
    cb2[1] = [np.inf, 1.0, np.nan, 2.0]
    buf2 = _random_w4(rng, R, C, cb=cb2)
    res2 = _w4_or_skip(buf2, (R, C))
    assert not res2.gemm_ok                              # nonfinite table entries stay on the fused law
    _gemm_on(monkeypatch)
    X = rng.normal(size=(16, C)).astype(np.float32)
    res2.gemm(mx.array(X))
    assert res2.last_path == "fused"


@pytest.mark.parametrize("pool", [[0], [30, 100], [191, 120], [255, 120]])
def test_exceptional_exponents_route_to_fused(pool, monkeypatch):
    """Blocks below 2^-96 or at 2^64 and above (and the infinite scale 255)
    keep the fused kernels' split-accumulator law: such a matrix is never
    sent through the float32 decode, and its multi-token output is the
    fused-only output bit for bit."""
    rng = np.random.default_rng(14)
    R, C = 8, 512
    buf = _random_w4(rng, R, C, exp_pool=np.array(pool))
    res = _w4_or_skip(buf, (R, C))
    assert not res.gemm_ok
    X = rng.normal(size=(16, C)).astype(np.float32)
    _gemm_off(monkeypatch)
    y0 = np.asarray(res.gemm(mx.array(X)))
    _gemm_on(monkeypatch)
    y1 = np.asarray(res.gemm(mx.array(X)))
    assert res.last_path == "fused"
    assert np.array_equal(y0, y1, equal_nan=True)
    bs = _random_s16(rng, R, C, exp_pool=np.array(pool))
    assert not qmetal.S16Metal(bs, (R, C)).gemm_ok


@pytest.mark.parametrize("T,R,C", [(16, 33, 768), (64, 70, 1024), (256, 17, 5120), (40, 6, 17408)])
def test_gemm_path_matches_oracle_and_fused(T, R, C, monkeypatch):
    """The decode-once GEMM output agrees with the float64 oracle within the
    GEMM bound and with the fused output within both bounds; the fused
    control is bit-identical to the path with the switch off."""
    rng = np.random.default_rng(200 + T)
    X = rng.normal(size=(T, C)).astype(np.float32)
    X64 = X.astype(np.float64)
    for codec, dec in (("w4", qcodec.dec_w4), ("s16", qcodec.dec_s16)):
        buf = _random_w4(rng, R, C) if codec == "w4" else _random_s16(rng, R, C)
        res = _w4_or_skip(buf, (R, C)) if codec == "w4" else qmetal.S16Metal(buf, (R, C))
        W64 = dec(buf, (R, C)).astype(np.float64)
        ref = X64 @ W64.T
        absdot = np.abs(X64) @ np.abs(W64).T
        _gemm_off(monkeypatch)
        y_fused = np.asarray(res.gemm(mx.array(X)), dtype=np.float64)
        assert res.last_path == "fused"
        _gemm_on(monkeypatch)
        y = np.asarray(res.gemm(mx.array(X)), dtype=np.float64)
        assert res.last_path == "gemm"
        assert y.shape == (T, R)
        assert np.isfinite(y).all()
        assert (np.abs(y - ref) <= RTOL_GEMM * np.abs(ref) + ATOL_COND_GEMM * absdot).all()
        assert (np.abs(y - y_fused) <= RTOL_GEMM * np.abs(ref) + (ATOL_COND_GEMM + ATOL_COND) * absdot).all()
        _gemm_off(monkeypatch)
        assert np.array_equal(np.asarray(res.gemm(mx.array(X))), y_fused)


def test_gemm_threshold_and_control(monkeypatch):
    rng = np.random.default_rng(15)
    R, C = 9, 512
    buf = _random_w4(rng, R, C)
    res = _w4_or_skip(buf, (R, C))
    # per-codec defaults (the measured crossovers): W4 16, S16 32
    monkeypatch.setattr(qmetal._Resident, "prefill_gemm", True)
    monkeypatch.setattr(qmetal._Resident, "gemm_min_t", None)
    res.gemm(mx.array(rng.normal(size=(15, C)).astype(np.float32)))
    assert res.last_path == "fused"
    res.gemm(mx.array(rng.normal(size=(16, C)).astype(np.float32)))
    assert res.last_path == "gemm"
    rs = qmetal.S16Metal(_random_s16(rng, R, C), (R, C))
    rs.gemm(mx.array(rng.normal(size=(31, C)).astype(np.float32)))
    assert rs.last_path == "fused"
    rs.gemm(mx.array(rng.normal(size=(32, C)).astype(np.float32)))
    assert rs.last_path == "gemm"
    _gemm_on(monkeypatch, min_t=32)
    res.gemm(mx.array(rng.normal(size=(31, C)).astype(np.float32)))
    assert res.last_path == "fused"
    res.gemm(mx.array(rng.normal(size=(32, C)).astype(np.float32)))
    assert res.last_path == "gemm"
    res.gemv(mx.array(rng.normal(size=C).astype(np.float32)))
    assert res.last_path == "fused"
    _gemm_off(monkeypatch)
    res.gemm(mx.array(rng.normal(size=(256, C)).astype(np.float32)))
    assert res.last_path == "fused"


def test_gemm_slabs_and_squeeze(monkeypatch):
    """Row slabs (bounded temporaries) give the same output as one slab,
    the 1-d input still squeezes, and no decoded matrix stays on the
    resident afterwards."""
    rng = np.random.default_rng(16)
    R, C = 37, 1024
    buf = _random_w4(rng, R, C)
    res = _w4_or_skip(buf, (R, C))
    X = rng.normal(size=(8, C)).astype(np.float32)
    _gemm_on(monkeypatch)
    y_one = np.asarray(res.gemm(mx.array(X)), dtype=np.float64)
    monkeypatch.setattr(qmetal, "GEMM_SLAB_BYTES", C * 4 * 8)        # eight-row slabs
    y_slab = np.asarray(res.gemm(mx.array(X)), dtype=np.float64)
    assert res.last_path == "gemm"
    W64 = qcodec.dec_w4(buf, (R, C)).astype(np.float64)
    ref = X.astype(np.float64) @ W64.T
    absdot = np.abs(X.astype(np.float64)) @ np.abs(W64).T
    for y in (y_one, y_slab):
        assert (np.abs(y - ref) <= RTOL_GEMM * np.abs(ref) + ATOL_COND_GEMM * absdot).all()
    y1 = res.gemm(mx.array(X[0]))
    assert y1.shape == (R,)
    assert not any(isinstance(v, mx.array) and v.dtype == mx.float32 for v in vars(res).values())


def test_path_counters(monkeypatch):
    rng = np.random.default_rng(17)
    R, C = 5, 512
    res = _w4_or_skip(_random_w4(rng, R, C), (R, C))
    assert (res.calls_fused, res.calls_gemm) == (0, 0)
    _gemm_on(monkeypatch, min_t=16)
    res.gemm(mx.array(rng.normal(size=(16, C)).astype(np.float32)))
    res.gemm(mx.array(rng.normal(size=(4, C)).astype(np.float32)))
    res.gemv(mx.array(rng.normal(size=C).astype(np.float32)))
    assert (res.calls_fused, res.calls_gemm) == (2, 1)
    other = qmetal.S16Metal(_random_s16(rng, R, C), (R, C))
    other.gemv(mx.array(rng.normal(size=C).astype(np.float32)))
    assert qmetal.path_counts([res, other]) == {"fused": 3, "gemm": 1}


# ---- scoped GEMM dispatch (the qualified prefill entry point enables the
# path for its own call only; nothing else inherits it from a row count) ----

def test_gemm_scope_gates_dispatch_not_row_count(monkeypatch):
    """With the process default off, a 64-row call takes the fused path;
    inside gemm_scope(True) the same call takes the GEMM path; after the
    scope exits the next call is fused again."""
    _gemm_off(monkeypatch)
    rng = np.random.default_rng(7)
    res = _w4_or_skip(_random_w4(rng, 24, 256), (24, 256))
    x = mx.array(rng.standard_normal((64, 256)).astype(np.float32))
    assert qmetal.gemm_scope_state() is None
    y0 = res.gemm(x); mx.eval(y0)
    assert res.last_path == "fused"
    with qmetal.gemm_scope(True):
        assert qmetal.gemm_scope_state() is True
        y1 = res.gemm(x); mx.eval(y1)
        assert res.last_path == "gemm"
    assert qmetal.gemm_scope_state() is None
    y2 = res.gemm(x); mx.eval(y2)
    assert res.last_path == "fused"
    assert mx.array_equal(y0, y2)
    assert res.calls_fused == 2 and res.calls_gemm == 1


def test_gemm_scope_false_is_a_fused_only_control(monkeypatch):
    """gemm_scope(False) forces the fused kernels even when the process
    default (or an outer scope) has the GEMM path on."""
    _gemm_on(monkeypatch, min_t=1)
    rng = np.random.default_rng(8)
    res = qmetal.S16Metal(_random_s16(rng, 20, 128), (20, 128))
    x = mx.array(rng.standard_normal((40, 128)).astype(np.float32))
    with qmetal.gemm_scope(True):
        with qmetal.gemm_scope(False):
            y = res.gemm(x); mx.eval(y)
            assert res.last_path == "fused"
        assert qmetal.gemm_scope_state() is True
        y = res.gemm(x); mx.eval(y)
        assert res.last_path == "gemm"
    with qmetal.gemm_scope(False):
        y = res.gemm(x); mx.eval(y)
        assert res.last_path == "fused"


def test_gemm_scope_restores_on_exception_and_keeps_small_tiles_fused(monkeypatch):
    _gemm_off(monkeypatch)
    rng = np.random.default_rng(9)
    res = _w4_or_skip(_random_w4(rng, 16, 128), (16, 128))
    with pytest.raises(RuntimeError):
        with qmetal.gemm_scope(True):
            raise RuntimeError("boom")
    assert qmetal.gemm_scope_state() is None
    with qmetal.gemm_scope(True):
        # the per-codec threshold still applies inside the scope: a decode
        # step (one row) and an 8-row tile stay on the fused kernels
        for t in (1, 8):
            y = res.gemm(mx.array(rng.standard_normal((t, 128)).astype(np.float32))); mx.eval(y)
            assert res.last_path == "fused"
        y = res.gemm(mx.array(rng.standard_normal((16, 128)).astype(np.float32))); mx.eval(y)
        assert res.last_path == "gemm"


def test_two_d_batch_after_prefill_scope_stays_fused(monkeypatch):
    """A 2-D call of min_t rows or more reaching _run after a prefill scope
    has exited dispatches by the process default (fused), on a real
    resident, through the model-style gemm entry."""
    _gemm_off(monkeypatch)
    rng = np.random.default_rng(11)
    res = _w4_or_skip(_random_w4(rng, 8, 128), (8, 128))
    x = mx.array(rng.standard_normal((32, 128)).astype(np.float32))
    with qmetal.gemm_scope(True):
        y_in = res.gemm(x); mx.eval(y_in)
        assert res.last_path == "gemm"
    y_out = res.gemm(x); mx.eval(y_out)
    assert res.last_path == "fused"
    y_ctrl = res.gemm(x.reshape(4, 8, 128)); mx.eval(y_ctrl)    # a 3-D batch flattens to 32 rows: still fused
    assert res.last_path == "fused"
    assert mx.array_equal(y_ctrl, y_out)
