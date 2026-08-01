"""llmopt.intmath — the promoted deterministic-birth core must be
bit-equivalent to the certified scratch references (table shas
pinned to the cross-lab digests; optimizer checked step-for-step
against the R2 implementation)."""
import math
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, "scratch")

from llmopt.intmath import (IntAdamW, Q, build_exp_table,  # noqa: E402
                            build_silu_tables, int_mm,
                            isqrt_newton, mean_square_q32, rdiv,
                            rms_isqrt_q16, table_sha)


def test_table_shas_pinned():
    silu, dsilu = build_silu_tables()
    assert table_sha(silu).startswith("24499877ab63ee6b")
    assert table_sha(dsilu).startswith("967943f938fc924f")
    assert table_sha(build_exp_table()).startswith("9b8649244ca8c235")


def test_rdiv_equals_p3_form_all_integers():
    """The program-wide rdiv unification (axiom proof, relay
    2026-07-31-4): r1 form == P3 form, odd divisors included."""
    x = torch.arange(-1000, 1001, dtype=torch.int64)
    for d in (1, 2, 3, 7, 10, 512, 513):
        p3 = torch.where(x >= 0, (2 * x + d) // (2 * d),
                         -((-2 * x + d) // (2 * d)))
        assert torch.equal(rdiv(x, d), p3), f"d={d}"


def test_int_mm_exact_vs_bigint():
    torch.manual_seed(0)
    a = torch.randint(-(1 << 20), 1 << 20, (5, 7), dtype=torch.int64)
    w = torch.randint(-(1 << 20), 1 << 20, (3, 7), dtype=torch.int64)
    y = int_mm(a, w)
    for i in range(5):
        for j in range(3):
            ref = sum(int(a[i, k]) * int(w[j, k]) for k in range(7))
            assert int(y[i, j]) == ref


def test_isqrt_exact_floor():
    x = torch.cat([torch.arange(0, 500, dtype=torch.int64),
                   torch.tensor([2 ** 40, 2 ** 40 + 1,
                                 (10 ** 9) ** 2 - 1, (10 ** 9) ** 2],
                                dtype=torch.int64)])
    r = isqrt_newton(x)
    assert torch.all(r * r <= x)
    assert torch.all((r + 1) * (r + 1) > x)


@pytest.mark.parametrize("dim", [1, 2, 64, 128])
def test_mean_square_q32_bitidentical_to_legacy_in_safe_range(dim):
    """Factoring Q^2 before multiplication changes no safe bit."""
    torch.manual_seed(dim)
    x = torch.randint(-46340, 46341, (17, dim), dtype=torch.int64)
    s2 = (x * x).sum(-1, keepdim=True)
    legacy = (s2 // dim) * (1 << 32) // (Q * Q) + 42950
    assert torch.equal(mean_square_q32(x, dim), legacy)


def test_rms_factoring_recovers_booked_w4c_overflow():
    """ACLAMP=49152 used to wrap m40 negative and produce isq=0."""
    dim = 128
    x = torch.full((3, dim), 49152, dtype=torch.int64)
    s2 = (x * x).sum(-1, keepdim=True)
    legacy = (s2 // dim) * (1 << 32) // (Q * Q) + 42950
    assert torch.all(legacy < 0)  # reproduces BR-W4c's failure mechanism

    mean = 49152 ** 2
    oracle = mean * ((1 << 32) // (Q * Q)) + 42950
    got = mean_square_q32(x, dim)
    assert torch.all(got == oracle)
    assert torch.all(rms_isqrt_q16(x, dim) == math.isqrt(oracle))


def test_mean_square_rejects_rounding_contract_change():
    with pytest.raises(ValueError, match="exactly divide"):
        mean_square_q32(torch.ones(1, 4, dtype=torch.int64), 4, q=500)


def test_intadamw_matches_r2_reference():
    """Step-for-step equivalence with the certified scratch R2
    optimizer (shift=0, lr 1/20) on random grads."""
    from detbwd_r2_adamw import IntAdamW as R2Ref
    torch.manual_seed(4)
    w1 = torch.randint(-Q, Q + 1, (8, 8), dtype=torch.int64)
    w2 = w1.clone()
    a = IntAdamW([w1], shift=0, lrn=1, lrd=20)
    b = R2Ref([w2])
    for _ in range(25):
        g = torch.randint(-Q, Q + 1, (8, 8), dtype=torch.int64)
        a.step([g.clone()])
        b.step([g.clone()])
        assert torch.equal(w1, w2)


def test_intadamw_shift_widens_updates():
    """R3a's mechanism from fixed state: a small-but-real moment
    floors to zero at shift=0 and updates at shift=8."""
    outs = {}
    for shift in (0, 8):
        w = torch.zeros(4, dtype=torch.int64)
        opt = IntAdamW([w], shift=shift, lrd=1000)
        opt.m = [torch.full((4,), 100, dtype=torch.int64)]
        opt.v = [torch.full((4,), 2000, dtype=torch.int64)]
        opt.t = 100                     # bias corrections ~ 1
        opt.step([torch.zeros(4, dtype=torch.int64)])
        outs[shift] = opt.nz_last
    assert outs[0] == 0.0 and outs[8] == 1.0
