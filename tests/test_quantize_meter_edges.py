"""Capacity-meter boundary behaviour (rehabilitation 2C, step 5).

Ordinary inputs keep their exact pre-change values (regression pins
captured from the meter at commit 99503ed1 on torch 2.12.1 CPU).
Undefined statistics are explicit: ``meter_detailed`` names them in
``flags`` and the numeric slot carries NaN, never a fabricated finite
value; ``meter`` keeps its (M, kurtosis) tuple; invalid input (empty,
non-2-D, non-finite) raises ValueError.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from llmopt.quantize.meter import (MeterResult, meter, meter_detailed,
                                   meter_group, meter_group_detailed)


def _fixtures():
    torch.manual_seed(0)
    g = torch.randn(64, 32)
    h = g.clone()
    h[torch.rand_like(h) > 0.98] *= 25
    u = torch.rand(50, 40) - 0.5
    r = torch.where(torch.rand(32, 64) < 0.5, -1.0, 1.0)
    return g, h, u, r


G, H, U, R = _fixtures()

# (tensor, M, kurtosis) exactly as the pre-change meter returned them.
REGRESSION = [
    ("gauss", G, 1.6702032655055077, 3.2057228088378906),
    ("heavy", H, 4.760847961368769, 193.13136291503906),
    ("uniform", U, 0.9848146783185294, 1.839089274406433),
    ("rademacher", R, 1.1072612757107303, 0.9993323683738708),
    ("single_row", G[:1], 1.0432524370391922, 2.0764198303222656),
    ("bf16", G.bfloat16(), 1.67139360621089, 3.2045764923095703),
]


@pytest.mark.parametrize("name,w,m_pin,k_pin", REGRESSION, ids=[r[0] for r in REGRESSION])
def test_ordinary_inputs_keep_their_values(name, w, m_pin, k_pin):
    m, k = meter(w)
    assert m == m_pin
    assert k == k_pin
    d = meter_detailed(w)
    assert isinstance(d, MeterResult)
    assert (d.m_bits, d.kurtosis, d.flags) == (m, k, ())


def test_group_keeps_its_value():
    assert meter_group([G, H]) == (3.215525613437138, 98.16854286193848, 4096)


# --- analytical fixtures ---------------------------------------------------

def test_gaussian_kurtosis_is_three():
    torch.manual_seed(1)
    _, k = meter(torch.randn(400, 500))
    assert abs(k - 3.0) < 0.05


def test_uniform_kurtosis_is_nine_fifths():
    torch.manual_seed(2)
    _, k = meter(torch.rand(400, 500))
    assert abs(k - 1.8) < 0.02


def test_rademacher_is_exact():
    """+-1 rows of 16: unbiased row sigma sqrt(16/15), grid ceil(2/sigma)=2,
    codes +-2 -> span 5 -> 3 bits, entropy 1 bit -> M = 2. The meter's
    kurtosis is the biased fourth moment over the UNBIASED variance
    squared, so a two-point law gives ((N-1)/N)^2 with N=64, not 1."""
    w = torch.tensor([[1.0, -1.0] * 8] * 4)
    m, k = meter(w)
    assert m == 2.0
    assert abs(k - (63 / 64) ** 2) < 1e-6


# --- undefined statistics are explicit -----------------------------------

@pytest.mark.parametrize("w", [torch.full((8, 8), 0.5), torch.zeros(4, 4),
                               torch.full((1, 16), -3.0)],
                         ids=["constant", "zeros", "constant_row"])
def test_constant_tensor_is_flagged_not_measured(w):
    d = meter_detailed(w)
    assert "constant" in d.flags
    assert math.isnan(d.kurtosis)
    assert math.isnan(d.m_bits)
    m, k = meter(w)
    assert math.isnan(m) and math.isnan(k)


def test_near_constant_tensor_reports_clamped_rows():
    torch.manual_seed(3)
    w = torch.full((8, 8), 1.0) + 1e-12 * torch.randn(8, 8)
    d = meter_detailed(w)
    assert "sigma_clamped" in d.flags
    assert math.isnan(d.m_bits)
    # the spread is below float32 resolution around 1.0: constant in fp32
    assert "constant" in d.flags


def test_near_constant_but_resolvable_spread_measures_kurtosis_in_float64():
    torch.manual_seed(4)
    w = (1e-12 * torch.randn(64, 64)).double()  # (1e-12)^4 underflows fp32
    d = meter_detailed(w)
    assert math.isfinite(d.kurtosis) and abs(d.kurtosis - 3.0) < 0.5
    assert "kurtosis_float64" in d.flags
    assert "sigma_clamped" in d.flags  # 1e-12 is under the 1e-8 grid floor
    assert math.isfinite(d.m_bits)


def test_single_column_has_no_row_sigma():
    d = meter_detailed(G[:, :1])
    assert "single_column" in d.flags
    assert math.isnan(d.m_bits)
    assert math.isfinite(d.kurtosis)


def test_large_magnitudes_do_not_overflow_kurtosis():
    d = meter_detailed(G * 1e30)
    assert math.isfinite(d.kurtosis)
    assert abs(d.kurtosis - 3.2057228088378906) < 1e-3
    assert "kurtosis_float64" in d.flags


@pytest.mark.parametrize("w,match", [
    (torch.zeros(0, 4), "empty"),
    (torch.zeros(4, 0), "empty"),
    (torch.zeros(5), "2-D"),
    (torch.zeros(2, 3, 4), "2-D"),
    (torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), "non-finite"),
    (torch.tensor([[1.0, float("inf")], [0.0, 1.0]]), "non-finite"),
], ids=["empty_rows", "empty_cols", "1d", "3d", "nan", "inf"])
def test_invalid_input_raises(w, match):
    with pytest.raises(ValueError, match=match):
        meter(w)
    with pytest.raises(ValueError, match=match):
        meter_detailed(w)


# --- group semantics --------------------------------------------------------

def test_group_excludes_undefined_tensors_and_counts_them():
    const = torch.full((16, 16), 2.0)
    m, k, n = meter_group([G, const, H])
    assert (m, k, n) == meter_group([G, H])
    d = meter_group_detailed([G, const, H])
    assert d.n_params == n and d.n_excluded == const.numel()
    assert d.excluded_flags == {"constant": 1, "sigma_clamped": 1}
    assert d.m_bits == m and d.kurtosis == k


def test_group_of_only_undefined_tensors_is_nan():
    m, k, n = meter_group([torch.zeros(3, 3)])
    assert math.isnan(m) and math.isnan(k) and n == 0


def test_cpu_and_mps_agree_when_available():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    cpu = meter(G)
    mps = meter(G.to("mps"))
    assert cpu == mps


def test_requires_grad_tensor_is_accepted():
    w = G.clone().requires_grad_(True)
    assert meter(w) == meter(G)
