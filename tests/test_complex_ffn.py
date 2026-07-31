import pytest
import torch

from llmopt.train.complex_ffn import ComplexFFN


def test_matches_true_complex_arithmetic():
    """The (re|im)-half layout must compute genuine C-arithmetic:
    rebuild the forward with torch.complex64 and compare."""
    torch.manual_seed(0)
    m = ComplexFFN(d=16, ffn=32).eval()
    h = torch.randn(2, 5, 16)
    f2 = m.f2

    def cplx(zr, zi):
        return torch.complex(zr, zi)

    zg = m.gate(h)
    zu = m.up(h)
    g = cplx(zg[..., :f2], zg[..., f2:])
    u = cplx(zu[..., :f2], zu[..., f2:])
    mag = torch.sqrt(g.real**2 + g.imag**2 + 1e-12)
    act = torch.relu(mag + m.mod_b) / mag
    p = (g * act.to(g.dtype)) * u                # complex product
    y_ref = m.down(torch.cat([p.real, p.imag], -1))
    assert torch.allclose(m(h), y_ref, atol=1e-6)


def test_phase_equivariance_of_product():
    """Rotating BOTH complex inputs by e^{i*theta} rotates the
    modReLU-gated product by e^{2i*theta} (magnitudes untouched) —
    the property that makes this a rotational block at all."""
    torch.manual_seed(1)
    f2 = 8
    g = torch.complex(torch.randn(4, f2), torch.randn(4, f2))
    u = torch.complex(torch.randn(4, f2), torch.randn(4, f2))
    th = torch.tensor(0.7)
    rot = torch.polar(torch.ones(()), th)

    def gated_product(g, u, b=torch.zeros(f2)):
        mag = g.abs().clamp(min=1e-12)
        act = torch.relu(mag + b) / mag
        return (g * act.to(g.dtype)) * u

    p0 = gated_product(g, u)
    p1 = gated_product(g * rot, u * rot)
    assert torch.allclose(p1, p0 * torch.polar(torch.ones(()),
                                              2 * th), atol=1e-5)


def test_grad_flows_and_shapes():
    m = ComplexFFN(d=8, ffn=16)
    h = torch.randn(3, 7, 8, requires_grad=True)
    y = m(h)
    assert y.shape == (3, 7, 8)
    y.sum().backward()
    assert h.grad is not None
    assert m.mod_b.grad is not None


def test_odd_ffn_rejected():
    with pytest.raises(ValueError):
        ComplexFFN(d=8, ffn=15)
