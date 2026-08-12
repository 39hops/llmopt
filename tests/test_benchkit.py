"""Behavior tests for llmopt.search.benchkit + llmopt.common.quant
(spec 2026-08-12 §4.2 helper adoption)."""
import pytest

torch = pytest.importorskip("torch")
sp = pytest.importorskip("sympy")

from llmopt.common.quant import ternary  # noqa: E402
from llmopt.common.seed import srng  # noqa: E402
from llmopt.search.benchkit import NnueEval, _check, _root  # noqa: E402


@pytest.mark.parametrize("kind", ["diff", "int"])
def test_root_check_roundtrip(kind):
    prob, truth = _root(srng("benchkit-test", 1), 1, kind)
    assert isinstance(prob, (sp.Derivative, sp.Integral))
    ans = prob.doit()
    assert _check(kind, ans, truth)


def test_check_rejects_wrong_answer():
    prob, truth = _root(srng("benchkit-test", 2), 1, "diff")
    assert not _check("diff", prob.doit() + 1, truth)


def test_nnue_eval_shape():
    from llmopt.search.features import N_FEATURES
    net = NnueEval()
    out = net(torch.zeros(3, N_FEATURES))
    assert out.shape == (3,)


def test_ternary_values_and_signs():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(4, 16, generator=g)
    q = ternary(w)
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    # each row's outputs live in {-s, 0, +s}
    for r in range(4):
        vals = set(q[r].tolist())
        assert vals <= {-float(s[r]), 0.0, float(s[r])}
    nz = q != 0
    assert torch.equal(torch.sign(q[nz]), torch.sign(w[nz]))
