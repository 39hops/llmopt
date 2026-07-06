import sympy as sp

from llmopt.search.features import N_FEATURES, featurize

x = sp.Symbol("x")


def test_shape_and_determinism():
    e = sp.Integral(2 * x * sp.cos(x**2), x) + sp.Derivative(x**3, x)
    v = featurize(e)
    assert len(v) == N_FEATURES
    assert v == featurize(e)
    assert all(isinstance(f, float) for f in v)


def test_solved_vs_unsolved_visible():
    solved = featurize(3 * x**2 + sp.cos(x))
    unsolved = featurize(sp.Integral(3 * x**2 + sp.cos(x), x))
    assert solved != unsolved


def test_scales_with_unsolved_mass():
    small = featurize(sp.Integral(x, x))
    big = featurize(sp.Integral(x, x) + sp.Integral(sp.sin(x) * sp.exp(x), x))
    # summed-unsolved-ops feature must grow
    assert sum(big) > sum(small)
