"""llmopt.lab.anatomy — projection and scaling invariants."""
import pytest

torch = pytest.importorskip("torch")

from llmopt.lab import anatomy  # noqa: E402


def _w(n=64, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_project_shapes():
    W = _w()
    for method in ("pca", "sphere", "polar"):
        xs, ys, mag = anatomy.project(W, method)
        assert xs.shape == ys.shape == mag.shape == (64,)


def test_project_unknown_method_raises():
    with pytest.raises(ValueError):
        anatomy.project(_w(), "tsne")


def test_polar_x_is_angle():
    xs, _, _ = anatomy.project(_w(), "polar")
    assert xs.min() >= -torch.pi and xs.max() <= torch.pi


def test_rank_scale_uniform():
    mag = _w().norm(dim=1)
    r = anatomy.rank_scale(mag)
    assert float(r.min()) == 0.0 and float(r.max()) == 1.0
    # ranks preserve the magnitude ordering
    assert torch.equal(r.argsort(), mag.argsort())


def test_neuron_rows_pools_family(tmp_path):
    sd = {"blocks.0.gate.weight": torch.ones(4, 3),
          "blocks.1.gate.weight": torch.zeros(5, 3),
          "blocks.0.up.weight": torch.ones(9, 3)}
    p = tmp_path / "toy.pt"
    torch.save(sd, p)
    label, W = anatomy.neuron_rows(str(p), "gate.weight")
    assert W.shape == (9, 3)
    assert "2 layers" in label


def test_neuron_rows_missing_key_raises(tmp_path):
    p = tmp_path / "toy.pt"
    torch.save({"emb.weight": torch.ones(2, 2)}, p)
    with pytest.raises(ValueError):
        anatomy.neuron_rows(str(p), "gate.weight")


def test_render_writes_both_modes(tmp_path):
    pytest.importorskip("matplotlib")
    outs = anatomy.render_dot_views(
        _w(), str(tmp_path / "toy"), "TOY", source_label="toy rows",
        provenance="toy", dpi=40)
    assert [o.endswith(m + ".png") for o, m in zip(outs, ("dark", "light"))]
    for o in outs:
        assert (tmp_path / o.split("/")[-1]).stat().st_size > 0
