"""Export-profile sizes are pinned: one full-res render fans out to
the README (1600w quantized), LinkedIn (1200x627 letterboxed), and
source copies. Spec 2026-08-13 archive-rebirth program, Phase A."""
import pytest

pytest.importorskip("PIL")


def test_export_profiles(tmp_path):
    from PIL import Image

    from llmopt.figures.export import export_profiles
    src = tmp_path / "fig.png"
    Image.new("RGB", (3200, 1400), "#101010").save(src)
    out = export_profiles(src, tmp_path, "fig")
    readme = Image.open(out["readme"])
    linkedin = Image.open(out["linkedin"])
    assert readme.width == 1600
    assert (linkedin.width, linkedin.height) == (1200, 627)
    assert out["source"].exists()
