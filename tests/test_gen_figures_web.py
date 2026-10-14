"""The figure pipeline: one serializer, order-independent generators,
a --check that verifies content rather than timestamps, and explicit
limits on what the raster check can claim without Chrome.

Fixtures copy docs/figures.json and docs/assets/web into a temp dir and
drive scripts/gen_figures_web.py at the copies with --spec/--out, so
nothing here writes into the repo.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gen_figures_web.py"
SPEC = ROOT / "docs" / "figures.json"
ASSETS = ROOT / "docs" / "assets" / "web"
FINDINGS = ROOT / "docs" / "FINDINGS.md"
sys.path.insert(0, str(ROOT / "scripts"))
import gen_figures_web as gfw  # noqa: E402

from llmopt.figures import spec as figspec  # noqa: E402
from llmopt.figures.figsvg import render_spec  # noqa: E402

HAS_CHROME = gfw.chrome_available()


def run_check(spec: Path, out: Path, *extra: str, findings: Path = FINDINGS):
    env = dict(os.environ)
    env.pop(gfw.NO_RASTER_ENV, None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--spec", str(spec),
         "--out", str(out), "--findings", str(findings), *extra],
        cwd=ROOT, capture_output=True, text=True, env=env)


@pytest.fixture
def copies(tmp_path):
    spec = tmp_path / "figures.json"
    shutil.copy(SPEC, spec)
    out = tmp_path / "web"
    shutil.copytree(ASSETS, out)
    return spec, out


# ---- serializer and order independence ----------------------------------

def test_figures_json_is_a_fixed_point_of_the_shared_recount():
    text = SPEC.read_text()
    assert figspec.recount(text, FINDINGS.read_text()) == text


def test_both_generators_produce_identical_bytes_in_either_order(tmp_path):
    """gen_readme's figure rewrite and gen_figures_web's recount are
    the same function; applying them in either order to a stale copy
    yields the same bytes, and a second application changes nothing."""
    import importlib.util
    s = importlib.util.spec_from_file_location("gen_readme", ROOT / "scripts" / "gen_readme.py")
    gr = importlib.util.module_from_spec(s)
    s.loader.exec_module(gr)
    stale = json.loads(SPEC.read_text())
    stale["honesty_ledger"]["parts"][0]["value"] = 1
    stale["honesty_ledger"]["fence"] = re.sub(
        r"\d+ curated claims", "1 curated claims", stale["honesty_ledger"]["fence"])
    stale_text = json.dumps(stale, indent=4, ensure_ascii=True)  # a foreign serialization
    via_readme = gr.render_figure(stale_text)
    via_web = figspec.recount(stale_text, FINDINGS.read_text())
    assert via_readme == via_web == SPEC.read_text()
    assert gr.render_figure(via_web) == figspec.recount(via_readme, FINDINGS.read_text()) == via_web


def test_write_mode_then_check_is_clean_on_copies(copies, tmp_path):
    """A full write into the copies (SVG only, raster skipped) leaves
    --check green: the generator is its own fixed point."""
    spec, out = copies
    r = subprocess.run([sys.executable, str(SCRIPT), "--no-raster", "--spec", str(spec),
                        "--out", str(out), "--findings", str(FINDINGS)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert spec.read_text() == SPEC.read_text()
    for p in ASSETS.glob("*.svg"):
        assert (out / p.name).read_text() == p.read_text(), p.name
    r = run_check(spec, out, "--no-raster")
    assert r.returncode == 0, r.stdout


# ---- --check on the real tree -------------------------------------------

def test_check_is_clean_on_the_repo_without_raster():
    r = run_check(SPEC, ASSETS, "--no-raster")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "raster: NOT verified" in r.stdout
    assert "[figures] current" in r.stdout


def test_check_never_claims_raster_without_chrome(copies, monkeypatch):
    """Without Chrome the check must say the raster is unverified, in
    those words, even though every PNG is present and sized right."""
    spec, out = copies
    monkeypatch.setattr(gfw, "CHROME", "/nonexistent/chrome")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gfw.check(spec, out, FINDINGS, raster=True)
    assert rc == 0
    assert "raster: NOT verified on this machine (no Chrome" in buf.getvalue()
    assert "verified by re-render" not in buf.getvalue()


@pytest.mark.skipif(not HAS_CHROME, reason="raster verification needs Chrome")
def test_check_verifies_raster_by_pixel_comparison_when_chrome_is_present():
    r = run_check(SPEC, ASSETS)
    assert r.returncode == 0, r.stdout + r.stderr
    n = 2 * len(gfw.figures(json.loads(SPEC.read_text())))
    assert f"raster: verified by re-render, {n}/{n} pixel-identical" in r.stdout


# ---- mutations the check must catch ------------------------------------

def test_one_digit_numeric_mutation_is_caught(copies):
    spec, out = copies
    d = json.loads(spec.read_text())
    d["loss_floor"]["points"][0]["value"] = 0.4365  # was 0.4364
    spec.write_text(figspec.dump(d))
    r = run_check(spec, out, "--no-raster")
    assert r.returncode == 1
    assert "loss_floor.svg: differs from a fresh render" in r.stdout
    assert "loss_floor-dark.svg: differs" in r.stdout
    assert "merge_space.svg" not in r.stdout  # only the mutated figure


def test_missing_png_is_caught(copies):
    spec, out = copies
    (out / "ssm_star.png").unlink()
    r = run_check(spec, out, "--no-raster")
    assert r.returncode == 1 and "ssm_star.png: missing" in r.stdout


def test_missing_svg_is_caught(copies):
    spec, out = copies
    (out / "ssm_star-dark.svg").unlink()
    r = run_check(spec, out, "--no-raster")
    assert r.returncode == 1 and "ssm_star-dark.svg: missing" in r.stdout


def test_newer_timestamp_does_not_rescue_a_stale_png(copies):
    """A PNG newer than its SVG but carrying another figure's raster
    (wrong dimensions) is stale; the header check catches it with no
    Chrome at all."""
    spec, out = copies
    shutil.copy(out / "ssm_star.png", out / "loss_floor.png")
    future = time.time() + 3600
    os.utime(out / "loss_floor.png", (future, future))
    r = run_check(spec, out, "--no-raster")
    assert r.returncode == 1
    assert "loss_floor.png: header size" in r.stdout


@pytest.mark.skipif(not HAS_CHROME, reason="pixel comparison needs Chrome")
def test_same_size_stale_png_is_caught_only_by_pixels(copies):
    """Dark and light renders of one figure have identical dimensions,
    so swapping them defeats every header or timestamp check; only a
    re-render comparison sees it."""
    spec, out = copies
    shutil.copy(out / "ssm_star-dark.png", out / "ssm_star.png")
    future = time.time() + 3600
    os.utime(out / "ssm_star.png", (future, future))
    assert run_check(spec, out, "--no-raster").returncode == 0  # invisible without raster
    r = run_check(spec, out)
    assert r.returncode == 1 and "ssm_star.png: " in r.stdout
    assert "pixel(s) differ from a fresh Chrome capture" in r.stdout


def test_stale_ledger_count_in_spec_is_caught(copies):
    spec, out = copies
    d = json.loads(spec.read_text())
    d["honesty_ledger"]["parts"][0]["value"] += 1
    spec.write_text(figspec.dump(d))
    r = run_check(spec, out, "--no-raster")
    assert r.returncode == 1 and "counts or serialization are stale" in r.stdout


def test_foreign_serialization_is_caught(copies):
    spec, out = copies
    spec.write_text(json.dumps(json.loads(spec.read_text()), indent=2, ensure_ascii=True) + "\n")
    r = run_check(spec, out, "--no-raster")
    assert r.returncode == 1 and "serialization are stale" in r.stdout


def test_malformed_spec_is_an_error_not_a_pass(copies):
    spec, out = copies
    spec.write_text("{not json")
    r = run_check(spec, out, "--no-raster")
    assert r.returncode == 2 and "cannot be parsed" in r.stdout


def test_unrenderable_figure_is_reported_not_skipped(copies):
    spec, out = copies
    d = json.loads(spec.read_text())
    d["ssm_star"]["kind"] = "no_such_kind"
    spec.write_text(figspec.dump(d))
    r = run_check(spec, out, "--no-raster")
    assert r.returncode == 1 and "ssm_star.svg: render failed" in r.stdout


def test_fence_that_would_be_cropped_is_refused():
    d = json.loads(SPEC.read_text())["ssm_star"]
    d["fence"] = " ".join(["provenance"] * 60)
    with pytest.raises(ValueError, match="at most 2 fit"):
        render_spec(d)


def test_check_writes_nothing(copies):
    spec, out = copies
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    before["figures.json"] = spec.read_bytes()
    run_check(spec, out, "--no-raster")
    after = {p.name: p.read_bytes() for p in out.iterdir()}
    after["figures.json"] = spec.read_bytes()
    assert after == before
