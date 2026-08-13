"""Guards for the animation pipeline (banked repair, 2026-08-13).

Two failure modes these catch:
  * a committed data/anim npz citing checkpoint shas that no longer
    match the files on disk (a stale npz ships a figure built from the
    wrong weights, silently);
  * a scene in the shipping registry whose asset set is missing or
    stale-named in docs/assets/anim/.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "anim"
ASSETS = ROOT / "docs" / "assets" / "anim"


def _meta(name):
    f = DATA / f"{name}.npz"
    if not f.exists():
        pytest.skip(f"{f} not present")
    return json.loads(str(np.load(f, allow_pickle=False)["meta"]))


def _sha8(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


@pytest.mark.parametrize("scene", ["crystal", "morph"])
def test_npz_cited_checkpoint_shas_match_disk(scene):
    """provenance carries '<ckpt>.pt <sha8>' pairs; re-hash each."""
    prov = _meta(scene)["provenance"]
    pairs = re.findall(r"([\w.]+\.pt) ([0-9a-f]{8})", prov)
    assert pairs, f"no checkpoint citations parsed from: {prov}"
    for name, sha in pairs:
        ckpt = ROOT / "checkpoints" / name
        if not ckpt.exists():
            pytest.skip(f"{name} not on this machine")
        assert _sha8(ckpt) == sha, (
            f"{scene}.npz cites {name} {sha} but the file on disk hashes "
            f"{_sha8(ckpt)} — the npz is STALE, regenerate it")


def test_atlas_npz_arms_match_ledger():
    meta = _meta("atlas")
    assert meta["arms"] == {"full": 189, "control": 217, "carriers": 244}
    assert "of 360" in meta["unit"]


def test_registry_scenes_ship_complete_snake_case_triplets():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from render_anim import SCENES
    for stem, (_, _, has_light) in SCENES.items():
        assert (ROOT / "scripts" / "anim" / f"{stem}.py").exists()
        want = [f"{stem}.mp4", f"{stem}.gif", f"{stem}_poster.png"]
        if has_light:
            want += [f"{stem}_light.mp4", f"{stem}_light.gif",
                     f"{stem}_light_poster.png"]
        missing = [w for w in want if not (ASSETS / w).exists()]
        assert not missing, f"{stem}: missing assets {missing}"
        assert "-" not in stem
