"""Tests for llmopt/lab/catalog.py + scripts/gen_catalog.py.

All on tmp_path fakes — never touches the real checkpoints/ tree.
torch-dependent cases skip cleanly when torch is absent (house rule:
optional-dep tests skip, never fail).
"""
import hashlib
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from llmopt.lab.catalog import parent_ids, scan_checkpoint, sha256_file  # noqa: E402

try:
    import torch
except Exception:
    torch = None


def _load_gen_catalog():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen_catalog", os.path.join(REPO, "scripts", "gen_catalog.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- lineage

def test_grown_parent_is_birth():
    sib = ["crown_c_birth_s2.pt", "crown_c_grown_s2.pt", "crown_c_grown_s3.pt"]
    assert parent_ids("crown_c_grown_s2.pt", sib) == ["crown_c_birth_s2.pt"]
    # seed s3 has no birth twin present -> no parent
    assert parent_ids("crown_c_grown_s3.pt", sib) == []


def test_latent_falls_back_birth():
    sib = ["x_birth_s1.pt", "x_latent_s1.pt"]
    assert parent_ids("x_latent_s1.pt", sib) == ["x_birth_s1.pt"]
    sib2 = ["x_birth_s1.pt", "x_grown_s1.pt", "x_latent_s1.pt"]
    assert parent_ids("x_latent_s1.pt", sib2) == ["x_grown_s1.pt"]


def test_ep_chain():
    sib = ["gallery19m_s1.pt", "gallery19m_s1_ep0.pt", "gallery19m_s1_ep1.pt"]
    assert parent_ids("gallery19m_s1_ep1.pt", sib) == ["gallery19m_s1_ep0.pt"]
    # _ep0 -> bare-stem edge dropped 2026-08-11 (review C1: the bare
    # stem can be the rolling latest file — edge pointed at the future)
    assert parent_ids("gallery19m_s1_ep0.pt", sib) == []


def test_3ep_parent_is_plain():
    sib = ["mathnative_110m_gen4_std.pt", "mathnative_110m_gen4_std_3ep.pt",
           "mathnative_110m_gen3_std.pt"]
    got = parent_ids("mathnative_110m_gen4_std_3ep.pt", sib)
    assert "mathnative_110m_gen4_std.pt" in got


def test_gen_chain():
    sib = ["mathnative_19m_gen8.pt", "mathnative_19m_gen9A.pt"]
    assert parent_ids("mathnative_19m_gen9A.pt", sib) == ["mathnative_19m_gen8.pt"]
    assert parent_ids("mathnative_19m_gen8.pt", sib) == []


def test_no_self_parent():
    assert parent_ids("a_birth_s1.pt", ["a_birth_s1.pt"]) == []


# ----------------------------------------------------------------- sha

def test_sha_streaming(tmp_path):
    p = tmp_path / "big.bin"
    data = os.urandom(1024) * 300
    p.write_bytes(data)
    assert sha256_file(str(p), chunk=1000) == hashlib.sha256(data).hexdigest()


# ------------------------------------------------------ scan + script

def _fake_house_ckpt(path, d=16, layers=2, vocab=11, ffn=32):
    sd = {"emb.weight": torch.zeros(vocab, d)}
    for i in range(layers):
        sd[f"blocks.{i}.qkv.weight"] = torch.zeros(3 * d, d)
        sd[f"blocks.{i}.gate.weight"] = torch.zeros(ffn, d)
    sd["head.weight"] = torch.zeros(vocab, d)
    torch.save(sd, path)


def _mk_repo(tmp_path):
    root = tmp_path / "repo"
    ck = root / "checkpoints"
    (ck / "confirmed" / "calib").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "docs" / "RESULTS.md").write_text(
        "VERDICT: `crown_c_birth_s2.pt` banked.\n")
    return root, ck


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_scan_checkpoint_row(tmp_path):
    root, ck = _mk_repo(tmp_path)
    p = ck / "crown_c_birth_s2.pt"
    _fake_house_ckpt(str(p))
    (ck / "crown_c_grown_s2.pt").write_bytes(b"notatorch")
    (ck / "crown_c_birth_s2.pt.ep").write_text("ep3\n")
    row = scan_checkpoint(str(p), str(root), {"crown_c_birth_s2.pt"})
    assert row["path"] == "checkpoints/crown_c_birth_s2.pt"
    assert row["cited"] is True
    assert row["ep_marker"] == "ep3"
    assert row["arch"] == {"vocab": 11, "d_model": 16, "layers": 2,
                           "ffn": 32, "heads": None}
    # grown sibling row: non-torch payload -> arch None, parent = birth
    row2 = scan_checkpoint(str(ck / "crown_c_grown_s2.pt"), str(root), set())
    assert row2["arch"] is None
    assert row2["parent_ids"] == ["crown_c_birth_s2.pt"]
    assert row2["cited"] is False


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_gen_catalog_update_skips_unchanged(tmp_path):
    gc = _load_gen_catalog()
    root, ck = _mk_repo(tmp_path)
    a, b = ck / "a_birth_s1.pt", ck / "a_grown_s1.pt"
    _fake_house_ckpt(str(a))
    _fake_house_ckpt(str(b))
    out = root / "data" / "catalog" / "models.jsonl"
    gc.main(["--root", str(root), "--out", str(out)])
    rows = {json.loads(x)["path"]: json.loads(x)
            for x in out.read_text().splitlines()}
    assert len(rows) == 2
    sha_a = rows["checkpoints/a_birth_s1.pt"]["sha256"]
    assert sha_a == sha256_file(str(a))
    assert rows["checkpoints/a_grown_s1.pt"]["parent_ids"] == ["a_birth_s1.pt"]

    # tamper the stored sha of the UNCHANGED file; --update must keep it
    rows["checkpoints/a_birth_s1.pt"]["sha256"] = "SENTINEL"
    # touch b so it re-hashes
    b.write_bytes(b"changed-not-a-torch-file")
    with open(out, "w") as f:
        for r in rows.values():
            f.write(json.dumps(r) + "\n")
    gc.main(["--root", str(root), "--out", str(out), "--update"])
    rows2 = {json.loads(x)["path"]: json.loads(x)
             for x in out.read_text().splitlines()}
    assert rows2["checkpoints/a_birth_s1.pt"]["sha256"] == "SENTINEL"
    assert rows2["checkpoints/a_grown_s1.pt"]["sha256"] == sha256_file(str(b))


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_gen_catalog_no_sha_and_limit(tmp_path):
    gc = _load_gen_catalog()
    root, ck = _mk_repo(tmp_path)
    for n in ("m1.pt", "m2.pt"):
        _fake_house_ckpt(str(ck / n))
    out = root / "o.jsonl"
    gc.main(["--root", str(root), "--out", str(out), "--no-sha",
             "--limit", "1"])
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["sha256"] is None


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_manifest_cross_check_raises(tmp_path):
    gc = _load_gen_catalog()
    root, ck = _mk_repo(tmp_path)
    p = ck / "confirmed" / "calib" / "c1.pt"
    _fake_house_ckpt(str(p))
    (ck / "MANIFEST.jsonl").write_text(json.dumps(
        {"path": "confirmed/calib/c1.pt", "sha256": "0" * 64,
         "bytes": 1}) + "\n")
    out = root / "o.jsonl"
    with pytest.raises(RuntimeError, match="sha mismatch"):
        gc.main(["--root", str(root), "--out", str(out)])
    # matching sha passes
    (ck / "MANIFEST.jsonl").write_text(json.dumps(
        {"path": "confirmed/calib/c1.pt",
         "sha256": sha256_file(str(p))}) + "\n")
    gc.main(["--root", str(root), "--out", str(out)])
    assert out.exists()
