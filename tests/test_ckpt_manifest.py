import importlib.util
import json
import os

spec = importlib.util.spec_from_file_location(
    "ckpt_manifest",
    os.path.join(os.path.dirname(__file__), "..", "scripts",
                 "ckpt_manifest.py"))
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "w.pt"
    p.write_bytes(b"\x00" * 1024 + b"weights")
    import hashlib
    assert cm.sha256(str(p)) == hashlib.sha256(
        p.read_bytes()).hexdigest()


def test_scan_filters_extensions(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.safetensors").write_bytes(b"x")
    found = sorted(os.path.relpath(p, tmp_path)
                   for p in cm.scan(str(tmp_path)))
    assert found == ["a.pt", os.path.join("sub", "c.safetensors")]


def test_manifest_curation_preserved(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    (root / "confirmed" / "cat").mkdir(parents=True)
    (root / "confirmed" / "cat" / "m.pt").write_bytes(b"model")
    monkeypatch.setattr(cm, "ROOT", str(root))
    monkeypatch.setattr(cm, "MANIFEST", str(root / "MANIFEST.jsonl"))
    monkeypatch.setattr("sys.argv", ["ckpt_manifest.py"])
    cm.main()
    rows = [json.loads(ln) for ln in open(root / "MANIFEST.jsonl")]
    assert len(rows) == 1 and rows[0]["bytes"] == 5
    # curate, regenerate, curation must survive
    rows[0]["category"] = "math-test"
    rows[0]["note"] = "kept"
    with open(root / "MANIFEST.jsonl", "w") as f:
        f.write(json.dumps(rows[0]) + "\n")
    cm.main()
    rows = [json.loads(ln) for ln in open(root / "MANIFEST.jsonl")]
    assert rows[0]["category"] == "math-test"
    assert rows[0]["note"] == "kept"
