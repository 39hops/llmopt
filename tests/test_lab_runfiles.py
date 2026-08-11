

def test_hash_module_is_canonical():
    """One digest semantics (grok cross-check adoption 2026-08-11):
    catalog/merge/runfiles all route through lab.hash."""
    from llmopt.lab import catalog, hash as lab_hash, merge, runfiles
    assert catalog.sha256_file is lab_hash.sha256_file
    assert merge._sha256("/dev/null") == lab_hash.sha256_file("/dev/null")
    assert runfiles._git_sha() == lab_hash.git_sha(short=True)
    assert merge._git_sha() == lab_hash.git_sha()
    assert lab_hash.git_sha() != "unknown"  # anchored to THIS repo


def test_jsonl_semantics(tmp_path):
    from llmopt.lab.jsonl import append_jsonl, read_jsonl, write_jsonl
    p = tmp_path / "t.jsonl"
    write_jsonl(p, [{"a": 1}, {"b": 2}])
    append_jsonl(p, {"c": 3})
    assert read_jsonl(p) == [{"a": 1}, {"b": 2}, {"c": 3}]
    # blank lines skipped; malformed raises with line number
    p.write_text('{"a": 1}\n\n{"b": 2}\n')
    assert len(read_jsonl(p)) == 2
    p.write_text('{"a": 1}\nnot json\n')
    import pytest as _pt
    with _pt.raises(ValueError, match=":2:"):
        read_jsonl(p)
    # atomic write leaves no tmp behind
    assert not list(tmp_path.glob("*.tmp"))
