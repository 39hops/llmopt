

def test_hash_module_is_canonical():
    """One digest semantics (grok cross-check adoption 2026-08-11):
    catalog/merge/runfiles all route through lab.hash."""
    from llmopt.lab import catalog, hash as lab_hash, merge, runfiles
    assert catalog.sha256_file is lab_hash.sha256_file
    assert merge._sha256("/dev/null") == lab_hash.sha256_file("/dev/null")
    assert runfiles._git_sha() == lab_hash.git_sha(short=True)
    assert merge._git_sha() == lab_hash.git_sha()
    assert lab_hash.git_sha() != "unknown"  # anchored to THIS repo
