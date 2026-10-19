"""llmopt.lab.qprefill: chunked prefill through the model's own caches,
final-position head by default, full-position head for research."""
import pytest

mx = pytest.importorskip("mlx.core")

from llmopt.lab import qprefill  # noqa: E402


class _Cache:
    """A cache object in the mlx-lm shape: state is a tuple of arrays and
    the offset counts tokens seen."""
    def __init__(self):
        self.offset = 0
        self.keys = mx.zeros((0, 4))

    @property
    def state(self):
        return (self.keys,)


class _Body:
    """Hidden state = running sum of token ids seen so far (depends on
    every earlier token through the cache), one row per position."""
    def __init__(self):
        self.calls = []

    def __call__(self, ids, cache):
        c = cache[0]
        self.calls.append(int(ids.shape[1]))
        base = float(c.keys[-1, 0]) if c.keys.shape[0] else 0.0      # running sum carried in the cache
        run = mx.cumsum(ids.astype(mx.float32), axis=1) + base
        h = mx.stack([run, run * 2, run * 3, run * 4], axis=-1)
        c.keys = mx.concatenate([c.keys, h[0]], axis=0)
        c.offset += int(ids.shape[1])
        return h


class _Head:
    def __init__(self):
        self.rows = []

    def __call__(self, h):
        self.rows.append(int(h.shape[1]))
        return h @ mx.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]])


class _LM:
    def __init__(self):
        self.model = _Body()
        self.lm_head = _Head()


class _Model:
    def __init__(self):
        self.language_model = _LM()


def test_final_head_matches_full_head_and_state_advances():
    ids = list(range(1, 11))
    m_full, m_final = _Model(), _Model()
    c_full, c_final = [_Cache()], [_Cache()]
    lg_full = qprefill.prefill_chunked(m_full, ids, c_full, chunk=4, head="full")
    lg_final = qprefill.prefill_chunked(m_final, ids, c_final, chunk=4, head="final")
    assert lg_full.shape == (1, 2, 2) and lg_final.shape == (1, 1, 2)      # last chunk has 2 tokens
    assert mx.array_equal(lg_full[:, -1:, :], lg_final)
    assert m_full.language_model.model.calls == [4, 4, 2] == m_final.language_model.model.calls
    assert m_full.language_model.lm_head.rows == [4, 4, 2]
    assert m_final.language_model.lm_head.rows == [1, 1, 1]
    assert c_full[0].offset == c_final[0].offset == 10
    assert mx.array_equal(c_full[0].keys, c_final[0].keys)
    # one-shot reference: the chunked cache equals the whole-prompt cache
    m_one, c_one = _Model(), [_Cache()]
    lg_one = qprefill.prefill_chunked(m_one, ids, c_one, chunk=100, head="full")
    assert mx.array_equal(lg_one[:, -1:, :], lg_final)
    assert mx.array_equal(c_one[0].keys, c_final[0].keys)


def test_on_chunk_receives_every_chunk_and_none_head_returns_none():
    ids = list(range(1, 8))
    seen = []
    m, c = _Model(), [_Cache()]
    qprefill.prefill_chunked(m, ids, c, chunk=3, head="full", on_chunk=lambda s, lg: seen.append((s, int(lg.shape[1]))))
    assert seen == [(0, 3), (3, 3), (6, 1)]
    m2, c2 = _Model(), [_Cache()]
    assert qprefill.prefill_chunked(m2, ids, c2, chunk=3, head="none") is None
    assert m2.language_model.lm_head.rows == [] and c2[0].offset == 7


def test_refuses_bad_arguments():
    m, c = _Model(), [_Cache()]
    with pytest.raises(ValueError):
        qprefill.prefill_chunked(m, [1, 2], c, chunk=2, head="last")
    with pytest.raises(ValueError):
        qprefill.prefill_chunked(m, [1, 2], c, chunk=0)
    with pytest.raises(ValueError):
        qprefill.split_body_head(object())


class _Resident:
    """Stand-in for a qmetal resident: records the scope state it saw on
    each call."""
    def __init__(self):
        self.seen = []

    def __call__(self):
        from llmopt.lab import qmetal
        self.seen.append(qmetal.gemm_scope_state())


def test_matmul_argument_scopes_the_gemm_path_to_the_call():
    from llmopt.lab import qmetal
    ids = list(range(1, 8))
    for matmul, expect in (("gemm", True), ("fused", False), (None, None)):
        m, c = _Model(), [_Cache()]
        res = _Resident()
        inner = m.language_model.model

        class _SpyBody:
            calls = inner.calls

            def __call__(self, ids_, cache):
                res()
                return inner(ids_, cache)
        m.language_model.model = _SpyBody()
        assert qmetal.gemm_scope_state() is None
        qprefill.prefill_chunked(m, ids, c, chunk=3, head="final", matmul=matmul)
        assert res.seen == [expect, expect, expect]
        assert qmetal.gemm_scope_state() is None
    with pytest.raises(ValueError):
        qprefill.prefill_chunked(_Model(), ids, [_Cache()], chunk=3, matmul="auto")


def test_refuses_batched_or_non_integer_prompts():
    for bad in ([[1, 2, 3]], [1, 2.5, 3], "123", [], [1, [2]]):
        with pytest.raises(ValueError):
            qprefill.prefill_chunked(_Model(), bad, [_Cache()], chunk=2)


def test_serving_config_and_memory_report():
    cfg = qprefill.ServingConfig()
    assert cfg.chunk == 256 and cfg.matmul == "gemm" and cfg.cache_limit_bytes == 2 * 1024 ** 3
    prev = qprefill.configure_serving_memory(cfg)
    try:
        assert isinstance(prev, int)
        assert mx.set_cache_limit(prev) == cfg.cache_limit_bytes      # the limit was applied
    finally:
        mx.set_cache_limit(prev)
    rep = qprefill.memory_report()
    assert set(rep) >= {"active_bytes", "peak_bytes", "cache_bytes", "footprint_bytes", "rss_peak_bytes"}
    assert rep["footprint_bytes"] == rep["active_bytes"] + rep["cache_bytes"]
    assert all(isinstance(v, int) and v >= 0 for v in rep.values())


def test_on_chunk_runs_outside_the_scope_and_head_runs_inside():
    from llmopt.lab import qmetal
    seen_head, seen_cb = [], []
    m, c = _Model(), [_Cache()]
    inner_head = m.language_model.lm_head

    class _SpyHead:
        rows = inner_head.rows

        def __call__(self, h):
            seen_head.append(qmetal.gemm_scope_state())
            return inner_head(h)
    m.language_model.lm_head = _SpyHead()
    qprefill.prefill_chunked(m, list(range(1, 8)), c, chunk=3, head="full", matmul="gemm",
                             on_chunk=lambda s, lg: seen_cb.append(qmetal.gemm_scope_state()))
    assert seen_head == [True, True, True]
    assert seen_cb == [None, None, None]        # a decode or batched call issued by the callback sees no scope


def test_exception_inside_prefill_leaves_no_scope():
    from llmopt.lab import qmetal

    class _Boom:
        def __call__(self, ids, cache):
            raise RuntimeError("boom")
    m = _Model()
    m.language_model.model = _Boom()
    with pytest.raises(RuntimeError):
        qprefill.prefill_chunked(m, [1, 2, 3], [_Cache()], chunk=2, matmul="gemm")
    assert qmetal.gemm_scope_state() is None


def test_scope_is_thread_local():
    import threading
    from llmopt.lab import qmetal
    inside = threading.Event()
    release = threading.Event()
    other = {}

    def worker():
        inside.wait()
        other["state"] = qmetal.gemm_scope_state()
        release.set()
    t = threading.Thread(target=worker)
    t.start()
    with qmetal.gemm_scope(True):
        inside.set()
        release.wait()
    t.join()
    assert other["state"] is None


def test_serving_memory_refuses_a_process_wide_gemm_default(monkeypatch):
    from llmopt.lab import qmetal
    monkeypatch.setattr(qmetal._Resident, "prefill_gemm", True)
    with pytest.raises(RuntimeError):
        qprefill.configure_serving_memory(qprefill.ServingConfig())
    monkeypatch.setattr(qmetal._Resident, "prefill_gemm", False)
    monkeypatch.setattr(qmetal, "PREFILL_GEMM", True)
    with pytest.raises(RuntimeError):
        qprefill.configure_serving_memory(qprefill.ServingConfig())
