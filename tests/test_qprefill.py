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
