"""Deterministic decode: two runs hash-identical, greedy stable,
bounds honored — on a tiny random crystal, CPU only."""
import hashlib

import pytest

torch = pytest.importorskip("torch")

from llmopt.decoding.deterministic import DeterministicLM, make_tables
from llmopt.train.mathnative import build_model


@pytest.fixture()
def tiny(tmp_path):
    torch.manual_seed(7)
    d, layers, ffn, heads, vocab = 16, 2, 32, 2, 11
    m = build_model(vocab, d=d, layers=layers, heads=heads, ffn=ffn)
    sd = {k: v.detach() for k, v in m.state_dict().items()}
    path = str(tmp_path / "tables.pt")
    make_tables(sd, d, heads, path)
    return path, d, layers, ffn, heads


def _trace(path, d, layers, ffn, heads):
    lm = DeterministicLM(path, d, layers, ffn, heads, "cpu")
    ids = [1, 4, 2, 9, 3]
    h = hashlib.sha256()
    past = None
    for pos, t in enumerate(ids):
        lg, past = lm.step(torch.tensor(t), past, pos)
        h.update(lg.numpy().tobytes())
    toks = lm.greedy(ids, 8)
    return h.hexdigest(), toks, lm.max_partial


def test_bit_identical_runs(tiny):
    h1, t1, b1 = _trace(*tiny)
    h2, t2, b2 = _trace(*tiny)
    assert h1 == h2
    assert t1 == t2
    assert b1 == b2 and b1 < (1 << 24)


def test_logits_are_integers(tiny):
    path, d, layers, ffn, heads = tiny
    lm = DeterministicLM(path, d, layers, ffn, heads, "cpu")
    lg, _ = lm.step(torch.tensor(1), None, 0)
    assert lg.dtype == torch.int64
