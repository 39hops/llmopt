"""Token accounting, stop handling, context limits and session isolation of
llmopt.lab.qserve, pinned on a deterministic stub model (no artifact
needed; mlx required for the arrays) and, when LLMOPT_QWEN_LIVE=1 and the
arm C artifact is present, on the real tower."""
from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")

from llmopt.lab import qserve  # noqa: E402

V = 64
STOP = V - 1
D = 4


class _Cache:
    def __init__(self):
        self.fed: list = []

    @property
    def offset(self):
        return len(self.fed)

    @property
    def state(self):
        return (mx.array([len(self.fed)]),)


class _Body:
    """hidden[s] encodes the token fed at s and the count of tokens fed so far."""
    def __call__(self, ids, cache):
        toks = [int(t) for t in ids[0].tolist()]
        rows = []
        for t in toks:
            cache[0].fed.append(t)
            rows.append([float(t), float(len(cache[0].fed)), 0.0, 0.0])
        return mx.array([rows])


def _rule(tok: int, count: int) -> int:
    """next token = (7 * tok + count) mod V; the stub's deterministic law."""
    return (7 * tok + count) % V


class _Head:
    def __call__(self, h):
        out = []
        for row in h[0].tolist():
            nxt = _rule(int(row[0]), int(row[1]))
            logits = [0.0] * V
            logits[nxt] = 1.0
            out.append(logits)
        return mx.array([out])


class _LM:
    def __init__(self):
        self.model = _Body()
        self.lm_head = _Head()


class _Model:
    def __init__(self):
        self.language_model = _LM()

    def make_cache(self):
        return [_Cache()]

    def __call__(self, ids, cache=None):
        return self.language_model.lm_head(self.language_model.model(ids, cache))


class _Tok:
    eos_token_id = STOP

    def convert_tokens_to_ids(self, t):
        return STOP - 1 if t == qserve.ENDOFTEXT else -1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kw):
        # user text encodes byte-wise; an assistant turn re-encodes the ids its
        # decode printed (invertible), followed by the stop token, as the real
        # template renders a past turn verbatim and closes it with <|im_end|>
        ids = []
        for m in messages:
            if m["role"] == "assistant":
                ids += [1] + [int(x) for x in m["content"].split()] + [STOP]
            else:
                ids += [1] + [int(c) % 50 + 2 for c in m["content"].encode()] + [STOP]
        return ids + [1]


@pytest.fixture
def engine(monkeypatch):
    from llmopt.lab import qmetal
    monkeypatch.setattr(qmetal, "VERSION", 5)
    return qserve.Engine(model=_Model(), tokenizer=_Tok(), context_cap=64, cache_limit_bytes=qserve.CACHE_LIMIT_BYTES)


def _expected(prompt, max_tokens):
    """the stub's own trajectory: prefill then greedy steps, stopping on STOP."""
    fed = list(prompt)
    cur = _rule(fed[-1], len(fed))
    out = []
    while cur not in (STOP, STOP - 1):
        out.append(cur)
        if len(out) >= max_tokens:
            return out, "max_tokens", fed
        fed.append(cur)
        cur = _rule(cur, len(fed))
    return out, "stop_token", fed


def test_first_token_from_prefill_then_each_fed_once(engine):
    prompt = [3, 5, 8]
    res = engine.generate_ids(prompt, max_tokens=5)
    exp, reason, fed = _expected(prompt, 5)
    assert res.generated_ids == exp and res.stop_reason == reason
    assert res.cache.__class__ is list and res.cache[0].fed == fed            # every token fed exactly once, in order
    assert res.cache_tokens == len(fed) and res.fed_generated == len(exp) - (reason == "max_tokens")
    assert res.last_token_in_cache is (reason == "stop_token")


def test_max_tokens_leaves_last_token_unfed(engine):
    prompt = [3, 5, 8]
    res = engine.generate_ids(prompt, max_tokens=2)
    assert res.stop_reason == "max_tokens" and res.n_generated == 2
    assert res.cache[0].fed == prompt + res.generated_ids[:1]
    assert res.cache_tokens == len(prompt) + 1 and not res.last_token_in_cache


def test_stop_token_is_neither_emitted_nor_fed(engine):
    # find a prompt whose trajectory hits STOP within the budget
    for seed in range(2, 60):
        exp, reason, fed = _expected([seed], 40)
        if reason == "stop_token":
            break
    else:
        pytest.skip("no stop trajectory in the stub")
    res = engine.generate_ids([seed], max_tokens=40)
    assert res.stop_reason == "stop_token" and res.stop_token_id in engine.stop_ids
    assert STOP not in res.generated_ids and res.cache[0].fed == fed and res.cache_tokens == len(fed)


def test_context_cap_refuses_without_truncating(engine):
    with pytest.raises(qserve.ContextLimitError):
        engine.generate_ids(list(range(2, 62)), max_tokens=10)
    with pytest.raises(qserve.InputError):
        engine.generate_ids([], max_tokens=4)
    with pytest.raises(qserve.InputError):
        engine.generate_ids([1, "x"], max_tokens=4)
    with pytest.raises(qserve.InputError):
        engine.generate_ids([1, 2], max_tokens=0)
    with pytest.raises(qserve.InputError):
        engine.render_chat([{"role": "user", "content": "   "}])
    with pytest.raises(qserve.InputError):
        engine.render_chat([{"role": "robot", "content": "hi"}])


def test_session_replay_accounting_and_isolation(engine):
    s = qserve.Session(engine, mode="replay")
    r1 = s.ask("hello", max_tokens=3)
    r2 = s.ask("again", max_tokens=3)
    assert r1.mode == "fresh" and r2.mode == "fresh" and r2.cache is None and s.cache is None   # replay keeps no cache
    assert r2.cache_offset_measured == r2.n_prompt + r2.fed_generated == r2.cache_tokens         # the fresh cache held exactly this turn's feed
    assert s.fed_ids == r2.prompt_ids + r2.generated_ids[: r2.fed_generated] and r2.cache_tokens == len(s.fed_ids)
    t = qserve.Session(engine, mode="replay")
    assert t.messages == [] and t.ask("hello", max_tokens=3).generated_ids == r1.generated_ids   # isolation: a new session repeats turn 1
    s.reset()
    assert s.messages == [] and s.cache is None and s.fed_ids == []


def test_session_incremental_equals_replay_when_prefix_holds(engine):
    """the stub's template renders history verbatim, so the prefix check holds
    and the incremental cache must produce the same tokens as a replay."""
    a = qserve.Session(engine, mode="incremental")
    b = qserve.Session(engine, mode="replay")
    ra1, rb1 = a.ask("hello", 3), b.ask("hello", 3)
    ra2, rb2 = a.ask("more", 3), b.ask("more", 3)
    assert ra1.generated_ids == rb1.generated_ids
    assert ra2.mode == "incremental", (ra2.mode, ra2.note)   # the stub's template is invertible, so the prefix must hold
    assert ra2.generated_ids == rb2.generated_ids
    assert a.cache[0].fed == a.fed_ids and ra2.cache_offset_measured == len(a.fed_ids)   # one cache, every token fed exactly once
    assert b.cache is None                                    # replay holds no cache between turns


def test_immediate_stop_and_budget_of_one(engine):
    for seed in range(2, 60):
        exp, reason, fed = _expected([seed], 1)
        if reason == "stop_token" and not exp:
            r = engine.generate_ids([seed], max_tokens=1)
            assert r.n_generated == 0 and r.fed_generated == 0 and r.cache_tokens == 1 and not r.last_token_in_cache
            break
    r = engine.generate_ids([3, 5, 8], max_tokens=1)
    assert r.n_generated == 1 and r.fed_generated == 0 and r.cache_tokens == 3 and r.cache_offset_measured == 3


def test_refused_turn_leaves_the_session_unchanged(engine):
    s = qserve.Session(engine, mode="incremental")
    s.ask("hello", max_tokens=3)
    n = len(s.messages)
    with pytest.raises(qserve.ContextLimitError):
        s.ask("x" * 200, max_tokens=3)
    assert len(s.messages) == n and s.cache is None


LIVE = os.environ.get("LLMOPT_QWEN_LIVE") == "1" and os.path.exists(os.path.expanduser("~/qwen_whole0t/C/manifest.json"))


@pytest.fixture(scope="module")
def live_engine():
    if not LIVE:
        pytest.skip("set LLMOPT_QWEN_LIVE=1 with the arm C artifact present")
    return qserve.Engine()


def test_live_one_turn_and_stop(live_engine):
    r = live_engine.chat([{"role": "user", "content": "Reply with the single word: yes"}], max_tokens=16, enable_thinking=False)
    assert r.n_generated >= 1 and r.stop_reason in ("stop_token", "max_tokens")
    if r.stop_reason == "stop_token":
        assert r.stop_token_id in live_engine.stop_ids and r.cache_tokens == r.n_prompt + r.n_generated


def test_live_two_turns_refer_back(live_engine):
    s = qserve.Session(live_engine, mode="replay", enable_thinking=False)
    r1 = s.ask("My favourite colour is teal. Remember it and reply only: noted.", max_tokens=16)
    r2 = s.ask("What is my favourite colour? Answer with one word.", max_tokens=16)
    assert "teal" in r2.text.lower(), (r1.text, r2.text)


def test_live_incremental_matches_replay(live_engine):
    a = qserve.Session(live_engine, mode="incremental", enable_thinking=False)
    b = qserve.Session(live_engine, mode="replay", enable_thinking=False)
    for i, q in enumerate(("Name one prime number below ten. One word.", "And one above ten. One word.")):
        ra, rb = a.ask(q, 12), b.ask(q, 12)
        assert ra.generated_ids == rb.generated_ids, (ra.mode, ra.note, ra.text, rb.text)
        if i == 1:
            assert ra.mode == "incremental", (ra.mode, ra.note)
            assert ra.cache_offset_measured == ra.cache_tokens == len(a.fed_ids)
