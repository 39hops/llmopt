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


THINK_OPEN, THINK_CLOSE, DROPPED = 60, 59, 58     # <think>, </think>, and a special token decode drops


class _Tok:
    eos_token_id = STOP

    def convert_tokens_to_ids(self, t):
        return STOP - 1 if t == qserve.ENDOFTEXT else -1

    def decode(self, ids, skip_special_tokens=True):
        names = {THINK_OPEN: "<think>", THINK_CLOSE: "</think>"}
        return " ".join(names.get(i, str(i)) for i in ids if not (skip_special_tokens and i == DROPPED))

    @staticmethod
    def _enc(text):
        names = {"<think>": THINK_OPEN, "</think>": THINK_CLOSE}
        return [names.get(x) if x in names else int(x) for x in text.split()] if text else []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, enable_thinking=None, **kw):
        # mirrors the STRUCTURE of the pinned template (not its whitespace,
        # trims, BPE merges or injected reasoning system message, which only
        # the live tests see): user text encodes byte-wise; a past assistant
        # turn renders <think> reasoning </think> content <|im_end|>
        # (invertible through decode); the generation prompt opens a think
        # block (thinking on or unset) or an empty one (thinking off)
        ids = []
        for m in messages:
            if m["role"] == "assistant":
                ids += [1, THINK_OPEN] + self._enc(m.get("reasoning_content", "")) + [THINK_CLOSE] + self._enc(m["content"]) + [STOP]
            else:
                ids += [1] + [int(c) % 50 + 2 for c in m["content"].encode()] + [STOP]
        return ids + ([1, THINK_OPEN, THINK_CLOSE] if enable_thinking is False else [1, THINK_OPEN])


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
    a = qserve.Session(engine, mode="incremental", enable_thinking=False)
    b = qserve.Session(engine, mode="replay", enable_thinking=False)
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


def _pair(engine, thinking=False):
    return qserve.Session(engine, mode="incremental", enable_thinking=thinking), qserve.Session(engine, mode="replay", enable_thinking=thinking)


def _same(a, b, q, n):
    ra, rb = a.ask(q, n), b.ask(q, n)
    assert ra.generated_ids == rb.generated_ids and ra.stop_reason == rb.stop_reason, (ra.mode, ra.note, ra.text, rb.text)
    assert ra.cache_offset_measured == ra.cache_tokens == len(a.fed_ids)
    return ra, rb


def _find(kind, budget=40):
    """a one-token user text whose stub trajectory ends the given way."""
    for c in range(33, 127):
        exp, reason, _ = _expected([1, c % 50 + 2, STOP, 1, THINK_OPEN, THINK_CLOSE], budget)
        if kind == "stop" and reason == "stop_token" and exp:
            return chr(c)
        if kind == "empty" and reason == "stop_token" and not exp:
            return chr(c)
        if kind == "budget" and reason == "max_tokens":
            return chr(c)
    pytest.skip(f"no {kind} trajectory in the stub")


def test_incremental_after_stop_token_feeds_the_stop_exactly_once(engine):
    a, b = _pair(engine)
    q = _find("stop")
    ra, _ = _same(a, b, q, 40)
    assert ra.stop_reason == "stop_token" and ra.last_token_in_cache
    n_fed, prefix = len(a.fed_ids), list(a.fed_ids)
    ra2, _ = _same(a, b, "next", 3)
    assert ra2.mode == "incremental" and ra2.prompt_ids[0] == STOP           # the un-fed stop token is the first new id
    assert a.cache[0].fed == a.fed_ids and a.fed_ids[:n_fed] == prefix and a.cache[0].fed.count(STOP) == a.fed_ids.count(STOP)


def test_incremental_after_budget_stop_feeds_the_last_token_once(engine):
    a, b = _pair(engine)
    ra, _ = _same(a, b, _find("budget", 3), 3)
    assert ra.stop_reason == "max_tokens" and not ra.last_token_in_cache
    last = ra.generated_ids[-1]
    ra2, _ = _same(a, b, "next", 3)
    assert ra2.mode == "incremental" and ra2.prompt_ids[0] == last             # the un-fed last token is fed first, once
    assert a.cache[0].fed == a.fed_ids


def test_incremental_after_empty_completion(engine):
    a, b = _pair(engine)
    ra, _ = _same(a, b, _find("empty"), 40)
    assert ra.n_generated == 0 and ra.stop_reason == "stop_token"
    ra2, _ = _same(a, b, "next", 3)
    assert ra2.mode == "incremental" and a.cache[0].fed == a.fed_ids


def test_thinking_on_and_off_both_modes(engine):
    # thinking off: the generation prompt's empty think block equals the past
    # turn's rendering, so the cache is reused; thinking on (True or unset)
    # with a reply that never closed its think block: the template re-renders
    # the turn with a </think> the model never emitted, so the prefix breaks
    # and the turn is replayed, never continued from the cache
    for thinking in (True, False, None):
        a, b = _pair(engine, thinking)
        ra, _ = _same(a, b, "hello", 6)
        ra2, _ = _same(a, b, "more", 6)
        assert a.cache[0].fed == a.fed_ids
        assert THINK_CLOSE not in ra.generated_ids
        if thinking is False:
            assert ra2.mode == "incremental", (thinking, ra2.mode)
        else:
            assert ra2.mode == "replay-fallback", (thinking, ra2.mode)


def test_thinking_on_with_a_closed_block_reuses_the_cache(engine, monkeypatch):
    # the stub emits </think> as its second generated token; the stored turn
    # splits into reasoning and content, the template re-renders them around
    # the close, and the rendering extends the fed ids exactly
    import sys
    orig = _rule
    monkeypatch.setattr(sys.modules[__name__], "_rule", lambda t, c: THINK_CLOSE if c == 10 else orig(t, c))
    a, b = _pair(engine, True)
    ra, _ = _same(a, b, "hello", 6)
    assert ra.generated_ids[1] == THINK_CLOSE and a.messages[1]["reasoning_content"] and a.messages[1]["content"]
    ra2, _ = _same(a, b, "more", 6)
    assert ra2.mode == "incremental" and a.cache[0].fed == a.fed_ids


def test_changed_rendering_forces_replay_fallback_never_a_guess(engine):
    a, b = _pair(engine, False)
    _same(a, b, "hello", 3)
    a.system = b.system = None
    a.messages.insert(0, {"role": "system", "content": "be brief"})   # the transcript's rendering changed: no longer extends the fed ids
    b.messages.insert(0, {"role": "system", "content": "be brief"})
    ra2, _ = _same(a, b, "more", 3)
    assert ra2.mode == "replay-fallback" and ra2.note and a.cache[0].fed == a.fed_ids == ra2.prompt_ids + ra2.generated_ids[: ra2.fed_generated]
    ra3, _ = _same(a, b, "again", 3)
    assert ra3.mode == "incremental"                        # the fresh cache is reused from here


def test_dropped_special_token_in_generation_falls_back(engine, monkeypatch):
    # a generated special token that decode drops cannot be re-rendered, so
    # the next turn's rendering does not extend the fed ids: replay, not reuse
    a, b = _pair(engine)
    orig = _rule
    import sys
    monkeypatch.setattr(sys.modules[__name__], "_rule", lambda t, c: DROPPED if c == 10 else orig(t, c))   # the first decode step emits it
    ra, _ = _same(a, b, "hello", 4)
    assert DROPPED in ra.generated_ids and "58" not in ra.text
    ra2, _ = _same(a, b, "more", 3)
    assert ra2.mode == "replay-fallback" and a.cache[0].fed == a.fed_ids


def test_cap_refusal_then_recovery_in_incremental_mode(engine):
    a, b = _pair(engine)
    _same(a, b, "hello", 3)
    for s in (a, b):
        with pytest.raises(qserve.ContextLimitError):
            s.ask("x" * 200, 3)
    assert a.cache is None and len(a.messages) == 2
    ra, _ = _same(a, b, "more", 3)
    assert ra.mode == "fresh" and ra.prompt_ids == engine.render_chat(a.messages[:-1], False)   # the discarded cache is rebuilt from the whole transcript
    ra2, _ = _same(a, b, "again", 3)
    assert ra2.mode == "incremental"


def test_reset_and_isolation_in_incremental_mode(engine):
    a, b = _pair(engine)
    r1, _ = _same(a, b, "hello", 3)
    _same(a, b, "more", 3)
    c = qserve.Session(engine, mode="incremental", enable_thinking=False)
    assert c.ask("hello", 3).generated_ids == r1.generated_ids and c.cache[0].fed == c.fed_ids and len(c.fed_ids) < len(a.fed_ids)
    a.reset()
    assert a.cache is None and a.fed_ids == [] and a.messages == []
    assert a.ask("hello", 3).generated_ids == r1.generated_ids and a.ask("hello", 3).mode == "incremental"


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


def test_live_thinking_on_two_turns_both_modes(live_engine):
    # the pinned template re-renders a past turn from reasoning_content and
    # content; whether that rendering extends the fed ids is measured here,
    # and the two modes must agree token for token either way
    a = qserve.Session(live_engine, mode="incremental", enable_thinking=True)
    b = qserve.Session(live_engine, mode="replay", enable_thinking=True)
    for i, q in enumerate(("What is 6 times 7? Answer briefly.", "Add 10 to that. Answer briefly.")):
        ra, rb = a.ask(q, 256), b.ask(q, 256)
        assert ra.generated_ids == rb.generated_ids and ra.stop_reason == rb.stop_reason, (ra.mode, ra.note, ra.text[-80:], rb.text[-80:])
        assert ra.cache_offset_measured == ra.cache_tokens == len(a.fed_ids)
        if i == 1:
            # a closed block that stopped on a stop token re-renders exactly on this template (measured in the serving smoke); either mode is correct, reuse is expected
            assert ra.mode == ("incremental" if ("</think>" in a.turns[0].text and a.turns[0].stop_reason == "stop_token") else "replay-fallback"), (ra.mode, ra.note)
            assert "</think>" not in a.messages[1].get("content", "")   # the split moved the close out of the stored content


def test_live_budget_stop_then_follow_up_incremental(live_engine):
    a = qserve.Session(live_engine, mode="incremental", enable_thinking=False)
    b = qserve.Session(live_engine, mode="replay", enable_thinking=False)
    ra, rb = a.ask("Write one long sentence about rivers.", 6), b.ask("Write one long sentence about rivers.", 6)
    assert ra.stop_reason == rb.stop_reason == "max_tokens" and ra.generated_ids == rb.generated_ids and not ra.last_token_in_cache
    ra2, rb2 = a.ask("Now say only: done", 8), b.ask("Now say only: done", 8)
    assert ra2.generated_ids == rb2.generated_ids, (ra2.mode, ra2.note, ra2.text, rb2.text)
    assert ra2.mode == "incremental" and ra2.prompt_ids[0] == ra.generated_ids[-1]   # the un-fed last token fed exactly once
    assert ra2.cache_offset_measured == ra2.cache_tokens == len(a.fed_ids)


def test_incremental_is_the_default_session_mode(engine):
    # VERDICT QWEN-CHAT-DEFAULT-0: incremental by default; replay stays an explicit option
    s = qserve.Session(engine, enable_thinking=False)
    assert s.mode == "incremental"
    assert qserve.Session(engine, mode="replay", enable_thinking=False).mode == "replay"
