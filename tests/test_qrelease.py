"""Release integration checks for the chat defaults (QWEN-LOW-DEFAULT-1):
the pinned tokenizer in FRESH subprocesses, the planning-turn-7 tokenizer
mismatch fixture (tests/fixtures/qwen_planning_turn7.json, extracted from the
frozen receipt of QWEN-LOW-RELEASE-0), the CLI's help and mode resolution,
and, under LLMOPT_QWEN_LIVE=1 with the arm C artifact present, the fallback
and the replay equivalence on the real tower. Skips without the vendor
tokenizer."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.expanduser("~/qwen_vendor")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "qwen_planning_turn7.json")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(VENDOR, "tokenizer_config.json")), reason="vendor tokenizer absent")
LIVE = os.environ.get("LLMOPT_QWEN_LIVE") == "1" and os.path.exists(os.path.expanduser("~/qwen_whole0t/C/manifest.json"))
MSGS = [{"role": "user", "content": "What is 2 + 2?"}]
PINNED_XHIGH, PINNED_OFF = ("a2c3268e72cb", 60), ("b1d7a1c8ffb7", 20)      # tests/test_qtemplate.py pins


def sub(code: str) -> dict:
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-1500:]
    return json.loads(p.stdout.strip().splitlines()[-1])


PRE = ("import json, sys\nsys.path.insert(0, '.'); sys.path.insert(0, 'tests'); sys.path.insert(0, 'scripts')\n"
       "from transformers import AutoTokenizer\nimport test_qserve as T\nfrom llmopt.lab import qserve, qmetal\nqmetal.VERSION = 5\n"
       f"tok = AutoTokenizer.from_pretrained({VENDOR!r})\n"
       "eng = qserve.Engine(model=T._Model(), tokenizer=tok, context_cap=8192)\n"      # the stub model with the REAL tokenizer and template
       "import hashlib\nsha = lambda ids: hashlib.sha256(str(ids).encode()).hexdigest()[:12]\n")


def test_no_flags_render_equals_explicit_low_and_overrides_keep_their_pinned_renderings():
    out = sub(PRE + (
        f"m = {MSGS!r}\nimport qwen_chat as q\n"
        "s = qserve.Session(eng)\n"
        "r_default = eng.render_chat(m, s.enable_thinking, s.reasoning_effort)\n"
        "r_low = eng.render_chat(m, True, 'low')\n"
        "r_off = eng.render_chat(m, *q.resolve_mode(False, True, None))\n"
        "r_xhigh = eng.render_chat(m, *q.resolve_mode(False, False, 'xhigh'))\n"
        "r_tmpl = eng.render_chat(m)\n"
        "print(json.dumps({'default_is_low': r_default == r_low, 'off': [sha(r_off), len(r_off)], 'xhigh': [sha(r_xhigh), len(r_xhigh)],"
        " 'xhigh_is_template_default': r_xhigh == r_tmpl, 'default_budget': qserve.DEFAULT_MAX_TOKENS, 'mode': s.mode}))\n"))
    assert out["default_is_low"] and out["xhigh_is_template_default"] and out["mode"] == "incremental" and out["default_budget"] == 768
    assert tuple(out["off"]) == PINNED_OFF and tuple(out["xhigh"]) == PINNED_XHIGH


def test_cli_help_names_the_fast_option_and_the_defaults():
    p = subprocess.run([sys.executable, "scripts/qwen_chat.py", "--help"], capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert p.returncode == 0
    assert "--no-thinking" in p.stdout and "fast" in p.stdout.lower() and "768" in p.stdout and "--replay" in p.stdout and "--reasoning-effort" in p.stdout


def test_turn7_fixture_tokens_differ_from_reencoded_text_and_the_prefix_check_detects_it():
    fx = json.load(open(FIXTURE))
    out = sub(PRE + (
        f"fx = json.load(open({FIXTURE!r}))\n"
        "t6 = fx['turns'][5]\n"
        "re_ids = [int(i) for i in tok(t6['text'], add_special_tokens=False)['input_ids']]\n"
        "gen = t6['generated_ids']\n"
        "first = next((i for i, (a, b) in enumerate(zip(gen, re_ids)) if a != b), None)\n"
        # the session's view: messages from the stored turns, the fed ids = rendered turn-6 prompt + its generated ids (a stop-token turn feeds everything)
        "msgs = []\n"
        "for t in fx['turns'][:6]:\n"
        "    msgs.append({'role': 'user', 'content': t['user']})\n"
        "    reasoning, content = qserve.split_thinking(t['text'])\n"
        "    msgs.append({'role': 'assistant', 'content': content, **({'reasoning_content': reasoning} if reasoning else {})})\n"
        "p6 = eng.render_chat(msgs[:11], True, 'low')\n"      # the six user turns and five replies: turn 6's prompt\n"
        "fed = p6 + gen\n"
        "p7 = eng.render_chat(msgs + [{'role': 'user', 'content': fx['turns'][6]['user']}], True, 'low')\n"
        "extends = len(p7) > len(fed) and p7[:len(fed)] == fed\n"
        "brk = next((i for i, (a, b) in enumerate(zip(fed, p7)) if a != b), None)\n"
        "import hashlib as H\n"
        "print(json.dumps({'first_diff_in_generation': first, 'gen_at': gen[first:first + 2], 're_at': re_ids[first:first + 1], 'decoded': tok.decode(gen[first:first + 2]),"
        " 'p7_sha': H.sha256(json.dumps(p7).encode()).hexdigest(), 'decode_roundtrip': tok.decode(gen, skip_special_tokens=True) == t6['text'],"
        " 'p6_len': len(p6), 'p7_len': len(p7), 'fed_len': len(fed), 'extends': extends, 'break_at': brk, 'break_offset_in_generation': (brk - len(p6)) if brk is not None else None}))\n"))
    assert out["first_diff_in_generation"] == fx["mismatch"]["fed_offset_from_turn6_generation"] == 47
    assert out["gen_at"] == [1, 2523] and out["re_at"] == [20946] and out["decoded"] == '"You'       # the exact ids: emitted apart, re-encoded merged
    assert out["p7_len"] == fx["turns"][6]["prompt_len"] == fx["replay_turn7_prompt_len"] == 1851
    assert out["p7_sha"] == fx["replay_turn7_prompt_sha256"] and out["decode_roundtrip"] is True      # the rendering IS the receipt's prompt; the stored text is the ids' decode
    assert out["extends"] is False and out["break_offset_in_generation"] == 47                   # the prefix check detects it: replay, never reuse
    assert fx["turns"][6]["mode"] == "replay-fallback" and fx["turns"][6]["generated_ids"] == fx["replay_turn7_generated_ids"]   # the frozen receipt: the fallback matched replay


def test_turn7_fixture_session_on_the_stub_model_with_the_real_tokenizer_replays():
    """Session.ask itself (not a re-implementation) on the stub model with the pinned
    tokenizer: a session holding the fixture's fed ids asks turn 7 and must fall back."""
    out = sub(PRE + (
        f"fx = json.load(open({FIXTURE!r}))\n"
        "msgs = []\n"
        "for t in fx['turns'][:6]:\n"
        "    msgs.append({'role': 'user', 'content': t['user']})\n"
        "    reasoning, content = qserve.split_thinking(t['text'])\n"
        "    msgs.append({'role': 'assistant', 'content': content, **({'reasoning_content': reasoning} if reasoning else {})})\n"
        "p6 = eng.render_chat(msgs[:11], True, 'low')\n"
        "fed = p6 + fx['turns'][5]['generated_ids']\n"
        "cache = eng.model.make_cache(); cache[0].fed = list(fed)\n"
        "s = qserve.Session(eng)\n"
        "s.messages, s.fed_ids, s.cache = list(msgs), list(fed), cache\n"
        "r = s.ask(fx['turns'][6]['user'], 4)\n"
        "print(json.dumps({'mode': r.mode, 'note_has_reason': 'does not extend' in r.note, 'prompt_len': r.n_prompt, 'cache_fed_equals_fed_ids': s.cache[0].fed == s.fed_ids}))\n"))
    assert out["mode"] == "replay-fallback" and out["note_has_reason"] and out["prompt_len"] == 1851 and out["cache_fed_equals_fed_ids"]


@pytest.fixture(scope="module")
def live_engine():
    if not LIVE:
        pytest.skip("set LLMOPT_QWEN_LIVE=1 with the arm C artifact present")
    from llmopt.lab import qserve
    return qserve.Engine()


def test_live_turn7_fallback_replays_to_the_reference_with_valid_offsets(live_engine):
    """The real tower: a session holding exactly the fed ids of the fixture's six
    turns asks turn 7; the rendering does not extend the fed ids, the session
    replays, the tokens equal a replay session's and the offsets hold."""
    from llmopt.lab import qserve
    fx = json.load(open(FIXTURE))
    msgs = []
    for t in fx["turns"][:6]:
        msgs.append({"role": "user", "content": t["user"]})
        reasoning, content = qserve.split_thinking(t["text"])
        msgs.append({"role": "assistant", "content": content, **({"reasoning_content": reasoning} if reasoning else {})})
    p6 = live_engine.render_chat(msgs[:11], True, "low")
    fed = p6 + fx["turns"][5]["generated_ids"]
    assert len(fed) == fx["turns"][5]["cache_tokens"]
    # a cache holding exactly `fed`: feed it with a budget of one (the one generated token stays un-fed)
    warm = live_engine.generate_ids(fed, max_tokens=1)
    assert warm.cache_tokens == len(fed) == warm.cache_offset_measured
    a = qserve.Session(live_engine)                      # the new default: thinking low, incremental
    a.messages, a.fed_ids, a.cache = list(msgs), list(fed), warm.cache
    b = qserve.Session(live_engine, mode="replay")
    b.messages = list(msgs)
    ra, rb = a.ask(fx["turns"][6]["user"], 768), b.ask(fx["turns"][6]["user"], 768)
    assert ra.mode == "replay-fallback" and "does not extend" in ra.note
    assert ra.generated_ids == rb.generated_ids == fx["replay_turn7_generated_ids"]
    assert ra.cache_offset_measured == ra.cache_tokens == len(a.fed_ids) and rb.cache_offset_measured == rb.cache_tokens
    assert ra.prompt_ids == rb.prompt_ids and len(ra.prompt_ids) == 1851


def test_live_default_session_multi_turn_incremental_equals_replay(live_engine):
    from llmopt.lab import qserve
    a, b = qserve.Session(live_engine), qserve.Session(live_engine, mode="replay")
    assert a.enable_thinking is True and a.reasoning_effort == "low" and b.enable_thinking is True and b.reasoning_effort == "low"
    turns = [("Which element has the chemical symbol Fe? One word.", 128), ("What is its atomic number? Number only.", 128), ("Reply with just: thanks.", 64)]
    for i, (q, n) in enumerate(turns):
        ra, rb = a.ask(q, n), b.ask(q, n)
        assert ra.generated_ids == rb.generated_ids and ra.stop_reason == rb.stop_reason, (i, ra.mode, ra.note)
        assert ra.cache_offset_measured == ra.cache_tokens == len(a.fed_ids)
        assert "</think>" in ra.text
        if i > 0 and "</think>" in a.turns[-2].text:
            assert ra.mode == "incremental", (i, ra.mode, ra.note)                 # a closed low turn re-renders exactly here, or the fallback reason is on record
    # an incompatible change mid-session never reuses the cache
    a.reasoning_effort = "xhigh"
    r = a.ask("And the symbol for gold? One word.", 128)
    assert r.mode == "replay-fallback" and r.cache_offset_measured == r.cache_tokens == len(a.fed_ids)
    # a budget stop, then a context-cap refusal, then recovery and reset
    r2 = a.ask("Write one long sentence about rivers.", 5)
    assert r2.stop_reason == "max_tokens" and not r2.last_token_in_cache
    with pytest.raises(qserve.ContextLimitError):
        a.ask("x " * (live_engine.context_cap + 64), 64)                          # longer than the cap on its own, whatever the history
    assert a.cache is None
    r3 = a.ask("Reply with just: ok.", 64)
    assert r3.mode == "fresh"
    a.reset()
    assert a.messages == [] and a.fed_ids == [] and a.cache is None
