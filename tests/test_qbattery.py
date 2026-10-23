"""The battery's validators pinned on synthetic replies (no model), the
collection's determinism, and the code sandbox's isolation (macOS only)."""
from __future__ import annotations

import json
import sys

import os
import collections
import pytest

from llmopt.lab import qbattery


def test_collection_is_deterministic_and_balanced():
    a, b = qbattery.build(), qbattery.build()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    kinds = {}
    for t in a["tasks"]:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    assert kinds == {"arith": 10, "json": 8, "code": 8, "instruct": 8, "twoturn": 6, "degen": 6}
    assert len({t["id"] for t in a["tasks"]}) == len(a["tasks"])


def _task(kind, idx=0):
    return [t for t in qbattery.build()["tasks"] if t["kind"] == kind][idx]


def test_arith_takes_the_last_integer_and_accepts_separators():
    t = _task("arith")
    e = t["expected"]
    assert qbattery.score(t, [f"Let me compute. The result is {e:,}."], [{}])["status"] == "pass"
    assert qbattery.score(t, [f"{e} or maybe {e + 1}"], [{}])["status"] == "fail"
    assert qbattery.score(t, ["no digits here"], [{}])["status"] == "fail"


def test_json_requires_schema_and_values():
    t = _task("json")
    good = dict(t["expected"])
    assert qbattery.score(t, ["Sure:\n```json\n" + json.dumps(good) + "\n```"], [{}])["status"] == "pass"
    bad = dict(good)
    k = next(iter(bad))
    bad[k] = "wrong" if not isinstance(bad[k], str) else bad[k] + "x"
    assert qbattery.score(t, [json.dumps(bad)], [{}])["status"] == "fail"
    assert qbattery.score(t, ["not json"], [{}])["status"] == "fail"


def test_instruct_checks_are_programs():
    lines = _task("instruct", 0)
    assert qbattery.score(lines, ["Pacific\nAtlantic\nIndian"], [{}])["status"] == "pass"
    assert qbattery.score(lines, ["Pacific\nAtlantic"], [{}])["status"] == "fail"
    caps = _task("instruct", 2)
    assert qbattery.score(caps, ["RAIN FALLS SOFTLY."], [{}])["status"] == "pass"
    assert qbattery.score(caps, ["Rain falls softly."], [{}])["status"] == "fail"
    numbered = _task("instruct", 4)
    assert qbattery.score(numbered, ["1. kale\n2. leek\n3. corn\n4. pea\n5. yam"], [{}])["status"] == "pass"
    assert qbattery.score(numbered, ["kale, leek, corn, pea, yam"], [{}])["status"] == "fail"


def test_twoturn_reads_turn_two():
    t = _task("twoturn")
    assert qbattery.score(t, ["noted.", f"It was {t['expected']}."], [{}, {}])["status"] == "pass"
    assert qbattery.score(t, ["noted.", "I do not remember."], [{}, {}])["status"] == "fail"


def test_degen_requires_a_stop_token_and_exact_counts():
    t = _task("degen")
    w, n = t["params"]["word"], t["params"]["n"]
    ok = " ".join([w] * n)
    assert qbattery.score(t, [ok], [{"stop_reason": "stop_token"}])["status"] == "pass"
    assert qbattery.score(t, [ok], [{"stop_reason": "max_tokens"}])["status"] == "fail"
    assert qbattery.score(t, [" ".join([w] * (n + 1))], [{"stop_reason": "stop_token"}])["status"] == "fail"


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_code_sandbox_passes_correct_code_and_isolates():
    t = _task("code", 0)
    good = "```python\ndef reverse_words(s):\n    return ' '.join(reversed(s.split()))\n```"
    assert qbattery.score(t, [good], [{}])["status"] == "pass"
    wrong = "```python\ndef reverse_words(s):\n    return s\n```"
    assert qbattery.score(t, [wrong], [{}])["status"] == "fail"
    net = "```python\nimport socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)\ndef reverse_words(s):\n    return ' '.join(reversed(s.split()))\n```"
    assert qbattery.score(t, [net], [{}])["status"] == "fail"
    home_file = os.path.join(os.path.expanduser("~"), ".zshrc")     # a real file under the user's home, resolved on the host
    r = qbattery.run_code_sandboxed(f"import os\nprint(os.path.exists('/Users'))\nopen({home_file!r}).read()", [], 5)
    assert r["status"] == "fail" and "PermissionError" in (r.get("stdout", "") + r.get("stderr", ""))
    r = qbattery.run_code_sandboxed("open('/tmp/qbat_probe', 'w').write('x')", [], 5)
    assert r["status"] == "fail"
    r = qbattery.run_code_sandboxed("while True:\n    pass", [], 2)
    assert r["status"] == "fail" and (r.get("timed_out") or r.get("returncode") not in (0, None))
    r = qbattery.run_code_sandboxed("print('ALL-TESTS-PASSED-0000')", ["assert False"], 5)
    assert r["status"] == "fail"


def test_code_without_sandbox_is_hold(monkeypatch):
    monkeypatch.setattr(qbattery, "sandbox_available", lambda: False)
    t = _task("code", 0)
    assert qbattery.score(t, ["```python\ndef reverse_words(s):\n    return s\n```"], [{}])["status"] == "hold"


@pytest.mark.skipif(not qbattery.sandbox_available(), reason="no sandbox on this platform")
def test_code_candidate_syntax_error_is_a_model_failure_not_a_harness_error():
    # a reply cut off by the token budget leaves an unterminated function; the
    # harness must still start, so the verdict is fail (the candidate's), not error
    r = qbattery.run_code_sandboxed("def f(n):\n    return (n +", ["assert f(1) == 2"], 5)
    assert r["status"] == "fail" and "SyntaxError" in r.get("stdout", "") + r.get("stderr", "")


def test_thinking_scores_the_final_answer_after_the_closed_think_block():
    t = {"id": "arith-x", "kind": "arith", "turns": ["What is 2 + 2? Answer with the number only."], "expected": 4}
    res = [{"stop_reason": "stop_token"}]
    assert qbattery.score_thinking(t, ["Let me add. 2 + 2 = 5, no, 4.\n</think>\n\n4"], res)["status"] == "pass"
    # the number only appears inside the reasoning: no final answer, fail, never a search of the hidden text
    v = qbattery.score_thinking(t, ["Let me add. 2 + 2 = 4 so the answer is 4 and"], [{"stop_reason": "max_tokens"}])
    assert v["status"] == "fail" and "did not terminate" in v["why"]
    # a closed block with a wrong final answer fails on the final answer
    assert qbattery.score_thinking(t, ["2 + 2 = 4\n</think>\n\n5"], res)["status"] == "fail"
    assert qbattery.split_final("a\n</think>\n\nb</think>c") == ("a\n</think>\n\nb", "c")
    assert qbattery.split_final("no close") == (None, None)
    assert qbattery.split_final("a</think>\n<think>more and") == (None, None)      # reopened and never closed: no final answer
    # a two-turn task scores turn 2 only, so an unclosed turn 1 is not a fail by itself
    t2 = {"id": "twoturn-x", "kind": "twoturn", "turns": ["Remember 7.", "What was it?"], "expected": "7"}
    v = qbattery.score_thinking(t2, ["thinking forever about 7", "ok\n</think>\n\n7"], [{"stop_reason": "max_tokens"}, {"stop_reason": "stop_token"}])
    assert v["status"] == "pass", v


def test_format_ok_is_the_contract_of_the_task_kind():
    assert qbattery.format_ok({"kind": "arith"}, "148710") and not qbattery.format_ok({"kind": "arith"}, "The answer is 148710")
    assert qbattery.format_ok({"kind": "json"}, ' {"a": 1} ') and not qbattery.format_ok({"kind": "json"}, 'Here: {"a": 1}')
    assert qbattery.format_ok({"kind": "code"}, "x") is None


# ---------------------------------------------------------------- v3 validator branches (QWEN-LOW-RELEASE-0)

def test_v3_instruct_checks():
    ck = lambda c, p, text: qbattery.check_instruct({"check": c, "params": p}, [text])[0]  # noqa: E731
    assert ck("sentences", {"n": 2}, "Rain falls. Rivers rise.") and not ck("sentences", {"n": 2}, "Rain falls. Rivers rise. Seas fill.")
    assert ck("starts_with", {"word": "indeed"}, "Indeed, the sky is blue.") and not ck("starts_with", {"word": "indeed"}, "The sky, indeed, is blue.")
    assert ck("no_commas", {}, "The sky is blue and wide.") and not ck("no_commas", {}, "The sky, blue and wide.")
    assert ck("bullets", {"n": 3}, "- a\n- b\n- c") and not ck("bullets", {"n": 3}, "- a\n- b") and not ck("bullets", {"n": 3}, "1. a\n2. b\n3. c")
    assert ck("alphabetical", {"n": 3}, "apple\nbanana\ncherry") and not ck("alphabetical", {"n": 3}, "banana\napple\ncherry")
    assert ck("ends_question", {}, "What time is it?") and not ck("ends_question", {}, "It is noon.")
    assert ck("exact_words", {"n": 4}, "one two three four") and not ck("exact_words", {"n": 4}, "one two three")
    assert ck("all_lower", {}, "all quiet here.") and not ck("all_lower", {}, "All quiet here.")
    assert ck("include_two", {"words": ["river", "bridge"]}, "A river under a bridge.") and not ck("include_two", {"words": ["river", "bridge"]}, "A river.")


def test_v3_degen_checks():
    ok = lambda c, p, text, extra={}: qbattery.check_degen({"check": c, "params": p, **extra}, [text], [{"stop_reason": "stop_token"}])[0]  # noqa: E731
    assert ok("repeat_phrase", {"phrase": "blue sky", "n": 3}, "blue sky, blue sky, blue sky") and not ok("repeat_phrase", {"phrase": "blue sky", "n": 3}, "blue sky, blue sky")
    assert ok("countdown", {"a": 10, "b": 5}, "10, 9, 8, 7, 6, 5") and not ok("countdown", {"a": 10, "b": 5}, "10 9 8 7 6")
    assert ok("hyphen_spell", {"word": "river"}, "R-I-V-E-R") and ok("hyphen_spell", {"word": "river"}, "r-i-v-e-r") and not ok("hyphen_spell", {"word": "river"}, "RIVER")
    assert ok("one_word", {}, "Tokyo", {"expected": "tokyo"}) and not ok("one_word", {}, "It is Tokyo.", {"expected": "tokyo"})
    assert ok("letters", {"n": 4}, "a, b, c, d") and ok("letters", {"n": 4}, "A,B,C,D") and not ok("letters", {"n": 4}, "a, b, c")


def test_v3_json_dict_values_compare_case_insensitively():
    t = {"schema": {"rows_per_category": "dict", "total_qty": "integer"}, "expected": {"rows_per_category": {"tools": 2, "paint": 1}, "total_qty": 9}}
    assert qbattery.check_json(t, ['{"rows_per_category": {"Tools": 2, "paint": 1}, "total_qty": 9}'])[0]
    assert not qbattery.check_json(t, ['{"rows_per_category": {"tools": 1, "paint": 1}, "total_qty": 9}'])[0]
    assert not qbattery.check_json(t, ['{"rows_per_category": [2, 1], "total_qty": 9}'])[0]


def test_v3_collection_builds_disjoint_from_v2_and_its_code_references_pass():
    from llmopt.lab import qbattery3
    c = qbattery3.build()
    assert c["version"] == 3 and len(c["tasks"]) == 60
    kinds = collections.Counter(t["kind"] for t in c["tasks"])
    assert kinds == {"arith": 12, "json": 12, "code": 12, "instruct": 10, "twoturn": 8, "degen": 6}
    v2 = qbattery.build()
    assert not ({t["turns"][0] for t in c["tasks"]} & {t["turns"][0] for t in v2["tasks"]})       # no shared prompt text
    ov = qbattery3.overlap_report(c, v2)
    assert ov["shared_prompts"] == 0 and ov["shared_code_functions"] == 0
    assert qbattery3.build() == c                                                                   # deterministic
    for t in c["tasks"]:
        if t["kind"] == "code":
            ns = {}
            exec(t["reference"], ns)                                                                # the reference solution passes the hidden tests
            for a in t["tests"]:
                exec(a, ns)


def test_v3_validator_edge_cases_from_the_prefreeze_review():
    ck = lambda c, p, text: qbattery.check_instruct({"check": c, "params": p}, [text])[0]  # noqa: E731
    assert not ck("alphabetical", {"n": 5}, "1. apple\n2. banana\n3. cherry\n4. date\n5. fig")      # numbering is not 'nothing else'
    assert not ck("alphabetical", {"n": 3}, "apple\napple\napple") and ck("alphabetical", {"n": 3}, "apple\nbanana\ncherry")
    assert ck("exact_words", {"n": 12}, "The ever-changing sea stretches endlessly, whispering secrets to the moonlit shore tonight.")
    assert ck("ends_question", {}, "**Want to grab lunch tomorrow?**") and ck("ends_question", {}, 'She asked, "Lunch?"')
    assert ck("sentences", {"n": 2}, "Light scatters, e.g. blue light. It is bright.")
    dg = lambda c, p, text: qbattery.check_degen({"check": c, "params": p}, [text], [{"stop_reason": "stop_token"}])[0]  # noqa: E731
    assert dg("letters", {"n": 6}, "a, b, c, d, e, f.") and dg("hyphen_spell", {"word": "river"}, "r - i - v - e - r")
    assert dg("repeat_phrase", {"phrase": "red door", "n": 2}, "'red door', 'red door'")


def test_v3_frozen_file_equals_the_generator_and_its_recorded_sha():
    import hashlib
    from llmopt.lab import qbattery3
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "evalsets", "qwen_battery_v3.json")
    frozen = json.load(open(path))
    assert frozen == qbattery3.build()
    ov = json.load(open(path.replace("qwen_battery_v3.json", "qwen_battery_v3_overlap.json")))
    assert ov["v3_sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest() and ov["shared_prompts"] == 0
    assert len(ov["near_relatives_of_v2_families"]) >= 10                                           # the independence caveat is on the record


def test_lowchat_addressing_checks():
    import importlib.util
    spec = importlib.util.spec_from_file_location("lowchat", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "qwen_lowchat.py"))
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    a = lambda check, text: lc.addressed(check, text)[0]  # noqa: E731
    assert a(("exact", "done"), "Done.") and not a(("exact", "done"), "Not done yet.")
    assert a(("yes_no", "yes"), "Yes, Python's sort is stable.") and not a(("yes_no", "yes"), "No, it is not stable.")
    assert a(("re", [r"\bles\b", r"\bjour\b"]), "Jour 1 : Lisbonne") and not a(("re", [r"\bles\b", r"\bjour\b"]), "A possible two-line plan for Sintra.")
    assert a(("number", 440), "The new total is 440.") and not a(("number", 440), "The total is 400.")
    assert a(("lines", 2), "one\ntwo") and not a(("lines", 2), "one")
    assert a(("min_words", 3), "a b c d") and not a(("min_words", 3), "a b") and not a(("any", ["x"]), "")
    assert len(lc.CONVERSATIONS) == 4 and all(len(c["turns"]) == 8 for c in lc.CONVERSATIONS) and lc.CONVERSATIONS[0]["name"] == "travel"
