"""The battery's validators pinned on synthetic replies (no model), the
collection's determinism, and the code sandbox's isolation (macOS only)."""
from __future__ import annotations

import json
import sys

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
    r = qbattery.run_code_sandboxed("import os\nprint(os.path.exists('/Users'))\nopen('/Users/artin/.zshrc').read()", [], 5)
    assert r["status"] == "fail" and "PermissionError" in r.get("stderr", "")
    r = qbattery.run_code_sandboxed("open('/tmp/qbat_probe', 'w').write('x')", [], 5)
    assert r["status"] == "fail"


def test_code_without_sandbox_is_hold(monkeypatch):
    monkeypatch.setattr(qbattery, "sandbox_available", lambda: False)
    t = _task("code", 0)
    assert qbattery.score(t, ["```python\ndef reverse_words(s):\n    return s\n```"], [{}])["status"] == "hold"
