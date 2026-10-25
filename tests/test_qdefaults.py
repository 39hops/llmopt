"""The actual defaults of Session and the CLI with no flags supplied, checked
in FRESH subprocesses (VERDICT QWEN-LOW-DEFAULT-1: thinking on at the pinned
template's reasoning_effort low, 768 tokens, incremental reuse), on the
deterministic stub engine of tests/test_qserve.py."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("mlx.core")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(code: str) -> dict:
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-1200:]
    return json.loads(p.stdout.strip().splitlines()[-1])


STUB = (
    "import json, sys\n"
    "sys.path.insert(0, 'tests'); sys.path.insert(0, '.')\n"
    "import test_qserve as T\n"
    "from llmopt.lab import qserve, qmetal\n"
    "qmetal.VERSION = 5\n"
    "eng = qserve.Engine(model=T._Model(), tokenizer=T._Tok(), context_cap=64)\n"
)


def test_session_with_no_arguments_is_thinking_low_incremental_768():
    out = run(STUB + (
        "s = qserve.Session(eng)\n"
        "explicit = qserve.Session(eng, enable_thinking=True, reasoning_effort='low')\n"
        "off = qserve.Session(eng, enable_thinking=False)\n"
        "tmpl = qserve.Session(eng, enable_thinking=None)\n"
        "m = [{'role': 'user', 'content': 'hi'}]\n"
        "print(json.dumps({'thinking': s.enable_thinking, 'effort': s.reasoning_effort, 'mode': s.mode, 'budget': qserve.DEFAULT_MAX_TOKENS,"
        " 'default_equals_explicit_low': eng.render_chat(m, s.enable_thinking, s.reasoning_effort) == eng.render_chat(m, True, 'low'),"
        " 'default_differs_from_template_default': eng.render_chat(m, s.enable_thinking, s.reasoning_effort) != eng.render_chat(m, None),"
        " 'off_effort': off.reasoning_effort, 'tmpl_effort': tmpl.reasoning_effort, 'tmpl_thinking': tmpl.enable_thinking}))\n"))
    assert out == {"thinking": True, "effort": "low", "mode": "incremental", "budget": 768, "default_equals_explicit_low": True,
                   "default_differs_from_template_default": True, "off_effort": None, "tmpl_effort": None, "tmpl_thinking": None}


def test_effort_only_implies_thinking_and_effort_with_no_thinking_is_refused():
    out = run(STUB + (
        "e = qserve.Session(eng, reasoning_effort='low')\n"
        "try:\n    qserve.Session(eng, enable_thinking=False, reasoning_effort='low'); refused = False\n"
        "except qserve.InputError:\n    refused = True\n"
        "print(json.dumps({'effort_only_thinking': e.enable_thinking, 'refused': refused}))\n"))
    assert out == {"effort_only_thinking": True, "refused": True}


def test_cli_resolves_no_flags_to_thinking_low_and_keeps_overrides():
    out = run("import json, sys\nsys.path.insert(0, 'scripts')\nimport qwen_chat as q\n"
              "print(json.dumps({'none': q.resolve_mode(False, False, None), 'off': q.resolve_mode(False, True, None), 'xhigh': q.resolve_mode(False, False, 'xhigh'),"
              " 'thinking_flag': q.resolve_mode(True, False, None), 'medium': q.resolve_mode(True, False, 'medium')}))\n")
    assert out == {"none": [True, "low"], "off": [False, None], "xhigh": [True, "xhigh"], "thinking_flag": [True, "low"], "medium": [True, "medium"]}


def test_cli_default_budget_is_768_and_incompatible_flags_are_refused():
    p = subprocess.run([sys.executable, "scripts/qwen_chat.py", "--prompt", "x", "--no-thinking", "--reasoning-effort", "low", "--art-dir", "/nonexistent"],
                       capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert p.returncode == 2 and "no effect with --no-thinking" in p.stderr
    out = run("import json, sys, argparse\nsys.path.insert(0, 'scripts')\nimport qwen_chat as q\n"
              "src = open('scripts/qwen_chat.py').read()\n"
              "print(json.dumps({'budget_default_768': '\"--max-tokens\", type=int, default=768' in src}))\n")
    assert out == {"budget_default_768": True}


def test_changing_mode_or_effort_mid_session_never_reuses_an_incompatible_cache():
    out = run(STUB + (
        "s = qserve.Session(eng)\n"
        "s.ask('hello', 6)\n"
        "s.reasoning_effort = 'xhigh'\n"
        "r = s.ask('more', 6)\n"
        "s.enable_thinking = False; s.reasoning_effort = None\n"
        "r2 = s.ask('again', 6)\n"
        "print(json.dumps({'effort_change': r.mode, 'thinking_change': r2.mode, 'cache_equals_fed': s.cache[0].fed == s.fed_ids}))\n"))
    assert out == {"effort_change": "replay-fallback", "thinking_change": "replay-fallback", "cache_equals_fed": True}


def test_default_recurrence_is_the_kernel_and_a_supplied_model_records_none():
    """QWEN-GDN-PROMOTE-0: the library default selects the Metal recurrence
    kernel by name; the stub engine (a supplied model) makes no selection."""
    out = run(STUB + (
        "print(json.dumps({'default': qserve.DEFAULT_RECURRENCE, 'stub_recurrence': eng.info.get('recurrence'),"
        " 'paths': list(__import__('llmopt.lab.qtower', fromlist=['x']).RECURRENCE_PATHS)}))\n"))
    assert out["default"] == "kernel" and out["stub_recurrence"] is None and out["paths"] == ["loop", "kernel"]


def test_kernel_recurrence_refuses_a_chunk_other_than_256():
    out = run(STUB + (
        "try:\n"
        "    qserve.Engine(model=T._Model(), tokenizer=T._Tok(), context_cap=64, chunk=512)\n"
        "    refused = False\n"
        "except RuntimeError as e:\n"
        "    refused = 'qualified at chunk 256' in str(e)\n"
        "ok_loop = qserve.Engine(model=T._Model(), tokenizer=T._Tok(), context_cap=64, chunk=512, recurrence='loop').cfg.chunk\n"
        "print(json.dumps({'refused': refused, 'loop_chunk': ok_loop}))\n"))
    assert out["refused"] is True and out["loop_chunk"] == 512
