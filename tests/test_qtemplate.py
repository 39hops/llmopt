"""The pinned Qwen chat template's reasoning_effort control, verified
mechanically on the real tokenizer in a FRESH subprocess per rendering
(no state shared with the test process), and the Engine's routing of the
argument (QWEN-EFFORT-AND-DEEPCHAT-0, study A). Skips without the vendor
tokenizer."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

VENDOR = os.path.expanduser("~/qwen_vendor")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(VENDOR, "tokenizer_config.json")), reason="vendor tokenizer absent")

MSGS = [{"role": "user", "content": "What is 2 + 2?"}]
# measured 2026-10-23 on the pinned tokenizer (tokenizer_config.json sha recorded by the Engine): the ids of MSGS under each control
PINNED = {"xhigh": ("a2c3268e72cb", 60), "low": ("d093289e3799", 48), "medium": ("6568c97a0178", 18), "off": ("b1d7a1c8ffb7", 20)}
XHIGH_TEXT = "Reasoning effort is set to xhigh. Please think carefully"
LOW_TEXT = "Reasoning effort is set to low. Keep your thinking brief"


def render(**kw):
    """ids and decoded text from a fresh interpreter: the template itself, no Engine."""
    code = ("import json, sys\nfrom transformers import AutoTokenizer\n"
            f"t = AutoTokenizer.from_pretrained({VENDOR!r})\n"
            f"ids = t.apply_chat_template({MSGS!r}, add_generation_prompt=True, tokenize=True, **{kw!r})\n"
            "ids = list(ids['input_ids']) if hasattr(ids, 'keys') else list(ids)\n"
            "print(json.dumps({'ids': [int(i) for i in ids], 'text': t.decode(ids)}))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-800:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def sha(ids):
    return hashlib.sha256(str(ids).encode()).hexdigest()[:12]


def test_omitted_effort_is_xhigh_and_matches_the_pinned_rendering():
    omitted, true_, xhigh = render(), render(enable_thinking=True), render(enable_thinking=True, reasoning_effort="xhigh")
    assert omitted["ids"] == true_["ids"] == xhigh["ids"]
    assert (sha(xhigh["ids"]), len(xhigh["ids"])) == PINNED["xhigh"]
    assert XHIGH_TEXT in xhigh["text"] and xhigh["text"].endswith("<|im_start|>assistant\n<think>\n")


def test_low_and_medium_render_the_documented_variants():
    low, med = render(enable_thinking=True, reasoning_effort="low"), render(enable_thinking=True, reasoning_effort="medium")
    assert (sha(low["ids"]), len(low["ids"])) == PINNED["low"] and LOW_TEXT in low["text"] and low["text"].endswith("<think>\n")
    assert (sha(med["ids"]), len(med["ids"])) == PINNED["medium"]
    assert "Reasoning effort" not in med["text"] and med["text"].startswith("<|im_start|>user\n") and med["text"].endswith("<think>\n")   # medium injects no instruction


def test_no_thinking_rendering_is_unchanged_and_ignores_effort_in_the_template():
    off = render(enable_thinking=False)
    assert (sha(off["ids"]), len(off["ids"])) == PINNED["off"] and off["text"].endswith("<think>\n\n</think>\n\n")
    assert render(enable_thinking=False, reasoning_effort="low")["ids"] == off["ids"]   # the template ignores it: the Engine must refuse instead


def test_unsupported_effort_is_refused_by_the_template():
    code = (f"from transformers import AutoTokenizer\nt = AutoTokenizer.from_pretrained({VENDOR!r})\n"
            f"t.apply_chat_template({MSGS!r}, add_generation_prompt=True, tokenize=True, enable_thinking=True, reasoning_effort='high')\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert p.returncode != 0 and "Unexpected reasoning effort" in p.stderr
