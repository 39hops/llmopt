"""Frozen-output pins for the mathgen generator (rehabilitation 2C).

tests/fixtures/mathgen_pins.json was written by scripts/gen_mathgen_pins.py
from the generator BEFORE the 2C guards landed. Every dataset digest,
single-maker case and deterministic-retry case must reproduce exactly;
a mismatch means the seed arithmetic, prompt ordering, retry branch or
answer text changed for a previously successful input.
"""
import json
from pathlib import Path

import pytest

sp = pytest.importorskip("sympy")

from llmopt.mathgen.digest import CONTRACT, dataset_digest
from llmopt.mathgen.problems import _resolve_maker, make_dataset

PINS = json.loads((Path(__file__).parent / "fixtures" / "mathgen_pins.json").read_text())


def _dataset(entry):
    args = {k: (tuple(v) if isinstance(v, list) else v) for k, v in entry["args"].items()}
    exclude = frozenset()
    if entry["exclude_from"]:
        src = next(d for d in PINS["datasets"] if d["name"] == entry["exclude_from"])
        exclude = frozenset(p.prompt for p in _dataset(src))
    return make_dataset(**args, exclude=exclude)


def test_pin_file_records_the_digest_contract():
    assert PINS["contract"] == CONTRACT
    assert PINS["skipped"] == []


@pytest.mark.parametrize("entry", PINS["datasets"], ids=lambda e: e["name"])
def test_dataset_digest_unchanged(entry):
    ds = _dataset(entry)
    assert len(ds) == entry["count"]
    assert ds[0].prompt == entry["first_prompt"]
    assert ds[-1].prompt == entry["last_prompt"]
    assert dataset_digest(ds) == entry["sha256"]


@pytest.mark.parametrize("case", [c for c in PINS["cases"] if c["level"] <= 3],
                         ids=lambda c: f"{c['kind']}-L{c['level']}-s{c['seed']}")
def test_low_level_case_unchanged(case):
    p = _resolve_maker(case["kind"])(case["level"], case["seed"])
    assert (p.prompt, p.answer) == (case["prompt"], case["answer"])


@pytest.mark.parametrize("case", [c for c in PINS["cases"] if c["level"] > 3],
                         ids=lambda c: f"{c['kind']}-L{c['level']}-s{c['seed']}")
def test_deep_level_case_unchanged(case):
    p = _resolve_maker(case["kind"])(case["level"], case["seed"])
    assert (p.prompt, p.answer) == (case["prompt"], case["answer"])


@pytest.mark.parametrize("case", PINS["retry_cases"],
                         ids=lambda c: f"{c['kind']}-L{c['level']}-s{c['seed']}")
def test_retry_case_unchanged(case):
    mk = _resolve_maker(case["kind"])
    p = mk(case["level"], case["seed"])
    assert (p.prompt, p.answer) == (case["prompt"], case["answer"])
    # the retry branch is deterministic: the retried seed yields the same prompt
    assert mk(case["level"], case["retried_seed"]).prompt == case["prompt"]
