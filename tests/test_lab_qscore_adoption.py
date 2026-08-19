"""Adoption guard for llmopt.lab.qscore.

The X/K scoring math was adopted verbatim from the frozen
results-cited driver scratch/qwen_model1_score.py (2026-08-19, after
its third in-place surface extension). While the two files coexist,
every adopted symbol must stay CHARACTER-IDENTICAL between them — a
fix lands in both in the same commit or in neither.

The identity comparison is PURE TEXT (parsed with ast, never
imported), so it runs everywhere, including hosts with no torch.
teacher_receipt_block is NEW lab code (renamed sha fields, auditor
N2), so it is behavior-tested here, not identity-guarded.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch" / "qwen_model1_score.py"
LAB = ROOT / "llmopt" / "lab" / "qscore.py"

ADOPTED = ["MARGIN_EDGES", "SMALL_N", "log_softmax", "mean_ce",
           "mean_forward_kl", "perturb_ulp", "sensitivity_floor",
           "margin_bin", "teacher_margins_top1", "flip_table",
           "sha_arr", "fsha"]


def _top_level_sources(path: Path) -> dict[str, str]:
    """name -> source text, for top-level defs/classes/assignments.

    Mirrors inspect.getsource's span (decorators included) without
    importing the module.
    """
    text = path.read_text()
    lines = text.splitlines()
    out: dict[str, str] = {}

    def span(node, first):
        return "\n".join(lines[first - 1:node.end_lineno])

    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            first = min([node.lineno]
                        + [d.lineno for d in node.decorator_list])
            out[node.name] = span(node, first)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = span(node, node.lineno)
    return out


@pytest.fixture(scope="module")
def pair():
    return _top_level_sources(SCRATCH), _top_level_sources(LAB)


@pytest.mark.parametrize("name", ADOPTED)
def test_adopted_source_identical(pair, name):
    scr, lab = pair
    assert name in scr, f"{name} missing from {SCRATCH}"
    assert name in lab, f"{name} missing from {LAB}"
    assert lab[name] == scr[name], (
        f"{name} drifted between {SCRATCH} and {LAB}; fix BOTH copies "
        "in the same commit")


def test_qscore_imports_without_torch():
    import llmopt.lab.qscore as q
    for name in ADOPTED + ["teacher_receipt_block"]:
        assert hasattr(q, name)


def test_teacher_receipt_block_renamed_fields_are_derived():
    from llmopt.lab.qscore import teacher_receipt_block
    man = {"code_commit": "abc1234", "revision": "r0",
           "records": {"corpus": {"sha256": "c-rec"},
                       "prefixes": {"sha256": "p-rec"}},
           "inputs": {"corpus_sha256": "c-in",
                      "prefixes_sha256": "p-in"}}
    b = teacher_receipt_block(man, "logs/qwenteacher_v2")
    assert b == {"dir": "logs/qwenteacher_v2", "code_commit": "abc1234",
                 "revision": "r0",
                 "corpus_record_sha": "c-rec",
                 "prefix_record_sha": "p-rec",
                 "corpus_input_sha": "c-in",
                 "prefixes_input_sha": "p-in"}
    # legacy ambiguous names must not reappear
    assert "corpus_sha" not in b and "prefix_sha" not in b


def test_mean_ce_matches_hand_computation():
    """Behavior pin independent of the scratch fixture suite: uniform
    logits over V=4 give CE = ln(4) at every position."""
    from llmopt.lab.qscore import mean_ce
    lg = np.zeros((3, 4), dtype=np.float32)
    assert mean_ce(lg, [0, 1, 2], 4) == pytest.approx(np.log(4.0))
