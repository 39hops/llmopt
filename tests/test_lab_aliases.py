"""Phase 5 move shims: old llmopt.lab.<name> paths must alias the
moved modules with full fidelity (frozen scripts/scratch import the
old paths)."""
import importlib

import pytest

PAIRS = [
    ("llmopt.lab.figstyle", "llmopt.figures.figstyle"),
    ("llmopt.lab.figsvg", "llmopt.figures.figsvg"),
    ("llmopt.lab.figures", "llmopt.figures.figures"),
    ("llmopt.lab.anatomy", "llmopt.figures.anatomy"),
    ("llmopt.lab.runlog", "llmopt.runs.receipts"),
    ("llmopt.runs.runlog", "llmopt.runs.receipts"),
    ("llmopt.lab.traj", "llmopt.runs.traj"),
    ("llmopt.lab.runfiles", "llmopt.runs.runfiles"),
    ("llmopt.lab.lake", "llmopt.runs.lake"),
]


@pytest.mark.parametrize("old,new", PAIRS)
def test_alias_is_same_module(old, new):
    try:
        target = importlib.import_module(new)
    except ImportError as e:
        pytest.skip(f"optional dep missing for {new}: {e}")
    assert importlib.import_module(old) is target
