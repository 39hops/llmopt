"""Behavior guard for the pincer_labels_v2 migration (results-
hardening spec, migration appendix): the UNIQUE/AMBIG/MISS decision
structure must match pincer_r1b_labels.py v1 exactly. v1 counts
inline (len==1 -> unique; truthy -> ambig; else miss); v2 extracts
that as classify(). This test pins the behavior so the migration
never drifts from the frozen original's semantics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "scratch"))


def test_classify_matches_v1_decision_structure():
    from pincer_labels_v2 import classify
    # v1: len(names)==1 -> unique; elif names -> ambig; else miss
    assert classify(["r1"]) == "unique"
    assert classify(["r1", "r2"]) == "ambig"
    assert classify(["r1", "r1"]) == "ambig"  # duplicates: v1 counts len
    assert classify([]) == "miss"


def test_v1_source_still_carries_the_inline_logic():
    """If the frozen v1 file ever changes its decision block, this
    guard forces a human look at the v2 mirror."""
    src = (Path(__file__).resolve().parent.parent
           / "scratch" / "pincer_r1b_labels.py").read_text()
    assert "if len(names) == 1:" in src
    assert "elif names:" in src
