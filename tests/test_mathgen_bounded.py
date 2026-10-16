"""make_dataset is bounded: an exhausted prompt space raises a typed
error with the cell diagnostics instead of looping forever, invalid
arguments are rejected up front, and successful generation is unchanged
(rehabilitation 2C). Low-cardinality cells measured 2026-10-16: drec L1
has 24 distinct prompts over 600 seeds, taylor L1 has 15.
"""
import time

import pytest

sp = pytest.importorskip("sympy")

from llmopt.mathgen.problems import DatasetExhausted, make_dataset


def _all_prompts(kind, level, seeds=600):
    from llmopt.mathgen.problems import _resolve_maker
    mk = _resolve_maker(kind)
    return frozenset(mk(level, s).prompt for s in range(seeds))


def test_exclude_all_raises_within_two_seconds():
    excl = _all_prompts("drec", 1)
    assert len(excl) == 24
    t = time.monotonic()
    with pytest.raises(DatasetExhausted):
        make_dataset(1, kinds=("drec",), levels=(1,), seed=0, exclude=excl)
    assert time.monotonic() - t < 2.0


def test_request_beyond_cardinality_raises():
    with pytest.raises(DatasetExhausted):
        make_dataset(30, kinds=("drec",), levels=(1,), seed=0)


def test_exhaustion_error_carries_diagnostics():
    excl = _all_prompts("taylor", 1)
    with pytest.raises(DatasetExhausted) as info:
        make_dataset(5, kinds=("taylor",), levels=(1,), seed=3, exclude=excl)
    err = info.value
    assert err.n == 5
    assert err.produced == 0
    assert err.seed == 3
    assert err.exclude_size == len(excl)
    assert err.consecutive_rejects == err.max_consecutive_rejects
    assert err.attempts >= err.consecutive_rejects
    assert err.cells == (("taylor", 1),)
    assert err.rejects_by_cell[("taylor", 1)] == err.attempts
    msg = str(err)
    for token in ("taylor", "level 1", "n=5", "exclude=15", "seed=3"):
        assert token in msg, msg


def test_partial_output_is_reported_not_returned():
    # 24 prompts exist; asking for 30 produces 24 then exhausts.
    with pytest.raises(DatasetExhausted) as info:
        make_dataset(30, kinds=("drec",), levels=(1,), seed=0)
    assert info.value.produced == 24
    assert len(info.value.partial) == 24
    assert len({p.prompt for p in info.value.partial}) == 24


def test_cap_is_overridable():
    excl = _all_prompts("drec", 1)
    with pytest.raises(DatasetExhausted) as info:
        make_dataset(1, kinds=("drec",), levels=(1,), exclude=excl,
                     max_consecutive_rejects=7)
    assert info.value.attempts == 7


@pytest.mark.parametrize("kw,match", [
    (dict(n=-1), "n must be"),
    (dict(n=3, kinds=()), "kinds"),
    (dict(n=3, levels=()), "levels"),
    (dict(n=3, max_consecutive_rejects=0), "max_consecutive_rejects"),
    (dict(n=3, kinds=("no_such_kind",)), "no_such_kind"),
])
def test_invalid_arguments_are_rejected_before_generation(kw, match):
    with pytest.raises((ValueError, KeyError), match=match):
        make_dataset(**kw)


def test_zero_rows_is_empty():
    assert make_dataset(0) == []


# --- metamorphic / property checks on successful generation -----------

@pytest.mark.parametrize("seed", range(5))
def test_prefix_property(seed):
    short = make_dataset(20, seed=seed)
    long = make_dataset(45, seed=seed)
    assert [p.prompt for p in long[:20]] == [p.prompt for p in short]


@pytest.mark.parametrize("seed", range(5))
def test_outputs_are_unique_and_disjoint_from_exclude(seed):
    ev = make_dataset(40, seed=seed + 100)
    excl = frozenset(p.prompt for p in ev)
    tr = make_dataset(60, seed=seed, exclude=excl)
    prompts = [p.prompt for p in tr]
    assert len(set(prompts)) == 60
    assert not (set(prompts) & excl)


def test_unrelated_exclusions_do_not_change_output():
    base = make_dataset(40, seed=11)
    noisy = make_dataset(40, seed=11, exclude=frozenset({"not a prompt", "x"}))
    assert [p.prompt for p in base] == [p.prompt for p in noisy]


def test_positional_signature_unchanged():
    # Historical callers pass n positionally and everything else by keyword.
    ds = make_dataset(3, kinds=("differentiate",), levels=(1,), seed=2)
    assert [p.kind for p in ds] == ["differentiate"] * 3
