"""make_split: eval first, train with exclude=eval prompts, so the two
halves are disjoint by construction and not by seed distance
(rehabilitation 2C, step 4). The metamorphic check runs 200 cases
including the low-cardinality cells (drec L1: 24 prompts, taylor L1: 15,
taylor L2: 54) where seed bands alone collide.
"""
import pytest

sp = pytest.importorskip("sympy")

from llmopt.mathgen.problems import DatasetExhausted, make_dataset, make_split

LOW_CARD = [(("drec",), (1,)), (("taylor",), (1,)), (("taylor",), (2,)),
            (("drec", "taylor"), (1, 2)), (("dsum",), (1,))]
WIDE = [(("differentiate", "integrate", "limit"), (1, 2, 3)),
        (("integrate",), (1,)), (("cdiff", "cint"), (1, 2)),
        (("tangent_line", "definite_integral"), (1, 2, 3)),
        (("second_derivative", "continuity", "limit_traced"), (1, 2))]


def _cases():
    cases = []
    for seed in range(20):
        for kinds, levels in WIDE:
            cases.append((kinds, levels, seed, 30, 10))
    for seed in range(20):
        for kinds, levels in LOW_CARD:
            n_eval = 4
            n_train = 8 if kinds == ("taylor",) and levels == (1,) else 12
            cases.append((kinds, levels, seed, n_train, n_eval))
    return cases


CASES = _cases()
assert len(CASES) == 200


@pytest.mark.parametrize("kinds,levels,seed,n_train,n_eval", CASES,
                         ids=lambda v: str(v) if not isinstance(v, tuple) else "+".join(map(str, v)))
def test_split_is_disjoint_and_full(kinds, levels, seed, n_train, n_eval):
    train, ev = make_split(n_train, n_eval, kinds=kinds, levels=levels,
                           train_seed=seed, eval_seed=seed + 1)
    tp, ep = {p.prompt for p in train}, {p.prompt for p in ev}
    assert len(train) == n_train and len(ev) == n_eval
    assert len(tp) == n_train and len(ep) == n_eval
    assert not (tp & ep)


def test_split_eval_is_the_plain_dataset_and_train_is_the_excluded_one():
    train, ev = make_split(50, 30, train_seed=0, eval_seed=99)
    assert [p.prompt for p in ev] == [p.prompt for p in make_dataset(30, seed=99)]
    expected = make_dataset(50, seed=0, exclude=frozenset(p.prompt for p in ev))
    assert [p.prompt for p in train] == [p.prompt for p in expected]


def test_split_rejects_shared_seed():
    with pytest.raises(ValueError, match="train_seed"):
        make_split(10, 5, train_seed=3, eval_seed=3)


def test_split_raises_when_the_cell_cannot_hold_both_halves():
    with pytest.raises(DatasetExhausted):
        make_split(20, 10, kinds=("drec",), levels=(1,), train_seed=0, eval_seed=1)


def test_seed_bands_alone_collide_on_low_cardinality_cells():
    """The reason make_split exists: distinct seeds do not separate."""
    a = {p.prompt for p in make_dataset(12, kinds=("drec",), levels=(1,), seed=0)}
    b = {p.prompt for p in make_dataset(12, kinds=("drec",), levels=(1,), seed=1)}
    assert a & b
