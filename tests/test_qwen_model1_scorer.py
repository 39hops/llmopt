"""Fixtures for the MODEL1-TREE scorer math and the mechanical tree
walker (scratch/qwen_model1_score.py, scratch/qwen_tree_adjudicate.py)
— pinned BEFORE any 27B forward runs (qualification ladder: the model
run is never the first test of new code).

Every branch of the registered tree is exercised on synthetic
receipts: T1/T2/T3/T4, NONMONOTONIC, INSTRUMENT-ALARM, the degenerate
X_Z <= 0 stop, and the 5x-floor fence turning a 20% winner into
NO-FIRE.
"""
import importlib.util
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load("qwen_model1_score", "scratch/qwen_model1_score.py")
tr = _load("qwen_tree_adjudicate", "scratch/qwen_tree_adjudicate.py")


# ------------------------------------------------------------- math
def test_mean_ce_hand_value():
    # 3 positions, vocab 2 live of 3; logits[:-1] v ids[1:]
    lg = np.array([[0.0, 0.0, 99.0],
                   [2.0, 0.0, 99.0],
                   [5.0, 5.0, 5.0]], np.float32)
    ids = [0, 1, 0]
    # pos0 -> target 1: CE = log(1+1) = log 2
    # pos1 -> target 0: CE = log(1+e^-2)
    want = (np.log(2.0) + np.log1p(np.exp(-2.0))) / 2
    assert sc.mean_ce(lg, ids, v_live=2) == pytest.approx(want, rel=1e-12)


def test_mean_ce_refuses_nonfinite_and_dead_target():
    lg = np.array([[0.0, np.inf], [0.0, 0.0]], np.float32)
    with pytest.raises(SystemExit, match="non-finite"):
        sc.mean_ce(lg, [0, 1], 2)
    with pytest.raises(SystemExit, match="live vocab"):
        sc.mean_ce(np.zeros((2, 3), np.float32), [0, 2], 2)


def test_forward_kl_zero_on_identity_and_hand_value():
    lg = np.random.default_rng(0).normal(size=(4, 5)).astype(np.float32)
    assert sc.mean_forward_kl(lg, lg, 5) == pytest.approx(0.0, abs=1e-12)
    t = np.array([[np.log(0.75), np.log(0.25)]], np.float32)
    a = np.array([[np.log(0.5), np.log(0.5)]], np.float32)
    want = 0.75 * np.log(1.5) + 0.25 * np.log(0.5)
    assert sc.mean_forward_kl(t, a, 2) == pytest.approx(want, rel=1e-6)


def test_kl_direction_is_forward():
    # teacher concentrated, arm flat: forward KL weighs teacher mass
    t = np.array([[10.0, 0.0]], np.float32)
    a = np.array([[0.0, 0.0]], np.float32)
    fwd = sc.mean_forward_kl(t, a, 2)
    rev = sc.mean_forward_kl(a, t, 2)
    assert fwd != pytest.approx(rev, rel=1e-3)  # direction matters


def test_sensitivity_floor_positive_and_small():
    rec = np.random.default_rng(1).normal(size=(6, 8)) \
        .astype(np.float16)
    ids = list(range(6))

    def fn(a):
        return sc.mean_ce(a, ids[:6], 8)
    f = sc.sensitivity_floor(fn, rec)
    assert 0 < f < 1e-2


def test_margins_top1_and_flip_table():
    tl = np.array([[1.0, 0.99, 0.0],       # margin 0.01 -> bin 0
                   [3.0, 1.0, 0.0],        # margin 2.0  -> bin 7
                   [0.5, 0.0, 0.47]], np.float16)  # margin 0.03 -> bin 1
    top1, m = sc.teacher_margins_top1(tl, v_live=3)
    assert top1.tolist() == [0, 0, 0]
    assert m == pytest.approx([0.01, 2.0, 0.03], abs=1e-3)
    tab = sc.flip_table(top1, m, np.array([1, 0, 0]))
    assert tab[0] == [1, 1] and tab[7] == [1, 0] and tab[1] == [1, 0]
    assert sum(n for n, _ in tab) == 3


def test_margin_bin_edges():
    assert sc.margin_bin(0.0) == 0
    assert sc.margin_bin(0.02) == 1
    assert sc.margin_bin(4.99) == 7
    assert sc.margin_bin(5.0) == 8
    assert sc.margin_bin(1e9) == 8


# ------------------------------------------------- tree projection
def _receipt(X, K, f_X=1e-6, f_K=1e-6, **over):
    r = {"smoke": False, "device_actual": "cpu",
         "traversal": {"linear_attn": 48, "full_attn": 16},
         "qualification": {"ok": True},
         "teacher": {"dir": "logs/qwenteacher_v2"},
         "ce_teacher_nats": 2.5, "X": X, "K": K,
         "f_X": f_X, "f_K": f_K, "v_live": 151669}
    r.update(over)
    return r


def _run(XA, XB, XC, KA, KB, KC, **kw):
    from llmopt.lab.prereg import adjudicate_prereg, load
    rc = {"A": _receipt(XA, KA, **kw), "B": _receipt(XB, KB, **kw),
          "C": _receipt(XC, KC, **kw)}
    obs = tr.build_observations(rc)
    prereg = load(os.path.join(REPO, "docs/preregs/qwen-model1-tree.json"))
    outcomes = {o.bar_id: o.outcome
                for o in adjudicate_prereg(prereg, obs)}
    return tr.walk(outcomes), obs


def test_branch_t1():
    (b, _), _o = _run(0.5, 0.3, 0.28, 0.5, 0.3, 0.28)
    assert b == "T1"


def test_branch_t2():
    (b, _), _o = _run(0.5, 0.4, 0.2, 0.5, 0.4, 0.2)
    assert b == "T2"


def test_branch_t3_parity():
    (b, _), _o = _run(0.50, 0.48, 0.46, 0.50, 0.48, 0.46)
    assert b == "T3"


def test_branch_t4_cumulative():
    # two ~13% steps: neither crosses 20%, cumulative ~24% does
    (b, _), _o = _run(0.50, 0.435, 0.378, 0.50, 0.435, 0.378)
    assert b == "T4"


def test_branch_nonmonotonic():
    (b, r), _o = _run(0.3, 0.5, 0.5, 0.3, 0.3, 0.3)
    assert b == "NONMONOTONIC"


def test_instrument_alarm_uniform_damage():
    (b, r), _o = _run(1.5, 1.2, 1.1, 0.5, 0.4, 0.35)
    assert b == "INSTRUMENT-ALARM"


def test_instrument_alarm_alignment():
    from llmopt.lab.prereg import adjudicate_prereg, load
    rc = {a: _receipt(0.5, 0.5) for a in "ABC"}
    for a in rc:
        rc[a]["ce_teacher_nats"] = 12.4
    obs = tr.build_observations(rc)
    prereg = load(os.path.join(REPO, "docs/preregs/qwen-model1-tree.json"))
    outcomes = {o.bar_id: o.outcome
                for o in adjudicate_prereg(prereg, obs)}
    assert tr.walk(outcomes)[0] == "INSTRUMENT-ALARM"


def test_degenerate_x_books_unresolved():
    (b, _), obs = _run(-0.01, -0.02, -0.02, 0.5, 0.3, 0.3)
    assert b == "UNRESOLVED"
    assert any(not c["admissible"] for c in obs["contrasts"].values())


def test_floor_fence_blocks_trigger():
    # 40% relative improvement but delta 0.2 under a 0.1 floor
    # (needs > 5x = 0.5): the step must NOT fire -> parity path
    (b, _), _o = _run(0.5, 0.3, 0.3, 0.5, 0.3, 0.3, f_X=0.1, f_K=0.1)
    assert b == "T3"


def test_measurement_invalid_when_receipts_disagree():
    rc = {a: _receipt(0.5, 0.5) for a in "ABC"}
    rc["B"]["ce_teacher_nats"] = 2.6
    obs = tr.build_observations(rc)
    assert obs["measurement_valid"] is False


def test_inadmissible_arm_unresolves_tree():
    from llmopt.lab.prereg import adjudicate_prereg, load
    rc = {a: _receipt(0.5, 0.4) for a in "ABC"}
    rc["C"]["traversal"] = {"linear_attn": 47, "full_attn": 16}
    obs = tr.build_observations(rc)
    prereg = load(os.path.join(REPO, "docs/preregs/qwen-model1-tree.json"))
    outcomes = {o.bar_id: o.outcome
                for o in adjudicate_prereg(prereg, obs)}
    assert tr.walk(outcomes)[0] in ("UNRESOLVED", "INSTRUMENT-ALARM")


def test_prereg_document_validates():
    from llmopt.lab.prereg import load
    doc = load(os.path.join(REPO, "docs/preregs/qwen-model1-tree.json"))
    assert len(doc["bars"]) == 12


def test_arm_dependent_f_k_does_not_invalidate():
    rc = {a: _receipt(0.5, 0.4) for a in "ABC"}
    rc["A"]["f_K"] = 2.6e-4
    rc["B"]["f_K"] = 5.3e-5
    obs = tr.build_observations(rc)
    assert obs["measurement_valid"] is True
    assert obs["f_K"] == pytest.approx(2.6e-4)
