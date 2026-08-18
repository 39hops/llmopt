"""Fixtures for the ATTN-ATTRIB-1 observations builder + registered
resolution rule (scratch/qwen_attrib_adjudicate.py) — every
resolution outcome pinned on synthetic receipts before any
recomposed artifact scores."""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "qwen_attrib_adjudicate",
    os.path.join(REPO, "scratch/qwen_attrib_adjudicate.py"))
aa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aa)


def _receipt(X, K):
    return {"smoke": False, "device_actual": "cpu",
            "traversal": {"linear_attn": 48, "full_attn": 16},
            "teacher": {"dir": "logs/qwenteacher_v2"},
            "ce_teacher_nats": 1.064, "X": X, "K": K,
            "f_X": 3.7e-5, "f_K": 2.6e-4, "v_live": 248077}


def _comp(name, gib):
    return {"name": name, "bytes_added": int(gib * 2 ** 30)}


def _run(XF, XL, XQ, KF, KL, KQ):
    from llmopt.lab.prereg import adjudicate_prereg, load
    rc = {"B": _receipt(0.834, 0.338), "C": _receipt(0.249, 0.162),
          "F": _receipt(XF, KF), "L": _receipt(XL, KL),
          "Q": _receipt(XQ, KQ)}
    comp = {"F": _comp("F", 0.39), "L": _comp("L", 1.29),
            "Q": _comp("Q", 0.59)}
    obs = aa.build_observations(rc, comp)
    prereg = load(os.path.join(REPO,
                               "docs/preregs/qwen-attn-attrib-1.json"))
    outcomes = {o.bar_id: o.outcome
                for o in adjudicate_prereg(prereg, obs)}
    return aa.resolution(outcomes, obs["measurements"]), obs, outcomes


def test_l_dominant():
    (res, _), obs, _o = _run(0.70, 0.35, 0.80, 0.30, 0.20, 0.36)
    assert res == "L-DOMINANT"


def test_f_dominant():
    (res, _), obs, _o = _run(0.35, 0.70, 0.80, 0.20, 0.30, 0.36)
    assert res == "F-DOMINANT"


def test_mixed_when_metrics_disagree():
    (res, _), obs, _o = _run(0.35, 0.70, 0.80, 0.30, 0.20, 0.36)
    assert res == "MIXED/UNRESOLVED"


def test_alarm_on_bracket_violation():
    # X_F above B past floors -> recomposer alarm
    (res, _), obs, _o = _run(0.95, 0.35, 0.80, 0.30, 0.20, 0.36)
    assert res == "INSTRUMENT-ALARM"
    assert obs["measurements"]["1"]["value"] >= 1


def test_iso_bars_exclusive():
    # K_Q well above K_B -> IO-WINS-ISO-K fires, ATTN mirror does not
    (_, _), obs, out = _run(0.70, 0.35, 0.80, 0.30, 0.20, 0.40)
    assert out[8] == "FIRE" and out[9] == "NO-FIRE"
    # X_Q below X_B -> ATTN-WINS-ISO-X fires
    assert out[11] == "FIRE" and out[10] == "NO-FIRE"


def test_per_byte_extras():
    (_, _), obs, _o = _run(0.70, 0.35, 0.80, 0.30, 0.20, 0.36)
    u = obs["per_byte"]
    assert u["F"]["U_X_nat_per_gib"] == pytest.approx(
        (0.834 - 0.70) / 0.39, rel=1e-6)
    assert u["L"]["gib_added"] == pytest.approx(1.29)


def test_inadmissible_without_compose_receipt():
    from llmopt.lab.prereg import adjudicate_prereg, load
    rc = {"B": _receipt(0.834, 0.338), "C": _receipt(0.249, 0.162),
          "F": _receipt(0.7, 0.3), "L": _receipt(0.35, 0.2),
          "Q": _receipt(0.8, 0.36)}
    comp = {"F": _comp("F", 0.39), "L": _comp("L", 1.29)}  # Q missing
    obs = aa.build_observations(rc, comp)
    assert obs["arms"]["Q"]["admissible"] is False
    prereg = load(os.path.join(REPO,
                               "docs/preregs/qwen-attn-attrib-1.json"))
    out = {o.bar_id: o.outcome for o in adjudicate_prereg(prereg, obs)}
    assert out[8] == "UNRESOLVED"


def test_prereg_validates():
    from llmopt.lab.prereg import load
    doc = load(os.path.join(REPO,
                            "docs/preregs/qwen-attn-attrib-1.json"))
    assert len(doc["bars"]) == 11
