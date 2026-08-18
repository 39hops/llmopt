"""Fixtures for the IO-ATTRIB-1 builder + resolution rule."""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "qwen_ioattrib_adjudicate",
    os.path.join(REPO, "scratch/qwen_ioattrib_adjudicate.py"))
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)


def _receipt(X, K):
    return {"smoke": False, "device_actual": "cpu",
            "traversal": {"linear_attn": 48, "full_attn": 16},
            "teacher": {"dir": "logs/qwenteacher_v2"},
            "ce_teacher_nats": 1.064, "X": X, "K": K,
            "f_X": 3.7e-5, "f_K": 2.6e-4, "v_live": 248077}


def _run(XD, XE, KD, KE):
    from llmopt.lab.prereg import adjudicate_prereg, load
    rc = {"A": _receipt(1.061, 0.472), "B": _receipt(0.834, 0.338),
          "D": _receipt(XD, KD), "E": _receipt(XE, KE)}
    comp = {a: {"name": a, "bytes_added": 317847584} for a in "DE"}
    obs = io.build_observations(rc, comp)
    prereg = load(os.path.join(REPO,
                               "docs/preregs/qwen-io-attrib-1.json"))
    out = {o.bar_id: o.outcome for o in adjudicate_prereg(prereg, obs)}
    return io.resolution(out, obs["measurements"]), obs, out


def test_d_dominant():
    (res, _), obs, _ = _run(0.90, 1.00, 0.38, 0.44)
    assert res == "D-DOMINANT"


def test_e_dominant():
    (res, _), obs, _ = _run(1.00, 0.90, 0.44, 0.38)
    assert res == "E-DOMINANT"


def test_mixed():
    (res, _), obs, _ = _run(0.90, 1.00, 0.44, 0.38)
    assert res == "MIXED/UNRESOLVED"


def test_alarm_on_out_of_bracket():
    (res, _), obs, _ = _run(1.30, 1.00, 0.38, 0.44)
    assert res == "INSTRUMENT-ALARM"
