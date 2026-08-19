"""Fixtures for the MODEL-2 builder + engine-side precedence."""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "qwen_model2_adjudicate",
    os.path.join(REPO, "scratch/qwen_model2_adjudicate.py"))
m2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2)


def _receipt(X, K, chain=None):
    r = {"smoke": False, "device_actual": "cpu",
         "traversal": {"linear_attn": 48, "full_attn": 16},
         "teacher": {"dir": "logs/qwenteacher_m2"},
         "ce_teacher_nats": 1.1, "X": X, "K": K,
         "f_X": 3.7e-5, "f_K": 2.6e-4, "v_live": 248077}
    if chain is not None:
        r["qualification"] = {"chain_sha256": chain}
    return r


def _compose(a):
    return {"name": a, "bytes_added": m2.BAND_BYTES,
            "recipe": {"base": "FLe", "donor": "C",
                       "mark": ".linear_attn.",
                       "layers": m2.BAND_LAYERS[a],
                       "n_expected": 48},
            "promoted_keys": 48,
            "out_chain_sha256": f"chain-{a}",
            "base": {"chain_sha256": m2._frozen_chain_sha("FLe")},
            "donor": {"chain_sha256": m2._frozen_chain_sha("C")}}


def _run(x, k, break_arm=None):
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation, load)
    rc = {a: _receipt(x[a], k[a], chain=f"chain-{a}")
          for a in ("PX", "PK", "FLe", "C")}
    comp = {a: _compose(a) for a in ("PX", "PK")}
    if break_arm:
        comp[break_arm]["recipe"]["donor"] = "B"
    obs = m2.build_observations(rc, comp)
    prereg = load(os.path.join(REPO,
                               "docs/preregs/qwen-model2-alloc-1.json"))
    outs = adjudicate_prereg(prereg, obs)
    ref = adjudicate_refutation(prereg, obs, bar_outcomes=outs)
    return {o.bar_id: o.outcome for o in outs}, ref, obs


REG = dict(x={"FLe": 0.36, "PX": 0.232, "PK": 0.364, "C": 0.26},
           k={"FLe": 0.23, "PX": 0.207, "PK": 0.190, "C": 0.17})


def test_crossover_and_transport_fire():
    out, ref, obs = _run(REG["x"], REG["k"])
    assert out[1] == "FIRE" and out[2] == "FIRE" and out[3] == "FIRE"
    assert out[4] == "FIRE" and out[5] == "FIRE"
    assert ref == "NOT-REFUTED"


def test_reversed_x_leg_refutes():
    x = dict(REG["x"], PX=0.40, PK=0.30)   # late beats mid on X
    out, ref, obs = _run(x, REG["k"])
    assert out[2] == "NO-FIRE"
    assert ref == "REFUTED"


def test_sanity_miss_suppresses_refutation_via_registered_precedence():
    x = dict(REG["x"], PX=0.40, PK=0.30)
    out, ref, obs = _run(x, REG["k"], break_arm="PX")
    assert out[1] == "NO-FIRE"
    assert ref.startswith("UNADJUDICATED (precedence: bar 1")


def test_transport_drift_without_crossover_failure():
    x = dict(REG["x"], PX=0.30)  # dX(PX|FLe)=0.06, off band; crossover holds
    out, ref, obs = _run(x, REG["k"])
    assert out[2] == "FIRE" and out[4] == "NO-FIRE"
    assert ref == "NOT-REFUTED"
