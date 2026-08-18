"""Fixtures for the LBAND-1 builder + bar-wise resolution."""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "qwen_lband_adjudicate",
    os.path.join(REPO, "scratch/qwen_lband_adjudicate.py"))
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)


def _receipt(X, K):
    return {"smoke": False, "device_actual": "cpu",
            "traversal": {"linear_attn": 48, "full_attn": 16},
            "teacher": {"dir": "logs/qwenteacher_v2"},
            "ce_teacher_nats": 1.064, "X": X, "K": K,
            "f_X": 3.7e-5, "f_K": 2.6e-4, "v_live": 248077}


def _run(band_x, band_k):
    """band_x/band_k: dicts arm -> (X, K) for the six arms."""
    from llmopt.lab.prereg import adjudicate_prereg, load
    rc = {"B": _receipt(0.834, 0.338), "C": _receipt(0.249, 0.162),
          "F": _receipt(0.520, 0.264)}
    for a in lb.B_ARMS + lb.F_ARMS:
        rc[a] = _receipt(band_x[a], band_k[a])
    comp = {a: {"name": a, "bytes_added": 461373440}
            for a in lb.B_ARMS + lb.F_ARMS}
    obs = lb.build_observations(rc, comp)
    prereg = load(os.path.join(REPO, "docs/preregs/qwen-lband-1.json"))
    out = {o.bar_id: o.outcome for o in adjudicate_prereg(prereg, obs)}
    return lb.resolution(out), obs, out


def test_structure_and_conditioning():
    # late band clearly best from B; F-conditional value differs
    bx = {"BLe": 0.80, "BLm": 0.75, "BLl": 0.60,
          "FLe": 0.50, "FLm": 0.46, "FLl": 0.42}
    bk = {"BLe": 0.33, "BLm": 0.30, "BLl": 0.25,
          "FLe": 0.26, "FLm": 0.24, "FLl": 0.20}
    (res, _), obs, out = _run(bx, bk)
    assert out[1] == "FIRE" and out[2] == "FIRE"
    assert "B-X STRUCTURE" in res and "COND-X FIRES" in res
    # refutation gap in rec units: (0.234 - 0.084)/0.585 > 0.2
    assert obs["band_gap_rec_units_baseB"] > 0.2


def test_flat_at_grain():
    # bands within a floor of each other -> structure bars quiet
    bx = {"BLe": 0.700000, "BLm": 0.700010, "BLl": 0.700020,
          "FLe": 0.470000, "FLm": 0.470010, "FLl": 0.470020}
    bk = {"BLe": 0.300000, "BLm": 0.300010, "BLl": 0.300005,
          "FLe": 0.250000, "FLm": 0.250008, "FLl": 0.250004}
    (res, _), obs, out = _run(bx, bk)
    assert out[1] == "FIRE" and out[2] == "NO-FIRE"
    assert "B-X FLAT" in res


def test_alarm_on_out_of_bracket():
    bx = {"BLe": 0.90, "BLm": 0.75, "BLl": 0.60,  # BLe above B+5f
          "FLe": 0.50, "FLm": 0.46, "FLl": 0.42}
    bk = {"BLe": 0.33, "BLm": 0.30, "BLl": 0.25,
          "FLe": 0.26, "FLm": 0.24, "FLl": 0.20}
    (res, why), obs, out = _run(bx, bk)
    assert res == "INSTRUMENT-ALARM"


def test_conditioning_pairs_best_b_band():
    # best B band on X is BLl; conditioning must compare FLl v BLl
    bx = {"BLe": 0.80, "BLm": 0.75, "BLl": 0.60,
          "FLe": 0.50, "FLm": 0.46, "FLl": 0.42}
    bk = {"BLe": 0.33, "BLm": 0.30, "BLl": 0.25,
          "FLe": 0.26, "FLm": 0.24, "FLl": 0.20}
    _, obs, _ = _run(bx, bk)
    assert "BLl" in obs["measurements"]["5"]["provenance"]
    dX = obs["dX"]
    expect = abs(dX["FLl"] - dX["BLl"]) / obs["f_X"]
    assert abs(obs["measurements"]["5"]["value"] - expect) < 1e-9
