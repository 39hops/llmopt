"""Minimal reproduction registry tests (the full SHA arm is an acceptance run)."""
from llmopt.reproduce import _runner_env, available


def test_gravmoe_rb1_contract_and_pin():
    spec = available()["gravmoe-rb1"]
    assert spec["env"] == {
        "RJOB_LOCAL": "1", "COND": "1", "QK": "1",
        "LN": "0", "LD": "1", "STEPS": "2000",
    }
    assert spec["expected_sha"].startswith("c6766da235cf0b76")


def test_gravmoe_grb1_includes_gate_and_qk():
    spec = available()["gravmoe-grb1"]
    assert spec["env"]["GATE"] == "1"
    assert spec["env"]["COND"] == "1"
    assert spec["env"]["QK"] == "1"


def test_all_pinned_arms_are_exposed():
    choices = available()
    assert len(choices) == 16
    assert all(len(spec["expected_sha"]) == 64 for spec in choices.values())


def test_ambient_experiment_knobs_cannot_poison_pin():
    spec = available()["gravmoe-rb1"]
    env = _runner_env(spec, {"PATH": "/bin", "DIM": "999", "SHIFT": "3",
                             "GATE": "1", "EXPORT": "bad"})
    assert env["PATH"] == "/bin"
    assert "DIM" not in env and "SHIFT" not in env
    assert "GATE" not in env and "EXPORT" not in env
    assert env["COND"] == "1" and env["QK"] == "1"
