"""Minimal reproduction registry tests (the full SHA arm is an acceptance run)."""
import os
import subprocess

from llmopt.reproduce import ROOT, _runner_env, available


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
                             "GATE": "1", "EXPORT": "bad",
                             "ANSWER_ONLY": "1"})
    assert env["PATH"] == "/bin"
    assert "DIM" not in env and "SHIFT" not in env
    assert "GATE" not in env and "EXPORT" not in env
    assert "ANSWER_ONLY" not in env
    assert env["COND"] == "1" and env["QK"] == "1"


def test_pin_launcher_scrubs_ambient_answer_only(tmp_path):
    audit = tmp_path / "answer-only-audit.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"${ANSWER_ONLY-unset}\" >> \"$ANSWER_ONLY_AUDIT\"\n"
        "printf '[gmoe] FINAL trajectory sha fake\\n'\n"
    )
    fake_python.chmod(0o755)

    source = (ROOT / "scratch" / "p4_arms_0801.sh").read_text()
    launcher_source = source.replace(
        "PY=.venv/bin/python", f"PY={fake_python}", 1)
    assert launcher_source != source
    launcher = tmp_path / "p4_arms_0801.sh"
    launcher.write_text(launcher_source)

    env = dict(os.environ)
    env.update(ANSWER_ONLY="1", ANSWER_ONLY_AUDIT=str(audit))
    proc = subprocess.run(
        ["bash", str(launcher)], cwd=tmp_path, env=env,
        text=True, capture_output=True,
    )
    assert proc.returncode == 1  # The fake runtime deliberately emits fake SHAs.
    assert audit.read_text().splitlines() == ["unset"] * 16
