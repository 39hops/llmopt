"""Minimal reproduction registry tests (the full SHA arm is an acceptance run)."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

from llmopt.reproduce import ROOT, _runner_env, available


def test_gravmoe_rb1_contract_and_pin():
    spec = available()["gravmoe-rb1"]
    assert {key: spec["env"][key] for key in (
        "RJOB_LOCAL", "COND", "QK", "LN", "LD", "STEPS"
    )} == {"RJOB_LOCAL": "1", "COND": "1", "QK": "1",
           "LN": "0", "LD": "1", "STEPS": "2000"}
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


def test_all_pins_select_committed_sha_checked_window_family():
    for spec in available().values():
        gate = spec["env"].get("GATE") == "1"
        family = "grb1" if gate else "rb1"
        windows_path = Path(spec["windows_path"])
        contract_path = Path(spec["contract_path"])

        assert windows_path == (
            ROOT / "scratch" / "detbwd_gmoe_ref" / f"{family}_windows.bin"
        )
        assert contract_path == (
            ROOT / "scratch" / "detbwd_gmoe_ref" / f"{family}_contract.json"
        )
        assert windows_path.is_file() and contract_path.is_file()
        contract = json.loads(contract_path.read_text())
        assert hashlib.sha256(windows_path.read_bytes()).hexdigest() == \
            contract["windows_sha"]
        assert spec["env"]["WINDOWS_BIN"] == str(windows_path)
        assert spec["env"]["WINDOWS_CONTRACT"] == str(contract_path)
        assert spec["env"].get("TRAJECTORY_ONLY") == ("1" if gate else None)


def test_shared_nongate_windows_are_contract_identical():
    # A0/CA0/RB1 exports share their exact committed window bytes, so the
    # RB1 window family is valid for every non-gate arm by contract SHA.
    ref_dir = ROOT / "scratch" / "detbwd_gmoe_ref"
    expected = json.loads((ref_dir / "rb1_contract.json").read_text())[
        "windows_sha"
    ]
    for family in ("a0", "ca0", "rb1"):
        contract = json.loads((ref_dir / f"{family}_contract.json").read_text())
        assert contract["windows_sha"] == expected


def test_s1_stays_gate_scheduled_sampling_on_committed_windows():
    spec = available()["gravmoe-s1"]
    assert spec["env"]["GATE"] == "1"
    assert spec["env"]["SS"] == "1"
    assert spec["env"]["TRAJECTORY_ONLY"] == "1"
    assert Path(spec["windows_path"]).name == "grb1_windows.bin"
    assert "DIET" not in " ".join(spec["env"])


def test_ambient_experiment_knobs_cannot_poison_pin():
    spec = available()["gravmoe-rb1"]
    env = _runner_env(spec, {"PATH": "/bin", "DIM": "999", "SHIFT": "3",
                             "GATE": "1", "EXPORT": "bad",
                             "ANSWER_ONLY": "1",
                             "WINDOWS_BIN": "/tmp/hostile.bin",
                             "WINDOWS_CONTRACT": "/tmp/hostile.json",
                             "TRAJECTORY_ONLY": "1"})
    assert env["PATH"] == "/bin"
    assert "DIM" not in env and "SHIFT" not in env
    assert "GATE" not in env and "EXPORT" not in env
    assert "ANSWER_ONLY" not in env
    assert env["COND"] == "1" and env["QK"] == "1"
    assert env["WINDOWS_BIN"] == spec["env"]["WINDOWS_BIN"]
    assert env["WINDOWS_CONTRACT"] == spec["env"]["WINDOWS_CONTRACT"]
    assert "TRAJECTORY_ONLY" not in env


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
