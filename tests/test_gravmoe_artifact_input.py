import json
import os
import subprocess
import sys

import pytest

from llmopt.reproduce import CONTRACT_ENV, ROOT


RUNNER = ROOT / "scratch" / "detbwd_gravmoe.py"
REF = ROOT / "scratch" / "detbwd_gmoe_ref"


def runner_env(**updates):
    env = dict(os.environ)
    for key in CONTRACT_ENV:
        env.pop(key, None)
    env.update(RJOB_LOCAL="1", STEPS="1", **updates)
    return env


def run_runner(env):
    return subprocess.run(
        [sys.executable, str(RUNNER)], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=60,
    )


def test_runner_refuses_artifact_with_wrong_raw_sha(tmp_path):
    contract = json.loads((REF / "rb1_contract.json").read_text())
    contract["windows_sha"] = "0" * 64
    bad_contract = tmp_path / "contract.json"
    bad_contract.write_text(json.dumps(contract))

    proc = run_runner(runner_env(
        WINDOWS_BIN=str(REF / "rb1_windows.bin"),
        WINDOWS_CONTRACT=str(bad_contract),
    ))

    assert proc.returncode != 0
    assert "window artifact SHA mismatch" in proc.stdout + proc.stderr


def test_runner_trains_from_valid_artifact_without_diet_sidecar():
    proc = run_runner(runner_env(
        WINDOWS_BIN=str(REF / "rb1_windows.bin"),
        WINDOWS_CONTRACT=str(REF / "rb1_contract.json"),
    ))

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[gmoe] artifact windows" in proc.stdout
    assert "[gmoe] cycle-mean" in proc.stdout


@pytest.mark.parametrize("provided", ["WINDOWS_BIN", "WINDOWS_CONTRACT"])
def test_runner_requires_artifact_paths_together(provided):
    value = REF / (
        "rb1_windows.bin" if provided == "WINDOWS_BIN" else "rb1_contract.json"
    )
    proc = run_runner(runner_env(**{provided: str(value)}))

    assert proc.returncode != 0
    assert "WINDOWS_BIN and WINDOWS_CONTRACT must be provided together" in (
        proc.stdout + proc.stderr
    )


def test_trajectory_only_requires_artifact_inputs():
    proc = run_runner(runner_env(TRAJECTORY_ONLY="1"))

    assert proc.returncode != 0
    assert "TRAJECTORY_ONLY requires committed artifact inputs" in (
        proc.stdout + proc.stderr
    )


def test_trajectory_only_requires_gate_mode():
    proc = run_runner(runner_env(
        TRAJECTORY_ONLY="1",
        WINDOWS_BIN=str(REF / "rb1_windows.bin"),
        WINDOWS_CONTRACT=str(REF / "rb1_contract.json"),
    ))

    assert proc.returncode != 0
    assert "TRAJECTORY_ONLY requires GATE=1" in proc.stdout + proc.stderr


def test_gate_artifact_mode_requires_explicit_trajectory_only():
    proc = run_runner(runner_env(
        GATE="1",
        WINDOWS_BIN=str(REF / "grb1_windows.bin"),
        WINDOWS_CONTRACT=str(REF / "grb1_contract.json"),
    ))

    assert proc.returncode != 0
    assert "gate artifact inputs require TRAJECTORY_ONLY=1" in (
        proc.stdout + proc.stderr
    )


def test_s1_derives_marker_and_recorded_splits_from_artifact_rows():
    proc = run_runner(runner_env(
        GATE="1", COND="1", SS="1", TRAJECTORY_ONLY="1",
        WINDOWS_BIN=str(REF / "grb1_windows.bin"),
        WINDOWS_CONTRACT=str(REF / "grb1_contract.json"),
    ))

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[gmoe] marker ids [4, 26]" in proc.stdout
    assert "splits [15, 10, 15, 15, 19, 15, 12, 15]" in proc.stdout
    assert "SymPy solve scoring requires the uncommitted row text" in proc.stdout
    assert "[gate] TRAIN" not in proc.stdout
    assert "[gate] HELDOUT" not in proc.stdout
