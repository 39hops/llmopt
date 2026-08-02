import hashlib
import json
import os
import struct
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


def run_runner(env, *, optimized=False):
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.append(str(RUNNER))
    return subprocess.run(
        command, cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=60,
    )


def write_sha_consistent_rows(tmp_path, rows):
    rows_raw = b"".join(struct.pack("<33q", *row) for row in rows)
    windows_raw = b"".join(
        struct.pack("<64q", *(row[:32] + row[1:])) for row in rows
    )
    contract = json.loads((REF / "grb1_contract.json").read_text())
    contract["windows_sha"] = hashlib.sha256(windows_raw).hexdigest()
    contract["windows_rows_sha"] = hashlib.sha256(rows_raw).hexdigest()
    windows_path = tmp_path / "windows.bin"
    contract_path = tmp_path / "contract.json"
    windows_path.write_bytes(windows_raw)
    contract_path.write_text(json.dumps(contract))
    return windows_path, contract_path


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


def test_optimized_runner_refuses_sha_consistent_s1_split_drift(tmp_path):
    values = struct.unpack(
        "<512q", (REF / "grb1_windows.bin").read_bytes()
    )
    rows = []
    for offset in range(0, len(values), 64):
        tok = list(values[offset:offset + 32])
        tgt = values[offset + 32:offset + 64]
        rows.append(tok + [tgt[-1]])
    rows[0][16:18] = [4, 26]  # Later valid marker moves split 15 to split 18.
    windows_path, contract_path = write_sha_consistent_rows(tmp_path, rows)

    proc = run_runner(runner_env(
        GATE="1", COND="1", SS="1", TRAJECTORY_ONLY="1",
        WINDOWS_BIN=str(windows_path),
        WINDOWS_CONTRACT=str(contract_path),
    ), optimized=True)

    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "S1 split-position contract violated" in proc.stdout + proc.stderr
