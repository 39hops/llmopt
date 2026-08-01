"""One-command reproduction of pinned llmopt results.

Usage:
    python -m llmopt.reproduce --list
    python -m llmopt.reproduce gravmoe-rb1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "scratch" / "detbwd_gmoe_ref" / "pins.json"
RUNNER = ROOT / "scratch" / "detbwd_gravmoe.py"
FINAL_PREFIX = "[gmoe] FINAL trajectory sha "
CONTRACT_ENV = {
    "ACLAMP", "COND", "DHEAD", "DIM", "E", "EXPORT", "FFN", "GATE",
    "GB", "K", "LD", "LN", "NBLK", "QK", "SCHED", "SHIFT", "SS",
    "SSW", "STEPS", "TAU", "TWIN",
}


def _arm_env(arm: str, pin: dict) -> dict[str, str]:
    """Resolve the pin schema plus its declared gravmoe cell contract."""
    env = {"RJOB_LOCAL": "1"}
    if arm.startswith("CA"):
        env["COND"] = "1"
    elif arm.startswith("GA"):
        env.update(GATE="1", COND="1")
    elif arm.startswith("RB"):
        env.update(COND="1", QK="1")
    elif arm == "GRB1":
        env.update(GATE="1", COND="1", QK="1")
    elif arm == "S1":
        env.update(GATE="1", COND="1", SS="1")
    elif arm.startswith("A"):
        env["COND"] = "0"
    else:
        raise ValueError(f"unknown gravmoe arm contract: {arm}")

    for key in ("LN", "LD", "STEPS", "SHIFT", "TAU", "SS"):
        if key in pin:
            env[key] = str(pin[key])
    return env


def available() -> dict[str, dict]:
    pins = json.loads(PINS.read_text())
    return {f"gravmoe-{arm.lower()}": {
        "arm": arm,
        "expected_sha": pin["final_sha"],
        "env": _arm_env(arm, pin),
    } for arm, pin in pins.items()}


def _runner_env(spec: dict, base: dict[str, str] | None = None) -> dict[str, str]:
    """Remove ambient experiment knobs before applying the pinned contract."""
    env = dict(os.environ if base is None else base)
    for key in CONTRACT_ENV:
        env.pop(key, None)
    env.update(spec["env"])
    env["PYTHONUNBUFFERED"] = "1"
    return env


def reproduce(name: str) -> int:
    choices = available()
    if name not in choices:
        print(f"unknown reproduction: {name}", file=sys.stderr)
        print("use --list to show available names", file=sys.stderr)
        return 2
    spec = choices[name]
    env = _runner_env(spec)
    command = [sys.executable, str(RUNNER)]
    print(f"REPRO {name} arm={spec['arm']}", flush=True)
    print("ENV " + " ".join(f"{k}={v}" for k, v in sorted(spec["env"].items())),
          flush=True)
    print(f"EXPECTED {spec['expected_sha']}", flush=True)
    started = time.monotonic()
    proc = subprocess.Popen(
        command, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    actual = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith(FINAL_PREFIX):
            actual = line[len(FINAL_PREFIX):].strip()
    rc = proc.wait()
    elapsed = time.monotonic() - started
    print(f"ELAPSED {elapsed:.1f}s")
    if rc != 0:
        print(f"FAIL {name}: runner exited {rc}")
        return 1
    if actual != spec["expected_sha"]:
        print(f"FAIL {name}: expected {spec['expected_sha']} got {actual or 'MISSING'}")
        return 1
    print(f"PASS {name} {actual}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?")
    parser.add_argument("--list", action="store_true", dest="list_repros")
    args = parser.parse_args(argv)
    choices = available()
    if args.list_repros:
        for name, spec in choices.items():
            print(f"{name:20s} {spec['expected_sha']}")
        return 0
    if not args.name:
        parser.error("provide a reproduction name or --list")
    return reproduce(args.name)


if __name__ == "__main__":
    raise SystemExit(main())
