"""Pinned-reproduction input validation (rehabilitation 2C, step 6):
missing files and malformed pins fail with repo-bound messages before
any runner starts, and every environment key the pinned runner tree
reads is covered by the CONTRACT_ENV scrub.
"""
import ast
import json
from pathlib import Path

import pytest

from llmopt import reproduce as R

RUNNER_TREE = [
    "scratch/detbwd_gravmoe.py", "scratch/detbwd_mb.py",
    "scratch/detbwd_diet.py", "scratch/detbwd_r1.py",
    "scratch/detbwd_r2b.py", "scratch/detbwd_r3_qw.py",
    "llmopt/window_artifact.py",
]
# Set by the launcher itself, never ambient: not part of the scrub set.
LAUNCHER_OWNED = {"RJOB_LOCAL", "PYTHONUNBUFFERED"}


def _environ_keys(src: str) -> set:
    """Every key read through os.environ / os.getenv, with a literal
    loop over a tuple of constants expanded; anything dynamic is
    returned as '<dynamic:...>' so the test fails loudly on it."""
    tree = ast.parse(src)
    loop_vars = {}
    aliases = {"environ"}  # names bound to os.environ (plain or via __import__)
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension) and isinstance(node.iter, ast.Tuple) \
                and isinstance(node.target, ast.Name):
            loop_vars[node.target.id] = [c.value for c in node.iter.elts
                                         if isinstance(c, ast.Constant)]
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) \
                and node.value.attr == "environ":
            aliases |= {tg.id for tg in node.targets if isinstance(tg, ast.Name)}
    keys = set()

    def is_environ(expr) -> bool:
        return (isinstance(expr, ast.Attribute) and expr.attr == "environ") or \
            (isinstance(expr, ast.Name) and expr.id in aliases)

    def key_of(expr):
        if isinstance(expr, ast.Constant):
            return {expr.value}
        if isinstance(expr, ast.Name) and expr.id in loop_vars:
            return set(loop_vars[expr.id])
        return {f"<dynamic:{ast.unparse(expr)}>"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            f = node.func
            env_method = is_environ(f.value) and f.attr in ("get", "setdefault", "pop")
            getenv = isinstance(f.value, ast.Name) and f.value.id == "os" and f.attr == "getenv"
            if (env_method or getenv) and node.args:
                keys |= key_of(node.args[0])
        if isinstance(node, ast.Subscript) and is_environ(node.value):
            keys |= key_of(node.slice)
    return keys


def test_every_runner_environment_key_is_scrubbed():
    read = set()
    for rel in RUNNER_TREE:
        read |= _environ_keys((R.ROOT / rel).read_text())
    dynamic = {k for k in read if str(k).startswith("<dynamic")}
    assert not dynamic, f"unresolvable environ reads: {dynamic}"
    uncovered = read - R.CONTRACT_ENV - LAUNCHER_OWNED
    assert not uncovered, f"runner reads env keys the pin scrub does not clear: {uncovered}"


def test_contract_env_keys_are_all_read_or_documented():
    read = set()
    for rel in RUNNER_TREE:
        read |= _environ_keys((R.ROOT / rel).read_text())
    assert R.CONTRACT_ENV - read == set(), (
        "CONTRACT_ENV names keys nothing in the runner tree reads: "
        f"{sorted(R.CONTRACT_ENV - read)}")


def test_scanner_sees_environ_aliases():
    src = ('import os\n_env = __import__("os").environ\nE2 = os.environ\n'
           'a = _env.get("ALPHA")\nb = E2["BETA"]\nc = os.getenv("GAMMA")\n'
           'd = os.environ.setdefault("DELTA", "1")\n')
    assert _environ_keys(src) == {"ALPHA", "BETA", "GAMMA", "DELTA"}


# --- missing files ----------------------------------------------------------

def test_missing_pins_file_names_the_repo_path(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "PINS", tmp_path / "pins.json")
    with pytest.raises(R.ReproduceInputError) as info:
        R.available()
    msg = str(info.value)
    assert "pins.json" in msg and "scratch/detbwd_gmoe_ref" in msg
    assert "git checkout" in msg and str(tmp_path) in msg


def test_missing_runner_is_reported_before_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "RUNNER", tmp_path / "detbwd_gravmoe.py")
    launched = []
    monkeypatch.setattr(R.subprocess, "Popen",
                        lambda *a, **k: launched.append(a) or pytest.fail("launched"))
    rc = R.main(["gravmoe-rb1"])
    assert rc == 2 and not launched


def test_missing_windows_artifact_is_reported_before_launch(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "REFERENCE_DIR", tmp_path)
    pins_copy = tmp_path / "pins.json"
    pins_copy.write_text(R.PINS.read_text())
    monkeypatch.setattr(R, "PINS", pins_copy)
    monkeypatch.setattr(R.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("launched"))
    rc = R.main(["gravmoe-rb1"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "rb1_windows.bin" in err


def test_unknown_name_still_returns_two(capsys):
    assert R.main(["gravmoe-nope"]) == 2
    assert "unknown reproduction" in capsys.readouterr().err


# --- malformed pins ---------------------------------------------------------

@pytest.mark.parametrize("mutate,match", [
    (lambda pins: pins.__setitem__("RB1", {**pins["RB1"], "final_sha": "abc"}), "final_sha"),
    (lambda pins: pins["RB1"].pop("final_sha"), "final_sha"),
    (lambda pins: pins.__setitem__("RB1", {**pins["RB1"], "STEPS": "2000x"}), "STEPS"),
    (lambda pins: pins.__setitem__("ZZ9", pins["RB1"]), "ZZ9"),
    (lambda pins: pins.clear(), "no arms"),
], ids=["short_sha", "no_sha", "steps_not_int", "unknown_arm", "empty"])
def test_malformed_pins_raise_typed_error(monkeypatch, tmp_path, mutate, match):
    pins = json.loads(R.PINS.read_text())
    mutate(pins)
    bad = tmp_path / "pins.json"
    bad.write_text(json.dumps(pins))
    monkeypatch.setattr(R, "PINS", bad)
    with pytest.raises(R.PinSchemaError, match=match):
        R.available()


def test_pins_that_are_not_json_raise_typed_error(monkeypatch, tmp_path):
    bad = tmp_path / "pins.json"
    bad.write_text("{not json")
    monkeypatch.setattr(R, "PINS", bad)
    with pytest.raises(R.PinSchemaError, match="JSON"):
        R.available()


def test_committed_pins_validate():
    assert len(R.available()) == 16
    assert Path(R.PINS).is_file()
