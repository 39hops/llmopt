"""lab.config guards: casts raise, unknown prefixed vars error (the
typo-takes-default kill), bools are strict, Optional handles "none",
echo emits the resolved config as one banner + one jsonl line, and
the environment is read at call time (F2, the env-at-import hazard).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from llmopt.lab.config import ConfigError, LabConfig


@dataclass
class Cfg(LabConfig):
    frac: float = 1.0
    seed: int = 1234
    gate_only: bool = False
    tag: str | None = None


def test_defaults_and_casts():
    env = {"GT_FRAC": "0.453", "GT_SEED": "777", "GT_GATE_ONLY": "true",
           "GT_TAG": "crest"}
    c = Cfg.from_env("GT", env)
    assert (c.frac, c.seed, c.gate_only, c.tag) == (0.453, 777, True, "crest")
    assert Cfg.from_env("GT", {}) == Cfg()  # pure defaults


def test_unknown_knob_is_fatal():
    with pytest.raises(ConfigError, match="GT_GATEONLY"):
        Cfg.from_env("GT", {"GT_GATEONLY": "1"})  # the typo class
    # other prefixes pass through untouched
    Cfg.from_env("GT", {"OTHER_GATEONLY": "1", "PATH": "/x"})


def test_bad_cast_raises():
    with pytest.raises(ConfigError, match="GT_FRAC"):
        Cfg.from_env("GT", {"GT_FRAC": "half"})
    with pytest.raises(ConfigError, match="bool"):
        Cfg.from_env("GT", {"GT_GATE_ONLY": "maybe"})


def test_optional_none():
    assert Cfg.from_env("GT", {"GT_TAG": "none"}).tag is None


def test_call_time_not_import_time(monkeypatch):
    monkeypatch.setenv("GT_SEED", "999")
    assert Cfg.from_env("GT").seed == 999  # reads os.environ NOW


def test_echo_banner_and_jsonl(tmp_path, capsys):
    out = tmp_path / "run.jsonl"
    c = Cfg.from_env("GT", {"GT_SEED": "42"})
    banner = c.echo(jsonl=str(out))
    assert "seed=42" in banner and banner in capsys.readouterr().out
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows == [{"cfg": "Cfg", "frac": 1.0, "seed": 42,
                     "gate_only": False, "tag": None}]
