"""lab.config — typed env-var config for arm drivers (spec
2026-08-05-llmopt-lab-extraction module 2). Kills the
typo-takes-default class: 237 bare os.environ.get sites across 65+
scratch files meant a misspelled knob silently ran the WRONG
EXPERIMENT with a clean exit. Here the contract is loud both ways:

- casts RAISE (a malformed value dies at init, never coerces),
- UNKNOWN prefixed env vars ERROR (the typo'd knob is caught before
  any tokens are spent, listing the valid field names),
- the RESOLVED config (after casts and defaults) echoes as one
  banner line + one jsonl line at init — pre-reg cross-checks need
  what the run actually used, not what the launcher shell intended.

Also fixes the env-at-import hazard (lab spec F2, the gt2_jaccard
burn): from_env reads the environment at CALL time, so drivers bind
config where they construct it, never at module import.

Usage:
    @dataclass
    class GateCfg(LabConfig):
        frac: float = 1.0
        seed: int = 1234
        gate_only: bool = False
        out: str = "logs/gate.jsonl"

    cfg = GateCfg.from_env("GT8")   # reads GT8_FRAC, GT8_SEED, ...
    cfg.echo(jsonl="logs/gate.jsonl")
"""
from __future__ import annotations

import dataclasses
import json
import os
import types
import typing
from dataclasses import dataclass

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(RuntimeError):
    """Raised loudly at init: unknown knob or malformed value."""


def _cast(name: str, raw: str, typ) -> object:
    origin = typing.get_origin(typ)
    if origin in (typing.Union, types.UnionType):  # Optional[T] / T | None
        members = [t for t in typing.get_args(typ) if t is not type(None)]
        if raw.lower() == "none":
            return None
        return _cast(name, raw, members[0])
    if typ is bool:
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ConfigError(
            f"config field {name}: bool wants one of "
            f"{sorted(_TRUE | _FALSE)}, got {raw!r}")
    try:
        return typ(raw)
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"config field {name}: cannot cast {raw!r} to "
            f"{getattr(typ, '__name__', typ)}: {e}") from e


@dataclass
class LabConfig:
    """Subclass with typed fields + defaults; construct via from_env."""

    @classmethod
    def from_env(cls, prefix: str, environ: dict | None = None):
        env = os.environ if environ is None else environ
        hints = typing.get_type_hints(cls)
        fields = {f.name: hints[f.name]
                  for f in dataclasses.fields(cls)}
        tag = prefix.rstrip("_") + "_"
        known = {tag + name.upper(): name for name in fields}
        unknown = sorted(k for k in env
                         if k.startswith(tag) and k not in known)
        if unknown:
            raise ConfigError(
                f"unknown {tag}* env vars {unknown} — valid knobs: "
                f"{sorted(known)} (typo-takes-default is the failure "
                "class this kills; fix the name, don't ignore this)")
        kw = {name: _cast(env_key, env[env_key], fields[name])
              for env_key, name in known.items() if env_key in env}
        return cls(**kw)

    def echo(self, jsonl: str | None = None) -> str:
        """One banner line + (optionally) one jsonl line: the RESOLVED
        config, i.e. what the run actually uses."""
        d = dataclasses.asdict(self)
        banner = "[cfg] " + type(self).__name__ + " " + \
            " ".join(f"{k}={v}" for k, v in d.items())
        print(banner, flush=True)
        if jsonl is not None:
            with open(jsonl, "a") as f:
                f.write(json.dumps({"cfg": type(self).__name__, **d})
                        + "\n")
        return banner
