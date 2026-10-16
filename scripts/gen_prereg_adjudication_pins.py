"""Freeze the adjudicator's output on every historical pre-registration
that has a booked observations document (rehabilitation phase 2D,
step 1).

For each (pre-reg, observations) pair the pin file records:

- ``outcomes``: the ``adjudicate_prereg`` result, one row per bar with
  the bar id, bar name, outcome and the full reason chain;
- ``refutation``: the ``adjudicate_refutation`` string, or null;
- ``stdout`` and ``returncode`` of ``scripts/adjudicate.py`` run as a
  subprocess, byte for byte;
- ``prereg_sha256`` and ``observations_sha256`` of the two input files,
  so a later run can tell "the adjudicator changed" from "the inputs
  changed".

The pair registry is explicit (``PAIRS``) rather than globbed: an
observations document is paired with the pre-reg its driver scored,
and two pre-regs (homeo-actuator, headswap-impulse) have both an
online and an offline observations document.

The file is a freeze: the script refuses to overwrite an existing pin
file unless ``--force`` is given. Tests assert against it in
tests/test_prereg_adjudication_pins.py.

Usage:
    .venv/bin/python scripts/gen_prereg_adjudication_pins.py [--force] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmopt.lab.prereg import (adjudicate_prereg,  # noqa: E402
                               adjudicate_refutation, load)

OUT = ROOT / "tests" / "fixtures" / "prereg_adjudication_pins.json"
CONTRACT = "prereg-adjudication-pins-v1"

# (pre-reg path, observations path), both repo-relative.
PAIRS: tuple[tuple[str, str], ...] = (
    ("docs/preregs/ex5-layermatch-0.json", "logs/ex5/ex5_observations.json"),
    ("docs/preregs/ex6-phase-0.json", "logs/ex6/ex6_observations.json"),
    ("docs/preregs/ex6-b43-knife-0.json",
     "logs/ex6b43knife/ex6b43knife_observations.json"),
    ("docs/preregs/ex6-depth-0.json", "logs/ex6depth/ex6depth_observations.json"),
    ("docs/preregs/ex6-depth-1.json",
     "logs/ex6depth1/ex6depth1_observations.json"),
    ("docs/preregs/ex6-temporal-0.json",
     "logs/ex6temporal/ex6temporal_observations.json"),
    ("docs/preregs/qwen-attn-attrib-1.json",
     "logs/qwenattrib/attrib_observations.json"),
    ("docs/preregs/qwen-io-attrib-1.json",
     "logs/qwenattrib/ioattrib_observations.json"),
    ("docs/preregs/qwen-lband-1.json", "logs/qwenattrib/lband_observations.json"),
    ("docs/preregs/qwen-model2-alloc-1.json",
     "logs/qwenattrib/model2_observations.json"),
    ("docs/preregs/qwen-cheap-readout-0.json",
     "logs/qwencheapread/census_observations.json"),
    ("docs/preregs/qwen-cycle-impulse-0.json",
     "logs/qwencycle/impulse_observations.json"),
    ("docs/preregs/qwen-ble-freegen-2.json",
     "logs/qweneffort2/ble2_observations.json"),
    ("docs/preregs/qwen-homeo-actuator-0.json",
     "logs/qwenhomeo/homeo_observations.json"),
    ("docs/preregs/qwen-homeo-actuator-0.json",
     "logs/qwenhomeo/homeo_observations_offline.json"),
    ("docs/preregs/qwen-headswap-impulse-0.json",
     "logs/qwenhsimpulse/impulse_observations.json"),
    ("docs/preregs/qwen-headswap-impulse-0.json",
     "logs/qwenhsimpulse/impulse_observations_offline.json"),
    ("docs/preregs/qwen-loop-state-0.json",
     "logs/qwenloopstate/loopstate_observations.json"),
    ("docs/preregs/qwen-loop-state-1-headswap.json",
     "logs/qwenloopstate1/headswap_observations.json"),
    ("docs/preregs/qwen-mips-census-0.json", "logs/qwenmips/mips_observations.json"),
    ("docs/preregs/qwen-model1-tree.json", "logs/qwenmodel1/tree_observations.json"),
    ("docs/preregs/qwen-rk-census-0.json", "logs/qwenrouter/rk_observations.json"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjudicate_pair(prereg_rel: str, obs_rel: str) -> dict:
    """Run both the library path and the CLI on one pair."""
    prereg_path = ROOT / prereg_rel
    obs_path = ROOT / obs_rel
    prereg = load(prereg_path)
    obs = json.loads(obs_path.read_text())
    outcomes = adjudicate_prereg(prereg, obs)
    refutation = adjudicate_refutation(prereg, obs, bar_outcomes=outcomes)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "adjudicate.py"),
         str(prereg_path), str(obs_path)],
        capture_output=True, text=True, cwd=ROOT, check=False)
    return {
        "prereg": prereg_rel,
        "observations": obs_rel,
        "prereg_sha256": _sha(prereg_path),
        "observations_sha256": _sha(obs_path),
        "outcomes": [{"bar_id": o.bar_id, "bar_name": o.bar_name,
                      "outcome": o.outcome, "reasons": list(o.reasons)}
                     for o in outcomes],
        "refutation": refutation,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }


def build() -> dict:
    return {"contract": CONTRACT,
            "pairs": [adjudicate_pair(p, o) for p, o in PAIRS]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.out.exists() and not a.force:
        print(f"refusing to overwrite {a.out}; pass --force", file=sys.stderr)
        return 2
    pins = build()
    a.out.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    n_pre = len({p["prereg"] for p in pins["pairs"]})
    n_bars = sum(len(p["outcomes"]) for p in pins["pairs"])
    print(f"wrote {a.out}: {len(pins['pairs'])} pairs over {n_pre} "
          f"pre-regs, {n_bars} bar outcomes")
    for p in pins["pairs"]:
        tally = {}
        for o in p["outcomes"]:
            tally[o["outcome"]] = tally.get(o["outcome"], 0) + 1
        print(f"  {p['prereg'].split('/')[-1]:36s} rc={p['returncode']} "
              f"{tally} ref={p['refutation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
