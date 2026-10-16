"""Freeze the mathgen generator's observable output into
tests/fixtures/mathgen_pins.json (rehabilitation phase 2C, step 1).

The pin file records, under the digest contract in
``llmopt.mathgen.digest.CONTRACT``:

- ``datasets``: ``make_dataset`` calls (default kinds and levels at
  n=200 seed=0, the ALL_KINDS and lazily registered kinds, and one call
  whose ``exclude`` set forces the skip path) with their digests;
- ``cases``: single-maker calls spanning every kind, levels 1 to 9
  where the maker supports them, several seeds, with the full prompt
  and answer text;
- ``retry_cases``: seeds whose maker took its deterministic
  ``seed + 1_000_003`` retry branch (detected by the retried seed
  producing the same prompt), pinned with their text.

High-level cases (L4 and up) are drawn under a fork-isolated wall; a
seed that does not finish inside the wall is not pinned and is listed
under ``skipped`` so the omission is visible.

The file is a freeze: the script refuses to overwrite an existing pin
file unless ``--force`` is given. Tests assert against it in
tests/test_mathgen_pins.py.

Usage:
    .venv/bin/python scripts/gen_mathgen_pins.py [--force] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmopt.mathgen import linalg, mechanics, multivar, ntheory, odes  # noqa: E402
from llmopt.mathgen.digest import CONTRACT, dataset_digest  # noqa: E402
from llmopt.mathgen.problems import ALL_KINDS, _resolve_maker, make_dataset  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "mathgen_pins.json"
LAZY_KINDS = (tuple(linalg.MAKERS) + tuple(odes.MAKERS) + tuple(ntheory.MAKERS)
              + tuple(multivar.MAKERS) + tuple(mechanics.MAKERS))
# Makers whose level reaches the L4..L9 expression families.
DEEP_KINDS = ("differentiate", "integrate", "second_derivative")
RETRY_KINDS = {  # makers with a deterministic seed + 1_000_003 retry site
    "integrate": (1, 2), "cint": (1, 2), "taylor": (2, 3),
    "definite_integral": (1, 2), "tangent_line": (1, 2), "limit": (1, 2),
}
RETRY_STEP = 1_000_003
HIGH_WALL = 30.0


def _isolated(kind: str, level: int, seed: int, wall: float):
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def _w():
        p = _resolve_maker(kind)(level, seed)
        q.put((p.prompt, p.answer))

    proc = ctx.Process(target=_w)
    proc.start()
    proc.join(wall)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return None
    try:
        return q.get(timeout=10)
    except Exception:
        return None


def _case(kind, level, seed, prompt, answer, **extra):
    return {"kind": kind, "level": level, "seed": seed,
            "prompt": prompt, "answer": answer, **extra}


def build() -> dict:
    t0 = time.time()
    pins: dict = {"contract": CONTRACT, "datasets": [], "cases": [],
                  "retry_cases": [], "skipped": []}

    def dataset(name, **kw):
        exclude_from = kw.pop("exclude_from", None)
        exclude = frozenset()
        if exclude_from:
            src = next(d for d in pins["datasets"] if d["name"] == exclude_from)
            exclude = frozenset(p.prompt for p in make_dataset(
                **{k: v for k, v in src["args"].items()}))
        ds = make_dataset(**kw, exclude=exclude)
        args = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}
        pins["datasets"].append({
            "name": name, "args": args, "exclude_from": exclude_from,
            "count": len(ds), "sha256": dataset_digest(ds),
            "first_prompt": ds[0].prompt, "last_prompt": ds[-1].prompt,
        })

    dataset("default-200-seed0", n=200, seed=0)
    dataset("default-100-seed99", n=100, seed=99)
    dataset("all-kinds-60-seed5", n=60, kinds=ALL_KINDS, seed=5)
    dataset("all-kinds-120-seed1", n=120, kinds=ALL_KINDS, seed=1)
    dataset("lazy-kinds-51-seed3", n=51, kinds=LAZY_KINDS, seed=3)
    dataset("default-30-seed1", n=30, seed=1)
    dataset("default-50-seed1-excl30", n=50, seed=1,
            exclude_from="default-30-seed1")
    dataset("integrate-only-40-seed7", n=40, kinds=("integrate",), seed=7)
    dataset("l4-mixed-12-seed2", n=12, kinds=("differentiate", "integrate"),
            levels=(4,), seed=2)

    # Single-maker cases: every kind, levels 1..3, seeds 0, 1, 17.
    for kind in ALL_KINDS + LAZY_KINDS:
        for level in (1, 2, 3):
            for seed in (0, 1, 17):
                p = _resolve_maker(kind)(level, seed)
                pins["cases"].append(_case(kind, level, seed, p.prompt, p.answer))

    # Deep levels under a wall: first two finishing seeds per cell.
    for kind in DEEP_KINDS:
        for level in range(4, 10):
            got = 0
            for seed in range(0, 40):
                if got == 2:
                    break
                res = _isolated(kind, level, seed, HIGH_WALL)
                if res is None:
                    pins["skipped"].append({"kind": kind, "level": level,
                                            "seed": seed, "wall_s": HIGH_WALL})
                    continue
                pins["cases"].append(_case(kind, level, seed, *res,
                                           wall_s=HIGH_WALL))
                got += 1

    # Retry cases: seed s retried iff maker(level, s) == maker(level, s + STEP).
    for kind, levels in RETRY_KINDS.items():
        mk = _resolve_maker(kind)
        for level in levels:
            found = 0
            for seed in range(0, 4000):
                if found == 2:
                    break
                p = mk(level, seed)
                if p.prompt == mk(level, seed + RETRY_STEP).prompt:
                    pins["retry_cases"].append(_case(
                        kind, level, seed, p.prompt, p.answer,
                        retried_seed=seed + RETRY_STEP))
                    found += 1
    pins["build_seconds"] = round(time.time() - t0, 1)
    return pins


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing pin file")
    args = ap.parse_args(argv)
    if args.out.exists() and not args.force:
        print(f"refusing to overwrite {args.out} (pass --force)", file=sys.stderr)
        return 2
    pins = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(pins['datasets'])} datasets, "
          f"{len(pins['cases'])} cases, {len(pins['retry_cases'])} retry cases, "
          f"{len(pins['skipped'])} skipped, {pins['build_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
