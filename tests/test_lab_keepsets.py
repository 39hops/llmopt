"""lab.keepsets guards. Three tiers:

1. Source identity vs the frozen scratch/gt2_jaccard.py (always on).
2. Synthetic battery locking the regeneration-sensitive rules —
   DROP_TAIL first-row-per-(prompt,layer), GATE_ONLY string-prompt
   exclusion, stable-sort tie-break at the keep boundary (always on).
3. Full acceptance vs the BOOKED GT2-REVIEW-2 stats and the
   byte-frozen checkpoints/gt2_*_arm0_decode.json dumps — ~15s
   (measured), always on where the TRAJ artifacts exist, clean
   skip elsewhere.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from llmopt.lab import keepsets  # noqa: E402


@pytest.fixture(scope="module")
def frozen():
    spec = importlib.util.spec_from_file_location(
        "scratch_gt2_jaccard", ROOT / "scratch" / "gt2_jaccard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_source_identity(frozen):
    for name in ("_frac", "_flag", "decode_counts", "keep", "jmean",
                 "coverage"):
        assert inspect.getsource(getattr(keepsets, name)) == \
            inspect.getsource(getattr(frozen, name)), name


def _write_traj(path):
    rows = [
        # prompt 0, layer 0: first decode row is the mislabeled tail
        {"phase": "decode", "prompt": 0, "layer": 0, "topk": [1, 2]},
        {"phase": "decode", "prompt": 0, "layer": 0, "topk": [3, 4]},
        {"phase": "decode", "prompt": 0, "layer": 0, "topk": [3, 5]},
        # prompt phase rows never count
        {"phase": "prompt", "prompt": 0, "layer": 0, "topk": [9, 9]},
        # string prompt = probe row: excluded when gate_only
        {"phase": "decode", "prompt": "probe", "layer": 0, "topk": [7]},
        # second (prompt, layer) cell gets its own first-row drop
        {"phase": "decode", "prompt": 1, "layer": 0, "topk": [6]},
        {"phase": "decode", "prompt": 1, "layer": 0, "topk": [6]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_drop_tail_and_gate_only_rules(tmp_path, frozen):
    p = tmp_path / "traj.jsonl"
    _write_traj(p)
    got = keepsets.decode_counts(p, gate_only=True, drop_tail=True)
    assert {li: dict(row) for li, row in got.items()} == \
        {0: {3: 2, 4: 1, 5: 1, 6: 1}}   # tails dropped, probe excluded
    loose = keepsets.decode_counts(p, gate_only=False, drop_tail=False)
    assert loose[0][7] == 1 and loose[0][1] == 1  # both rules off
    # parity with the frozen implementation on the same file
    for go in (True, False):
        for dt in (True, False):
            assert keepsets.decode_counts(p, gate_only=go, drop_tail=dt) \
                == frozen.decode_counts(p, gate_only=go, drop_tail=dt)


def test_keep_tie_break_is_stable(frozen):
    # experts 2 and 3 tie at the boundary: stable sort keeps LOWER id
    counts = {0: {5: 10, 1: 9, 2: 3, 3: 3}}
    k = keepsets.keep(counts, n=8, top_k=3, frac=0.0)
    assert k == {0: {5, 1, 2}} == frozen.keep(counts, n=8, top_k=3,
                                              frac=0.0)


def test_jmean_and_coverage(frozen):
    ka, kb = {0: {1, 2, 3}, 1: {4, 5, 6}}, {0: {2, 3, 7}, 1: {4, 5, 6}}
    assert keepsets.jmean(ka, kb) == frozen.jmean(ka, kb) == (0.75, 0.5)
    demand = {0: {1: 10, 7: 30}, 1: {4: 60}}
    assert keepsets.coverage(demand, ka) == \
        frozen.coverage(demand, ka) == 0.7


BOOKED = {("math", "phys"): 0.8013, ("math", "code"): 0.5331,
          ("phys", "code"): 0.5280}
BOOKED_NULLS = {"math": 0.9205, "phys": 0.8670, "code": 0.6364}
TRAJ = {"math": "logs/opus/moe_gt1_traj_v2.jsonl",
        "phys": "logs/opus/gt2_phys_traj.jsonl",
        "code": "logs/opus/gt2_code_traj.jsonl"}


def test_full_acceptance_booked_stats_and_dump_bytes(tmp_path):
    """~15s measured on the Mac (2026-08-06 first pass: all booked
    stats + all three dumps byte-identical); skips where the TRAJ
    artifacts are absent."""
    for p in TRAJ.values():
        if not (ROOT / p).exists():
            pytest.skip(f"TRAJ artifact missing: {p}")
    counts = {d: keepsets.decode_counts(ROOT / p, gate_only=True,
                                        drop_tail=True)
              for d, p in TRAJ.items()}
    keeps = {d: keepsets.keep(c, frac=0.453) for d, c in counts.items()}
    for (a, b), want in BOOKED.items():
        got, _ = keepsets.jmean(keeps[a], keeps[b])
        assert round(got, 4) == want, (a, b, got)
    for d, p in TRAJ.items():
        half = lambda par: (lambda r: isinstance(r["prompt"], int)
                            and r["prompt"] % 2 == par)
        ke = keepsets.keep(keepsets.decode_counts(
            ROOT / p, half(0), gate_only=True, drop_tail=True), frac=0.453)
        ko = keepsets.keep(keepsets.decode_counts(
            ROOT / p, half(1), gate_only=True, drop_tail=True), frac=0.453)
        got, _ = keepsets.jmean(ke, ko)
        assert round(got, 4) == BOOKED_NULLS[d], (d, got)
    # DUMP_DECODE byte-identity: DROP_TAIL=0 reproduces the demand
    # logs the D4/PHYS-B/cross arms consumed (frozen in checkpoints/)
    for d, p in TRAJ.items():
        ref = ROOT / f"checkpoints/gt2_{d}_arm0_decode.json"
        if not ref.exists():
            pytest.skip(f"frozen dump missing: {ref}")
        c = keepsets.decode_counts(ROOT / p, gate_only=True,
                                   drop_tail=False)
        out = tmp_path / f"{d}.json"
        json.dump({"counts": {str(li): [row.get(e, 0) for e in range(128)]
                              for li, row in sorted(c.items())},
                   "source": f"{p} decode-only gate-prompts-only"},
                  open(out, "w"))
        assert out.read_bytes() == ref.read_bytes(), d
