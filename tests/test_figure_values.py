"""Every transcribed number in docs/figures.json is re-read from the
RESULTS entry its fence_ids name.

The figure file is the single source for the published figures, but
its values were typed once from the ledger. This test types nothing:
it parses the verdict tables at the lines the index gives for each
figure's fence_ids and compares. A one-digit edit in either place
fails here; the honesty_ledger figure is recounted from FINDINGS by
its own test (tests/test_gen_readme.py) and is not re-read here.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (ROOT / "docs" / "RESULTS.md").read_text().splitlines()
INDEX = {json.loads(r)["id"]: json.loads(r)["line"]
         for r in (ROOT / "docs" / "results-index.jsonl").open()}
SPEC = json.loads((ROOT / "docs" / "figures.json").read_text())


def entry(eid: str, span: int = 120) -> str:
    """The text of one RESULTS entry (heading line to the next heading,
    capped at `span` lines), located by index id so a ledger shift
    cannot silently retarget the read."""
    start = INDEX[eid] - 1
    assert RESULTS[start].startswith("## "), (eid, RESULTS[start])
    out = [RESULTS[start]]
    for line in RESULTS[start + 1:start + span]:
        if line.startswith("## "):
            break
        out.append(line)
    return "\n".join(out)


def ids(name: str) -> list[str]:
    return SPEC[name]["fence_ids"]


def test_loss_floor_points_and_reference():
    text = entry(ids("loss_floor")[0])
    table = dict(re.findall(r"^\s+(d\d+)\s+(\d\.\d{4})\s+\{", text, re.M))
    assert table, "FLOOR-HK-1 width table not found"
    for pt in SPEC["loss_floor"]["points"]:
        assert pt["value"] == float(table[pt["label"]]), pt
    h32 = re.search(r"H_32=(\d\.\d+)", text).group(1)
    assert SPEC["loss_floor"]["reference"]["value"] == float(h32)
    assert h32 in SPEC["loss_floor"]["reference"]["label"]


def test_effective_context_series_and_x():
    text = entry(ids("effective_context")[0])
    header = re.search(r"^\s+k\s+(.+)$", text, re.M).group(1).split()
    ks = [int(k) for k in header if k.isdigit()]
    assert SPEC["effective_context"]["x"] == ks
    rows = {m.group(1): [float(v) for v in m.group(2).split()]
            for m in re.finditer(r"^\s+(d\d+)\s+((?:\d\.\d{4}\s*){7})$", text, re.M)}
    assert len(rows) == 4
    for s in SPEC["effective_context"]["series"]:
        assert s["y"] == rows[s["label"]][:len(ks)], s["label"]


def test_ssm_star_arms():
    text = entry(ids("ssm_star")[0])
    gates = {m.group(1): int(m.group(2))
             for m in re.finditer(r"^\s+(attn|ssm)_d64\s+\d\.\d+\s+\{[^}]*\}\s+=\s+(\d+)/120", text, re.M)}
    assert gates == {"attn": 38, "ssm": 2} or len(gates) == 2
    by = {a["label"]: a["value"] for a in SPEC["ssm_star"]["arms"]}
    assert by["attention"] == gates["attn"] and by["selective SSM"] == gates["ssm"]


def test_routing_crest_seed_pairs_means_and_controls():
    r5, r6 = (entry(e) for e in ids("routing_crest"))
    seeds = {int(m.group(1)): (int(m.group(2)), int(m.group(3))) for m in
             re.finditer(r"seed (\d+): full (\d+) \{[\d,]+\} -> 45\.3% (\d+)", r5)}
    assert len(seeds) == 3
    pairs = {p["seed"]: (p["full"], p["mask"]) for p in SPEC["routing_crest"]["seed_pairs"]}
    assert pairs == seeds
    full_mean = sum(f for f, _ in seeds.values()) / 3
    mask_mean = sum(m for _, m in seeds.values()) / 3
    by = {a["label"]: a["value"] for a in SPEC["routing_crest"]["arms"]}
    assert by["top 45.3% by demand (mean of 3)"] == round(mask_mean)
    assert by["paired full model (mean of 3)"] == round(full_mean)
    # the rounding is stated in the fence, not hidden (66.33 drawn as 66)
    assert f"{full_mean:.2f}" in SPEC["routing_crest"]["fence"]
    controls = re.findall(r"RULE=(random|anti)\s+(?:RULESEED=\d+)?[^:]*:\s+(\d+)/120", r6)
    assert [int(v) for _, v in controls] == [0, 0, 0] and len(controls) == 3
    assert [by["random mask, same fraction"], by["random mask, same fraction (2)"],
            by["anti-demand, same fraction"]] == [0, 0, 0]


def test_merge_space_arms():
    ms1, ms3, ms4, ms5 = (entry(e) for e in ids("merge_space"))
    births = {m.group(1): int(m.group(2)) for m in
              re.finditer(r"^\s+(s[1-4]) \{[^}]*\} = (\d+)/120", ms1, re.M)}
    assert len(births) == 4
    by = {a["label"]: a["value"] for a in SPEC["merge_space"]["arms"]}
    for s, v in births.items():
        assert by[f"birth {s}"] == v, s
    assert re.search(r"avg_12/13/14/23/24/34 .*?: ALL\s+\{3:0,4:0,5:0,6:0,7:0\} = 0/120", ms1, re.S)
    assert by["independent merge (×6)"] == 0
    same_init = int(re.search(r"avg_ord \(s1 \+ s1o\): \{[^}]*\} = (\d+)/120", ms3).group(1))
    assert by["same-init merge"] == same_init
    soup = int(re.search(r"soup4 \(pair_a\+pair_b uniform 4-way\):\s+\{[^}]*\} = (\d+)/120", ms4).group(1))
    assert by["4-way order-twin soup"] == soup
    assert "n=3" in ms5 and "n=3" in SPEC["merge_space"]["fence"]


@pytest.mark.parametrize("name", [k for k in SPEC if not k.startswith("_")])
def test_every_figure_has_a_fence_with_ids(name):
    assert SPEC[name].get("fence"), name
    assert "fence_ids" in SPEC[name], name
