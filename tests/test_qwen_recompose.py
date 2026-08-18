"""Fixtures for the ATTN-ATTRIB-1 recomposer core (scratch/
qwen_recompose.py) — synthetic two-shard artifacts pin key-level
promotion, dense re-offsetting, byte fidelity, conservation, and
the refuse paths BEFORE any 27B byte moves."""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "qwen_recompose", os.path.join(REPO, "scratch/qwen_recompose.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def _mans():
    base = {
        "t.attn.w": {"codec": "w4", "shape": [1, 4], "shard": "s1",
                     "off": 0, "len": 4},
        "t.ffn.w": {"codec": "w4", "shape": [1, 4], "shard": "s1",
                    "off": 4, "len": 4},
        "t.head.w": {"codec": "s16", "shape": [1, 4], "shard": "s2",
                     "off": 0, "len": 6},
        "t.vision.w": {"codec": "excluded", "shape": [2]},
    }
    donor = {
        "t.attn.w": {"codec": "s16", "shape": [1, 4], "shard": "s1",
                     "off": 0, "len": 8},
        "t.ffn.w": {"codec": "w4", "shape": [1, 4], "shard": "s1",
                    "off": 8, "len": 4},
        "t.head.w": {"codec": "s16", "shape": [1, 4], "shard": "s2",
                     "off": 0, "len": 6},
        "t.vision.w": {"codec": "excluded", "shape": [2]},
    }
    base_bytes = {"t.attn.w": b"BBBB", "t.ffn.w": b"FFFF",
                  "t.head.w": b"HHHHHH"}
    donor_bytes = {"t.attn.w": b"DDDDDDDD", "t.ffn.w": b"ffff",
                   "t.head.w": b"hhhhhh"}
    return base, donor, base_bytes, donor_bytes


def _readers(base_bytes, donor_bytes, base, donor):
    def by_entry(table, man):
        rev = {(e.get("shard"), e.get("off")): k
               for k, e in man.items() if e["codec"] != "excluded"}

        def rd(e):
            return table[rev[(e["shard"], e["off"])]]
        return rd
    return by_entry(base_bytes, base), by_entry(donor_bytes, donor)


def test_promoted_keys_selects_codec_diff_and_mark():
    base, donor, *_ = _mans()
    assert rc.promoted_keys(base, donor, ".attn.") == ["t.attn.w"]
    assert rc.promoted_keys(base, donor, ".head.") == []  # same codec


def test_compose_bytes_offsets_conservation():
    base, donor, bb, db = _mans()
    rd_b, rd_d = _readers(bb, db, base, donor)
    written = {}

    def ws(sh, rows):
        written[sh] = b"".join(p for _, _, p in rows)
    nm = rc.compose(base, donor, {"t.attn.w"}, rd_b, rd_d, ws)
    # promoted key carries donor codec/len/payload; dense offsets
    assert nm["t.attn.w"]["codec"] == "s16"
    assert nm["t.attn.w"]["off"] == 0 and nm["t.attn.w"]["len"] == 8
    assert nm["t.ffn.w"]["off"] == 8  # re-offset after longer payload
    assert written["s1"] == b"DDDDDDDD" + b"FFFF"
    assert written["s2"] == b"HHHHHH"
    assert nm["t.vision.w"]["codec"] == "excluded"
    assert set(nm) == set(base)


def test_compose_refuses_promoted_excluded():
    base, donor, bb, db = _mans()
    rd_b, rd_d = _readers(bb, db, base, donor)
    with pytest.raises(SystemExit, match="excluded key promoted"):
        rc.compose(base, donor, {"t.vision.w"}, rd_b, rd_d,
                   lambda s, r: None)


def test_compose_refuses_length_mismatch():
    base, donor, bb, db = _mans()
    db["t.attn.w"] = b"SHORT"  # 5 != entry len 8
    rd_b, rd_d = _readers(bb, db, base, donor)
    with pytest.raises(SystemExit, match="payload length"):
        rc.compose(base, donor, {"t.attn.w"}, rd_b, rd_d,
                   lambda s, r: None)


def test_recipes_match_registration():
    assert rc.RECIPES["F"]["n_expected"] == 64
    assert rc.RECIPES["L"]["n_expected"] == 144
    assert rc.RECIPES["Q"]["n_expected"] == 48
    assert rc.RECIPES["Q"]["base"] == "A"


def test_new_recipes_match_registration():
    assert rc.RECIPES["D"] == {"base": "A", "donor": "B",
                               "mark": "embed_tokens.weight",
                               "n_expected": 1}
    assert rc.RECIPES["E"]["n_expected"] == 1
    for b in "eml":
        assert rc.RECIPES[f"BL{b}"]["n_expected"] == 48
        assert rc.RECIPES[f"FL{b}"]["base"] == "F"
    # bands partition the 48 linear layers, disjoint 16/16/16... by
    # vendor index: 21+21+21 slots but only 48 are linear layers
    e, m, l = (rc.RECIPES[f"BL{b}"]["layers"] for b in "eml")
    assert not (e & m or m & l or e & l)


def test_promoted_keys_band_filter():
    base = {"model.layers.5.linear_attn.in_proj_qkv.weight":
            {"codec": "w4"},
            "model.layers.30.linear_attn.in_proj_qkv.weight":
            {"codec": "w4"}}
    donor = {k: {"codec": "s16"} for k in base}
    got = rc.promoted_keys(base, donor, ".linear_attn.",
                           layers=set(range(0, 21)))
    assert got == ["model.layers.5.linear_attn.in_proj_qkv.weight"]
