"""CENSUS (unregistered, free): what is actually IN DeepSeek-V4-Flash's
48 shards? Headers only -- ~8 MB of range reads, no weights.

Written because spec v3's load-bearing memory claim was derived by
SUBTRACTION and was wrong. It said "~27B non-routed params are fp8, i.e.
~27 GB -- which alone nearly fills the machine", and called that "the
stronger argument v2 failed to make". The 27B is 304.181 - 277.025,
where 277.025 = 43 layers x 256 x 25.17M. But the checkpoint carries
routed experts in FORTY-SIX blocks: layers.0..42 plus mtp.0/1/2, three
full multi-token-prediction blocks with 256 experts each. Nineteen of
the "27B dense params" are forgotten EXPERTS.

Every number this prints is a sum over shard headers, so it is
reproducible from the vendor artifact with no cached state.

Env: NSHARD (default 48). Usage: .venv/bin/python scratch/v4flash_census.py
"""
import collections
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import v4flash_rungA as RA  # noqa: E402

NSHARD = int(os.environ.get("NSHARD", "48"))
OUT = "logs/opus/v4_census.jsonl"
# bytes per element; I8 holds TWO fp4 codes, handled separately below
NEL = {"I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F16": 2,
       "F32": 4, "I32": 4, "I64": 8}


def is_routed(name):
    """Routed expert weight or scale -- NOT the always-on shared expert."""
    return ".experts." in name and ".shared_experts." not in name


def main():
    os.makedirs("logs/opus", exist_ok=True)
    rows, dtypes = [], collections.Counter()
    blocks = collections.Counter()      # routed params per top-level block
    tot = collections.Counter()
    for s in range(1, NSHARD + 1):
        RA.SHARD = s
        hdr, _, _ = RA.header()
        row = collections.Counter({"shard": s})
        for k, v in hdr.items():
            lo, hi = v["data_offsets"]
            nby = hi - lo
            dt = v["dtype"]
            assert dt in NEL, f"{k}: unknown dtype {dt}"
            # A packed-fp4 weight stores TWO codes per byte. Block scales
            # are metadata, not parameters -- counting their bytes as
            # params inflates the routed total by 3% (17/16 vs 1), which
            # is exactly the kind of unit slip this census exists to fix.
            n = (nby * 2 if dt == "I8"
                 else 0 if k.endswith(".scale") or ".weight_scale" in k
                 else nby // NEL[dt])
            kind = "routed" if is_routed(k) else "dense"
            row[f"{kind}_params"] += n
            row[f"{kind}_bytes"] += nby
            row["tensors"] += 1
            if kind == "dense":
                dtypes[dt] += nby
            else:
                blocks[k.split(".ffn.")[0]] += n
        rows.append(dict(row))
        tot.update(row)
        print(f"[cen] shard {s:2d}/{NSHARD} {row['tensors']:6d} tensors | "
              f"routed {row['routed_bytes']/1e9:6.3f} GB | dense "
              f"{row['dense_bytes']/1e9:6.3f} GB", flush=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"summary": True, "nshard": NSHARD,
                            "dense_dtype_bytes": dict(dtypes),
                            "expert_blocks": len(blocks),
                            "mtp_blocks": sorted(
                                b for b in blocks if b.startswith("mtp.")),
                            **{k: v for k, v in tot.items()
                               if k != "shard"}}) + "\n")
    rb, db = tot["routed_bytes"], tot["dense_bytes"]
    print(f"\n[cen] tensors {tot['tensors']:,} across {NSHARD} shards")
    print(f"[cen] routed experts : {tot['routed_params']/1e9:8.3f} B "
          f"params  {rb/1e9:8.3f} GB")
    print(f"[cen] NON-ROUTED     : {tot['dense_params']/1e9:8.3f} B "
          f"params  {db/1e9:8.3f} GB   <- spec v3 claimed ~27 B / ~27 GB")
    print(f"[cen] TOTAL ARTIFACT : {(rb+db)/1e9:8.3f} GB")
    print(f"[cen] expert-bearing blocks: {len(blocks)} "
          f"(MTP: {sorted(b for b in blocks if b.startswith('mtp.'))})")
    mtp = sum(v for b, v in blocks.items() if b.startswith("mtp."))
    print(f"[cen] MTP routed params {mtp/1e9:.3f} B -- these are the "
          f"'dense' params the subtraction lost")
    print("[cen] non-routed bytes by dtype: " + ", ".join(
        f"{k} {v/1e9:.3f} GB" for k, v in dtypes.most_common()))
    print(f"[cen] RECEIPT V4-RUNG-MINUS-1 says non-expert tensors are "
          f"F8_E4M3 + F8_E8M0; BF16 is {dtypes['BF16']/1e9:.3f} GB "
          f"({100*dtypes['BF16']/db:.0f}% of them)")


if __name__ == "__main__":
    main()
