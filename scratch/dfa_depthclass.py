"""WRITER-DFA-1 depth x module-class census (PRE-REG L68321 item 6 (viii),
B2 of L68543, P4 of L68644: MANDATORY conditional on FUNCTION-BAND pass).
For the DFA and the control specimen: the 8 x 5 table D_{l,c} =
gate(full) - gate(blocks.{l}.{c}.weight reverted to W_0) over
c in {qkv, o, gate, up, down}, 40 gates per specimen, 80 gates, rows
appended to logs/writerdfa1/gates.jsonl (op=revert_cell), the table to
logs/writerdfa1/depthclass.json (refuses to overwrite). Descriptive:
no cell bar, no cell selection. Requires depend.json with band_pass.
"""
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ["DEPEND_SET"] = "dfa"

import torch  # noqa: E402

from writertraj_depend import Gater, SPECIMENS, load_sd, revert, OUT  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402

CLASSES5 = ("qkv", "o", "gate", "up", "down")
TARGET = OUT / "depthclass.json"


def main():
    if TARGET.exists():
        raise SystemExit(f"REFUSING: {TARGET} exists")
    depend = json.load((OUT / "depend.json").open())
    assert depend["band"]["band_pass"], "census runs only on FUNCTION-BAND pass"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    G = Gater(dev)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    table = {}
    for name in ("DFA", "CTRL"):
        sd, w0 = load_sd(SPECIMENS[name]["path"]), load_sd(SPECIMENS[name]["w0"])
        assert state_digest(sd) == depend["specimens"][name]["state_digest"], f"{name} digest v depend.json"
        full = depend["full"][name]
        table[name] = {}
        for l in range(8):
            for c in CLASSES5:
                key = f"blocks.{l}.{c}.weight"
                t = G.gate(revert(sd, w0, [key]), f"{name}/revert_cell/B{l}_{c}", {"specimen": name, "op": "revert_cell", "block": l, "cls": c, "key": key})
                table[name][f"B{l}_{c}"] = full - t
    rec = {"prereg": "WRITER-DFA-1", "kind": "depthclass", "commit": G.commit, "device": dev, "started_utc": started,
           "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "classes": list(CLASSES5), "full": depend["full"], "table": table,
           "diff_DFA_minus_CTRL": {k: table["DFA"][k] - table["CTRL"][k] for k in table["DFA"]}, "n_gates": 80, "descriptive": True}
    TARGET.write_text(json.dumps(rec, indent=1))
    print(f"[depthclass] written {TARGET}")


if __name__ == "__main__":
    main()
