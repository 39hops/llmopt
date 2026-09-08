"""WRITER-TRAJECTORY-CENSUS-0, STAGE 0B (pre-reg RESULTS L67576, sealed by
AMENDMENT -SEAL L67810): finished-model learned-update DEPENDENCE
profiles and cross-writer component COMPATIBILITY. Specimens: A =
checkpoints/gallery19m_phase_s2.pt, B = checkpoints/gallery19m_backsched_s2.pt
(shared W_0 = the canonical seed-2 construction), N1 =
checkpoints/atomtraj1/stock_s6/step_15420.pt (first run), N2 =
/Users/artin/code/llmopt-repair/checkpoints/atomtraj1/stock_s6/step_15420.pt
(repair), W_0 = their byte-identical step_00000.pt (seed-6 regeneration
recorded as a check). For each specimen: gate(full), gate(W_0) once per
seed, and for each of the nine groups (BLOCK 0..7, OUTSIDE) and eight
CLASSES, gate(model with that group's delta reverted to W_0); D_l =
gate(full) - gate(reverted). Swaps: for each block l, the recipient with
block l's delta taken from the donor (A <- B, B <- A, N1 <- N2, N2 <- N1);
swap loss = gate(hybrid) - gate(recipient full). Every gate is
llmopt.lab.gate.gate_eval on mps (the standard 120). Rows stream to
logs/writertraj0/gates.jsonl; bars D-0, D-1, D-2 to
logs/writertraj0/depend.json. Refuses to overwrite. Runs only after
census.json exists (STAGE 0 preserved first).

Env: SMOKE=1 gates specimen N1 full and one revert at the 8-prompt proxy
tier and appends kind=depend rows to logs/writertraj0/smoke.jsonl.

Usage: .venv/bin/python scratch/writertraj_depend.py
"""
import datetime
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import CLASSES, state_digest  # noqa: E402
from llmopt.lab.gate import gate_eval  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
OUT = Path("logs/writertraj0")
REPAIR = Path("/Users/artin/code/llmopt-repair")
KEYS = sorted(sum(CLASSES.values(), []))
GROUPS = {f"BLOCK{l}": sorted(k for k in KEYS if k.startswith(f"blocks.{l}.")) for l in range(8)}
GROUPS["OUTSIDE"] = sorted(k for k in KEYS if not k.startswith("blocks."))
CLASS_GROUPS = {f"CLASS_{c}": sorted(ks) for c, ks in CLASSES.items()}
SPECIMENS = {
    "A": {"path": "checkpoints/gallery19m_phase_s2.pt", "seed": 2, "w0": "regen"},
    "B": {"path": "checkpoints/gallery19m_backsched_s2.pt", "seed": 2, "w0": "regen"},
    "N1": {"path": "checkpoints/atomtraj1/stock_s6/step_15420.pt", "seed": 6, "w0": "checkpoints/atomtraj1/stock_s6/step_00000.pt"},
    "N2": {"path": str(REPAIR / "checkpoints/atomtraj1/stock_s6/step_15420.pt"), "seed": 6, "w0": str(REPAIR / "checkpoints/atomtraj1/stock_s6/step_00000.pt")},
}
SWAP_PAIRS = [("A", "B"), ("B", "A"), ("N1", "N2"), ("N2", "N1")]   # (recipient, donor)


def load_sd(p):
    d = torch.load(p, map_location="cpu")
    return d["model"] if isinstance(d, dict) and "model" in d else d


def w0_seed(seed):
    tok = TM.MathTokenizer()
    torch.manual_seed(seed)
    m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


class Gater:
    def __init__(self, dev):
        self.tok = TM.MathTokenizer()
        self.dev = dev
        self.n = 8 if SMOKE else None
        self.commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.receipts = OUT / ("smoke.jsonl" if SMOKE else "gates.jsonl")

    def gate(self, sd, label, meta):
        m = TM.build_model(len(self.tok.vocab), d=384, layers=8, heads=6, ffn=1536)
        m.load_state_dict(sd)
        m = m.to(self.dev).eval()
        t0 = time.time()
        solves, valid = gate_eval(m, self.tok, self.dev, n=self.n) if self.n else gate_eval(m, self.tok, self.dev)
        row = {"kind": "gate", "label": label, "smoke": SMOKE, "solves": {str(k): int(v) for k, v in solves.items()},
               "total": int(sum(solves.values())), "valid_pct": round(float(valid), 2), "device": self.dev,
               "wall_s": round(time.time() - t0, 1), "state_digest": state_digest(sd), "code_commit": self.commit,
               "gated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **meta}
        with self.receipts.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[depend] {label}: {row['total']} ({row['wall_s']}s)", flush=True)
        del m
        return row["total"]


def revert(sd, w0, keys):
    out = {k: v.clone() for k, v in sd.items()}
    for k in keys:
        out[k] = w0[k].clone()
    return out


def swap(recipient, donor, w0r, w0d, keys):
    """Recipient with group keys replaced by W_0(recipient) + delta_donor."""
    out = {k: v.clone() for k, v in recipient.items()}
    for k in keys:
        out[k] = w0r[k] + (donor[k] - w0d[k])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not SMOKE:
        assert (OUT / "census.json").exists(), "STAGE 0 census must exist and be preserved before STAGE 0B"
        for f in ("gates.jsonl", "depend.json"):
            assert not (OUT / f).exists(), f"REFUSING: {OUT / f} exists"
    assert not os.environ.get("VOCAB_EXTRA"), "W_0 law requires VOCAB_EXTRA unset"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    G = Gater(dev)
    tree_dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    sds, w0s, digests = {}, {}, {}
    names = ["N1"] if SMOKE else list(SPECIMENS)
    for name in names:
        spec = SPECIMENS[name]
        sds[name] = load_sd(spec["path"])
        digests[name] = {"state_digest": state_digest(sds[name]), "path": spec["path"]}
        if spec["w0"] == "regen":
            w0s[name] = w0_seed(spec["seed"])
        else:
            w0s[name] = load_sd(spec["w0"])
            digests[name]["w0_matches_seed_regeneration"] = (state_digest(w0s[name]) == state_digest(w0_seed(spec["seed"])))
        digests[name]["w0_state_digest"] = state_digest(w0s[name])
    if not SMOKE:
        assert state_digest(w0s["N1"]) == state_digest(w0s["N2"]), "N1 / N2 step_0 digests differ"
    full, w0gate, dep = {}, {}, {}
    for name in names:
        full[name] = G.gate(sds[name], f"{name}/full", {"specimen": name, "op": "full"})
        seed = SPECIMENS[name]["seed"]
        if seed not in w0gate:
            w0gate[seed] = G.gate(w0s[name], f"W0_seed{seed}", {"specimen": name, "op": "w0", "seed": seed})
        dep[name] = {}
        groups = dict(list(GROUPS.items())[:1]) if SMOKE else {**GROUPS, **CLASS_GROUPS}
        for gname, keys in groups.items():
            t = G.gate(revert(sds[name], w0s[name], keys), f"{name}/revert/{gname}", {"specimen": name, "op": "revert", "group": gname})
            dep[name][gname] = full[name] - t
    swaps = {}
    if not SMOKE:
        for rec_name, don_name in SWAP_PAIRS:
            for l in range(8):
                keys = GROUPS[f"BLOCK{l}"]
                t = G.gate(swap(sds[rec_name], sds[don_name], w0s[rec_name], w0s[don_name], keys), f"{rec_name}<-{don_name}/BLOCK{l}",
                           {"specimen": rec_name, "op": "swap", "donor": don_name, "group": f"BLOCK{l}"})
                swaps[f"{rec_name}<-{don_name}/BLOCK{l}"] = t - full[rec_name]
    rec = {"prereg": "WRITER-TRAJECTORY-CENSUS-0", "stage": "0B", "smoke": SMOKE, "commit": G.commit, "device": dev,
           "started_utc": started, "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "specimens": digests, "tree_dirty": tree_dirty, "vocab_len": len(G.tok.vocab), "full": full, "w0_gate": {str(k): v for k, v in w0gate.items()}, "dependence": dep, "swap_loss": swaps}
    if not SMOKE:
        band = max(full.values()) - min(full.values())
        d0 = band <= 7
        def dist(x, y, groups):
            return sum((dep[x][g] - dep[y][g]) ** 2 for g in groups) ** 0.5
        dAB9, dN9 = dist("A", "B", GROUPS), dist("N1", "N2", GROUPS)
        dAB8, dN8 = dist("A", "B", CLASS_GROUPS), dist("N1", "N2", CLASS_GROUPS)
        cross = [v for k, v in swaps.items() if k.startswith("A<-B") or k.startswith("B<-A")]
        null = [v for k, v in swaps.items() if k.startswith("N1<-N2") or k.startswith("N2<-N1")]
        d2 = statistics.median(cross) < min(null)
        rec["bars"] = {"D-0": d0, "band": band, "profile_dist_9group_AB": dAB9, "profile_dist_9group_N1N2": dN9,
                       "profile_dist_8class_AB": dAB8, "profile_dist_8class_N1N2": dN8,
                       "D-1": (dAB9 > dN9) if d0 else None, "cross_swap_median": statistics.median(cross), "null_swap_min": min(null),
                       "D-2": d2 if d0 else None, "DEPENDENCE": (dAB9 > dN9) if d0 else None, "COMPATIBILITY": d2 if d0 else None,
                       "adjudicable": d0}
        (OUT / "depend.json").write_text(json.dumps(rec, indent=1))
        print("[depend] written; bars:", json.dumps({k: rec["bars"][k] for k in ("D-0", "D-1", "D-2")}), flush=True)
    else:
        with (OUT / "smoke.jsonl").open("a") as f:
            f.write(json.dumps({"kind": "depend", **rec}) + "\n")


if __name__ == "__main__":
    main()
