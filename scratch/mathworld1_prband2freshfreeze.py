"""FRESH-CHECKPOINT FREEZE RECEIPT for RENDER-ATLAS-FRESH-SEED-0
(RESULTS L65695). Written only after all four fresh training
receipts exist; records, for 4 seeds x 2 representations, the
checkpoint path, sha256, attempt, init sha, training instrument
commit and sha, adopted-instrument sha, runtime, and the training
receipt sha. Refuse-if-exists. No model is opened.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import completion_commit, start_provenance  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402

SEEDS = ["21001", "22001", "23001", "24001"]
RDIR = Path("logs/mathworld1/prband2fresh_train")
OUT = RDIR / "fresh_checkpoint_freeze.json"


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    gate(not OUT.exists(), f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/mathworld1_prband2freshfreeze.py",
                              "scratch/mathworld1_svpfofresh.py",
                              "scratch/mathworld1_svpforepl.py"])
    cks, attempts = [], {}
    for s in SEEDS:
        cands = sorted(RDIR.glob(f"train_s{s}*_receipt.json"))
        gate(len(cands) >= 1, f"NO TRAINING RECEIPT s{s}")
        rec = None
        att = []
        for c in cands:
            r = json.loads(c.read_text())
            att.append({"receipt": str(c), "attempt": r["attempt"], "mode": r["mode"],
                        "checkpoints": r.get("checkpoints", {})})
            if r["mode"] == "production":
                gate(rec is None, f"TWO PRODUCTION RECEIPTS s{s}")
                rec, rpath = r, c
        gate(rec is not None, f"NO PRODUCTION RECEIPT s{s}")
        attempts[s] = att
        gate(rec["seed"] == int(s) and rec["prereg_commit"] == "632e57dd5593cc33c3f5588066c8e6176b7b14dd", "RECEIPT SEED/PREREG")
        gate(rec["updates_per_arm"] == {"CANONICAL": 7020, "PARAM_FIRST": 7020}
             and rec["tokens_continuation"] == {"CANONICAL": 2021220, "PARAM_FIRST": 2021220}
             and rec["order_counts"] == {"CANONICAL_first": 3510, "PARAM_FIRST_first": 3510}
             and rec["n_params"] == 19142016 and rec["vocab"] == 340, f"TRAINING LAW s{s}")
        gate(fsha(rec["init_path"]) == rec["init_sha"], f"INIT SHA s{s}")
        for path, sha in rec["checkpoints"].items():
            gate(Path(path).exists() and fsha(path) == sha, f"CKPT {path}")
            rep = "CANONICAL" if "canonical" in path else "PARAM_FIRST"
            cks.append({"seed": s, "representation": rep, "path": path, "sha256": sha,
                        "attempt": rec["attempt"], "init_sha": rec["init_sha"],
                        "training_instrument_commit": rec["start"]["start_commit"],
                        "training_instrument_sha256": rec["start"]["file_sha256"]["scratch/mathworld1_svpfofresh.py"],
                        "adopted_instrument_sha256": rec["adopted_instrument"]["sha256"],
                        "runtime": rec["env"], "training_receipt": str(rpath),
                        "training_receipt_sha256": fsha(rpath), "wall_s": rec["wall_s"]})
    gate(len(cks) == 8 and len({c["sha256"] for c in cks}) == 8, "8 DISTINCT")
    fz = {"prereg": "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-FRESH-SEED-0",
          "seeds": SEEDS, "n_checkpoints": 8, "checkpoints": cks, "attempts": attempts,
          "start": START, "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(fz, indent=1))
    print(json.dumps([(c["seed"], c["representation"], c["sha256"][:16], c["attempt"]) for c in cks]))


if __name__ == "__main__":
    main()
