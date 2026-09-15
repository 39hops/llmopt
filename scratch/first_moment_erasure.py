"""FIRST-MOMENT-ERASURE-1 instrument (PRE-REG RESULTS L74943): selective
first-moment erasure on the reproducible stock-OneCycle writer A ALONE, at
the validated A@7200 arena, against the PINNED Stage-0 A control of
OPTIMIZER-MEMORY-ABLATION-1 (VERDICT -STAGE-0 L74856). Not a rescue or
completion of OMA1: no writer-B code path, no two-writer adjudication.

Thin sibling: every continuation / intervention mechanic is IMPORTED from
scratch/optimizer_memory_ablation.py (stream law, scheduler resume, bind,
run_leg, apply_arm, readouts, adjudication) and every shared source whose
behavior determines the continuation is PINNED by exact SHA256 (PINS
below); a mismatch refuses before any state is created. Changing shared
causal-path code therefore stops this instrument until an amendment /
requalification re-pins it.

Registered mode (MODE=treat; the only one):
  1. assert PINS (source shas), the locked Stage-0 receipt sha (docs/
     receipts.lock.json) and the locked desk receipt sha;
  2. assert the A anchor file sha / state digest, the Stage-0 leg-slices
     digest against the freshly reconstructed stream, n_pred literal
     0.894284652673395 == the desk receipt == the Stage-0 recompute;
  3. load the five pinned C snapshots (h = 1 / 5 / 20 / 100 / 900) and
     assert each state digest against the Stage-0 receipt; missing or
     drifted -> REFUSE (a control rerun is a separate booked act);
  4. ONLY THEN create treatment state: Z (exp_avg <- 0) and E (exp_avg <-
     0.99 exp_avg) legs on CPU (deterministic), readouts at the horizons,
     HELD-32 CE, the 120 gate on C(8100) / Z(8100) (descriptive), the
     single-writer ladder.                        -> logs/fme1/treat.json
No follow-up is launched. Snapshots under checkpoints/fme1/A/{Z,E}/.

SMOKE=1: OMA's synthetic non-target anchor machinery builds a smoke
Stage-0 receipt (writer A of the smoke pair only is consumed) under
checkpoints/oma1_smoke + logs/oma1/smoke<TAG>_stage0.json, then this
instrument's treat mode runs on it with its own smoke receipts under
logs/fme1/smoke<TAG>_* and snapshots under checkpoints/fme1_smoke.
Never writes a real path.

Usage: MODE=treat .venv/bin/python scratch/first_moment_erasure.py
       SMOKE=1 SMOKE_TAG=mech .venv/bin/python scratch/first_moment_erasure.py
"""
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_TAG = os.environ.get("SMOKE_TAG", "")
MODE = os.environ.get("MODE", "smoke" if SMOKE else "")

# ---------------------------------------------------------------- source pins (exact SHA256 at the seal; a mismatch refuses)
PINS = {
    "scratch/optimizer_memory_ablation.py": "99b6550dc6f0063d07650b5f1fd306ffc441ffc7654ac341868e7910a8de835b",
    "scratch/optimizer_geometry_desk.py": "8807b3af271c44cc40505cc1ec483203a604d47fed78469ab7ad511867e5c9b5",
    "scratch/update_geometry_census.py": "c76805379c50a1c54366e32c72e628e1d8f2ee59ebaa067d9e2946a3367ca864",
    "scratch/onecycle_component_audit.py": "2e45c1344d6c086117251c7524636c8561345383398a74e44eccd3969fe32a38",
    "scratch/birth19m_curric.py": "e5d97c684d21a811212893838491ee675c00624e692a2664e71cb03e7f00b162",
    "scratch/atomtraj_pins.py": "d91b8866bc761f0bc7aab6aaf6fe80b2dcd4d5f6395271b8f394b5e5d92aa63b",
    "scripts/train_mathnative.py": "530f110e792b6aa71965437bd9e98db6b62eaaa8e313027e876efab04e38634a",
    "llmopt/train/mathnative.py": "f8f5ca114d83d0dd854ebcb25de1a707e41e57822f06b15efecb0a1d87dcab61",
    "llmopt/lab/gate.py": "7c0b62eec0a0823c7b3fdab628534a8caef2e896937e65f24197f7865dcc084f",
    "scratch/tenet_d2_revdiet.py": "c6ecf518365163a7003f52faa44095ddb37d898246b8e418439a0a2b9d2e0762",      # D2 excision band (row set)
    "scripts/step_grpo_micro.py": "087015fc29abdb61cb12a4b181e93a90daab71e516cf70da59d8d3c6296b97a6",       # imported by gate_band_exprs
    "scripts/bench_step_tokens.py": "b5ffd4a6307a75704e57c96ba4d963c32d8622f366207b2e44b9b663ab7a6705",     # imported by gate_band_exprs
    "llmopt/lab/locator.py": "77232c9d16310bba26e9b67b1bd53d2e0ec0040740fec5d42c37433796b13fe6",
    "llmopt/common/device.py": "14cc39c1ad92a2a29acbe072a879a8e6e569462d3c2c2533aedb9dfc3d769b27",
}
# The OMA source that PRODUCED the pinned Stage-0 control and the desk expectation (commit b5e5c1b8 / 60b30517) differs from the
# pinned current source only by a receipt-label comment fold in mode_stage0 (commit 4803703d). The successor law is MECHANICAL:
# the concatenated source of every leg-path symbol must hash to LEG_PATH_SHA, which is the value at BOTH revisions.
OMA_SHA_AT_STAGE0 = "043535027455e09d6f07c23f0ac957bdffd8f65c72269518c87ac80672fd58da"
LEG_PATH_SYMBOLS = ["epoch_position", "epoch_order", "leg_slices", "future_stream", "batch_tensors", "loss_of", "make_sched", "group_record",
                    "resume_sched", "apply_arm", "assert_no_dropout", "sd_cpu", "flat", "sd_equal", "run_leg", "held_ce", "clipped_grad",
                    "bar1_law", "virtual_u", "readout", "adjudicate", "bind", "snapshot_dir", "save_snap", "one_leg"]
LEG_PATH_SHA = "9dfcc6a0e0f5031722b869acaf7f51b1288a47a8ebbd26e36cc898a8ad5991c8"
# The two locked receipts this rung is bound to, as source literals (a coordinated overwrite + re-lock cannot move them)
STAGE0_SHA = "6f5d368809e470acd1e86ba2e6b030b17f57496546233a711ae2aa870d6c5cf7"
DESK_SHA = "4fc044625422b23dacaa7d388175a9ab75afa482c56aacda9d8744728a605e4c"
ARENA = {"anchor": 7200, "leg": 900, "horizons": [1, 5, 20, 100, 900], "threads": 8}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pins(pins=PINS, root="."):
    """{path: (expected, actual)} for every mismatch; empty means all pinned sources are byte-identical to the seal."""
    bad = {}
    for rel, exp in pins.items():
        p = Path(root) / rel
        got = sha256_file(p) if p.exists() else None
        if got != exp:
            bad[rel] = (exp, got)
    return bad


_bad = check_pins()
if _bad:
    raise SystemExit("PIN MISMATCH (shared causal-path source changed since the seal; amend / requalify before any treatment): "
                     + json.dumps({k: {"expected": v[0][:16], "actual": (v[1] or "MISSING")[:16]} for k, v in _bad.items()}))

import inspect  # noqa: E402

import torch  # noqa: E402

import optimizer_memory_ablation as OMA  # noqa: E402  (SMOKE read from the same env)
import train_mathnative as TM  # noqa: E402
import update_geometry_census as UG  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402
from llmopt.lab.gate import gate_eval  # noqa: E402

WRITER = "A"                                   # the only writer this instrument knows
N_PRED = 0.894284652673395                     # sealed A BAR-1 expectation (AMENDMENT -SEAL L74731; logs/oma1/desk_bar1.json)
ANCHOR_SHA = "a0cdf244fcf44f05c6c5839a765ebba111d7c8b871b86dae16b6c8aed11d7af1"   # checkpoints/phase19m/m007200.pt
HORIZONS = OMA.HORIZONS
LEG = OMA.LEG
ANCHOR = OMA.ANCHOR
ARENA_TAG = os.environ.get("SMOKE_ARENA_TAG", SMOKE_TAG)     # refusal smokes may point at another smoke arena's Stage-0 receipt
STAGE0 = Path(f"logs/oma1/smoke{ARENA_TAG}_stage0.json" if SMOKE else "logs/oma1/stage0.json")
DESK = Path(f"logs/oma1/smoke{ARENA_TAG}_desk_bar1.json" if SMOKE else "logs/oma1/desk_bar1.json")
LOCK = Path("docs/receipts.lock.json")
OUT_DIR = Path("logs/fme1")
CK_DIR = Path("checkpoints/fme1_smoke" if SMOKE else "checkpoints/fme1")
RECEIPT = OUT_DIR / (f"smoke{SMOKE_TAG}_treat.json" if SMOKE else "treat.json")
STREAM = OUT_DIR / (f"smoke{SMOKE_TAG}_treat.jsonl" if SMOKE else "treat.jsonl")
assert OMA.WRITERS[WRITER]["kind"] == "stock"
assert ("oma1_smoke" in OMA.WRITERS[WRITER]["anchor"]) == SMOKE
if not SMOKE:
    assert (ANCHOR, LEG, HORIZONS) == (ARENA["anchor"], ARENA["leg"], ARENA["horizons"]), "arena literals"
OMA.CK_DIR = CK_DIR                            # this instrument's snapshots never land under checkpoints/oma1


def leg_path_sha(module=OMA, symbols=LEG_PATH_SYMBOLS):
    h = hashlib.sha256()
    for s in symbols:
        h.update(inspect.getsource(getattr(module, s)).encode())
    return h.hexdigest()


_lp = leg_path_sha()
if _lp != LEG_PATH_SHA:
    raise SystemExit(f"LEG-PATH SOURCE CHANGED: {_lp[:16]} v pinned {LEG_PATH_SHA[:16]} (amend / requalify before any treatment)")


# ---------------------------------------------------------------- pin laws (pure)
def locked_sha(lock, rel):
    return lock["receipts"][rel]["sha256"]


def data_file_shas():
    """sha256 of every corpus file the trainer's load_rows reads (recorded; the probe digest is the content assertion)."""
    import glob as _g
    files = sorted(set(_g.glob("data/micromodel_chains_shard*.jsonl") + ["data/step_chains.jsonl"] + _g.glob("data/micromodel_algebra_shard*.jsonl")
                       + _g.glob("data/micromodel_calc_l4_shard*.jsonl") + _g.glob("data/micromodel_v22_shard*.jsonl")))
    return {f: sha256_file(f) for f in files if Path(f).exists()}


def assert_provenance(stage0, desk, lock, anchor_sha_on_disk, threads, torch_version, numpy_version):
    """Every registered-mode provenance assertion, pure (no state created): the receipt shas against the source literals AND the
    lock, the OMA source that produced the receipts, the A anchor sha, Stage-0 A PASS, the sealed n_pred on both receipts, and the
    thread / version law of the pinned control. Returns the record to store."""
    s0 = sha256_file(STAGE0); dk = sha256_file(DESK)
    assert s0 == STAGE0_SHA == locked_sha(lock, str(STAGE0)), "Stage-0 receipt v the source literal / the receipt lock"
    assert dk == DESK_SHA == locked_sha(lock, str(DESK)), "desk receipt v the source literal / the receipt lock"
    assert stage0["source_sha256"] == desk["source_sha256"] == OMA_SHA_AT_STAGE0, "OMA source that produced the receipts"
    cb = stage0["cells"][WRITER]["C"]["bind"]
    assert cb["file_sha256"] == ANCHOR_SHA == anchor_sha_on_disk, "A anchor sha"
    assert stage0["verdict_per_writer"][WRITER] == "PASS", "Stage-0 A did not PASS"
    assert desk["cells"][WRITER]["law"]["n_pred"] == N_PRED == stage0["cells"][WRITER]["bar1_law"]["n_pred"], "sealed n_pred"
    assert stage0["threads"] == ARENA["threads"] == threads, "thread count v the pinned control"
    assert stage0["torch_version"] == torch_version and stage0["numpy_version"] == numpy_version, "torch / numpy v the pinned control"
    assert stage0["deterministic"] is True
    return {"stage0_sha256": s0, "desk_sha256": dk, "stage0_lock_verified": True, "oma_sha_at_stage0": OMA_SHA_AT_STAGE0, "leg_path_sha": LEG_PATH_SHA,
            "anchor_state_digest_receipt": cb["state_digest"], "threads": threads}


def verify_control(stage0, root=None, horizons=HORIZONS):
    """{h: {path, receipt_digest, file_digest, status}} for the pinned C snapshots of writer A, read from the Stage-0 receipt's
    own snapshot paths (under `root` when given); status in OK / MISSING / DRIFTED. Pure read; creates nothing."""
    snaps = stage0["cells"][WRITER]["C"]["snapshots"]
    out = {}
    for h in horizons:
        ent = snaps[str(h)]
        p = Path(root) / ent["path"] if root else Path(ent["path"])
        if not p.exists():
            out[h] = {"path": str(p), "receipt_digest": ent["state_digest"], "file_digest": None, "status": "MISSING"}
            continue
        sd = torch.load(p, map_location="cpu")["model"]
        got = state_digest(sd)
        fsha = sha256_file(p)
        ok = got == ent["state_digest"] and fsha == ent["sha256"]
        out[h] = {"path": str(p), "receipt_digest": ent["state_digest"], "file_digest": got, "receipt_sha256": ent["sha256"], "file_sha256": fsha, "status": "OK" if ok else "DRIFTED"}
    return out


def control_ok(v):
    return all(x["status"] == "OK" for x in v.values())


def single_writer_label(adj):
    if adj["bar1"] != "PASS":
        return "INSTRUMENT-FAULT"
    return f"{adj['path']}-{adj['sensitivity']}+FUNCTION-{adj['function']} [writer A only, one anchor, one seed lineage; gate descriptive]"


# ---------------------------------------------------------------- the registered mode
def mode_treat(tok, enc, starts, info, segs, d, held, rec, stream):
    stage0 = json.loads(STAGE0.read_text())
    desk = json.loads(DESK.read_text())
    rec["pins"] = {"sources": {k: v for k, v in PINS.items()}, "leg_path_symbols": LEG_PATH_SYMBOLS, "leg_path_sha_measured": leg_path_sha(), "stage0_receipt": str(STAGE0),
                   "stage0_sha256": sha256_file(STAGE0), "desk_receipt": str(DESK), "desk_sha256": sha256_file(DESK), "data_files": data_file_shas()}
    if not SMOKE:
        import numpy as _np
        lock = json.loads(LOCK.read_text())
        rec["pins"].update(assert_provenance(stage0, desk, lock, sha256_file(OMA.WRITERS[WRITER]["anchor"]), torch.get_num_threads(), torch.__version__, _np.__version__))
        assert stage0["verdict_per_writer"][WRITER] == "PASS"
    n_pred = float(desk["cells"][WRITER]["law"]["n_pred"])
    assert abs(n_pred - stage0["cells"][WRITER]["bar1_law"]["n_pred"]) <= 1e-12
    slices = OMA.leg_slices(starts, info["n_enc"], ANCHOR + 1, LEG)
    leg_digest = hashlib.sha256(json.dumps(slices).encode()).hexdigest()
    assert leg_digest == stage0["leg_slices_digest"], "future stream v the Stage-0 leg digest"
    rec["leg_slices_digest"] = leg_digest
    # ---- control pin: every C snapshot digest against the receipt BEFORE any treatment state exists
    ctrl = verify_control(stage0, None)
    rec["control_pin"] = ctrl
    if not control_ok(ctrl):
        raise SystemExit("CONTROL PIN REFUSED (missing or drifted C snapshot; a control rerun is a separate booked act): "
                         + json.dumps({str(h): x["status"] for h, x in ctrl.items()}))
    anchor_sd = torch.load(OMA.WRITERS[WRITER]["anchor"], map_location="cpu")["model"]
    rec["anchor"] = {"path": OMA.WRITERS[WRITER]["anchor"], "file_sha256": sha256_file(OMA.WRITERS[WRITER]["anchor"]), "state_digest": state_digest(anchor_sd)}
    if not SMOKE:
        assert rec["anchor"]["state_digest"] == stage0["cells"][WRITER]["C"]["bind"]["state_digest"] == rec["pins"]["anchor_state_digest_receipt"]
    W0 = OMA.flat(anchor_sd, segs, d)
    snapsC = {h: torch.load(ctrl[h]["path"], map_location="cpu")["model"] for h in HORIZONS}
    WC = {h: OMA.flat(snapsC[h], segs, d) for h in HORIZONS}
    mC = UG.build(tok, "cpu")
    ceC = {}
    for h in HORIZONS:
        mC.load_state_dict(snapsC[h]); ceC[h] = OMA.held_ce(mC, tok, held, "cpu")[0]
    rec["ce_held_C"] = {str(h): ceC[h] for h in HORIZONS}
    # ---- treatment state (Z, E) from here
    wrec = {"n_Z": {}, "n_E": {}, "dCE_Z": {}, "dCE_E": {}, "n_pred": n_pred, "h_end": LEG}
    rec["arms"] = {}
    for arm in ("Z", "E"):
        ca = {}
        model, _opt, snaps, _ = OMA.one_leg(WRITER, arm, tok, enc, slices, "cpu", segs, d, stream, ca, keep_opt_at_end=(arm == "Z"))
        assert ca["tensors_touched"] == len(UG.KEYS) == 59, ca["tensors_touched"]
        ca["readout"] = {}; ca["ce_held"] = {}
        for h in HORIZONS:
            r = OMA.readout(W0, WC[h], OMA.flat(snaps[h], segs, d), segs)
            ca["readout"][str(h)] = r
            model.load_state_dict(snaps[h]); ce = OMA.held_ce(model, tok, held, "cpu")[0]
            ca["ce_held"][str(h)] = ce
            wrec[f"n_{arm}"][h] = r["n"]; wrec[f"dCE_{arm}"][h] = ce - ceC[h]
        rec["arms"][arm] = ca
    dev_gate = "mps" if torch.backends.mps.is_available() else "cpu"
    rec["gate_device"] = dev_gate
    gate = {}
    zsd = torch.load(rec["arms"]["Z"]["snapshots"][str(LEG)]["path"], map_location="cpu")["model"]
    for arm, sd in (("C", snapsC[LEG]), ("Z", zsd)):
        gm = UG.build(tok, dev_gate); gm.load_state_dict(sd); gm.eval()
        solves, valid = gate_eval(gm, tok, dev_gate, n=(2 if SMOKE else None))
        gate[arm] = {"solves": solves, "total": int(sum(solves.values())), "valid_pct": round(valid, 2)}
    rec["gate"] = gate
    adj = OMA.adjudicate(wrec)
    rec["adjudication"] = adj
    rec["numbers"] = {k: ({str(h): v for h, v in val.items()} if isinstance(val, dict) else val) for k, val in wrec.items()}
    rec["label"] = single_writer_label(adj)
    print(f"[fme1] A: {adj} gate C {gate['C']['total']} Z {gate['Z']['total']} -> {rec['label']}", flush=True)
    return rec


def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(OMA.THREADS)
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RECEIPT.exists() or STREAM.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} or {STREAM} exists")
    if CK_DIR.exists() and any(CK_DIR.rglob("*.pt")):
        raise SystemExit(f"REFUSING: {CK_DIR} holds snapshots")
    assert MODE in ("treat", "smoke"), MODE
    tok = TM.MathTokenizer()
    assert len(tok.vocab) == 40 and not os.environ.get("VOCAB_EXTRA") and not os.environ.get("SEQ_CAP") and not os.environ.get("BIRTH_BS") and TM.BS == OMA.BS
    OMA.OA.assert_verbatim()
    segs, d, flat_digest = UG.flatten_law(UG.build(tok, "cpu"))
    assert d == 18_911_616
    enc, starts, info = OMA.future_stream(tok)
    ugc0 = json.loads(OMA.UGC0.read_text())
    assert info["n_enc"] == 164_490 and info["steps_per_epoch"] == 5_140 and len(starts) == 5_140
    assert info["probe64_digest"] == ugc0["probe"]["digest"]
    info["flatten_law_digest"] = flat_digest
    batches, probe = UG.probe_batches(tok)
    held = [batches[i] for i in probe["held"]]
    info["held_panel"] = {"n": len(held), "probe_digest": probe["digest"], "ugc0_digest": ugc0["probe"]["digest"]}
    if not SMOKE:
        assert probe["digest"] == ugc0["probe"]["digest"]
    if SMOKE and os.environ.get("SMOKE_REUSE") == "1":
        # refusal smoke: reuse an existing smoke Stage-0 arena (e.g. with a tampered control snapshot) and run treat on it
        assert STAGE0.exists() and DESK.exists(), "SMOKE_REUSE needs an existing smoke Stage-0 arena"
    elif SMOKE:
        # non-target arena: OMA's synthetic anchor pair + its smoke Stage 0 (writer A of that pair is the only thing consumed here)
        if STAGE0.exists() or DESK.exists():
            raise SystemExit(f"REFUSING: smoke Stage-0 artifacts {STAGE0} / {DESK} exist")
        dev_mps = "mps" if torch.backends.mps.is_available() else "cpu"
        OMA.build_smoke_anchor(tok, enc, starts, info, dev_mps)
        OMA.CK_DIR = Path("checkpoints/oma1_smoke")
        for m in ("desk-bar1", "stage0"):
            r0 = OMA.base_record(m, tok, info, "cpu", dev_mps)

            def s0(row, _m=m):
                with OMA.stream_path(_m).open("a") as f:
                    f.write(json.dumps(row) + "\n")

            r0 = OMA.mode_desk_bar1(tok, enc, starts, info, segs, d, "cpu", r0, s0) if m == "desk-bar1" else OMA.mode_stage0(tok, enc, starts, info, segs, d, "cpu", dev_mps, held, r0, s0)
            OMA.receipt_path(m).write_text(json.dumps(r0, indent=1) + "\n")
        OMA.CK_DIR = CK_DIR
    rec = OMA.base_record("fme1-treat", tok, info, "cpu", "mps" if torch.backends.mps.is_available() else "cpu")
    rec.update({"prereg": "FIRST-MOMENT-ERASURE-1", "kind": "first_moment_erasure/treat", "writer": WRITER, "n_pred_literal": N_PRED, "self_sha256": sha256_file(__file__), "ck_dir": str(CK_DIR)})

    def stream(row):
        with STREAM.open("a") as f:
            f.write(json.dumps(row) + "\n")

    t0 = time.time()
    rec = mode_treat(tok, enc, starts, info, segs, d, held, rec, stream)
    rec["wall_s"] = round(time.time() - t0, 1); rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    RECEIPT.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[fme1] done in {rec['wall_s']} s -> {RECEIPT}", flush=True)


if __name__ == "__main__":
    main()
