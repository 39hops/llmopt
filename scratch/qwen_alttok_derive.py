"""QWEN-ALTTOKEN-CONTROL-0 phase-1 derivation (DERIVATION-ONLY, no
treatment): the five BLe-gap-matched non-vendor control tokens at
the frozen HEADSWAP-IMPULSE loci, per the registered rule
(PRE-REG + AMENDMENT -FREEZE-PROTOCOL in docs/RESULTS.md; rule and
gate in docs/preregs/qwen-alttoken-control-0.params.json).

Per locus: full BLe s16-head logit vector z at the pinned
loop-state h; preconditions argmax(z)==frozen ble_token and
vendor-head argmax==frozen vendor_token; exclusions = {ble_top1,
vendor_token} + all vendor-tokenizer special ids + eos ids +
vendor-head top-256 at that h; control = argmin over remaining ids
of |(z[ble_top1]-z[c]) - g*| with g* = z[ble_top1]-z[vendor_token];
ties break to the lower id (np.argmin first-index). Match-quality
gate: abs_gap_error <= 0.05 logits per locus, else the receipt
records CONTROL-MATCH-FAILED and no treatment may run.

Writes logs/qwenalttok/control_table.json (refuse-if-exists) and
STOPS. Phase 2 requires that receipt's exact committed bytes in
HEAD and re-derives, refusing on mismatch.

    .venv/bin/python scratch/qwen_alttok_derive.py           (3080)
"""
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

ART = os.path.expanduser(os.environ.get("ART_DIR",
                                        "~/qwen_whole0t/BLe"))
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR",
                                         "~/qwen_vendor"))
VSLICE = os.path.expanduser(os.environ.get(
    "VENDOR_SLICE", "~/qwen_vendor_lmhead"))
IN = "logs/qwenloopstate"
OUT = "logs/qwenalttok"
HS_PARAMS = "docs/preregs/qwen-headswap-impulse-0.params.json"
AT_PARAMS = "docs/preregs/qwen-alttoken-control-0.params.json"
EOS_IDS = (248046, 248044)
GAP_MAX = 0.05
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    hs = _load("qwen_loop_state_headswap",
               "scratch/qwen_loop_state_headswap.py")
    at = json.load(open(AT_PARAMS))
    assert at["match_quality_gate"][
        "abs_gap_error_max_logits"] == GAP_MAX
    inj = json.load(open(HS_PARAMS))["injections"]
    os.makedirs(OUT, exist_ok=True)
    rcpt_path = os.path.join(OUT, "control_table.json")
    if os.path.exists(rcpt_path):
        raise SystemExit(f"REFUSING: {rcpt_path} exists")
    START = start_provenance(
        ["scratch/qwen_alttok_derive.py",
         "scratch/qwen_loop_state_headswap.py",
         "llmopt/lab/qcodec_fast.py", HS_PARAMS, AT_PARAMS,
         "docs/preregs/qwen-alttoken-control-0.json"],
        artifacts={"BLe": ART, "vendor_slice": VSLICE,
                   "vendor_checkout": VDIR})
    rows = [json.loads(x) for x in
            open(os.path.join(IN, "loopstate_rows.jsonl"))]
    byid = {r["id"]: r for r in rows}
    arrs, npz_shas = {}, {}
    for rid in sorted({j["item"] for j in inj}):
        arrs[rid], npz_shas[rid] = hs.load_npz(
            rid, byid[rid]["arrays_sha256"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(VDIR)
    specials = set(int(i) for i in tok.all_special_ids) | set(EOS_IDS)

    # the 3080's vendor dir holds tokenizer only; the attested
    # lm_head slice is the vendor head here (HEADSWAP-IMPULSE
    # pattern, tensor-byte identity attested in
    # logs/qwenloopstate1/vendor_slice_attestation.txt)
    from safetensors import safe_open
    with safe_open(os.path.join(VSLICE, "lm_head.safetensors"),
                   framework="pt", device="cpu") as f:
        Wv = f.get_tensor("lm_head.weight").float().numpy()
    Wb = hs.ble_head_rows()
    V = Wb.R
    print(f"[ad] vendor head {Wv.shape}, ble rows {V}", flush=True)

    # h per locus, in the injection-table order
    H = []
    for j in inj:
        a = arrs[j["item"]]
        pos = {int(p): i for i, p in enumerate(a["positions"])}
        H.append(a["h"].astype(np.float32)[pos[j["pos"]]])
    H = np.stack(H)                       # (5, d)

    # full BLe logits per locus, chunked over decoded s16 rows
    Z = np.empty((len(inj), V), np.float32)
    for lo in range(0, V, 16384):
        hi = min(lo + 16384, V)
        Z[:, lo:hi] = H @ Wb.rows(lo, hi).T
    # vendor logits per locus (for top-256 exclusion + precondition)
    Zv = np.empty((len(inj), V), np.float32)
    for lo in range(0, V, 16384):
        hi = min(lo + 16384, V)
        Zv[:, lo:hi] = H @ Wv[lo:hi].T

    table, failed = [], []
    for k, j in enumerate(inj):
        z, zv = Z[k], Zv[k]
        ble_top1 = int(z.argmax())
        if ble_top1 != j["ble_token"]:
            raise SystemExit(f"REFUSING: BLe argmax {ble_top1} != "
                             f"frozen ble_token {j['ble_token']} at "
                             f"item {j['item']} pos {j['pos']}")
        if int(zv.argmax()) != j["vendor_token"]:
            raise SystemExit(f"REFUSING: vendor argmax != frozen "
                             f"vendor_token at item {j['item']} "
                             f"pos {j['pos']}")
        vtop256 = set(int(i) for i in
                      np.argpartition(-zv, 256)[:256])
        excl = {ble_top1, j["vendor_token"]} | specials | vtop256
        g_star = float(z[ble_top1] - z[j["vendor_token"]])
        err = np.abs((z[ble_top1] - z) - g_star)
        mask = np.zeros(V, bool)
        mask[list(excl)] = True
        err[mask] = np.inf
        c = int(err.argmin())             # first index = lowest id tie
        achieved = float(z[ble_top1] - z[c])
        abs_err = float(abs(achieved - g_star))
        rank = int((z > z[c]).sum())      # 0-based descending rank
        row = {"item": j["item"], "pos": j["pos"],
               "ble_token": j["ble_token"],
               "vendor_token": j["vendor_token"],
               "control_id": c, "control_ble_rank": rank,
               "target_gap": g_star, "achieved_gap": achieved,
               "abs_gap_error": abs_err,
               "n_excluded": len(excl),
               "n_vendor_top256": len(vtop256),
               "n_specials": len(specials),
               "gate_pass": abs_err <= GAP_MAX}
        table.append(row)
        if not row["gate_pass"]:
            failed.append((j["item"], j["pos"], abs_err))
        print(f"[ad] item{j['item']}@{j['pos']} control={c} "
              f"rank={rank} g*={g_star:.4f} err={abs_err:.6f} "
              f"pass={row['gate_pass']}", flush=True)

    rcpt = {"note": "QWEN-ALTTOKEN-CONTROL-0 phase-1 derivation "
                    "receipt (the frozen control table; phase 2 "
                    "requires these exact committed bytes and "
                    "re-derives, refusing on mismatch)",
            "start": START,
            "completion_commit": completion_commit(),
            "input_npz_sha256": npz_shas,
            "gate": {"abs_gap_error_max_logits": GAP_MAX,
                     "all_pass": not failed},
            "status": ("CONTROL-MATCH-FAILED" if failed else "OK"),
            "table": table}
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(rcpt, indent=1) + "\n")
    print(f"[ad] receipt -> {rcpt_path} status={rcpt['status']}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
