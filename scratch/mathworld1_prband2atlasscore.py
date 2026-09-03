"""MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-SCORING-0
execution scorer (adopt-not-fork of scratch/mathworld1_prband2cf.py).

Scores the SAME 96 frozen states under all 720 PERSISTED global
role-permutation renderings of the atlas (read by (atlas_index,
pair_id, theta) from the sha-pinned, untracked
logs/mathworld1/prband2atlas/renders.jsonl; never regenerated, the
renderer is never imported) x the same four sha-pinned checkpoints
x FULL (mask 255), plus MASK0 (mask 0) on the frozen 16-policy
sanity subset. Scoring law, candidate semantic order,
continuations, T = 9 SUM, strict top-1 and the 1e-05 noise bound
are imported from the booked scorer unchanged.

Checkpoint-major execution, frozen (RESULTS L65585): per checkpoint
(A) sha + runtime gate, (B) RAW replay gate (atlas index 12 FULL,
384 rows, exact parsed-float equality against the tracked booked
stream; failure stops the whole run), (C) FULL over atlas_index
0..719 in ascending order with a flush at every policy boundary
(index 12 re-scored in sequence must equal its gate rows), (D)
MASK0 on the 16-policy subset, (E) chunk close with a receipt and
the stream sha, (F) independent completeness re-read of the chunk
stream. A chunk directory found without a complete receipt is a
partial artifact: it is renamed aside (kept for diagnosis, no
authority) and the checkpoint restarts from index 0 after
re-passing its RAW replay gate. Complete chunks are retained.

Full 9-token LP vectors are persisted only for the preregistered
audit policies (RAW 12, K_FIRST 480, LOW_PAIR_FIRST 300, and the
13 frozen non-anchor indices) and for every MASK0 subset row; all
other rows carry the exact SUM only.

No aggregation across checkpoints happens here (that is
scratch/mathworld1_prband2atlasagg.py, run only after all four
chunks are complete).

Usage:
    PRBAND2AS_SMOKE=1 .venv/bin/python scratch/mathworld1_prband2atlasscore.py
    .venv/bin/python scratch/mathworld1_prband2atlasscore.py
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_prband2score import (A0, B0, CKPTS,  # noqa: E402
                                             CODE_BASE, NAMES, NOISE,
                                             PRIMARY, SEM, TOK, VOCAB,
                                             ctup, fsha, top1_of)
from scratch.mathworld1_respath import masked_token_lps  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpchal import CTX  # noqa: E402
from scratch.mathworld1_svpcode import factor_decode  # noqa: E402
from scratch.mathworld1_svpforder import (PERM, pf_decode,  # noqa: E402
                                          pf_encode)

SMOKE = os.environ.get("PRBAND2AS_SMOKE") == "1"
PREREG = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-SCORING-0"
PREREG_COMMIT = "bb4fa2e0cb90c7fb69330e8bef7f7711107e47cd"
OUTDIR = Path("logs/mathworld1/prband2atlasscore_smoke" if SMOKE
              else "logs/mathworld1/prband2atlasscore")
SMOKE_RECEIPT = Path(
    "logs/mathworld1/prband2atlasscore_smoke/prband2atlasscore_receipt.json")
ATLAS = "logs/mathworld1/prband2atlas"
RENDERS = f"{ATLAS}/renders.jsonl"
MANIFEST = f"{ATLAS}/atlas_manifest.jsonl"
POLICIES = f"{ATLAS}/atlas_policies.jsonl"
VIEWS = "logs/mathworld1/prband2nuis/views.jsonl"
OLD_RAW = "logs/mathworld1/prband2score/raw_scores.jsonl"
PINS = {
    f"{ATLAS}/prband2atlas_receipt.json":
        "da51825d4966bba31b0f8a4123397ae5eb8e159c077789237cba12c088c48ad6",
    MANIFEST: "687b5e54e0da19bf057431eb4d44b755302c1963d18e13fb6d316fa99dd2f4b2",
    POLICIES: "b4a1c08308ca429d4bd7eb01210dfe469cd11eb9c9c527daf42997cec6d86c71",
    RENDERS: "2cac5570bc8eb6143a0a35797dafe1ea78147e6871ef93aceb87951e88419d8b",
    PRIMARY: "209391ef3b2e5c87308571d6ef309bb5724a214160caa1b7857f4a31f9112c34",
    OLD_RAW: "68d014ac2a0bf5b085941daeeae7def5a30506c7bb45c91630a96113b01cb31e",
    VIEWS: "677201ccc0cf34fbdf2b2e060146b68c157a2450926ac062f1c0f16cac8a72bb",
}
ANCHORS = {"RAW": (12, "8f1479a9402429ac18d1d1e55f803d02bd47313f8c69dfbfae6abd0a4f5f26f2",
                   ("HI_D", "HI_L", "K", "LO_D", "LO_L", "W")),
           "K_FIRST": (480, "12170b528662864717abfd685b6666da8567a56699d76621aac22a047e39aa4a",
                       ("K", "HI_D", "HI_L", "LO_D", "LO_L", "W")),
           "LOW_PAIR_FIRST": (300, "654ee0410fd5e2ec8dc74365e1cb89265d3a7320a9c1c4679885c294b7c18afb",
                              ("LO_D", "LO_L", "K", "HI_D", "HI_L", "W"))}
RAW_IDX = 12
SUBSET13 = [268, 524, 687, 43, 508, 405, 353, 456, 338, 486, 355, 293, 162]
SUBSET13_SHA = "19debfefe4d57a022a26e5fa68619d06c45d98657138875bdb4e7e626e30e452"
MASK0_SET = [12, 480, 300] + SUBSET13
LP_SET = set(MASK0_SET)
EPS_SCORE = NOISE          # 1e-05
EPS_D = 2e-05
TORCH_PIN = "2.12.1"
ANCHOR_EXPECT = {  # (T, B) for RAW / K_FIRST / LOW_PAIR_FIRST, booked L65490
    "19001|CANONICAL": {12: (29, 0), 480: (48, 0), 300: (82, 34)},
    "19001|PARAM_FIRST": {12: (48, 0), 480: (95, 47), 300: (54, 6)},
    "20001|CANONICAL": {12: (96, 48), 480: (48, 0), 300: (48, 0)},
    "20001|PARAM_FIRST": {12: (62, 14), 480: (47, 8), 300: (17, 0)}}
# smoke: one checkpoint, two states, a handful of policies
SMOKE_POLICIES = [0, 12, 300, 480, 268, 719]


def sgn(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def load_atlas(P):
    """Atlas authority gates; returns manifest rows, policies rows,
    renders {(idx, pair_id, theta): cur}."""
    man = [json.loads(l) for l in open(MANIFEST)]
    gate(len(man) == 720 and [m["atlas_index"] for m in man]
         == list(range(720)), "MANIFEST INDEX")
    gate(len({m["render_id"] for m in man}) == 720, "RENDER IDS")
    pol = {}
    for l in open(POLICIES):
        p = json.loads(l)
        gate(p["atlas_index"] not in pol, "DUP POLICY")
        pol[p["atlas_index"]] = p
    gate(sorted(pol) == list(range(720)), "POLICY INDEX")
    gate(all(p["eligible"] for p in pol.values()), "ELIGIBLE 720")
    gate(len({p["matrix_sha"] for p in pol.values()}) == 720, "MATRIX SHAS")
    for m in man:
        gate(pol[m["atlas_index"]]["render_id"] == m["render_id"]
             and pol[m["atlas_index"]]["roles"] == m["roles"], "MANIFEST/POLICY")
    order = [(p["pair_id"], p["theta"]) for p in P]
    pop = set(order)
    by = defaultdict(dict)
    n = 0
    for l in open(RENDERS):
        r = json.loads(l)
        k = (r["pair_id"], r["theta"])
        gate(k not in by[r["atlas_index"]], "DUP STATE IN POLICY")
        by[r["atlas_index"]][k] = r["cur"]
        n += 1
    gate(n == 69120 and sorted(by) == list(range(720)), "RENDER ROWS")
    for idx, rows in by.items():
        gate(set(rows) == pop and len(rows) == 96, f"STATE SET {idx}")
        sha = hashlib.sha256(json.dumps([rows[k] for k in order]).encode()
                             ).hexdigest()
        gate(sha == pol[idx]["matrix_sha"], f"MATRIX SHA {idx}")
    # anchors: byte identity against the pinned nuisance views
    views = {}
    for l in open(VIEWS):
        v = json.loads(l)
        views[(v["view"], v["pair_id"], v["theta"])] = v["cur"]
    for name, (idx, rid, roles) in ANCHORS.items():
        gate(man[idx]["render_id"] == rid and tuple(man[idx]["roles"]) == roles,
             f"ANCHOR {name} MANIFEST")
        gate(all(views[(name, k[0], k[1])] == by[idx][k] for k in order),
             f"ANCHOR {name} BYTES")
    gate(hashlib.sha256(json.dumps(SUBSET13).encode()).hexdigest()
         == SUBSET13_SHA, "SUBSET13 DIGEST")
    rest = sorted((m for m in man if m["atlas_index"] not in (12, 480, 300)),
                  key=lambda m: m["render_id"])
    gate([m["atlas_index"] for m in rest[:13]] == SUBSET13, "SUBSET13 LAW")
    renders = {(idx, k[0], k[1]): cur for idx, rows in by.items()
               for k, cur in rows.items()}
    return man, pol, renders


def main():
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN DRIFT {p}")
    if not SMOKE:
        gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
        sr = json.loads(SMOKE_RECEIPT.read_text())
        gate(sr.get("smoke") is True and sr.get("verdict") == "SMOKE OK",
             "SMOKE NOT GREEN")
        for pth, h in sr["start"]["file_sha256"].items():
            gate(fsha(pth) == h, f"SMOKE STALE {pth}")
    START = start_provenance(
        ["scratch/mathworld1_prband2atlasscore.py",
         "scratch/mathworld1_prband2cf.py", "scratch/mathworld1_prband2score.py",
         "scratch/mathworld1_respath.py", "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_svpforder.py", "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py", "scratch/mathworld1_svpbirth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    gate(torch.__version__ == TORCH_PIN, f"TORCH {torch.__version__}")
    t0 = time.monotonic()
    P = [json.loads(l) for l in open(PRIMARY)]
    gate(len(P) == 96 and len({p["pair_key"] for p in P}) == 48, "N/keys")
    gate(Counter(p["theta"] for p in P) == Counter({"SIN_LOW": 48,
                                                    "COS_LOW": 48}), "theta")
    for k in range(0, 96, 2):
        gate(P[k]["pair_key"] == P[k + 1]["pair_key"]
             and P[k]["theta"] == "SIN_LOW" and P[k + 1]["theta"] == "COS_LOW",
             "pair adjacency")
    for p in P:
        gate(sorted(tuple(t) for t in p["cand_tuples"]) == sorted(SEM),
             "SEMANTIC SET")
        gate(tuple(p["gold_tuple"]) == (A0 if p["theta"] == "SIN_LOW" else B0),
             "gold mapping")
    man, pol, renders = load_atlas(P)
    policies = SMOKE_POLICIES if SMOKE else list(range(720))
    mask0_set = [i for i in MASK0_SET if i in policies] if SMOKE else MASK0_SET
    if SMOKE:
        P = P[:2]
    n = len(P)
    old = {}
    for l in open(OLD_RAW):
        r = json.loads(l)
        if r["mask"] != 255:
            continue
        k = (r["seed"], r["representation"], r["state"], r["pair_id"],
             r["theta"], r["cur_sha"], tuple(r["candidate"]))
        gate(k not in old, "DUP OLD ROW")
        old[k] = (r["lps"], r["sum"], r["gold"])
    gate(len(old) == 1536, "OLD FULL ROWS")
    states = []
    for i, p in enumerate(P):
        by = {ctup(c): c for c in p["candidates"]}
        conts = {"CANONICAL": [], "PARAM_FIRST": []}
        for s in SEM:
            cz = by[s]["factor_code"]
            gate(factor_decode(cz) == s, "C RT")
            pz = pf_encode(s)
            gate(pf_decode(pz) == s and pz == [cz[PERM[i2]] for i2 in range(8)],
                 "PF RT / PERM")
            conts["CANONICAL"].append([CODE_BASE + x for x in cz] + [TOK.eos_id])
            conts["PARAM_FIRST"].append([CODE_BASE + x for x in pz]
                                        + [TOK.eos_id])
        states.append({"i": i, "pair_id": p["pair_id"], "theta": p["theta"],
                       "raw_cur_sha": p["cur_sha"], "gold": tuple(p["gold_tuple"]),
                       "g": 1 if p["theta"] == "SIN_LOW" else -1,
                       "conts": conts})
    for st in states:
        gate(renders[(RAW_IDX, st["pair_id"], st["theta"])] == P[st["i"]]["cur"],
             "RAW RENDER IS PRIMARY CUR")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    receipt = {"smoke": SMOKE, "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
               "n_states": n, "n_policies": len(policies),
               "pins": {p: fsha(p) for p in PINS},
               "anchors": {k: {"atlas_index": v[0], "render_id": v[1],
                               "roles": list(v[2])} for k, v in ANCHORS.items()},
               "mask0_subset": mask0_set, "subset13": SUBSET13,
               "subset13_sha": SUBSET13_SHA, "lp_subset": sorted(LP_SET),
               "semantic_order": [list(s) for s in SEM],
               "eps": {"EPS_SCORE": EPS_SCORE, "EPS_D": EPS_D},
               "semantic_beyond_all_surface_identifiable": False,
               "torch": torch.__version__, "chunks": {}}

    def finish(verdict):
        receipt["verdict"] = verdict
        receipt["wall_s"] = round(time.monotonic() - t0, 1)
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "prband2atlasscore_receipt.json").write_text(
            json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items()
                          if k not in ("start", "pins")}, indent=1)[:6000],
              flush=True)

    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")
    receipt["device"] = "mps"

    for seed, rep, path, sha_exp in (CKPTS[:1] if SMOKE else CKPTS):
        ck = f"{seed}|{rep}"
        cdir = OUTDIR / f"chunk_{seed}_{rep}"
        crec = cdir / "chunk_receipt.json"
        if crec.exists():
            cr = json.loads(crec.read_text())
            if cr.get("verdict") == "CHUNK COMPLETE" and \
                    fsha(str(cdir / "scores.jsonl")) == cr["scores_sha256"]:
                receipt["chunks"][ck] = {"retained": True, **cr}
                print(f"[RETAINED {ck}]", flush=True)
                continue
        if cdir.exists():
            aside = cdir.with_name(cdir.name + f"_partial_{int(time.time())}")
            cdir.rename(aside)
            receipt.setdefault("partials_set_aside", []).append(str(aside))
            print(f"[PARTIAL SET ASIDE {aside}]", flush=True)
        cdir.mkdir(parents=True)
        tc = time.monotonic()
        # (A) checkpoint sha + runtime
        gate(fsha(path) == sha_exp, f"CKPT SHA {path}")
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(path, weights_only=True))
        gate(sum(q.numel() for q in m.parameters()) == 19142016, "PARAMS")
        m.eval()
        m = m.to(dev)
        gate(len(m.blocks) == 8, "8 BLOCKS")
        dtype = str(next(m.parameters()).dtype)
        chunk = {"seed": seed, "representation": rep, "path": path,
                 "sha256": sha_exp, "device": "mps", "torch": torch.__version__,
                 "dtype": dtype, "n_states": n, "policies": len(policies)}
        out = open(cdir / "scores.jsonl", "w")
        def score_policy(idx, mask, arm, keep_lps):
            rows_out = []
            rid = man[idx]["render_id"]
            for st in states:
                cur = renders[(idx, st["pair_id"], st["theta"])]
                lps = masked_token_lps(m, dev, cur, st["conts"][rep], mask)
                for s, lp, cont in zip(SEM, lps, st["conts"][rep]):
                    gate(len(lp) == 9, "T!=9")
                    tot = float(sum(lp))
                    row = {"seed": seed, "representation": rep, "arm": arm,
                           "mask": mask, "atlas_index": idx, "render_id": rid,
                           "state": st["i"], "pair_id": st["pair_id"],
                           "theta": st["theta"],
                           "cur_sha": hashlib.sha256(cur.encode()).hexdigest(),
                           "candidate": list(s), "name": NAMES[s],
                           "sum": tot, "gold": list(st["gold"])}
                    if keep_lps:
                        row["lps"] = lp
                        row["continuation"] = cont
                    rows_out.append((st, s, lp, tot))
                    out.write(json.dumps(row) + "\n")
            out.flush()
            return rows_out

        # (B) RAW replay gate
        gate_rows = score_policy(RAW_IDX, 255, "FULL", True)
        exact = 0
        drift = 0.0
        for st, s, lp, tot in gate_rows:
            k = (seed, rep, st["i"], st["pair_id"], st["theta"],
                 st["raw_cur_sha"], s)
            gate(k in old, f"OLD ROW MISSING {k}")
            olp, osum, ogold = old[k]
            gate(ogold == list(st["gold"]), "OLD GOLD")
            exact += (olp == lp) and (osum == tot)
            drift = max(drift, abs(osum - tot),
                        max(abs(a - b) for a, b in zip(olp, lp)))
        chunk["raw_replay"] = {"rows": len(gate_rows), "exact": exact,
                               "max_abs_drift": drift,
                               "pass": exact == len(gate_rows) == n * 4}
        print(f"[REPLAY {ck}] exact {exact}/{len(gate_rows)} drift {drift}",
              flush=True)
        if not chunk["raw_replay"]["pass"]:
            out.close()
            chunk["verdict"] = "RAW-REPLAY FAILURE"
            crec.write_text(json.dumps(chunk, indent=1))
            receipt["chunks"][ck] = chunk
            finish("RAW-REPLAY FAILURE — ATLAS NOT SCORED")
            return
        gate_key = {(st["i"], s): (lp, tot) for st, s, lp, tot in gate_rows}
        # (C) FULL over all policies, ascending atlas_index
        T = {}
        Bc = {}
        seq_ok = True
        for pi, idx in enumerate(policies):
            rows = score_policy(idx, 255, "FULL", idx in LP_SET)
            if idx == RAW_IDX:
                seq_ok = all(gate_key[(st["i"], s)] == (lp, tot)
                             for st, s, lp, tot in rows)
            sc = defaultdict(dict)
            for st, s, lp, tot in rows:
                sc[st["i"]][s] = tot
            corr = []
            for st in states:
                w, tie, mg = top1_of(sc[st["i"]])
                corr.append(w == st["gold"])
            T[idx] = sum(corr)
            Bc[idx] = sum(corr[k] and corr[k + 1] for k in range(0, n, 2))
            if pi % 20 == 0 or pi == len(policies) - 1:
                print(f"[{ck}] policy {pi + 1}/{len(policies)} idx {idx} "
                      f"T {T[idx]} B {Bc[idx]} {time.monotonic() - tc:.0f}s",
                      flush=True)
        chunk["sequence_replay_pass"] = seq_ok
        if not seq_ok:
            out.close()
            chunk["verdict"] = "SEQUENCE-REPLAY FAILURE"
            crec.write_text(json.dumps(chunk, indent=1))
            receipt["chunks"][ck] = chunk
            finish("SEQUENCE-REPLAY FAILURE — CHUNK WITHOUT AUTHORITY")
            return
        # (D) MASK0 sanity subset
        m0 = {}
        for idx in mask0_set:
            rows = score_policy(idx, 0, "MASK0", True)
            for st, s, lp, tot in rows:
                m0.setdefault((st["i"], s), {})[idx] = tot
        spread = max(max(v.values()) - min(v.values()) for v in m0.values())
        tops = {}
        switches = flips = 0
        for st in states:
            per = {}
            for idx in mask0_set:
                per[idx] = {s: m0[(st["i"], s)][idx] for s in SEM}
            ws = {idx: top1_of(v)[0] for idx, v in per.items()}
            tops[st["i"]] = ws
            base = ws[mask0_set[0]]
            for idx in mask0_set[1:]:
                if ws[idx] != base:
                    a, b = per[mask0_set[0]], per[idx]
                    if abs(max(a.values()) - sorted(a.values())[-2]) >= EPS_D \
                            and abs(max(b.values()) - sorted(b.values())[-2]) >= EPS_D:
                        switches += 1
            d = {idx: per[idx][A0] - per[idx][B0] for idx in mask0_set}
            d0 = d[mask0_set[0]]
            for idx in mask0_set[1:]:
                if sgn(d[idx]) != sgn(d0) and abs(d0) >= EPS_D \
                        and abs(d[idx]) >= EPS_D:
                    flips += 1
        same_top = all(len(set(ws.values())) == 1 for ws in tops.values())
        chunk["mask0"] = {"policies": mask0_set, "max_spread": spread,
                          "spread_ok": spread <= EPS_SCORE,
                          "same_top_all_renders": same_top,
                          "robust_top_switches": switches,
                          "robust_margin_flips": flips,
                          "pass": spread <= EPS_SCORE and same_top
                          and switches == 0 and flips == 0}
        out.close()
        # anchor reproduction (descriptive gate, booked at L65490)
        anc = {}
        if not SMOKE:
            for name, (idx, _r, _ro) in ANCHORS.items():
                exp = ANCHOR_EXPECT[ck][idx]
                anc[name] = {"atlas_index": idx, "T": T[idx], "B": Bc[idx],
                             "expected_T": exp[0], "expected_B": exp[1],
                             "match": (T[idx], Bc[idx]) == exp}
        chunk["anchor_reproduction"] = anc
        chunk["anchor_reproduction_pass"] = all(a["match"] for a in anc.values())
        chunk["T"] = {str(i): T[i] for i in policies}
        chunk["B"] = {str(i): Bc[i] for i in policies}
        # (E)+(F) close, sha, independent completeness re-read
        chunk["scores_sha256"] = fsha(str(cdir / "scores.jsonl"))
        seen = Counter()
        keys = set()
        for l in open(cdir / "scores.jsonl"):
            r = json.loads(l)
            k = (r["arm"], r["atlas_index"], r["state"], tuple(r["candidate"]))
            keys.add(k)
            seen[r["arm"]] += 1
        # the RAW gate pass writes index 12 FULL once more than the sequence
        gate(seen["FULL"] == (len(policies) + 1) * n * 4, "FULL ROWS")
        gate(seen["MASK0"] == len(mask0_set) * n * 4, "MASK0 ROWS")
        gate(len(keys) == (len(policies) + len(mask0_set)) * n * 4, "KEYS")
        chunk["rows"] = dict(seen)
        chunk["distinct_keys"] = len(keys)
        chunk["wall_s"] = round(time.monotonic() - tc, 1)
        chunk["verdict"] = "CHUNK COMPLETE"
        crec.write_text(json.dumps(chunk, indent=1))
        receipt["chunks"][ck] = chunk
        print(f"[CHUNK COMPLETE {ck}] wall {chunk['wall_s']}s mask0 "
              f"{chunk['mask0']['pass']} anchors {chunk['anchor_reproduction_pass']}",
              flush=True)
        del m
        torch.mps.empty_cache()
    complete = [c for c in receipt["chunks"].values()
                if c.get("verdict") == "CHUNK COMPLETE"]
    ok = len(complete) == (1 if SMOKE else 4) and all(
        c["raw_replay"]["pass"] and c["mask0"]["pass"]
        and (SMOKE or c["anchor_reproduction_pass"]) for c in complete)
    finish("SMOKE OK" if (SMOKE and ok) else
           "ATLAS SCORED" if ok else "ATLAS INCOMPLETE OR GATE FAILURE")


if __name__ == "__main__":
    main()
