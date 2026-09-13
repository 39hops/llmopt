"""VERIFIED-ENDOGENOUS-DATA-CROSSFOSTER-1, CHAIN-DESK stage (PRE-REG
VERIFIED-ENDOGENOUS-DATA-CROSSFOSTER-1-CHAIN-DESK). Zero training. Asks
whether ITERATING the verified donor transition law amplifies the small
one-step donor difference (OBSERVATION -DONOR: 6.9 % of matched rows) into
a materially distinct endogenous history. Donors unchanged:
  A = forward OneCycle final, checkpoints/gallery19m_phase_s2.pt
  B = backward OneCycle final, checkpoints/gallery19m_backsched_s2.pt
(seed-2 W_0; state digests asserted against logs/writertraj0/census.json
exactly as in scratch/crossfoster_donor.py, whose loader is reused).

Root population (fresh band, frozen before any chain output is read):
roots (level, seed), seed = ROOT_BAND + 1000 * level + I0 + i, i <
N_PER_LEVEL[level], levels 3..7 in the atoms-shard mix (level 4 at 40 %),
enumerated by the fractional-progress order (every prefix carries the
mix); the same pre-generation exclusions as the donor stage (gate band,
base-diet cur / nxt strings, duplicate norm(cur), generation failure).

Paired verified-chain law, IDENTICAL for A and B and derived from the
120-gate chain (llmopt.lab.gate.gate_eval): each donor walks its OWN
chain from the shared root for at most H plies. At ply t (0-based) the
donor at state cur_X samples up to N_WAVES waves of WAVE = 8 candidates
with llmopt.lab.gate.sample_wave_lp (temperature 0.7, max_new 120) under
the COMMON seeds SAMPLE_BAND + 2000 * root_index + 128 * t + 8 * wave + b
(the same integers for both donors at the same (root, ply, wave, slot):
common random numbers suppress pure sampling noise; once the states
differ, later differences come through the model -> state -> model
loop). Within a ply candidates are deduplicated by whitespace-stripped
text; a candidate whose norm is in the donor's VISITED set (the root and
every prior state of that donor's chain: identity and revisit rejection,
the gate's law) is rejected before the oracle; llmopt.lab.verify.
verify_wave(cur_X, candidates) decides PASS / SOLVED. The CHOSEN CHILD is
the first PASS candidate in sampling order of the first wave that has
one (waves after a PASS wave are not sampled). A chosen SOLVED child ends
the chain (terminal "solved"; its edge is recorded with solved = true);
no PASS in N_WAVES waves ends the chain (terminal "stalled", no edge);
reaching H edges ends it (terminal "horizon"). A chain is the ordered
list of verified edges (cur_t, nxt_t); canonical state identity =
norm(state), canonical edge identity = (norm(cur), norm(nxt)).

Paired accessibility and matched dose: L_pair = min(L_A, L_B) verified
edges; a root is RETAINED iff L_pair >= L_MIN = 2 (both donors advance the
root at least twice: the calibration on the disjoint smoke band found
12-ply paired completion INACCESSIBLE, 0 of 62 pairs reaching the
horizon and none beyond 4 edges, so the multi-ply subset is the only
place iteration can show; ply-0-only pairs repeat the booked one-step
measurement and are excluded); each donor contributes
exactly its first L_pair edges (matched root support, matched row count,
matched ply positions). Chain-dose projection: roots in the fixed order,
retained chains appended until N_ROWS rows per library; the last chain
is truncated at the ply that makes the count exactly N_ROWS (both
donors identically); fewer than N_ROWS rows from the whole population:
CHAIN-INACCESSIBLE (libraries written for the record).

Divergence readouts (on the retained pairs, and the projected
libraries): first divergence ply t* (smallest t in 1..L_pair with
norm(state_A[t]) != norm(state_B[t])); fraction divergent by ply t
among pairs with L_pair >= t; endpoint canonical overlap (state at
L_pair equal); row-position overlap (edge t of A equals edge t of B);
canonical edge-set overlap of the projected libraries; terminal / solve
divergence; the EFFECTIVE DONOR-SPECIFIC ROWS of the projected
libraries: rows of D_A_chain whose canonical edge is not in D_B_chain's
canonical edge set and vice versa (the bar reads the SMALLER side).

Stage law (pure, adjudicate): CHAIN-INACCESSIBLE if the projection is
short; else CHAIN-CONTRAST-FIRES iff min-side donor-specific rows >=
N_SPECIFIC_FIRE; else CHAIN-DEGENERATE.

Precondition REPLAY-EXACT (within-process): after the stage, BOTH donors'
chains are re-walked on the first N_REPLAY retained roots (same seeds,
same law) and must reproduce every edge byte-for-byte (the mps sampler was measured
bit-exact within and across processes on 240 samples before sealing);
any mismatch books the stage NOT-ADJUDICABLE (recorded, not raised).

Outputs: data/crossfoster1_chain/D_A_chain.jsonl, D_B_chain.jsonl
(refuse-if-exists; atoms-shard schema + chain fields), logs/crossfoster1/
chain.json (accounting, per-root chain log, readouts, digests),
logs/crossfoster1/chain.jsonl. SMOKE=1: own root band (8,650,000) and
sample band (200,000,000), SMOKE_PER_LEVEL / SMOKE_ROWS / SMOKE_WAVES /
SMOKE_H knobs, outputs under data/crossfoster1_chain_smoke<SMOKE_TAG>/ and
logs/crossfoster1/chain_smoke<SMOKE_TAG>.jsonl. Usage:
.venv/bin/python scratch/crossfoster_chain.py
"""
import collections
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

import sympy as sp  # noqa: E402
import torch  # noqa: E402

import birth19m_curric as C  # noqa: E402
import crossfoster_donor as CD  # noqa: E402  (loader, digests, sha helper, CENSUS; its own constants are not used here)
import train_mathnative as TM  # noqa: E402
from llmopt.lab.gate import sample_wave_lp  # noqa: E402
from llmopt.lab.gen import _gen_isolated  # noqa: E402
from llmopt.lab.verify import verify_wave  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_TAG = os.environ.get("SMOKE_TAG", "")
LEVELS = [3, 4, 5, 6, 7]
N_PER_LEVEL = {lv: int(os.environ.get("SMOKE_PER_LEVEL", "4")) for lv in LEVELS} if SMOKE else {3: 3600, 4: 9600, 5: 3600, 6: 3600, 7: 3600}   # population 24,000 (shard mix x 4; the calibration's 0.39 rows per processed root needs about 15,500 roots for 6,000)
N_ROWS = int(os.environ.get("SMOKE_ROWS", "8")) if SMOKE else 6000        # the atoms-shard dose, per library
H = int(os.environ.get("SMOKE_H", "12")) if SMOKE else 12                 # horizon (the gate's 12 plies)
L_MIN = 2                                                                 # paired accessibility: min(L_A, L_B) >= L_MIN (MULTI-PLY pairs only: ply-0-only pairs are the booked one-step measurement)
N_WAVES = int(os.environ.get("SMOKE_WAVES", "4")) if SMOKE else 4         # candidate budget per ply: up to 4 waves x 8 (early stop after the first PASS wave)
WAVE = 8
N_SPECIFIC_FIRE = 3000                                                    # min-side donor-specific rows of the 6,000 projected: half the dose
N_REPLAY = 2 if SMOKE else 20                                             # REPLAY-EXACT precondition: donor A re-walked on the first N_REPLAY retained roots
ROOT_BAND = 8_650_000 if SMOKE else 8_600_000                             # fresh bands (8.7M is step_grpo's run-3 gate band; 8.8M / 8.9M are the one-step stage's)
SAMPLE_BAND = 200_000_000 if SMOKE else 100_000_000                       # 100.0M .. 148.0M registered (2000 per root x 24,000 roots); 200M smoke (above the registered span)
I0 = 0
OUT_DATA = Path(f"data/crossfoster1_chain_smoke{SMOKE_TAG}" if SMOKE else "data/crossfoster1_chain")
OUT_LOG = Path("logs/crossfoster1")
OUT = OUT_LOG / (f"chain_smoke{SMOKE_TAG}.jsonl" if SMOKE else "chain.json")
LOG_EVERY = 5 if SMOKE else 100


def adjudicate(n_rows_projected, specific_min_side):
    """Pure stage law. CHAIN-INACCESSIBLE if the projection is short of
    N_ROWS; else CHAIN-CONTRAST-FIRES iff the smaller side's effective
    donor-specific rows >= N_SPECIFIC_FIRE (inclusive); else CHAIN-DEGENERATE."""
    if n_rows_projected < N_ROWS:
        return "CHAIN-INACCESSIBLE"
    if specific_min_side >= N_SPECIFIC_FIRE:
        return "CHAIN-CONTRAST-FIRES"
    return "CHAIN-DEGENERATE"


def root_order():
    roots = [(lv, i) for lv in LEVELS for i in range(N_PER_LEVEL[lv])]
    return sorted(roots, key=lambda r: ((r[1] + 0.5) / N_PER_LEVEL[r[0]], r[0], r[1]))


def ply_seeds(root_index, t, w):
    return [SAMPLE_BAND + 2000 * root_index + 128 * t + WAVE * w + b for b in range(WAVE)]


def walk_chain(model, tok, dev, root_cur, root_index):
    """One donor's verified chain from root_cur under the frozen law.
    Returns (edges, terminal, stats); edges = [{"cur","nxt","solved","ply","sample_seed","wave"}]."""
    cur, visited, edges = root_cur, {norm(root_cur)}, []
    st = collections.Counter()
    terminal = "horizon"
    with torch.no_grad():
        for t in range(H):
            prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
            seen, chosen = set(), None
            for w in range(N_WAVES):
                seeds = ply_seeds(root_index, t, w)
                texts, _, _ = sample_wave_lp(model, tok, prompt, seeds, dev)
                st["samples"] += len(texts)
                cands, cand_seed = [], {}
                for txt, s in zip(texts, seeds):
                    key = txt.replace(" ", "")
                    if not txt or key in seen:
                        continue
                    seen.add(key)
                    st["distinct"] += 1
                    if norm(txt) in visited:
                        st["visited_rejected"] += 1
                        continue
                    cands.append(txt)
                    cand_seed[txt] = (s, w)
                wv = verify_wave(cur, cands) if cands else {}
                for txt in cands:                     # sampling order within the wave
                    ok, solved = wv.get(txt, (False, False))
                    if ok:
                        st["pass"] += 1
                        if chosen is None:
                            chosen = (txt, bool(solved), cand_seed[txt])
                if chosen is not None:
                    st["waves_to_pass"] += w + 1
                    break
            if chosen is None:
                terminal = "stalled"
                st["stall_ply_sum"] += t
                break
            txt, solved, (s, w) = chosen
            edges.append({"cur": cur, "nxt": txt, "solved": solved, "ply": t, "sample_seed": s, "wave": w})
            if solved:
                terminal = "solved"
                break
            cur = txt
            visited.add(norm(txt))
    return edges, terminal, st


def first_divergence(edges_a, edges_b, l_pair):
    """Smallest 1-based ply t <= l_pair with norm(state_A[t]) != norm(state_B[t]); None if none."""
    for t in range(l_pair):
        if norm(edges_a[t]["nxt"]) != norm(edges_b[t]["nxt"]):
            return t + 1
    return None


def project(chains, n_rows):
    """Chain-dose projection: retained (root, L_pair) pairs in order; append
    L_pair rows per library until n_rows, truncating the last chain identically
    for both donors. Returns the list of (root_entry, n_edges_taken)."""
    taken, total = [], 0
    for ch in chains:
        if total >= n_rows:
            break
        k = min(ch["l_pair"], n_rows - total)
        taken.append((ch, k))
        total += k
    return taken, total


def readouts(chains, taken, libs):
    rd = {}
    n = len(chains)
    rd["retained_pairs"] = n
    if n:
        lp = [c["l_pair"] for c in chains]
        rd["l_pair_hist"] = dict(collections.Counter(lp))
        rd["l_pair_median"] = sorted(lp)[n // 2]
        rd["l_A_hist"] = dict(collections.Counter(c["l_A"] for c in chains))
        rd["l_B_hist"] = dict(collections.Counter(c["l_B"] for c in chains))
        rd["terminal_A"] = dict(collections.Counter(c["terminal_A"] for c in chains))
        rd["terminal_B"] = dict(collections.Counter(c["terminal_B"] for c in chains))
        rd["terminal_divergent_frac"] = sum(c["terminal_A"] != c["terminal_B"] for c in chains) / n
        rd["solved_divergent_frac"] = sum((c["terminal_A"] == "solved") != (c["terminal_B"] == "solved") for c in chains) / n
        rd["both_reach_horizon_frac"] = sum(c["terminal_A"] == "horizon" and c["terminal_B"] == "horizon" for c in chains) / n
        rd["first_divergence_hist"] = dict(collections.Counter(str(c["first_divergence"]) for c in chains))
        rd["divergent_frac_by_ply"] = {}
        for t in range(1, H + 1):
            elig = [c for c in chains if c["l_pair"] >= t]
            rd["divergent_frac_by_ply"][t] = {"eligible": len(elig), "frac": (sum(c["first_divergence"] is not None and c["first_divergence"] <= t for c in elig) / len(elig)) if elig else None}
        rd["endpoint_canonical_overlap_frac"] = sum(c["endpoint_equal"] for c in chains) / n
        rd["pair_ever_divergent_frac"] = sum(c["first_divergence"] is not None for c in chains) / n
    # projected libraries
    keyA = {(norm(r["cur"]), norm(r["nxt"])) for r in libs["A"]}
    keyB = {(norm(r["cur"]), norm(r["nxt"])) for r in libs["B"]}
    rd["projected_rows"] = {k: len(v) for k, v in libs.items()}
    rd["projected_roots"] = len(taken)
    rd["canonical_edge_overlap_rows"] = len(keyA & keyB)
    rd["unique_canonical"] = {"A": len(keyA), "B": len(keyB)}
    rd["donor_specific_rows"] = {"A": sum((norm(r["cur"]), norm(r["nxt"])) not in keyB for r in libs["A"]), "B": sum((norm(r["cur"]), norm(r["nxt"])) not in keyA for r in libs["B"])}
    rd["donor_specific_rows_min_side"] = min(rd["donor_specific_rows"].values())
    rd["row_position_overlap"] = sum(norm(a["cur"]) == norm(b["cur"]) and norm(a["nxt"]) == norm(b["nxt"]) for a, b in zip(libs["A"], libs["B"]))
    rd["projected_level_mix"] = dict(collections.Counter(r["level"] for r in libs["A"]))
    rd["projected_ply_mix"] = {k: dict(collections.Counter(r["ply"] for r in libs[k])) for k in libs}
    rd["solved_frac"] = {k: (sum(r["solved"] for r in libs[k]) / max(1, len(libs[k]))) for k in libs}
    rd["projected_rows_by_ply_specific"] = {k: dict(collections.Counter(r["ply"] for r in libs[k] if (norm(r["cur"]), norm(r["nxt"])) not in (keyB if k == "A" else keyA))) for k in libs}
    return rd


def main():
    OUT_LOG.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    if OUT_DATA.exists():
        raise SystemExit(f"REFUSING: {OUT_DATA} exists")
    if not SMOKE and (OUT_LOG / "chain.jsonl").exists():
        raise SystemExit("REFUSING: logs/crossfoster1/chain.jsonl exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if not SMOKE and dirty:
        raise SystemExit("REFUSING: registered chain stage on a dirty tree")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = TM.MathTokenizer()
    assert len(tok.vocab) == 40 and not os.environ.get("VOCAB_EXTRA")
    census = json.loads(CD.CENSUS.read_text())
    donors, drec = {}, {}
    for name in ("A", "B"):
        donors[name], drec[name] = CD.load_donor(name, tok, dev)
        assert drec[name]["state_digest"].startswith(CD.DONOR_DIGEST_PREFIX[name]), f"donor {name}: state digest {drec[name]['state_digest'][:16]} v sealed {CD.DONOR_DIGEST_PREFIX[name]}"
    assert census["artifacts"]["A_paths"]["15420"] == CD.DONORS["A"] and census["artifacts"]["B_paths"]["15420"] == CD.DONORS["B"], "donor paths v census"
    w0_digest = census["w0_seed2_state_digest"]
    assert w0_digest.startswith(CD.W0_DIGEST_PREFIX), "census W_0 digest v sealed"
    stock_rows = C.load_excised_rows()
    diet_norms = set()
    for r in stock_rows:
        diet_norms.add(norm(str(r["cur"])))
        diet_norms.add(norm(str(r["nxt"])))
    band = set(gate_band_exprs())
    rec = {"prereg": "VERIFIED-ENDOGENOUS-DATA-CROSSFOSTER-1-CHAIN-DESK", "stage": "chain", "kind": "crossfoster_chain", "smoke": SMOKE, "commit": commit, "tree_dirty": dirty,
           "source_sha256": CD.sha256_file(__file__), "donor_stage_source_sha256": CD.sha256_file(CD.__file__), "device": dev, "torch_version": torch.__version__, "donors": drec,
           "w0_seed2_state_digest_from_census": w0_digest, "census_sha256": CD.sha256_file(CD.CENSUS), "levels": LEVELS, "n_per_level": N_PER_LEVEL, "n_rows": N_ROWS, "H": H, "L_MIN": L_MIN,
           "n_waves": N_WAVES, "wave": WAVE, "n_specific_fire": N_SPECIFIC_FIRE, "n_replay": N_REPLAY, "root_band": ROOT_BAND, "i0": I0, "sample_band": SAMPLE_BAND,
           "seed_law": "SAMPLE_BAND + 2000 * root_index + 128 * ply + 8 * wave + b", "temperature": 0.7, "max_new": 120, "n_diet_rows": len(stock_rows), "n_diet_norms": len(diet_norms),
           "n_band_exprs": len(band), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    order = root_order()
    acc = collections.Counter()
    stats = {"A": collections.Counter(), "B": collections.Counter()}
    seen_roots = set()
    chains, root_log = [], []
    rows_so_far, processed = 0, 0
    t0 = time.time()
    for root_index, (lv, i) in enumerate(order):
        if rows_so_far >= N_ROWS:
            break
        processed += 1
        seed = ROOT_BAND + 1000 * lv + I0 + i
        p = _gen_isolated(lv, seed)
        if p is None:
            acc["gen_failed"] += 1
            continue
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        cn = norm(cur)
        if cn in band:
            acc["excluded_band"] += 1
            continue
        if cn in diet_norms:
            acc["excluded_diet"] += 1
            continue
        if cn in seen_roots:
            acc["excluded_dup"] += 1
            continue
        seen_roots.add(cn)
        out = {}
        for name in ("A", "B"):
            edges, terminal, st = walk_chain(donors[name], tok, dev, cur, root_index)
            out[name] = (edges, terminal)
            for k, v in st.items():
                stats[name][k] += v
        eA, tA = out["A"]
        eB, tB = out["B"]
        l_pair = min(len(eA), len(eB))
        entry = {"root_index": root_index, "level": lv, "root_seed": seed, "cur": cur, "l_A": len(eA), "l_B": len(eB), "terminal_A": tA, "terminal_B": tB, "l_pair": l_pair}
        if l_pair >= L_MIN:
            acc["retained"] += 1
            entry["retained"] = True
            entry["first_divergence"] = first_divergence(eA, eB, l_pair)
            entry["endpoint_equal"] = norm(eA[l_pair - 1]["nxt"]) == norm(eB[l_pair - 1]["nxt"])
            entry["edges_A"] = eA[:l_pair]
            entry["edges_B"] = eB[:l_pair]
            chains.append(entry)
            rows_so_far += l_pair
        else:
            okA, okB = len(eA) >= L_MIN, len(eB) >= L_MIN
            acc["inadequate_both" if not (okA or okB) else ("inadequate_A" if not okA else "inadequate_B")] += 1
            entry["retained"] = False
        root_log.append({k: v for k, v in entry.items() if k not in ("edges_A", "edges_B")})
        if processed % LOG_EVERY == 0:
            print(f"[chain] processed {processed} retained {acc['retained']} rows {rows_so_far} / {N_ROWS} acc {dict(acc)} ({time.time() - t0:.0f}s)", flush=True)
    wall_stage = time.time() - t0
    taken, total = project(chains, N_ROWS)
    libs = {"A": [], "B": []}
    for ch, k in taken:
        for name in ("A", "B"):
            for e in ch[f"edges_{name}"][:k]:
                libs[name].append({"cur": e["cur"], "nxt": e["nxt"], "level": ch["level"], "rule": "donor-chain", "source": f"donor-{name}", "root_index": ch["root_index"], "root_seed": ch["root_seed"],
                                   "ply": e["ply"], "sample_seed": e["sample_seed"], "wave": e["wave"], "solved": e["solved"], "l_pair": ch["l_pair"], "edges_taken": k})
    OUT_DATA.mkdir(parents=True, exist_ok=False)
    for name in ("A", "B"):
        pth = OUT_DATA / f"D_{name}_chain.jsonl"
        with pth.open("w") as f:
            for r in libs[name]:
                f.write(json.dumps(r) + "\n")
        rec[f"D_{name}_chain"] = {"path": str(pth), "rows": len(libs[name]), "file_sha256": CD.sha256_file(pth)}
    rd = readouts(chains, taken, libs)
    rd["adequacy"] = {"A": (acc["retained"] + acc["inadequate_B"]) / max(1, acc["retained"] + acc["inadequate_A"] + acc["inadequate_B"] + acc["inadequate_both"]),
                      "B": (acc["retained"] + acc["inadequate_A"]) / max(1, acc["retained"] + acc["inadequate_A"] + acc["inadequate_B"] + acc["inadequate_both"])}
    # REPLAY-EXACT precondition (within-process): BOTH donors re-walked on the first N_REPLAY retained roots
    t1 = time.time()
    replay = {"n": 0, "mismatch": 0}
    for ch in chains[:N_REPLAY]:
        for name in ("A", "B"):
            edges, terminal, _ = walk_chain(donors[name], tok, dev, ch["cur"], ch["root_index"])
            replay["n"] += 1
            if [(e["cur"], e["nxt"], e["solved"]) for e in edges[:ch["l_pair"]]] != [(e["cur"], e["nxt"], e["solved"]) for e in ch[f"edges_{name}"]] or len(edges) != ch[f"l_{name}"] or terminal != ch[f"terminal_{name}"]:
                replay["mismatch"] += 1
    replay["exact"] = replay["mismatch"] == 0 and replay["n"] == 2 * min(N_REPLAY, len(chains)) and replay["n"] > 0
    replay["wall_s"] = round(time.time() - t1, 1)
    rec.update({"processed_roots": processed, "population": len(order), "accounting": dict(acc), "sampling_stats": {k: dict(v) for k, v in stats.items()}, "readouts": rd, "replay": replay,
                "wall_stage_s": round(wall_stage, 1), "wall_s": round(time.time() - t0, 1), "root_log": root_log})
    law = adjudicate(total, rd["donor_specific_rows_min_side"])
    rec["status"] = law if replay["exact"] else "NOT-ADJUDICABLE-REPLAY"
    rec["law_ignoring_replay"] = law
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if SMOKE:
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[chain] smoke row appended: {rec['status']} rows {total} readouts {json.dumps({k: v for k, v in rd.items() if k != 'divergent_frac_by_ply'})[:600]}")
    else:
        OUT.write_text(json.dumps(rec, indent=1))
        with (OUT_LOG / "chain.jsonl").open("a") as f:
            f.write(json.dumps({k: v for k, v in rec.items() if k != "root_log"}) + "\n")
        print(f"[chain] written {OUT}: {rec['status']} rows {total} of {N_ROWS} from {len(taken)} roots ({processed} processed); specific {rd['donor_specific_rows']}")


if __name__ == "__main__":
    main()
