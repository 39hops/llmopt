"""MATH-CYBER-1 SVP-GRID-P2-MATERIALIZATION-QUALIFICATION-0 —
the FIRST and FINAL realization of HASH-P2 under the frozen law
of PERMUTATION-REPLICATION-PREREG-0 (commit
7976613b3fef18f54d953ac0404f377e5c74031b). ONE invocation:
implement the salt-changed Feistel exactly as frozen, qualify it
exhaustively, record the preregistered descriptive realization
census, write the receipt. NO model initialization, training,
checkpoint creation, calibration/heldout scoring, or model
inference anywhere in this file.

FROZEN P2 LAW (verbatim from the prereg): for round i and 12-bit
half x, salt_string =
"svp-feistel-p2-7976613b3fef18f54d953ac0404f377e5c74031b-r{i}-{x}"
round output = first two bytes of SHA256(salt_string),
big-endian, modulo 4096; same 4-round balanced 24-bit Feistel as
P1; ONLY the salt differs. P1 and FACTOR behavior are imported
UNCHANGED from the frozen svpcode module and re-qualified here.

ANTI-SELECTION (frozen): this is the first authorized P2
realization — no alternate salt, no reroll, no search, no
adaptation to ANY observed property. Whatever this law yields is
kept and disclosed. A qualification-bar failure books
P2-QUALIFICATION-INSTRUMENT-FAILURE and STOPS; the preregistered
bug-fix carve-out repairs code only, with the salt string
byte-identical forever.

HARD BARS (exhaustive over the full qualified semantic domain,
36 rules x 2 kinds x 64 ordinals x 3 param kinds x 64 indices =
884,736 tuples): every tuple encodes under P2 at width exactly 8
with every atom in 0..7; zero P2 collisions; exact P2
encode/decode roundtrip on all 884,736; P1 collision-freedom and
roundtrip re-established on the same enumeration (P1 unchanged);
FACTOR width/collision/roundtrip re-established (FACTOR
unchanged); Feistel-P2 inverse identity on a deterministic
stride sample of the full 2^24 payload space (construction note:
the balanced Feistel is bijective over all 2^24 payloads for ANY
round function; the stride sample is a code check, not the proof
of bijectivity). Frozen-population coverage: every training row
action (74,860) and every calibration/heldout candidate action
is in-domain and P2-roundtrips exactly — no model is scored and
no new model outcome is revealed.

DESCRIPTIVE-ONLY REALIZATION CENSUS (frozen list; NEVER gates,
can NEVER trigger regeneration): P1-v-P2 and FACTOR-v-P2
whole-code agreement over the qualified domain; prefix agreement
for prefix lengths 1..8; per-position x code-atom target
exposure over the exact frozen 74,860-row training population
for FACTOR/HASH-P1/HASH-P2; aggregate code-atom exposure per
arm; min/median/max direct summaries of those censuses.

P2 REALIZATION PIN for future three-arm drivers: sha256 over the
concatenated P2 codewords in canonical enumeration order
(p2_realization_sha), plus the salt template + this file's
committed sha — deterministic regeneration is the storage.

Outputs under logs/mathworld1/svpp2qual/ (refuse-if-exists):
svpp2qual_receipt.json, census.json.

    .venv/bin/python scratch/mathworld1_svpp2qual.py         (Mac)
"""
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_actiontok import OPCODE_ORDER  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import (ORD_MAX,  # noqa: E402
                                        WIDTH, factor_decode,
                                        factor_symbols,
                                        hash_decode,
                                        hash_symbols, in_domain,
                                        int_to_sym, sym_to_int)

PREREG_SHA = "7976613b3fef18f54d953ac0404f377e5c74031b"
SALT_TEMPLATE = ("svp-feistel-p2-" + PREREG_SHA + "-r{i}-{x}")
PINS = {
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
        "d8402351d468e8",
    "logs/mathworld1/svpdiet/balanced_grid_train.jsonl":
        "0ef3d8a880a7e07712d8de757bc1670df12701e487b856"
        "b44c97f8db16cb3759",
    "logs/mathworld1/svpdiet3/covered_calibration.jsonl":
        "af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b0850979"
        "4d0fb7978db3",
    "logs/mathworld1/svpdiet3/heldout_test16.jsonl":
        "a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec13881"
        "b4509df46ddb",
}
OUTDIR = Path("logs/mathworld1/svpp2qual")
N_DOMAIN_EXPECT = 36 * 2 * 64 * 3 * 64  # 884,736
N_TRAIN_EXPECT = 74860

_R2_CACHE = {}


def _round2(i, x):
    k = (i, x)
    if k not in _R2_CACHE:
        h = hashlib.sha256(
            SALT_TEMPLATE.format(i=i, x=x).encode()).digest()
        _R2_CACHE[k] = int.from_bytes(h[:2], "big") % 4096
    return _R2_CACHE[k]


def feistel2(v, inverse=False):
    L, R = v >> 12, v & 0xFFF
    rounds = range(4)
    if not inverse:
        for i in rounds:
            L, R = R, L ^ _round2(i, R)
    else:
        for i in reversed(rounds):
            L, R = R ^ _round2(i, L), L
    return (L << 12) | R


def hash2_symbols(rule, sk, so, pk, pi):
    return int_to_sym(feistel2(
        sym_to_int(factor_symbols(rule, sk, so, pk, pi))))


def hash2_decode(sym):
    return factor_decode(int_to_sym(
        feistel2(sym_to_int(sym), inverse=True)))


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    # gate-assert the resolved prereg sha from git log (the
    # frozen procedural control on the salt)
    full = subprocess.run(
        ["git", "rev-parse", PREREG_SHA + "^{commit}"],
        capture_output=True, text=True)
    gate(full.returncode == 0
         and full.stdout.strip() == PREREG_SHA,
         "PREREG COMMIT NOT FOUND")
    subj = subprocess.run(
        ["git", "log", "-1", "--format=%s", PREREG_SHA],
        capture_output=True, text=True).stdout
    gate("PERMUTATION-REPLICATION-PREREG-0" in subj,
         f"PREREG COMMIT SUBJECT MISMATCH: {subj!r}")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    START = start_provenance(
        ["scratch/mathworld1_svpp2qual.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_svpbirth.py",
         "llmopt/lab/provenance.py"])

    t0 = time.time()
    # HARD BARS: exhaustive semantic-domain qualification for
    # FACTOR (unchanged), P1 (unchanged), P2 (first realization)
    seen_f, seen_1, seen_2 = set(), set(), set()
    n_dom = 0
    agree_12 = 0     # P1-v-P2 whole-code agreement
    agree_f2 = 0     # FACTOR-v-P2 whole-code agreement
    prefix_12 = [0] * WIDTH
    prefix_f2 = [0] * WIDTH
    p2_stream = hashlib.sha256()
    for r in OPCODE_ORDER:
        for sk in ("W", "I"):
            for so in range(-1, ORD_MAX + 1):
                for pk in ("none", "u_choice", "term_index"):
                    for pi in range(-1, ORD_MAX + 1):
                        n_dom += 1
                        tup = (r, sk, so, pk, pi)
                        fs = factor_symbols(*tup)
                        gate(len(fs) == WIDTH
                             and all(0 <= s < 8 for s in fs),
                             "F WIDTH/ALPHABET")
                        fi = sym_to_int(fs)
                        gate(fi not in seen_f, "F COLLISION")
                        seen_f.add(fi)
                        gate(factor_decode(fs) == tup,
                             "F ROUNDTRIP")
                        h1 = hash_symbols(*tup)
                        i1 = sym_to_int(h1)
                        gate(i1 not in seen_1, "P1 COLLISION")
                        seen_1.add(i1)
                        gate(hash_decode(h1) == tup,
                             "P1 ROUNDTRIP")
                        h2 = hash2_symbols(*tup)
                        gate(len(h2) == WIDTH
                             and all(0 <= s < 8 for s in h2),
                             "P2 WIDTH/ALPHABET")
                        i2 = sym_to_int(h2)
                        gate(i2 not in seen_2, "P2 COLLISION")
                        seen_2.add(i2)
                        gate(hash2_decode(h2) == tup,
                             "P2 ROUNDTRIP")
                        p2_stream.update(bytes(h2))
                        # descriptive agreement tallies (never
                        # gates)
                        if h1 == h2:
                            agree_12 += 1
                        if fs == h2:
                            agree_f2 += 1
                        for k in range(WIDTH):
                            if h1[k] != h2[k]:
                                break
                            prefix_12[k] += 1
                        for k in range(WIDTH):
                            if fs[k] != h2[k]:
                                break
                            prefix_f2[k] += 1
    gate(n_dom == N_DOMAIN_EXPECT, f"DOMAIN SIZE {n_dom}")
    gate(len(seen_f) == len(seen_1) == len(seen_2) == n_dom,
         "INJECTIVITY COUNT")
    p2_realization_sha = p2_stream.hexdigest()

    # Feistel-P2 inverse identity, deterministic stride sample
    # of the full 2^24 payload space (code check; bijectivity is
    # by construction)
    n_inv = 0
    for v in range(0, 2 ** 24, 97003):
        gate(feistel2(feistel2(v), inverse=True) == v,
             "FEISTEL2 INVERSE")
        n_inv += 1

    # frozen-population coverage: training rows + calibration +
    # heldout candidates all in-domain and P2-roundtrip
    def check_action(tup, tag):
        gate(in_domain(*tup), f"{tag} OUT OF DOMAIN {tup}")
        h2 = hash2_symbols(*tup)
        gate(hash2_decode(h2) == tup, f"{tag} P2 ROUNDTRIP")
        return h2

    n_train = 0
    exp_pos = {a: [Counter() for _ in range(WIDTH)]
               for a in ("FACTOR", "HASH_P1", "HASH_P2")}
    for src in ("data/matsub_paired.jsonl",
                "logs/mathworld1/svpdiet/"
                "balanced_grid_train.jsonl"):
        for l in open(src):
            row = json.loads(l)
            tup = (row["rule"], row["site_kind"],
                   row["site_ordinal"], row["param_kind"],
                   row["param_index"])
            h2 = check_action(tup, "TRAIN")
            fs = factor_symbols(*tup)
            h1 = hash_symbols(*tup)
            for k in range(WIDTH):
                exp_pos["FACTOR"][k][fs[k]] += 1
                exp_pos["HASH_P1"][k][h1[k]] += 1
                exp_pos["HASH_P2"][k][h2[k]] += 1
            n_train += 1
    gate(n_train == N_TRAIN_EXPECT, f"TRAIN ROWS {n_train}")
    n_eval_cands = 0
    for src in ("logs/mathworld1/svpdiet3/"
                "covered_calibration.jsonl",
                "logs/mathworld1/svpdiet3/heldout_test16.jsonl"):
        for l in open(src):
            row = json.loads(l)
            for c in row["candidates"]:
                tup = (c["rule"], c["site_kind"],
                       c["site_ordinal"], c["param_kind"],
                       c["param_index"])
                check_action(tup, "EVAL")
                n_eval_cands += 1

    # DESCRIPTIVE CENSUS artifact (never a gate)
    def agg(arm):
        c = Counter()
        for k in range(WIDTH):
            c.update(exp_pos[arm][k])
        return c

    def summary(counter):
        vals = [counter.get(a, 0) for a in range(8)]
        sv = sorted(vals)
        return {"min": sv[0], "median": sv[4],
                "max": sv[-1], "zero_atoms":
                sum(1 for v in vals if v == 0)}

    census = {
        "whole_code_agreement": {
            "P1_v_P2": agree_12, "FACTOR_v_P2": agree_f2,
            "domain": n_dom},
        "prefix_agreement_counts_len1to8": {
            "P1_v_P2": prefix_12, "FACTOR_v_P2": prefix_f2},
        "train_exposure_per_position": {
            a: [{str(k): v for k, v in sorted(c.items())}
                for c in exp_pos[a]]
            for a in exp_pos},
        "train_exposure_aggregate": {
            a: {str(k): v for k, v in sorted(agg(a).items())}
            for a in exp_pos},
        "train_exposure_summary": {
            a: summary(agg(a)) for a in exp_pos},
        "n_train_rows": n_train,
        "n_eval_candidate_actions": n_eval_cands,
        "note": ("descriptive only; never gates; can never "
                 "trigger P2 regeneration or modification")}
    OUTDIR.mkdir(parents=True)
    (OUTDIR / "census.json").write_text(
        json.dumps(census, indent=1))
    census_sha = fsha(OUTDIR / "census.json")

    receipt = {
        "prereg": "MATH-CYBER-1-SVP-GRID-P2-MATERIALIZATION-"
                  "QUALIFICATION-0",
        "verdict": "P2 MATERIALIZED + QUALIFIED",
        "prereg_commit_sha": PREREG_SHA,
        "salt_template": SALT_TEMPLATE,
        "round_law": ("first two bytes of sha256(salt), "
                      "big-endian, mod 4096; 4-round balanced "
                      "24-bit Feistel, halves of 12 bits"),
        "hard_bars": {
            "domain_enumeration_size": n_dom,
            "factor_injective_roundtrip": n_dom,
            "p1_injective_roundtrip": n_dom,
            "p2_injective_roundtrip": n_dom,
            "p2_width8_alphabet8": n_dom,
            "feistel2_inverse_stride_checks": n_inv,
            "train_rows_in_domain_p2_roundtrip": n_train,
            "eval_candidate_actions_in_domain_p2_roundtrip":
                n_eval_cands},
        "p2_realization_sha": p2_realization_sha,
        "census_sha": census_sha,
        "first_and_final_realization": True,
        "no_model_trained_no_score_produced": True,
        "wall_s": round(time.time() - t0, 1),
        "pins": {p: fsha(p) for p in PINS},
        "start": START,
        "completion_commit": completion_commit()}
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    (OUTDIR / "svpp2qual_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "hard_bars",
                       "p2_realization_sha", "wall_s")},
                     indent=1), flush=True)
    print(json.dumps({k: census[k] for k in
                      ("whole_code_agreement",
                       "prefix_agreement_counts_len1to8",
                       "train_exposure_summary")}, indent=1),
          flush=True)
    print("[svpp2qual] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
