"""MATH-CYBER-1 SVP-CODE-QUALIFY-0 — the FACTOR-OPAQUE /
HASH-OPAQUE encoding pair, defined and qualified BEFORE any
birth prereg or third-band materialization. Zero training, zero
checkpoint access, zero new band.

Design (frozen here):
- CODE ALPHABET: 8 dedicated atoms <c:0>..<c:7> (base-8),
  appended to ActionGCTok-332 in every arm -> vocab 340. No pad
  and no <c:end> token: the payload is ALWAYS exactly 8 symbols
  and the standing model EOS terminates the continuation, so
  T = 9 for every action in both opaque arms.
- QUALIFIED TREATMENT DOMAIN (registered; overflow = NO-FIRE):
  rule in OPCODE_ORDER (36); site_kind in {W, I} (W = the
  no-site sentinel); site_ordinal in {-1} u [0, 62]; param_kind
  in {none, u_choice, term_index}; param_index in {-1} u
  [0, 62]. Sentinels encode as value+1, so each of ordinal and
  param_index occupies 0..63. The engine does not hard-bound
  ordinals, so the domain is SCOPED, not proven; any
  out-of-domain action hard-exits (the registered instrument
  NO-FIRE), never silently truncates.
- FACTOR payload F(A), 24 bits as 8 base-8 symbols with factor
  boundaries preserved:
    [r1 r2] rule index 0..35 as two base-8 digits
    [k]     site_kind index (W=0, I=1)
    [o1 o2] site_ordinal+1 as two base-8 digits (0..63)
    [pk]    param_kind index (none=0, u_choice=1, term_index=2)
    [x1 x2] param_index+1 as two base-8 digits (0..63)
  Injective by construction (each coordinate occupies disjoint
  positions with full range).
- HASH(A) = P(F(A)) where P is a 4-round balanced Feistel
  bijection on the full 24-bit payload space (halves of 12
  bits; round function = sha256("svp-feistel-r{i}-{x}") mod
  2^12, constants frozen HERE, before any third band exists).
  P mixes information ACROSS positions (position-wise renaming
  would preserve factor boundaries and is explicitly not used).
  Bijective on all 2^24 payloads by the Feistel property;
  decode = inverse rounds. NOTE (registered wording): a
  deterministic bijection PRESERVES whole-action identity
  entropy; what the treatment alters is the tokenwise/
  autoregressive conditional structure — HASH does not "match
  entropy per token" and is never described that way.

Qualification bars (this driver, receipt
logs/mathworld1/svpcode_receipt.json):
 1 FACTOR-INJECTIVE on the full qualified domain (exhaustive
   36*2*64*3*64 enumeration: zero collisions).
 2 HASH-BIJECTIVE: Feistel inverse roundtrips the same full
   enumeration exactly; zero collisions.
 3 EVERY observed action (73,324 training rows + every
   candidate of BOTH frozen bands) is in-domain, encodes to
   width 8 over the 8-atom alphabet, and decodes back to its
   exact tuple, under BOTH encodings.
 4 SAME-WIDTH/SAME-ALPHABET: both encodings, every action.
 5 EXPOSURE: per-atom target-token counts over the 73,324
   training rows for both encodings; zero-exposure atoms,
   min nonzero, p10/median/max.
 6 FIXED-WIDTH RANK IDENTITY (design registration): with all
   candidate T identical (=9), mean-lp ranking == summed-lp
   ranking decision-by-decision in both opaque arms; registered
   as a hard instrument gate for the future scorer.

    .venv/bin/python scratch/mathworld1_svpcode.py            (Mac)
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_actiontok import OPCODE_ORDER  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402

PINS = {
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
        "d8402351d468e8",
    "logs/mathworld1/svpeval/decisions.jsonl":
        "f63100a62f3091d544750d679483009a473261c587f3165241"
        "406a86253858c6",
    "logs/mathworld1/svpeval2/decisions.jsonl":
        "89efbe0ea447ee937c0c130d5419112921a2dd6c2159c6c211"
        "2cfd5e92f79315",
}
RECEIPT = Path("logs/mathworld1/svpcode_receipt.json")
WIDTH = 8
ALPHABET = [f"<c:{i}>" for i in range(8)]
RULE_IDX = {r: i for i, r in enumerate(OPCODE_ORDER)}
KIND_IDX = {"W": 0, "I": 1}
PK_IDX = {"none": 0, "u_choice": 1, "term_index": 2}
ORD_MAX = 62  # scoped domain bound; overflow hard-exits


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def in_domain(rule, sk, so, pk, pi):
    return (rule in RULE_IDX and sk in KIND_IDX
            and -1 <= so <= ORD_MAX and pk in PK_IDX
            and -1 <= pi <= ORD_MAX)


def factor_symbols(rule, sk, so, pk, pi):
    """F(A): 8 base-8 symbols, factor boundaries preserved."""
    gate(in_domain(rule, sk, so, pk, pi),
         f"OUT OF QUALIFIED DOMAIN {(rule, sk, so, pk, pi)}")
    r = RULE_IDX[rule]
    o = so + 1
    x = pi + 1
    return [r // 8, r % 8, KIND_IDX[sk], o // 8, o % 8,
            PK_IDX[pk], x // 8, x % 8]


def factor_decode(sym):
    r = sym[0] * 8 + sym[1]
    o = sym[3] * 8 + sym[4]
    x = sym[6] * 8 + sym[7]
    gate(r < 36 and sym[2] < 2 and sym[5] < 3 and o < 64
         and x < 64, "FACTOR DECODE RANGE")
    inv_r = OPCODE_ORDER[r]
    inv_k = {v: k for k, v in KIND_IDX.items()}[sym[2]]
    inv_p = {v: k for k, v in PK_IDX.items()}[sym[5]]
    return (inv_r, inv_k, o - 1, inv_p, x - 1)


def sym_to_int(sym):
    v = 0
    for s in sym:
        v = v * 8 + s
    return v


def int_to_sym(v):
    out = []
    for _ in range(WIDTH):
        out.append(v % 8)
        v //= 8
    return out[::-1]


_ROUND_CACHE = {}


def _round(i, x):
    k = (i, x)
    if k not in _ROUND_CACHE:
        h = hashlib.sha256(
            f"svp-feistel-r{i}-{x}".encode()).digest()
        _ROUND_CACHE[k] = int.from_bytes(h[:2], "big") % 4096
    return _ROUND_CACHE[k]


def feistel(v, inverse=False):
    L, R = v >> 12, v & 0xFFF
    rounds = range(4)
    if not inverse:
        for i in rounds:
            L, R = R, L ^ _round(i, R)
    else:
        for i in reversed(rounds):
            L, R = R ^ _round(i, L), L
    return (L << 12) | R


def hash_symbols(rule, sk, so, pk, pi):
    return int_to_sym(feistel(
        sym_to_int(factor_symbols(rule, sk, so, pk, pi))))


def hash_decode(sym):
    return factor_decode(int_to_sym(
        feistel(sym_to_int(sym), inverse=True)))


def main():
    if RECEIPT.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN {p}")
    START = start_provenance(
        ["scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_svpbirth.py",
         "llmopt/lab/provenance.py"])

    # bars 1+2: exhaustive domain enumeration
    seen_f, seen_h = set(), set()
    n_dom = 0
    for r in OPCODE_ORDER:
        for sk in ("W", "I"):
            for so in range(-1, ORD_MAX + 1):
                for pk in ("none", "u_choice", "term_index"):
                    for pi in range(-1, ORD_MAX + 1):
                        n_dom += 1
                        fs = factor_symbols(r, sk, so, pk, pi)
                        gate(len(fs) == WIDTH
                             and all(0 <= s < 8 for s in fs),
                             "F WIDTH/ALPHABET")
                        fi = sym_to_int(fs)
                        gate(fi not in seen_f, "F COLLISION")
                        seen_f.add(fi)
                        gate(factor_decode(fs)
                             == (r, sk, so, pk, pi),
                             "F ROUNDTRIP")
                        hs = hash_symbols(r, sk, so, pk, pi)
                        gate(len(hs) == WIDTH
                             and all(0 <= s < 8 for s in hs),
                             "H WIDTH/ALPHABET")
                        hi = sym_to_int(hs)
                        gate(hi not in seen_h, "H COLLISION")
                        seen_h.add(hi)
                        gate(hash_decode(hs)
                             == (r, sk, so, pk, pi),
                             "H ROUNDTRIP")
    # full-space bijection spot: feistel inverse on random-free
    # deterministic sample of the 2^24 space
    for v in range(0, 2 ** 24, 97003):
        gate(feistel(feistel(v), inverse=True) == v,
             "FEISTEL NOT BIJECTIVE")

    # bar 3: every observed action (training + both bands)
    def obs_actions():
        for l in open("data/matsub_paired.jsonl"):
            r = json.loads(l)
            yield ("train", r["rule"], r["site_kind"],
                   r["site_ordinal"], r["param_kind"],
                   r["param_index"])
        for src, p in (("old", PINS_OLD), ("new", PINS_NEW)):
            for l in open(p):
                r = json.loads(l)
                for c in r.get("candidates", []):
                    yield (src, c["rule"], c["site_kind"],
                           c["site_ordinal"], c["param_kind"],
                           c["param_index"])

    PINS_OLD = "logs/mathworld1/svpeval/decisions.jsonl"
    PINS_NEW = "logs/mathworld1/svpeval2/decisions.jsonl"
    f_expo = Counter()
    h_expo = Counter()
    counts = Counter()
    for src, r, sk, so, pk, pi in obs_actions():
        counts[src] += 1
        fs = factor_symbols(r, sk, so, pk, pi)
        hs = hash_symbols(r, sk, so, pk, pi)
        gate(factor_decode(fs) == (r, sk, so, pk, pi), "F OBS")
        gate(hash_decode(hs) == (r, sk, so, pk, pi), "H OBS")
        if src == "train":
            f_expo.update(fs)
            h_expo.update(hs)
    gate(counts["train"] == 73324, "TRAIN N")

    def expo_report(c):
        vals = sorted(c.get(i, 0) for i in range(8))
        nz = [v for v in vals if v > 0]
        return {"per_atom": {f"<c:{i}>": c.get(i, 0)
                             for i in range(8)},
                "zero_exposure_atoms": 8 - len(nz),
                "min_nonzero": min(nz) if nz else None,
                "p10": vals[0], "median": vals[4],
                "max": vals[-1]}

    # pricing (T = WIDTH + 1 model eos)
    T = WIDTH + 1
    per_epoch = T * 73324
    receipt = {
        "alphabet": ALPHABET, "width": WIDTH,
        "eos_law": "payload + standing model EOS; T = 9 for "
                   "every action, both opaque arms",
        "vocab": {"base": 332, "code_atoms": 8, "total": 340},
        "domain": {"rules": 36, "kinds": 2,
                   "ordinal": [-1, ORD_MAX],
                   "param_kinds": 3,
                   "param_index": [-1, ORD_MAX],
                   "scoped_not_proven": True,
                   "overflow_law": "hard-exit NO-FIRE"},
        "bars": {
            "FACTOR_INJECTIVE": True,
            "HASH_BIJECTIVE": True,
            "ZERO_F_COLLISIONS": True,
            "ZERO_H_COLLISIONS": True,
            "ROUNDTRIP_EXACT": True,
            "SAME_WIDTH_ALPHABET": True,
            "ALL_OBSERVED_ENCODE_DECODE": True,
            "note": "all gate-enforced hard exits; receipt "
                    "existence = pass",
        },
        "domain_enumeration_size": n_dom,
        "observed_actions": dict(counts),
        "exposure_train": {"FACTOR": expo_report(f_expo),
                           "HASH": expo_report(h_expo)},
        "pricing": {
            "T_per_action": T,
            "continuation_tokens_per_epoch": per_epoch,
            "continuation_tokens_3_epochs": 3 * per_epoch,
            "canonical_program_3_epochs": 1341459,
            "state_3_epochs": 4871751,
            "ratio_v_canonical_program": round(
                3 * per_epoch / 1341459, 4),
            "ratio_v_state": round(
                4871751 / (3 * per_epoch), 4)},
        "rank_identity_registration":
            "fixed width => mean-lp ranking == summed-lp "
            "ranking per decision in both opaque arms; any "
            "disagreement is an instrument bug (hard gate in "
            "the future scorer)",
        "pins": {p: fsha(p) for p in PINS},
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("start", "pins")}, indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
