"""Successors-bridge acceptance (house side; axiom spec
2026-07-27-successors-bridge, relay -28-2). 500 string-seeded
gen-4-band roots (L1-L8), house derivation.successors vs
axiom_sym.successors, E4-taxonomy decomposition:
- MATCH: child sets equal (sympy-srepr normalized)
- HOUSE_ONLY / AXIOM_ONLY children (named, sampled)
- I-FENCE: complex-carrier states skipped (axiom domain fence)
- EXPIRED: axiom deadline states (censored, never counted false)
Soundness leg: every axiom-only child re-verified on the HOUSE
oracle (verify_edge) — axiom emissions must never fail it.
Throughput logged both sides.
"""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402

import axiom_sym as ax  # noqa: E402
from llmopt.search.derivation import State, successors, verify_edge  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402

SEED = 99_400_000
N_PER = 63  # x8 levels ~ 500
DEADLINE_MS = 15000

roots = []
for lv in range(1, 9):
    for i in range(N_PER):
        p = _gen_isolated(lv, SEED + 1000 * lv + i)
        if p is not None:
            roots.append((lv, sp.Integral(p._expr, sp.Symbol("x"))))
roots = roots[:500]
print(f"{len(roots)} roots", flush=True)


def norm(e):
    try:
        return sp.srepr(sp.sympify(e) if isinstance(e, str) else e)
    except Exception:
        return f"UNPARSED:{e}"


match = house_only_states = axiom_only_states = both_diff = 0
expired = ifence = 0
t_house = t_ax = 0.0
sound_checked = sound_fail = 0
diags = []
for si, (lv, root) in enumerate(roots):
    rs = sp.sstr(root)
    if "I" in {str(a) for a in root.atoms(sp.I)}:
        ifence += 1
        continue
    t0 = time.time()
    hk = {norm(s.expr) for _, s in successors(State(root))}
    t_house += time.time() - t0
    t0 = time.time()
    try:
        r = ax.successors(ax.parse_sstr(rs), True, DEADLINE_MS)
    except Exception as ex:
        diags.append((lv, rs[:60], f"AX_PARSE:{ex}"))
        ifence += 1
        continue
    t_ax += time.time() - t0
    if r["expired"]:
        expired += 1
        continue
    ak_pairs = [(n, str(c)) for n, c in r["rows"]]
    ak = {norm(c) for _, c in ak_pairs}
    if hk == ak:
        match += 1
        continue
    ho, ao = hk - ak, ak - hk
    if ho and ao:
        both_diff += 1
    elif ho:
        house_only_states += 1
    else:
        axiom_only_states += 1
    if len(diags) < 30:
        diags.append((lv, rs[:60],
                      f"house_only={len(ho)} axiom_only={len(ao)}"))
    # soundness: axiom-only children through the HOUSE oracle
    for n, cs in ak_pairs:
        if norm(cs) in ao and sound_checked < 200:
            sound_checked += 1
            try:
                ok = verify_edge(root, sp.sympify(cs))
            except Exception:
                ok = False
            if not ok:
                sound_fail += 1
                diags.append((lv, rs[:50], f"UNSOUND? {n}: {cs[:60]}"))
    if (si + 1) % 100 == 0:
        print(f"[{si+1}] match {match}", flush=True)

n_read = len(roots) - ifence - expired
print(f"\nACCEPTANCE CARD: read {n_read} | MATCH {match} "
      f"({100*match/max(n_read,1):.1f}%) | house-only {house_only_states} "
      f"| axiom-only {axiom_only_states} | both-diff {both_diff} "
      f"| expired {expired} | fence-skip {ifence}", flush=True)
print(f"soundness: {sound_checked} axiom-only children house-verified, "
      f"{sound_fail} failures", flush=True)
print(f"throughput: house {n_read/max(t_house,1e-9):.1f} st/s | "
      f"axiom {n_read/max(t_ax,1e-9):.1f} st/s", flush=True)
for d in diags[:30]:
    print("  DIAG", d, flush=True)
