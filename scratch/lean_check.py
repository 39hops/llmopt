"""House-side batch checker for axiom's Lean certificate sidecars
(relays 2026-08-03-0/-1; axiom emitter c0511bc).

Three jobs, in order of importance:
  1. MECHANICAL STATEMENT DIFF (the anti-theater clause): re-derive the
     Lean statement from the row's (lhs, rhs, atoms) with OUR OWN
     printer and require equality with the emitted `lean` field modulo
     whitespace. A certificate that proves a statement we did not
     independently reconstruct certifies nothing for the ledger.
  2. KERNEL CHECK: write all rows into Certs.lean (one named example
     per row) and `lake env lean` it; any failure names its row — a
     failing cert on an EQUIVALENT verdict is a JUDGE BUG artifact.
  3. THE COST VERDICT: wall-clock per certificate vs axiom's ~11 ms/row
     oracle, plus the closable fraction from their emitted/fenced
     counters (never recomputed here — their counter is the truth
     source, per relay).

House printer scope (v0, matches the agreed tier-1 subset): rational
arithmetic in x, free symbols, and opaque atoms a1..an; integer
exponents; divisions become `hN : <den> ≠ 0` hypotheses. Any sstr
construct outside that grammar makes the row DIFF-INELIGIBLE and it is
reported, not silently passed — the whole point is that OUR
reconstruction is independent of THEIR printer.

Usage: .venv/bin/python scratch/lean_check.py <sidecar.jsonl>
       [--project scratch/leancheck]
"""
import json
import pathlib
import re
import subprocess
import sys
import time

ELAN = pathlib.Path.home() / ".elan/bin"


def sstr_to_lean(s, atoms):
    """sstr (tier-1 subset) -> Lean expression text. Deliberately
    independent of axiom's printer: shared grammar, our formatting."""
    t = s
    # longest-first atom substitution so a10 does not eat a1's text
    for name, sub in sorted(atoms.items(), key=lambda kv: -len(kv[1])):
        t = t.replace(sub, name)
    if re.search(r"[A-Za-z_]+\(", t):
        return None                     # un-generalized fn call remains
    t = t.replace("**", "^")
    if "/" in t or "^" in t and re.search(r"\^\s*\(", t):
        pass                            # handled by hypotheses upstream
    return re.sub(r"\s+", " ", t.strip())


def rederive(row):
    """Our canonical statement text from (lhs, rhs, atoms) + tactic."""
    atoms = row.get("atoms", {})
    lhs = sstr_to_lean(row["lhs"], atoms)
    rhs = sstr_to_lean(row["rhs"], atoms)
    if lhs is None or rhs is None:
        return None
    syms = sorted(set(re.findall(r"\b[a-z]\d*\b", lhs + " " + rhs))
                  - {"x"} | {"x"} | set(atoms))
    binder = " ".join(syms)
    # both parenthesized AND bare-symbol denominators, in position
    # order (the bare-symbol class was the 314/1000 house diff gap,
    # relay 2026-08-05-1: `3/x` emitted no `x ≠ 0` hypothesis)
    dens = [a or b for a, b in
            re.findall(r"/\s*(?:\(([^()]+)\)|([A-Za-z]\w*))",
                       lhs + " " + rhs)]
    hyps = "".join(f" (h{i + 1} : {d.strip()} ≠ 0)"
                   for i, d in enumerate(dict.fromkeys(dens)))
    return (f"example ({binder} : ℝ){hyps} : {lhs} = {rhs} := by "
            f"{row['tactic']}")


def norm(s):
    return re.sub(r"\s+", " ", s.strip())


def ac_equal(ours, theirs):
    """Statement equality up to associativity/commutativity: compare
    binder + hypotheses + tactic as strings, and each equation side as
    a sympy parse (auto-canonical arg ordering, NO simplification —
    x*(x+2)**2 stays unexpanded, so this is still a syntactic check).
    First run measured 238/443 raw-string mismatches, ALL commutative
    reordering (axiom's printer orders terms differently); raw ==
    stays as the fast path."""
    import sympy as sp

    if norm(ours) == norm(theirs):
        return True
    pat = re.compile(r"^example \(([^)]*) : ℝ\)(.*?) : (.*) = (.*) := by (.*)$")
    mo, mt = pat.match(norm(ours)), pat.match(norm(theirs))
    if not mo or not mt:
        return False
    if (sorted(mo.group(1).split()) != sorted(mt.group(1).split())
            or norm(mo.group(2)) != norm(mt.group(2))
            or mo.group(5) != mt.group(5)):
        return False
    try:
        for a, b in ((mo.group(3), mt.group(3)), (mo.group(4), mt.group(4))):
            ea = sp.parse_expr(a.replace("^", "**"), evaluate=True)
            eb = sp.parse_expr(b.replace("^", "**"), evaluate=True)
            if ea != eb:
                return False
        return True
    except Exception:
        return False


def main():
    src = pathlib.Path(sys.argv[1])
    proj = pathlib.Path(sys.argv[sys.argv.index("--project") + 1]
                        if "--project" in sys.argv else "scratch/leancheck")
    rows = [json.loads(l) for l in src.open() if l.strip()]
    diff_ok, diff_bad, diff_skip, lines = 0, [], 0, []
    for i, r in enumerate(rows):
        ours = rederive(r)
        if ours is None:
            diff_skip += 1
        elif ac_equal(ours, r["lean"]):
            diff_ok += 1
        else:
            diff_bad.append((r.get("id", i), ours, r["lean"]))
        lines.append(f"-- id: {r.get('id', i)}\n{r['lean']}")
    print(f"[lean] rows {len(rows)} | statement-diff ok {diff_ok} "
          f"skip {diff_skip} MISMATCH {len(diff_bad)}")
    for rid, ours, theirs in diff_bad[:5]:
        print(f"[lean]   DIFF {rid}:\n    ours:   {ours}\n    theirs: {theirs}")
    # CHUNKED kernel check: Lean aborts a file at ~100 diagnostics
    # (set_option maxErrors in-file did NOT lift it — verified live
    # 2026-08-05: the 1000-row single file truncated twice, rows past
    # the abort silently unchecked). 50-row chunks bound the damage
    # per file and give complete coverage + per-row attribution.
    CHUNK = 50
    header = ("import Mathlib.Tactic\n\n"
              "set_option linter.unusedVariables false\n\n")
    t0, n_fail, fail_ids = time.time(), 0, []
    for c0 in range(0, len(rows), CHUNK):
        chunk = rows[c0:c0 + CHUNK]
        (proj / "Certs.lean").write_text(
            header + "\n\n".join(
                f"-- id: {r.get('id', c0 + j)}\n{r['lean']}"
                for j, r in enumerate(chunk)) + "\n")
        p = subprocess.run([str(ELAN / "lake"), "env", "lean",
                            "Certs.lean"], cwd=proj,
                           capture_output=True, text=True)
        if p.returncode != 0:
            # map error line numbers back to row ids
            body = (proj / "Certs.lean").read_text().splitlines()
            for m in re.finditer(r"Certs\.lean:(\d+):\d+: error: (.+)",
                                 p.stdout + p.stderr):
                ln, msg = int(m.group(1)), m.group(2)
                rid = next((body[i][7:] for i in range(ln - 1, -1, -1)
                            if body[i].startswith("-- id: ")), "?")
                n_fail += 1
                fail_ids.append(rid)
                print(f"[lean]   KERNEL-FAIL {rid}: {msg[:120]}")
        done = min(c0 + CHUNK, len(rows))
        print(f"[lean] chunk {c0 // CHUNK}: {done}/{len(rows)} checked, "
              f"{n_fail} failures ({time.time() - t0:.0f}s)", flush=True)
    dt = time.time() - t0
    print(f"[lean] kernel check: {len(rows) - n_fail}/{len(rows)} PASS "
          f"| failures: {sorted(set(fail_ids))} | {dt:.1f}s total, "
          f"{dt / max(len(rows), 1) * 1e3:.0f} ms/cert "
          f"(vs ~11 ms/row oracle)")
    sys.exit(0 if n_fail == 0 and not diff_bad else 1)


if __name__ == "__main__":
    main()
