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
    dens = re.findall(r"/\s*\(([^()]+)\)", lhs + " " + rhs)
    hyps = "".join(f" (h{i} : {d.strip()} ≠ 0)"
                   for i, d in enumerate(dict.fromkeys(dens)))
    return (f"example ({binder} : ℝ){hyps} : {lhs} = {rhs} := by "
            f"{row['tactic']}")


def norm(s):
    return re.sub(r"\s+", " ", s.strip())


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
        elif norm(ours) == norm(r["lean"]):
            diff_ok += 1
        else:
            diff_bad.append((r.get("id", i), ours, r["lean"]))
        lines.append(f"-- id: {r.get('id', i)}\n{r['lean']}")
    (proj / "Certs.lean").write_text(
        "import Mathlib.Tactic\n\n" + "\n\n".join(lines) + "\n")
    print(f"[lean] rows {len(rows)} | statement-diff ok {diff_ok} "
          f"skip {diff_skip} MISMATCH {len(diff_bad)}")
    for rid, ours, theirs in diff_bad[:5]:
        print(f"[lean]   DIFF {rid}:\n    ours:   {ours}\n    theirs: {theirs}")
    t0 = time.time()
    p = subprocess.run([str(ELAN / "lake"), "env", "lean", "Certs.lean"],
                       cwd=proj, capture_output=True, text=True)
    dt = time.time() - t0
    ok = p.returncode == 0
    print(f"[lean] kernel check: {'PASS' if ok else 'FAIL'} | "
          f"{dt:.1f}s total, {dt / max(len(rows), 1) * 1e3:.0f} ms/cert "
          f"(vs ~11 ms/row oracle)")
    if not ok:
        print(p.stdout[-2000:] or p.stderr[-2000:])
    sys.exit(0 if ok and not diff_bad else 1)


if __name__ == "__main__":
    main()
