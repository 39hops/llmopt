"""ANSWER-FORM-0 + CENSOR-0 — the two desk residues of the
EMITTER-DIVERGES suspect list (RULE-ABLATE bank, RIFF-LEDGER
L5281; ATOM-DOSE-LADDER-1 L29662).

THRESHOLDS, named before counting (committed before the run):

ANSWER-FORM-0 — same integrand, two emitters, diff the answers.
Sample N_AF sympy-shard rows; run axiom emit_chain on the SAME cur
(fork-walled); classes over CO-SOLVED integrands:
  identical    norm-string equal
  formdiff     equivalent_mod_const true, strings differ
  disagree     axiom answer not equivalent-mod-const to sympy's
  SUSPECT-LIVE  formdiff >= 30% of co-solved -> answer form is a
                live mechanism for EMITTER-DIVERGES; a
                normalization arm gets priced.
  SUSPECT-CLEAR formdiff < 10% -> answer form cleared.
  BETWEEN       10-30% -> judgement note, no verdict.

CENSOR-0 — what the 8s L4 wall removed. Re-run N_C fresh L4 seeds
(same generator, fresh band) at a 60s wall, record wall-clock:
  fast      success < 8s (would have survived the farm wall)
  censored  success in [8, 60) s -> the class the wall removed
  hang      no success at 60s
  BIMODAL-CONFIRMED censored < 2% of successes -> the farm's
                    bimodality claim holds; censoring benign.
  WALL-BIASED       censored >= 10% -> the wall removed a real
                    solvable band; shard rule mix is speed-biased
                    (a named EMITTER-DIVERGES mechanism).
  BETWEEN           2-10%.

Fork isolation throughout (NO sympy call is safely boxed by
SIGALRM); workers stream rows incrementally (killed-class
visibility). Rolling pool copied in shape from the frozen
scratch/farm_atoms_axiom.py.

    .venv/bin/python -u scratch/answerform0_censor0.py
"""
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

AXIOM_DIR = "/Users/artin/code/axiom/build-iv7"
AXIOM_SHA = "5a8ae70"
N_AF = int(os.environ.get("N_AF", "300"))
N_C = int(os.environ.get("N_C", "200"))
CENSOR_SEED_BASE = 91_700_000     # fresh band (farm used 72.0M)
WAVE = 10
OUT_AF = Path("logs/emitterdiv/answerform0.jsonl")
OUT_C = Path("logs/emitterdiv/censor0.jsonl")


def _heur_slot(sp):
    """The in-process sympy heurisch slot, verbatim from the frozen
    scratch/farm_atoms_axiom.py — without it axiom cannot emit the
    heurisch class and both desks would measure the wrong emitter."""
    from llmopt.search.rules import i_heurisch
    _ELEM = (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.atan,
             sp.asin, sp.acos, sp.Abs)

    def heur(node_sstr):
        try:
            node = sp.sympify(node_sstr)
            if not isinstance(node, sp.Integral):
                return []
            out = []
            for r in i_heurisch(node):
                if not [f for f in r.atoms(sp.Function)
                        if not isinstance(f, _ELEM)]:
                    out.append(sp.sstr(r))
            return out
        except Exception:
            return []
    return heur


def _af_worker(cur, nxt_sympy, level, q):
    sys.path.insert(0, AXIOM_DIR)
    import axiom_sym as ax
    import sympy as sp
    assert ax.INTERFACE_VERSION == 7 and ax.GIT_SHA.startswith(AXIOM_SHA)
    t0 = time.time()
    try:
        root = ax.parse_sstr(cur)
        r = ax.emit_chain(root, int(level), heurisch=_heur_slot(sp),
                          deadline_ms=45_000)
        rows = r.get("rows", [])
        if len(rows) != 1 or rows[0].get("source") != "axiom-oneply":
            q.put({"cur": cur, "cls": "axiom_no_emit",
                   "wall_s": round(time.time() - t0, 2)})
            return
        nxt_ax = rows[0]["nxt"]
        a = sp.sstr(sp.sympify(nxt_ax)).replace(" ", "")
        b = sp.sstr(sp.sympify(nxt_sympy)).replace(" ", "")
        if a == b:
            cls = "identical"
        elif ax.equivalent_mod_const(ax.parse_sstr(nxt_ax),
                                     ax.parse_sstr(nxt_sympy)):
            cls = "formdiff"
        else:
            cls = "disagree"
        q.put({"cur": cur, "cls": cls, "nxt_sympy": nxt_sympy,
               "nxt_axiom": nxt_ax,
               "wall_s": round(time.time() - t0, 2)})
    except Exception as e:
        q.put({"cur": cur, "cls": "error", "err": type(e).__name__,
               "wall_s": round(time.time() - t0, 2)})


def _c_worker(seed, q):
    sys.path.insert(0, AXIOM_DIR)
    import axiom_sym as ax
    import sympy as sp
    assert ax.INTERFACE_VERSION == 7 and ax.GIT_SHA.startswith(AXIOM_SHA)
    from llmopt.mathgen.problems import make_integrate
    t0 = time.time()
    try:
        p = make_integrate(4, seed)
        root = ax.parse_sstr(sp.sstr(
            sp.Integral(p._expr, sp.Symbol("x"))))
        r = ax.emit_chain(root, 4, heurisch=_heur_slot(sp),
                          deadline_ms=58_000)
        ok = len(r.get("rows", [])) == 1
        q.put({"seed": seed, "ok": ok,
               "wall_s": round(time.time() - t0, 2)})
    except Exception as e:
        q.put({"seed": seed, "ok": False, "err": type(e).__name__,
               "wall_s": round(time.time() - t0, 2)})


def rolling(jobs, worker, wall_s, out_path, tag):
    """Spawn-per-job rolling pool; kill at wall; stream rows."""
    out = out_path.open("a")
    pend, done = [], 0
    jobs = list(jobs)
    results = []
    while jobs or pend:
        while jobs and len(pend) < WAVE:
            args = jobs.pop()
            q = mp.Queue()
            pr = mp.Process(target=worker, args=(*args, q))
            pr.start()
            pend.append((pr, q, time.time(), args))
        nxt_pend = []
        for pr, q, t0, args in pend:
            if not q.empty():
                row = q.get()
                pr.join(1)
                results.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                done += 1
            elif time.time() - t0 > wall_s:
                pr.kill()
                pr.join(1)
                row = {"cls": "wallkill", "ok": False,
                       "wall_s": wall_s, "args0": str(args[0])[:60]}
                results.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                done += 1
            else:
                nxt_pend.append((pr, q, t0, args))
                continue
        pend = nxt_pend
        if done and done % 25 == 0:
            print(f"  [{tag}] {done} done", flush=True)
        time.sleep(0.25)
    out.close()
    return results


def main():
    os.makedirs("logs/emitterdiv", exist_ok=True)
    # ---- ANSWER-FORM-0
    rows = [json.loads(line)
            for line in open("data/micromodel_atoms_shard0.jsonl")]
    random.Random("answerform0-v1").shuffle(rows)
    sample = [(r["cur"], r["nxt"], r["level"]) for r in rows[:N_AF]]
    t0 = time.time()
    res = rolling(sample, _af_worker, 50, OUT_AF, "af")
    c = Counter(r["cls"] for r in res)
    cosolved = c["identical"] + c["formdiff"] + c["disagree"]
    print(f"[af] {dict(c)} co-solved={cosolved} "
          f"formdiff={100*c['formdiff']/max(cosolved,1):.2f}% "
          f"({time.time()-t0:.0f}s)", flush=True)
    # ---- CENSOR-0
    seeds = [(CENSOR_SEED_BASE + i,) for i in range(N_C)]
    t0 = time.time()
    res = rolling(seeds, _c_worker, 60, OUT_C, "c")
    okr = [r for r in res if r.get("ok")]
    fast = sum(1 for r in okr if r["wall_s"] < 8)
    cens = sum(1 for r in okr if 8 <= r["wall_s"] < 60)
    hang = len(res) - len(okr)
    print(f"[c] successes={len(okr)} fast(<8s)={fast} "
          f"censored(8-60s)={cens} "
          f"({100*cens/max(len(okr),1):.2f}% of successes) "
          f"hang/fail={hang} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
