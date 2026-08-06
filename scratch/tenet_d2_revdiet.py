"""TENET D2: the certified reversed diet + exclude-union semantics
(spec 2026-08-05-tenet-battery.md).

A gen-4 row reverses ONLY if nxt is a legal engine child of cur
(replay successors(State(cur), use_macros=True) and skeleton-match —
the R1b mechanism, ~67-68% of the corpus). The MISS population
(const-of-integration offsets, multi-rule skips — oracle-valid but
not single engine moves) is EXCLUDED, never silently reversed: its
reversal has no well-defined predecessor semantics. Outputs, at
matched dose (paired-arms doctrine):
  data/gen4_reverse_certified.jsonl  — certified rows, cur/nxt swapped
  data/gen4_forward_certified.jsonl  — SAME rows, forward (the
                                       matched-dose control diet)
  data/gen4_replay_status.jsonl      — per-row status stream +
                                       terminal record

Exclude-union law (D2 proper): reversal swaps prompt/target, so the
exclude set for ANY reversed corpus is the UNION of both directions'
normalized string sets — which equals {cur} U {nxt}, reversal-
invariant. The builder checks the 120 gate-band expressions against
that union and refuses to write diets on a hit.

Loud-failure contract: replay runs in a pool of subprocess
line-servers (the oracle_worker v3.2 pattern: per-row wall,
parent-side RSS watchdog via `ps -o rss=` — Darwin ignores
RLIMIT_AS/DATA; jetsam kills the LARGEST process), every anomaly
typed {TIMEOUT, MEMBOMB, CRASH} + counted + streamed; rows stream
per result, never buffer; a terminal record closes the status file
(analyses refuse a jsonl without one). Restart is lossless: existing
status rows are honored, only the remainder replays.

Usage:  .venv/bin/python scratch/tenet_d2_revdiet.py         # full
        N=300 .venv/bin/python scratch/tenet_d2_revdiet.py   # smoke
        WORKERS=9 WALL=30 override the pool.
"""
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

STATUS = Path("data/gen4_replay_status.jsonl")
REV_OUT = Path("data/gen4_reverse_certified.jsonl")
FWD_OUT = Path("data/gen4_forward_certified.jsonl")
WALL = float(os.environ.get("WALL", "30"))
MEM_CAP_KB = 3 << 20  # 3 GB per worker
norm = lambda s: s.replace(" ", "")  # noqa: E731


def worker_main():
    """Line server: json [cur, nxt] per line -> status."""
    import sympy as sp

    from llmopt.search.derivation import State, successors

    for line in sys.stdin:
        try:
            cur_s, nxt_s = json.loads(line)
            cur = sp.sympify(cur_s)
            target = norm(sp.sstr(sp.sympify(nxt_s)))
            names = [name for name, ch
                     in successors(State(cur), use_macros=True)
                     if norm(sp.sstr(ch.expr)) == target]
            status = ("unique" if len(names) == 1
                      else "ambig" if names else "miss")
        except Exception:
            status = "err"
        print(status, flush=True)


class Replayer:
    """One line-server with the v3.2 loud-failure contract."""

    def __init__(self):
        self.proc = None
        self.task = None       # (key, row) in flight
        self.deadline = 0.0

    def start(self):
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def send(self, key, row):
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        payload = json.dumps([row["cur"], row["nxt"]])
        try:
            self.proc.stdin.write(payload + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            self.proc = None
            return False
        self.task, self.deadline = (key, row), time.monotonic() + WALL
        return True

    def poll(self, counts):
        """-> (key, row, status) if this worker resolved, else None."""
        if self.task is None:
            return None
        ready, _, _ = select.select([self.proc.stdout], [], [], 0)
        if ready:
            line = self.proc.stdout.readline()
            key, row = self.task
            self.task = None
            if line == "":  # EOF: worker died mid-replay
                print(f"[d2] REPLAY-CRASH (eof) on {key[0][:40]} — "
                      "respawn, row excluded", flush=True)
                self.proc = None
                counts["crash"] += 1
                return (key, row, "crash")
            return (key, row, line.strip())
        rss = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.proc.pid)],
            capture_output=True, text=True).stdout.strip()
        if rss and int(rss) > MEM_CAP_KB:
            key, row = self.task
            print(f"[d2] REPLAY-MEMBOMB rss={int(rss) >> 10}MB on "
                  f"{key[0][:40]} — killed, row excluded", flush=True)
            self.kill()
            counts["membomb"] += 1
            return (key, row, "membomb")
        if time.monotonic() > self.deadline:
            key, row = self.task
            print(f"[d2] REPLAY-TIMEOUT ({WALL:.0f}s) on "
                  f"{key[0][:40]} — killed, row excluded", flush=True)
            self.kill()
            counts["timeout"] += 1
            return (key, row, "timeout")
        return None

    def check_sync(self, cur, nxt, counts):
        """Blocking single check -> status string (D1 reuses this:
        'unique'/'ambig' = nxt IS a legal engine child of cur)."""
        if not self.send(("sync", None), {"cur": cur, "nxt": nxt}):
            counts["crash"] += 1
            return "crash"
        while True:
            res = self.poll(counts)
            if res is not None:
                status = res[2]
                if status in ("unique", "ambig", "miss", "err"):
                    counts[status] += 1  # anomalies counted by poll
                return status
            time.sleep(0.02)

    def kill(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc = None
        self.task = None


def gate_band_exprs():
    import sympy as sp

    import step_grpo_micro as G
    from bench_step_tokens import _gen_isolated

    out = []
    for lv in G.GATE_LEVELS:
        for i in range(G.GATE_N):
            p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
            if p is not None:
                out.append(norm(f"Integral({sp.sstr(p._expr)}, x)"))
    return out


def main():
    from train_mathnative import load_rows

    rows = load_rows(gen4=True)
    rows = [r for r in rows if norm(r["cur"]) != norm(r["nxt"])]
    by_key, dups = {}, 0
    for r in rows:
        k = (norm(r["cur"]), norm(r["nxt"]))
        if k in by_key:
            dups += 1
        else:
            by_key[k] = r
    cap = int(os.environ.get("N", "0"))
    keys = list(by_key)[:cap] if cap else list(by_key)
    done = {}
    if STATUS.exists():  # lossless restart: honor prior rows
        for line in STATUS.open():
            d = json.loads(line)
            if "terminal" in d:
                continue
            done[(d["cur"], d["nxt"])] = d["status"]
    todo = [k for k in keys if k not in done]
    print(f"[d2] rows {len(rows)} unique {len(by_key)} (dups {dups}) "
          f"| scope {len(keys)} | resumed {len(keys) - len(todo)} "
          f"| todo {len(todo)}", flush=True)

    n_workers = int(os.environ.get(
        "WORKERS", str(max((os.cpu_count() or 4) - 2, 1))))
    pool = [Replayer() for _ in range(n_workers)]
    counts = {s: 0 for s in ("unique", "ambig", "miss", "err",
                             "timeout", "membomb", "crash")}
    for s in done.values():  # resumed rows count toward the census
        counts[s] = counts.get(s, 0) + 1
    it = iter(todo)
    pending = len(todo)
    t0 = time.time()
    with STATUS.open("a") as sf:
        while pending:
            for w in pool:
                if w.task is None:
                    k = next(it, None)
                    if k is not None:
                        w.send(k, by_key[k])
                res = w.poll(counts)
                if res is not None:
                    (kc, kn), _, status = res
                    counts[status] = counts.get(status, 0) + 1
                    sf.write(json.dumps(
                        {"cur": kc, "nxt": kn, "status": status})
                        + "\n")
                    sf.flush()
                    done[(kc, kn)] = status
                    pending -= 1
                    n_done = len(keys) - pending
                    if n_done % 2000 == 0:
                        rate = (n_done - (len(keys) - len(todo))) \
                            / max(time.time() - t0, 1e-9)
                        print(f"[d2] {n_done}/{len(keys)} "
                              f"({rate:.1f} rows/s) {counts}",
                              flush=True)
            time.sleep(0.02)
        sf.write(json.dumps({"terminal": True, "scope": len(keys),
                             "counts": counts,
                             "wall_s": round(time.time() - t0, 1)})
                 + "\n")
    for w in pool:
        w.kill()
    print(f"[d2] TERMINAL {counts} in {time.time() - t0:.0f}s",
          flush=True)

    certified = [k for k in keys if done[k] in ("unique", "ambig")]
    # D2 proper: the exclude-union law. Union of both directions'
    # normalized strings == {cur} U {nxt} (reversal-invariant). Any
    # certified row touching a gate-band expression is EXCLUDED
    # (exclude= doctrine: guard the split with prompt sets). First
    # full run caught 21/120 band expressions prompt-side in gen-4
    # (26 rows, L3-heavy small-space collision) — measured, booked,
    # excised here.
    band = set(gate_band_exprs())
    n0 = len(certified)
    certified = [k for k in certified
                 if k[0] not in band and k[1] not in band]
    if n0 - len(certified):
        print(f"[d2] exclude-union EXCISED {n0 - len(certified)} "
              f"certified rows touching gate-band expressions",
              flush=True)
    union = {k[0] for k in certified} | {k[1] for k in certified}
    hits = [b for b in band if b in union]
    if hits:  # excision must leave the union clean, or we stop
        print(f"[d2] EXCLUDE-VIOLATION persists: {len(hits)} band "
              f"hits after excision — DIETS NOT WRITTEN. First: "
              f"{hits[0][:60]}", flush=True)
        sys.exit(1)
    print(f"[d2] exclude-union clean: 0/{len(band)} band hits "
          f"({len(certified)} rows survive)", flush=True)
    with REV_OUT.open("w") as rf, FWD_OUT.open("w") as ff:
        for k in certified:
            r = by_key[k]
            base = {"level": r.get("level"), "source": "gen4-certified",
                    "hints": "none", "think": ""}
            ff.write(json.dumps({**base, "cur": r["cur"],
                                 "nxt": r["nxt"]}) + "\n")
            rf.write(json.dumps({**base, "cur": r["nxt"],
                                 "nxt": r["cur"]}) + "\n")
    print(f"[d2] wrote {len(certified)} certified rows -> "
          f"{REV_OUT} + {FWD_OUT} (matched dose)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_main()
    else:
        main()
