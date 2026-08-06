"""lab.oracle — the boxed oracle, v3.2 lineage (spec
2026-08-05-llmopt-lab-extraction module 1). Parent side of the
subprocess line-server in lab/oracle_worker.py; behavior ported
line-for-line from scratch/moe_gt1_arm2.check_isolated (which stays
frozen — booked verdicts cite it).

Contract (the loud-failure contract, every clause a 2026-08-05 burn):
- timeout -> kill + respawn-on-next-check + conservative REJECT,
  typed TIMEOUT event, printed and counted. Timeout is a FAILURE,
  never a skip — changing that silently moves accuracy.
- worker crash -> loud CRASH_EOF / CRASH_PIPE event + respawn +
  reject. A dead oracle must never look like a hard gate (the v2
  spawn-pool post-mortem).
- RSS watchdog: Darwin ignores RLIMIT_AS/DATA (verified live — a
  worker memory bomb ran uncapped and drove the machine to ~87MB
  free; jetsam kills the LARGEST process, i.e. the 30B driver, which
  is how four GT-6 runs died). The parent polls worker RSS every
  0.5s slice and kills a balloon early: typed MEMBOMB, reject.
- counters: self.counters[event] accumulates per Oracle; drivers put
  the census in their terminal record so a booking cannot quietly
  cite a run with a degraded oracle.

Worker test affordances (SLEEP/BOMB lines) stay — they are the
executable proof of the timeout and membomb paths.
"""
from __future__ import annotations

import base64
import pickle
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("oracle_worker.py")

EVENTS = ("TIMEOUT", "CRASH_EOF", "CRASH_PIPE", "MEMBOMB")


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    parsed: bool
    event: str | None  # None on a clean verdict, else one of EVENTS

    @property
    def timed_out(self) -> bool:
        """Legacy check_isolated semantics: TIMEOUT and MEMBOMB were
        the timed_out=True arm of the old ((ok, parsed), timed_out)."""
        return self.event in ("TIMEOUT", "MEMBOMB")


class Oracle:
    """Persistent boxed oracle. Usage:

        with Oracle(wall=20, mem_cap_gb=3) as o:
            r = o.check(problem, expr)   # CheckResult
        # o.counters -> {"TIMEOUT": n, ...} for the terminal record
    """

    def __init__(self, wall: int = 20, mem_cap_gb: int = 3):
        self.wall = wall
        self.mem_cap_kb = mem_cap_gb << 20
        self.counters: dict[str, int] = {e: 0 for e in EVENTS}
        self._proc: subprocess.Popen | None = None

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [sys.executable, str(WORKER)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, cwd=ROOT)
        return self._proc

    def _fail(self, event: str, detail: str) -> CheckResult:
        self.counters[event] += 1
        print(f"[oracle] {event} {detail} — booked as failure",
              flush=True)
        if self._proc is not None:
            self._proc.kill()
            self._proc = None
        return CheckResult(False, False, event)

    def check(self, problem, expr) -> CheckResult:
        pr = self._ensure()
        # pickle over a PRIVATE pipe to our own child (same trusted
        # codebase, both ends ours) — nothing untrusted touches it
        payload = base64.b64encode(pickle.dumps((problem, expr))).decode()
        try:
            pr.stdin.write(payload + "\n")
            pr.stdin.flush()
        except BrokenPipeError:
            return self._fail("CRASH_PIPE", "(pipe)")
        # sliced wait with the RSS watchdog (see module docstring)
        deadline = time.monotonic() + self.wall
        ready = None
        while time.monotonic() < deadline:
            ready, _, _ = select.select([pr.stdout], [], [], 0.5)
            if ready:
                break
            rss = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pr.pid)],
                capture_output=True, text=True).stdout.strip()
            if rss and int(rss) > self.mem_cap_kb:
                return self._fail("MEMBOMB", f"rss={int(rss) >> 10}MB")
        if not ready:
            return self._fail("TIMEOUT", f"wall={self.wall}s")
        line = pr.stdout.readline()
        if line == "":  # EOF: worker died mid-check
            return self._fail("CRASH_EOF", "(eof)")
        ok_s, _, parsed_s = line.strip().partition(",")
        return CheckResult(ok_s == "1", parsed_s == "1", None)

    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc = None

    def __enter__(self) -> "Oracle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
