"""Subprocess-isolated problem generation (rehabilitation 2C, step 3).

``MakerWorker`` owns one ``llmopt.mathgen.maker_worker`` child and
serves ``make(kind, level, seed)`` requests over a private pipe. The
child is a fresh CPU-only interpreter (not a fork), started once and
reused, so a dataset of cheap items pays one process start. The parent
owns the deadline: a maker that exceeds ``wall`` seconds is SIGKILLed
with its child and reported as ``MakerTimeout``; a child that dies is
``MakerCrashed`` with its exit code and stderr tail; a maker that
raises is ``MakerFailure`` carrying the traceback tail. No failure is
ever answered with a different sample.

Successful output is the pickled ``Problem`` the maker returned, so
prompt, answer, kind, level and the oracle payload are those of an
in-process call with the same arguments (pinned by
tests/test_mathgen_isolated.py against tests/fixtures/mathgen_pins.json).

Fork is deliberately not used: on macOS a fork of a driver holding
initialised Torch/MPS/MLX state has killed the PARENT (see
llmopt/lab/oracle_worker.py). The spawn/subprocess start is slower
(about 0.4 s) but independent of the parent's state.
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import select
import subprocess
import sys
import tempfile
import time
import weakref

from llmopt.common.repo import repo_root

ROOT = repo_root()
WORKER_MODULE = "llmopt.mathgen.maker_worker"


class MakerFailure(RuntimeError):
    """A maker request did not produce a Problem. Base of the typed
    failures; also raised directly when the maker itself raised."""

    def __init__(self, kind, level, seed, detail: str):
        self.kind, self.level, self.seed, self.detail = kind, level, seed, detail
        super().__init__(f"maker {kind} level={level} seed={seed}: {detail}")


class MakerTimeout(MakerFailure):
    """The maker exceeded the wall; the worker was SIGKILLed."""

    def __init__(self, kind, level, seed, wall: float):
        self.wall = wall
        super().__init__(kind, level, seed, f"exceeded wall {wall}s; worker killed")


class MakerCrashed(MakerFailure):
    """The worker process ended before replying."""

    def __init__(self, kind, level, seed, returncode, stderr_tail: str):
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        super().__init__(kind, level, seed,
                         f"worker exited {returncode}; stderr tail: "
                         f"{stderr_tail.strip()[-500:] or '(empty)'}")


class MakerWorker:
    """One reusable CPU-only child; see module docstring.

        with MakerWorker(wall=45) as w:
            p = w.make("integrate", 8, 1234)
    """

    def __init__(self, wall: float = 45.0):
        if wall <= 0:
            raise ValueError(f"wall must be positive, got {wall!r}")
        self.wall = float(wall)
        self._proc: subprocess.Popen | None = None
        self._stderr = None
        self._finalizer = weakref.finalize(self, MakerWorker._kill, self)

    # -- lifecycle ---------------------------------------------------------

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._drop()
            self._stderr = tempfile.TemporaryFile(mode="w+")
            self._proc = subprocess.Popen(
                [sys.executable, "-m", WORKER_MODULE],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, text=True, cwd=ROOT,
                env=dict(os.environ), start_new_session=True)
        return self._proc

    def _stderr_tail(self) -> str:
        if self._stderr is None:
            return ""
        try:
            self._stderr.seek(0)
            return self._stderr.read()[-2000:]
        except (OSError, ValueError):
            return ""

    def _drop(self) -> None:
        self._proc = None
        if self._stderr is not None:
            try:
                self._stderr.close()
            except OSError:
                pass
            self._stderr = None

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if proc is not None:
            for stream in (proc.stdin, proc.stdout):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        self._drop()

    def close(self) -> None:
        self._kill()

    def __enter__(self) -> "MakerWorker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- protocol ----------------------------------------------------------

    def send_nowait(self, kind: str, level: int, seed: int) -> None:
        """Queue a make request without waiting (used by cleanup tests)."""
        proc = self._ensure()
        proc.stdin.write(json.dumps({"op": "make", "kind": kind,
                                     "level": level, "seed": seed}) + "\n")
        proc.stdin.flush()

    def _await_line(self, proc: subprocess.Popen, wall: float) -> str | None:
        deadline = time.monotonic() + wall
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 0.5))
            if ready:
                return proc.stdout.readline()
            if proc.poll() is not None:
                return proc.stdout.readline()

    def _request(self, req: dict, kind, level, seed) -> tuple[str, bytes]:
        proc = self._ensure()
        try:
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            rc = proc.wait()
            tail = self._stderr_tail()
            self._kill()
            raise MakerCrashed(kind, level, seed, rc, tail) from None
        line = self._await_line(proc, self.wall)
        if line is None:
            self._kill()
            raise MakerTimeout(kind, level, seed, self.wall)
        if line == "":
            rc = proc.wait()
            tail = self._stderr_tail()
            self._kill()
            raise MakerCrashed(kind, level, seed, rc, tail)
        tag, _, payload = line.strip().partition(" ")
        return tag, base64.b64decode(payload)

    def make(self, kind: str, level: int, seed: int):
        """Return the Problem ``maker(kind)(level, seed)`` built in the
        worker, or raise a MakerFailure subclass."""
        tag, payload = self._request(
            {"op": "make", "kind": kind, "level": level, "seed": seed},
            kind, level, seed)
        if tag == "err":
            raise MakerFailure(kind, level, seed, payload.decode("utf-8", "replace"))
        if tag != "ok":
            self._kill()
            raise MakerFailure(kind, level, seed, f"malformed reply tag {tag!r}")
        return pickle.loads(payload)

    def status(self) -> dict:
        tag, payload = self._request({"op": "status"}, "status", None, None)
        return json.loads(payload.decode())


def make_problem_isolated(kind: str, level: int, seed: int, wall: float = 45.0):
    """One-shot: start a worker, build one problem, reap the worker."""
    with MakerWorker(wall=wall) as w:
        return w.make(kind, level, seed)
