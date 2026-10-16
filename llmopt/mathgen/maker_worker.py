"""CPU-only problem-maker worker for ``llmopt.mathgen.isolated``.

A fresh interpreter started by ``MakerWorker`` with ``python -m``; never
a fork of the parent, so a driver holding Torch, MPS or MLX state is not
duplicated (the fork-of-a-Metal-driver kill of 2026-08 lives in
llmopt/lab/oracle_worker.py's docstring). Imports sympy and
llmopt.mathgen only.

Protocol, one JSON object per line on stdin, one line per reply on stdout:

    {"op": "make", "kind": K, "level": L, "seed": S}
        -> "ok <base64(pickle(Problem))>"
        -> "err <base64(utf8(traceback tail))>"   (maker raised)
    {"op": "status"}
        -> "ok <base64(json({pid, torch_loaded, mlx_loaded}))>"

``kind`` is a registered mathgen kind, or ``module:function`` naming an
importable maker with the ``(level, seed) -> Problem`` signature (used
by the fault-injection tests). pickle is safe here: the only writer is
the parent over a private pipe.

A watchdog thread exits the process when its parent disappears (the
parent pid changes), so a hung maker cannot outlive a killed driver.
The address space is capped so a pathological simplify dies in here as
MemoryError rather than as system memory pressure.
"""
from __future__ import annotations

import base64
import importlib
import json
import os
import pickle
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MEM_CAP = 2 << 30


def _resolve(kind: str):
    if ":" in kind:
        module, _, func = kind.partition(":")
        return getattr(importlib.import_module(module), func)
    from llmopt.mathgen.problems import _resolve_maker
    return _resolve_maker(kind)


def _parent_watchdog(parent: int) -> None:
    while True:
        time.sleep(0.5)
        if os.getppid() != parent:
            os._exit(0)


def _reply(tag: str, payload: bytes) -> None:
    sys.stdout.write(f"{tag} {base64.b64encode(payload).decode()}\n")
    sys.stdout.flush()


def main() -> None:
    import resource
    for lim in ("RLIMIT_DATA", "RLIMIT_AS"):
        try:
            resource.setrlimit(getattr(resource, lim), (MEM_CAP, MEM_CAP))
        except (ValueError, OSError):
            pass
    threading.Thread(target=_parent_watchdog, args=(os.getppid(),),
                     daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "status":
            _reply("ok", json.dumps({
                "pid": os.getpid(),
                "torch_loaded": "torch" in sys.modules,
                "mlx_loaded": any(m == "mlx" or m.startswith("mlx.")
                                  for m in sys.modules),
            }).encode())
            continue
        try:
            problem = _resolve(req["kind"])(req["level"], req["seed"])
            _reply("ok", pickle.dumps(problem))
        except BaseException as exc:  # noqa: BLE001 - reported, not hidden
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            tail = "".join(traceback.format_exception(exc))[-2000:]
            _reply("err", tail.encode())


if __name__ == "__main__":
    main()
