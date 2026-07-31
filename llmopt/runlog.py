"""llmopt.runlog — standard run logging with honest wallclock.

Every line carries elapsed time since process start ("[+mm:ss.s]"),
so logs double as timing records. Components get child loggers via
get_logger("llmopt.<component>"); scripts and scratch may extend or
override freely (swap the formatter, add handlers, or ignore this
module entirely — it never touches the root logger).

    from llmopt.runlog import get_logger, timed
    log = get_logger("llmopt.pack")
    log.info("packing %s", name)
    with timed("gate eval", log):
        ...

Level via LLMOPT_LOG env (default INFO).
"""
import logging
import os
import sys
import time
from contextlib import contextmanager

_T0 = time.monotonic()


class ElapsedFormatter(logging.Formatter):
    """Prefixes every record with wallclock since process start."""

    def format(self, record):
        e = time.monotonic() - _T0
        record.elapsed = f"+{int(e // 60):02d}:{e % 60:04.1f}"
        return super().format(record)


DEFAULT_FMT = "[%(elapsed)s] %(name)s %(levelname)s %(message)s"


def get_logger(name="llmopt", level=None, stream=None, fmt=DEFAULT_FMT):
    """Idempotent: repeated calls return the same configured logger.
    Only the top-level 'llmopt' logger gets a handler; children
    propagate to it, so one handler serves the whole tree."""
    root = logging.getLogger("llmopt")
    if not root.handlers:
        h = logging.StreamHandler(stream or sys.stderr)
        h.setFormatter(ElapsedFormatter(fmt))
        root.addHandler(h)
        root.propagate = False
        root.setLevel(os.environ.get("LLMOPT_LOG", "INFO"))
    log = logging.getLogger(name)
    if level is not None:
        log.setLevel(level)
    return log


@contextmanager
def timed(label, log=None, level=logging.INFO):
    """Context manager: logs '<label> done in <t>s' on exit,
    '<label> FAILED after <t>s' if the body raises."""
    log = log or get_logger()
    t = time.monotonic()
    try:
        yield
    except BaseException:
        log.log(logging.ERROR, "%s FAILED after %.1fs",
                label, time.monotonic() - t)
        raise
    log.log(level, "%s done in %.1fs", label, time.monotonic() - t)
