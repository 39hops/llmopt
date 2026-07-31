import io
import logging

from llmopt.runlog import ElapsedFormatter, get_logger, timed


def _fresh(stream):
    root = logging.getLogger("llmopt")
    for h in list(root.handlers):
        root.removeHandler(h)
    return get_logger("llmopt.test", stream=stream)


def test_elapsed_prefix_and_hierarchy():
    buf = io.StringIO()
    log = _fresh(buf)
    log.info("hello %d", 7)
    out = buf.getvalue()
    assert out.startswith("[+")
    assert "llmopt.test INFO hello 7" in out
    # child shares the single top-level handler
    assert len(logging.getLogger("llmopt").handlers) == 1
    get_logger("llmopt.other")
    assert len(logging.getLogger("llmopt").handlers) == 1


def test_timed_success_and_failure():
    buf = io.StringIO()
    log = _fresh(buf)
    with timed("quick", log):
        pass
    assert "quick done in" in buf.getvalue()
    try:
        with timed("boom", log):
            raise ValueError
    except ValueError:
        pass
    assert "boom FAILED after" in buf.getvalue()


def test_formatter_units():
    f = ElapsedFormatter("[%(elapsed)s] %(message)s")
    rec = logging.LogRecord("x", logging.INFO, "", 0, "m", None, None)
    s = f.format(rec)
    assert s.startswith("[+") and "] m" in s
