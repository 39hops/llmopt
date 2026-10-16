"""Deliberately faulty makers for the isolated-worker tests. Reached by
the worker through its ``module:function`` kind form with this directory
on PYTHONPATH; never registered in llmopt.mathgen.problems._MAKERS."""
import os
import time


def hang(level, seed):
    time.sleep(600)


def crash(level, seed):
    os._exit(3)


def raises(level, seed):
    raise ValueError(f"maker refused level={level} seed={seed}")


def slow_then_ok(level, seed):
    from llmopt.mathgen.problems import make_differentiate
    time.sleep(0.3)
    return make_differentiate(level, seed)
