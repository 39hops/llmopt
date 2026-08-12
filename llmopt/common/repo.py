"""Checkout-only resource resolution.

Some package modules read repo-level files that a wheel never carries:
`llmopt.reproduce` executes `scratch/detbwd_gravmoe.py`, the figure
modules read `docs/figures.json` and `assets/fonts/`. Those files stay
at repo level on purpose — `docs/figures.json` is the single source of
truth for published numbers and duplicating it into package data would
recreate the dual-copy drift problem. The honest contract is therefore:
these subsystems require a source checkout and say so clearly, instead
of failing with a bare FileNotFoundError deep inside site-packages.
"""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the llmopt checkout root, or raise with a clear message.

    Walks up from this file looking for the checkout markers
    (pyproject.toml beside docs/). In an installed wheel the walk ends
    at site-packages and raises RuntimeError.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "docs").is_dir():
            return parent
    raise RuntimeError(
        "this llmopt subsystem needs a source checkout of the llmopt "
        "repository; installed wheels do not carry scratch/, docs/, or "
        "assets/. Clone https://github.com/39hops/llmopt and run "
        "pip install -e ."
    )
