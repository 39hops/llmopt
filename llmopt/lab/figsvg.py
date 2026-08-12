"""Moved to llmopt.figures.figsvg (Phase 5, 2026-08-12). This alias keeps
old imports working with full fidelity (privates included)."""
import sys

import llmopt.figures.figsvg as _m

sys.modules[__name__] = _m
