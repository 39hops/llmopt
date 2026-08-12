"""Moved to llmopt.runs.traj (Phase 5, 2026-08-12). This alias keeps
old imports working with full fidelity (privates included)."""
import sys

import llmopt.runs.traj as _m

sys.modules[__name__] = _m
