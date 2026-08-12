"""Renamed to llmopt.runs.receipts (Phase 7c, 2026-08-12): the module
writes experiment receipts, and root llmopt/runlog.py is general
logging — same name, different things. This alias keeps old imports
working with full fidelity (privates included)."""
import sys

import llmopt.runs.receipts as _m

sys.modules[__name__] = _m
