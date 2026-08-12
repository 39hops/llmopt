"""Shared test helpers.

artifact_or_skip: artifact-gated tests skip where a big untracked
artifact is absent (file-handoff convention), but under LLMOPT_FULL=1
absence is a FAILURE — so "skipped" on the artifact machine can never
read as "verified" (spec 2026-08-12 Phase 6.2).
"""
import os

import pytest


def artifact_or_skip(exists: bool, reason: str) -> None:
    if exists:
        return
    if os.environ.get("LLMOPT_FULL"):
        pytest.fail(f"LLMOPT_FULL=1 but artifact missing: {reason}")
    pytest.skip(reason)
