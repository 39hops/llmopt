"""gen_receipt_lock must be idempotent when nothing is accepted: a
plain regeneration keeps the previously recorded `_last_accept` block
(it is part of the reviewable record of WHY a sha changed), and
`--check` never writes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "gen_receipt_lock.py"


def _load():
    spec = importlib.util.spec_from_file_location("grl_mod", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plain_regen_preserves_last_accept():
    mod = _load()
    old = {"_doc": "x", "receipts": {},
           "_last_accept": {"reason": "why", "paths": ["logs/a/b.log"]}}
    payload = mod.make_payload(fresh={}, old_payload=old, accept=None,
                               changed=[])
    assert payload["_last_accept"] == old["_last_accept"]


def test_accept_replaces_last_accept():
    mod = _load()
    old = {"_doc": "x", "receipts": {},
           "_last_accept": {"reason": "old", "paths": []}}
    payload = mod.make_payload(fresh={}, old_payload=old, accept="new",
                               changed=["logs/x/y.json"])
    assert payload["_last_accept"] == {"reason": "new",
                                       "paths": ["logs/x/y.json"]}


def test_check_view_ignores_machine_local_and_pending_rows():
    """--check must be portable: a checkout that lacks the Mac's
    machine-local (local_only) receipts, or where a prereg-declared
    receipt is still pending, is not drift. Only rows the repository
    itself carries are compared."""
    mod = _load()
    old = {
        "logs/a/tracked.log": {"exists": True, "sha256": "aa", "bytes": 1,
                               "tracked": True, "source": "results"},
        "logs/b/local.log": {"exists": True, "sha256": "bb", "bytes": 1,
                             "tracked": False, "local_only": True,
                             "source": "results"},
        "logs/c/pending.log": {"exists": True, "pending": True,
                               "source": "prereg"},
        "logs/d/absent.log": {"exists": False, "source": "results"},
    }
    fresh = {
        "logs/a/tracked.log": dict(old["logs/a/tracked.log"]),
        "logs/b/local.log": {"exists": False, "source": "results"},
        "logs/c/pending.log": {"exists": False, "source": "prereg"},
        "logs/d/absent.log": {"exists": False, "source": "results"},
    }
    assert mod.check_view(old, old) == mod.check_view(fresh, old)
    # a TRACKED receipt whose bytes changed IS drift under --check
    fresh["logs/a/tracked.log"] = dict(old["logs/a/tracked.log"],
                                       sha256="zz")
    assert mod.check_view(fresh, old) != mod.check_view(old, old)
