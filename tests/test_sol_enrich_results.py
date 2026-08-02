import sys


sys.path.insert(0, "scripts")

import sol_enrich_results as S  # noqa: E402


CE_GATE = "2026-07-26-ce-gate-study-my-hypothesis-fails"
QK_COND = "2026-08-01-verdict-qk-cond-peaked-attention-is"
QK_RESCOPE = "2026-08-01-verdict-qk-rescope-amends-verdict-qk"


def _entries_by_id():
    return {entry["id"]: entry for entry in S.enrich()}


def test_null_entry_that_retracts_another_claim_stays_null():
    entry = _entries_by_id()[CE_GATE]
    assert entry["maturity"] == "null"


def test_rescope_retires_prior_doctrine_without_retracting_itself():
    entries = _entries_by_id()
    assert entries[QK_RESCOPE]["maturity"] == "measured"
    assert entries[QK_COND]["maturity"] == "superseded"
    assert entries[QK_COND]["inferred_superseded_by"] == [QK_RESCOPE]


def test_self_retraction_requires_the_current_entry_as_subject():
    assert S._is_self_retraction("RETRACTION QK-COND: claim withdrawn")
    assert S._is_self_retraction(
        "VERDICT QK-COND: this verdict is RETIRED after replication")
    assert not S._is_self_retraction(
        "VERDICT QK-RESCOPE: this entry RETIRES the earlier doctrine")
    assert not S._is_self_retraction(
        'VERDICT QK-RESCOPE: "peaked attention" is RETIRED as doctrine')


def test_target_retirement_requires_an_action_not_a_referenced_name():
    assert S._retires_amended_target(
        'VERDICT QK-RESCOPE: "peaked attention" is RETIRED as doctrine')
    assert not S._retires_amended_target(
        "VERDICT QK-COND: this verdict is RETIRED after replication")
    assert not S._retires_amended_target(
        "RIDER on VERDICT QK-RESCOPE: the gate leg is real")
