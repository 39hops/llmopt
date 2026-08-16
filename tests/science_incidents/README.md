# science_incidents — the regression half of graduation

House rule (RIFF-LEDGER 2026-08-16): **every auditor BLOCKER
graduates TWICE** — once into an executable invariant that refuses
it, once into a fixture here reproducing the original failure. A
class with only a documented law is **PROMOTED**; a class with both
is **GRADUATED**. The distinction exists because promotion demonstrably
does not prevent recurrence: `wrong_metric_population` recurred
within hours of being written into the ledger.

Each fixture reproduces a REAL incident with its real numbers, and
asserts the **reason code** rather than message text, so the status
table below is derived from the suite rather than claimed.

| incident_class | law | invariant | regression | status |
|---|---|---|---|---|
| smoke_row_in_real_receipt | yes | `.claude/hooks/smoke_guard.py` | yes | GRADUATED |
| wrong_metric_population | yes | `llmopt/lab/metrics.py` | yes | GRADUATED |
| metric_aggregation_mismatch | yes | `llmopt/lab/metrics.py` | yes | GRADUATED |
| contrast_not_adjudicable | yes | `llmopt/lab/metrics.py` | yes | GRADUATED |
| frozen_receipt_mutation | yes | pending | pending | PROMOTED |
| fp32_billed_fp16 | yes | pending | pending | PROMOTED |
| unserialized_arm | yes | pending | pending | PROMOTED |
| over_budget_comparator | yes | pending | pending | PROMOTED |
| moving_revision_literal | yes | partial (sha-pinned URL) | pending | PARTIAL |
| ternary_in_2bit_field | yes | pending | pending | PROMOTED |
| false_only_contrast | yes | pending | pending | PROMOTED |
| stale_results_anchor | yes | `tests/test_docs_integrity.py` | n/a (post-hoc) | PARTIAL |
| inherited_emitter_label | yes | partial | pending | PARTIAL |

**Success metric** is a RATE, not a count: `escaped_incidents /
opportunities_to_violate`, within class, before versus after
graduation — and refusals caught are positive evidence ("attempted
17, refused 17, escaped 0" says more than an absence of recurrence).

**Note on what "active" means.** An artifact existing is not an
invariant running, and neither inventory is authoritative: counting
hook files overcounts (`findings_headroom.py` is not a registered
hook), and counting `settings.json` entries undercounts (it is
invoked by `ledger_regen.py`). Status above tracks the invariant,
not the file.
