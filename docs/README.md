# docs/: the map

llmopt is a working research lab with an append-only evidence ledger,
not a packaged product. This directory holds everything from the
curated summary down to the raw record and the lab's own working notes.
The tiers below say which is which. Five minutes: the front
[README](../README.md), then [ENGINEERING](ENGINEERING.md), then
[METHODS](METHODS.md), then [FINDINGS](FINDINGS.md) browsed by section;
then run the command in [REPRODUCE](REPRODUCE.md).

## 1. Start here (curated, written for outside readers)

| file | what it is |
|---|---|
| [FINDINGS.md](FINDINGS.md) | Every curated claim, one bullet each, organized by topic. Each bullet carries exactly one maturity tag and its scope tags, and links to the ledger entry that supports it. |
| [../GLOSSARY.md](../GLOSSARY.md) | The maturity and scope vocabulary, then the house terms the ledger uses. Outsider terms first. |
| [ENGINEERING.md](ENGINEERING.md) | The systems actually implemented, each with code and test entry points and with booked measurements separated from development observations. |
| [METHODS.md](METHODS.md) | How a claim is made here: pre-registration, frozen instrument, run, receipt, adjudication, curated claim, then replication, null, amendment or retraction. |
| [REPRODUCE.md](REPRODUCE.md) | The one-command trajectory replay and what its PASS does and does not certify; the manual MoE recipes. |
| [paper/main.tex](paper/main.tex) | "Quantization at the Entropy Bound", with its RESULTS anchors inline. Its scope fence: the packing law holds on at-capacity house models and does not transport to Qwen2.5-0.5B. |
| [MEASURED-HISTORY.md](MEASURED-HISTORY.md), [EXTERNAL-REVIEWS.md](EXTERNAL-REVIEWS.md) | Appendices: the pre-August measured chronology and the digest of outside reviews. |

## 2. The record: where every claim resolves

| file | what it is |
|---|---|
| [RESULTS.md](RESULTS.md) | The append-only ledger. One `## ` heading per entry: `VERDICT`, `PRE-REG`, `AMENDMENT`, `OBSERVATION`, `RECEIPT`. Verdicts and numbers are not rewritten after booking; a correction is a new dated `AMENDMENT` that names its target. Cite an entry by commit SHA plus its heading. |
| [preregs/](preregs/) | Machine-readable pre-registrations for the later rungs, adjudicated by `scripts/adjudicate.py`. Earlier bars were recorded in prose inside RESULTS. |
| [receipts.lock.json](receipts.lock.json) | The sha256 of every receipt file a booked entry cites. Small receipts are force-added under `../logs/<run>/`; pinned checkpoints under `../checkpoints/`; large artifacts stay untracked and the lock records that. |
| [sol/RESULTS-SOL.md](sol/RESULTS-SOL.md) | The evidence record of the SOL adoption branch, named by its own RESULTS entries. |
| [figures.json](figures.json) | Every published number in a figure, transcribed from the RESULTS entry its `fence` names; the published SVG/PNG renderer (`llmopt/figures/figsvg.py`) reads this file, the matplotlib analysis renderer does not. |

## 3. Re-run it

[REPRODUCE.md](REPRODUCE.md) and [`../llmopt/reproduce.py`](../llmopt/reproduce.py):
sixteen pinned integer-training arms replay on CPU in about 80 seconds
each and are checked against committed trajectory digests. A PASS
certifies the pinned weight path and the teacher-forced readouts. It is
not a symbolic-correctness result and not a capability score.

## 4. Generated views and tooling (derived, not the record)

| file | what it is |
|---|---|
| [results-index.jsonl](results-index.jsonl) | Generated index of RESULTS entries (id, line, type) with hand-curated thread and link fields. Query with `scripts/results_query.py`. |
| [CODEMAP.md](CODEMAP.md) | Generated move gate for `scratch/` and `scripts/`: how documents and code reach each file, and its evidence class. |
| [SCOREBOARD.md](SCOREBOARD.md) | A generated view of live verdicts, frozen at 2026-07-26 and not regenerated since. FINDINGS and the index supersede it. |
| [sol/MATURITY-SUMMARY.md](sol/MATURITY-SUMMARY.md) | A generated branch view whose maturity labels are inferred proposals, not the curated tags. |
| [claims.deny.json](claims.deny.json) | The claim linter's list of superseded readings. |

## 5. Internal working notes (written for the lab's own use)

These are public but they are lab machinery, written in the ledger's
shorthand for the sessions that run the lab. Read them after the tiers
above, not instead of them.

- [BOARD.md](BOARD.md): the live status board, one line per thread, newest at the top.
- [THEORY.md](THEORY.md): house laws against their published lineage.
- [RIFF-LEDGER.md](RIFF-LEDGER.md): idea provenance; every proposed frame is recorded with its attribution and its breaks.
- [handoffs/](handoffs/): dated session handoffs, the resume artifacts.
- [superpowers/](superpowers/): pre-run specs, plans, and the relay traffic with the sibling C++ implementation.
- [LOOP-LOG.md](LOOP-LOG.md), [AXIOM-SURFACE.md](AXIOM-SURFACE.md), [hygiene-plan-2026-08-11.md](hygiene-plan-2026-08-11.md), [opus/](opus/), the paper drafts: closed records and working exhaust.
- [`../CLAUDE.md`](../CLAUDE.md) and [`../.claude/`](../.claude/): the guard layer. Day-to-day work is carried out by coding agents under mechanical guards (pre-registration adjudication, receipt locks, generated-document checks, auditor agents, editor hooks); every claim is checked by those guards and by one human operator.

## Conventions

- A ledger link has the form `RESULTS.md#L<n> "id:<entry-id>"`. The line must be an entry heading and the id must match the index; `tests/test_docs_integrity.py` enforces both.
- Every FINDINGS bullet carries one maturity tag from `[REPLICATED]`, `[MECHANISM-CONFIRMED]`, `[SINGLE-SEED]`, `[NULL]`, `[RETRACTED]`, plus scope tags. The tag is part of the claim.
- Counts on these pages are either generated or dated. A count with no date is a bug.
