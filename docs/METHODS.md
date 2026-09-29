# Methods

Every verdict in this repository is read against a threshold that was
written down before the run, is scored by a program or a named oracle,
and stays published when it fails. Observations and desks book
measurements without a bar and say so. This page explains that loop once; the claims
themselves are in [FINDINGS](FINDINGS.md).

## One example, followed to the end

The effective-context probe on the front page (the truncation curve on
the trained width ladder) had two registered bars. The pre-registration named both before anything
ran: [PRE-REG KEFF-PROBE-1](RESULTS.md#L27239 "id:2026-08-11-pre-reg-keff-probe-1-direct").
Bar 1 (deep positions use more than a hundred tokens of context at
every width) fired. Bar 2 (wider models separate further as context
grows) did not: the width gap is negative at k=8 and narrows again at
k=128. The verdict books both outcomes in one entry,
[VERDICT KEFF-PROBE-1](RESULTS.md#L27332 "id:2026-08-11-verdict-keff-probe-1-bar-1"),
and the curated bullet in FINDINGS carries the miss in the same
sentence as the hit, with its scope: one diet, one probe, loss only,
no gate, single seed. The front page shows the figure and the miss. A bar
that misses is reported with the same prominence as one that fires,
because a ledger that only records successes cannot be checked.

## The loop

**Question.** A question that will produce a verdict is admitted with
a bar it can fail. Counting questions on data already on disk run as
desks or censuses and are booked as observations, without a bar; ideas
without either are recorded in the idea ledger.

**Pre-registration.** The prediction, the arms, the decision bar and
the scope fences are booked as a `PRE-REG` entry before the run fires.
From mid-August 2026, registered rungs increasingly also carry the bar
as a JSON document in [preregs/](preregs/) that a program adjudicates
([`scripts/adjudicate.py`](../scripts/adjudicate.py)); bars before
that, and many since, are recorded in prose inside RESULTS and
adjudicated by reading. An
unregistered bar cannot be cleared after the fact.

**Frozen instrument.** The driver that runs the experiment is committed
before the run and its sha is recorded in the run's receipts. Files
cited by a booked verdict are the evidence record: [CODEMAP](CODEMAP.md)
classifies them and an editor guard asks before any edit. An instrument
adopted into the package is copied with a source-identity test, never
moved.

**Execution.** Paired arms run on one device with the same seeds and
one variable changed. Smoke runs write to their own paths. While a
registered run is live, an interlock refuses commits to the checkout.
Cross-device gate comparisons are forbidden outright.

**Receipt and provenance.** A run writes receipts whose provenance
fields (driver sha, checkpoint digests, code commit) are derived from
the artifacts it actually opened, never typed in. Every receipt a
booked entry cites is content-addressed in
[receipts.lock.json](receipts.lock.json); changing or deleting one fails
the test suite. Artifacts are named by logical locators, not machine
paths.

**Adjudication.** Each bar resolves to FIRE, NO-FIRE or UNRESOLVED
against its registration. A measurement that is not the registered
quantity is refused rather than reinterpreted. Verdict prose is linted
against the adjudicator's objects and a list of superseded readings
before booking.

**Curated claim.** The verdict is appended to RESULTS as a `VERDICT`
entry, and in the same commit one bullet is added to FINDINGS with
exactly one maturity tag and its scope tags, linked to the entry.
CI ratchets the backlog of verdicts without a bullet.

**Replication, null, amendment, retraction.** A claim moves to
`[REPLICATED]` only by a named route: three or more paired seeds for
small gate deltas, an independent device, or an independent
implementation. A bar that
misses is a `[NULL]`. A correction is a dated `AMENDMENT` entry that
names its target; a withdrawn conclusion is `[RETRACTED]` and stays
visible. Verdicts and numbers in RESULTS are not rewritten after they
are booked.

## Trajectory equality is not oracle correctness

The reproduction command replays a pinned integer training run and
compares its trajectory digest to a committed pin. A PASS certifies the
pinned weight path and the teacher-forced readouts. It is not symbolic
correctness evidence and not a capability score: free-run scoring by
the symbolic oracle needs row text that is not committed, so
artifact-backed gate arms run in an explicit trajectory-only mode
([REPRODUCE](REPRODUCE.md)). A gate-arm trajectory PASS is never
described as a free-run result.

Two further boundaries travel with every comparison. Arms are paired
on one device; cross-device gate deltas are not compared. Ordinary
float training on Apple-silicon MPS is run-level nondeterministic at a
fixed seed, so experiments on that substrate compare arms within one
run and never assert cross-run weight identity
([AMENDMENT SOFT-SPEED-1-PRECONDITION](RESULTS.md#L29985 "id:2026-08-15-amendment-soft-speed-1-precondition-target")).

## Maturity tags and scope fences

The tag is part of the claim. The five maturity tags and the stacking
scope tags are defined once, in the
[glossary](../GLOSSARY.md#evidence-maturity), together with the
controlled vocabulary a `[REGIME-SCOPED]` fence may use. A scope fence
names what was measured (device, seed count, format, regime), never a
conjectured class. The front page's generated line counts the tags
from FINDINGS every time it is built.

## Append-only, and how to cite

RESULTS is append-only in the sense that matters: a booked verdict or
number is never rewritten. A correction is a new dated `AMENDMENT`
naming the entry it corrects; the original stays in place with the
amendment pointing at it. The generated index keeps hand-curated links across
regenerations. To cite a claim, name the commit SHA and the entry
heading, in the form `RESULTS.md#L<n> "id:<entry-id>"`; the ledger is
living, so an unpinned citation is not reproducible.

## The guards, as code

[ENGINEERING §5](ENGINEERING.md#5-evidence-and-provenance-infrastructure)
lists the implementations: the pre-registration adjudicator, the claim
linter, the receipt lock and its mutation fixtures, the live-run
interlock, logical locators, the science-incidents suite (the rule is
that every auditor blocker becomes an executable invariant plus a
regression fixture; its status table shows which classes have both and
which are still pending), and the generated documents that refuse to
drift. What
"green" means is one script, [`scripts/check_source.sh`](../scripts/check_source.sh);
what it does not cover (the wheel and core-dependency jobs, GPU and
toolchain tests that skip) is stated in the script itself.
