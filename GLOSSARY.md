# llmopt glossary

This controlled vocabulary makes the living ledgers legible to external readers without turning their measured findings into broader claims; each source pointer identifies the entry that demonstrates the usage.

## Evidence maturity

Every finding carries exactly one maturity tag.

- `[RETRACTED]` — an earlier conclusion explicitly withdrawn or superseded by contrary evidence in its ledger record.
- `[NULL]` — a registered treatment did not clear its registered decision bar.
- `[MECHANISM-CONFIRMED]` — a causal arm, not a story, establishes the stated mechanism.
- `[REPLICATED]` — each tagged finding names the route by which it was reproduced: `n>=3` paired seeds for sub-1.5-sigma gate deltas, an independent device, or an independent implementation ([`docs/RESULTS.md:12697`](docs/RESULTS.md)).
- `[SINGLE-SEED]` — a finding based on one registered observation, run, or cell; it has no named replication route and carries an explicit `n=1` fence.

## Scope tags

Scope tags may stack: `[DEVICE-SCOPED]` names the measured device, `[FORMAT-BOUND]` the measured representation or data format, `[TEACHER-FORCED]` the teacher-forced readout, `[FREE-RUN-GATED]` the free-run oracle gate, and `[REGIME-SCOPED: controlled value]` the measured regime.

### Controlled scope vocabulary

`[REGIME-SCOPED: controlled value]` accepts only: `calculus search`; `closed-system math`; `house crystals`; `at-capacity house crystals`; `specified diet and recipe`; `deterministic integer battery`; `tested MoE recipes`; `measured deployment artifacts`; `Qwen2.5-0.5B`; `toy weight-space subjects`.

## Terms

- **crystal** — A trained model or checkpoint, especially its learned weights, treated as a compact, structured artifact ([`docs/RESULTS.md:7613`](docs/RESULTS.md)).
- **gate** — Unless qualified as a technical routing or identity gate, a declared evaluation used to accept, reject, or compare a model; capability gates run outputs and oracle-verify them ([`docs/RESULTS.md:14401`](docs/RESULTS.md)).
- **arm** — One treatment or configuration within an experiment ([`docs/RESULTS.md:14167`](docs/RESULTS.md)).
- **cell** — One named, usually preregistered experiment or comparison that may contain multiple arms; this is not a tensor location ([`docs/RESULTS.md:10184`](docs/RESULTS.md)).
- **battery** — A fixed, reusable suite of prompts, arms, or checks run as one instrument, so results compare across sessions, devices, or implementations; the deterministic integer battery is the canonical example ([`docs/RESULTS.md:12850`](docs/RESULTS.md)).
- **rung** — One ordered stage in a pre-planned ladder of experiments or capabilities, each gated on the one below it, whose result unlocks, blocks, or redirects the next ([`docs/RESULTS.md:14703`](docs/RESULTS.md)).
- **booked** — Entered into the authoritative results record with its prediction, verdict, caveat, or receipt; booked does not mean positive ([`docs/RESULTS.md:15081`](docs/RESULTS.md)).
- **banked** — Recorded as a future candidate awaiting a prerequisite or GO, not yet a result; the status is at the entry's time and can later change ([`docs/RESULTS.md:14254`](docs/RESULTS.md)).
- **fence** — An explicit boundary or invalidation condition limiting a claim, comparison, or launch; scope must use the [controlled scope vocabulary](#controlled-scope-vocabulary) to name what was measured, never a conjectured class, and spell `Qwen2.5-0.5B` exactly ([`docs/RESULTS.md:14767`](docs/RESULTS.md)).
- **diet** — The training data together with its mixture, exposure, and formatting recipe ([`docs/RESULTS.md:14045`](docs/RESULTS.md)).
- **ration** — The share or cap of a diet's training budget assigned to a level, grammar, or row class ([`docs/RESULTS.md:4226`](docs/RESULTS.md)).
- **birth** — A training run started from a new initialization, or the checkpoint it produces, distinct from continued growth or adaptation ([`docs/RESULTS.md:2281`](docs/RESULTS.md)).
- **crown** — The standing best production model or statistical champion class; a crown may remain tied ([`docs/RESULTS.md:3643`](docs/RESULTS.md)).
- **pin** — A fixed reference artifact, value, or hash defining a contract and expected to reproduce exactly; this includes selected contract constants and expected artifact or trajectory hashes ([`docs/RESULTS.md:14889`](docs/RESULTS.md)).
- **relay** — A provenance-carrying, file-based cross-lab handoff of specifications, artifacts, or verification receipts ([`docs/RESULTS.md:15015`](docs/RESULTS.md)).
- **house/axiom** — “House” is the llmopt-side lab/session and “axiom” the separate repo/session for independent implementation and cross-lab checks: independent code/lab paths, one human operator, Artin ([`docs/RESULTS.md:13898`](docs/RESULTS.md)).
- **riff** — An exploratory question, mechanism, analogy, or experiment idea recorded before possible promotion, testing, or retirement ([`docs/RIFF-LEDGER.md:2098`](docs/RIFF-LEDGER.md)).
- **twin** — A counterpart matched on a declared invariant—function, parameters, data, or representation—to isolate one difference; the matched invariant must always be named ([`docs/RESULTS.md:9859`](docs/RESULTS.md)).
- **HOLD/GO** — Run-control states: HOLD forbids launch until authority is given, while GO is explicit permission to fire; neither is an evidence-maturity label ([`docs/RESULTS.md:13747`](docs/RESULTS.md), [`docs/RESULTS.md:13747`](docs/RESULTS.md)).
- **demand log** — The recorded per-layer record of which experts a model actually routed to while running a declared corpus; masks and coalitions are derived from it, never from weights ([`docs/RESULTS.md:18176`](docs/RESULTS.md)).
- **carrier** — An expert in the top-demand-ranked exclusive set of a
  keep-set pair whose presence measurably moves gate capability;
  carriers are identified from a demand log and gate arms, never from
  weights ([`docs/RESULTS.md:21943`](docs/RESULTS.md)).
- **coalition** — The set of experts a given domain's demand log selects at a stated keep fraction; a coalition is a measured routing property of one vehicle, not a claim about expert function ([`docs/RESULTS.md:19254`](docs/RESULTS.md)).
- **keep-set** — The explicit list of experts a mask retains; the keep rule that builds it, and the fraction it keeps, are both part of any claim made with it ([`docs/RESULTS.md:20179`](docs/RESULTS.md)).
- **crest** — The masked keep fraction at which a gate score peaks, including above the paired full model; a crest is a gate observation on one vehicle and one domain, never a deployment recommendation ([`docs/RESULTS.md:18927`](docs/RESULTS.md)).
- **core / branch** — A core is the intersection shared by several measured coalitions against an independence null; a branch is a family of coalitions sharing their own core and separated from another branch by measured routing distance ([`docs/RESULTS.md:19852`](docs/RESULTS.md)).
- **amendment** — An entry that corrects, rescopes, or retracts part of a named earlier entry; the ledger is append-only, so an amendment names its target rather than editing it ([`docs/RESULTS.md:20006`](docs/RESULTS.md)).
- **pre-registration** — The declared prediction, arms, and decision bar recorded before a run fires; a verdict is read against its pre-registration, and an unregistered bar cannot be cleared after the fact ([`docs/RESULTS.md:21603`](docs/RESULTS.md)).
