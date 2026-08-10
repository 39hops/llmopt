# Handoff 2026-08-10-0 — the gcd wall is deleted; co-factor witness GO'd; authorship rule retired

Resume: BOARD header (MONDAY 08-10) -> this file -> RESULTS tail
(newest: PRE-REG COFACTOR-WITNESS 24298). Prior chain:
handoffs 2026-08-09-0 and -1.

## POLICY CHANGE (read first)
The Fable-only authorship rule is RETIRED (Artin, 2026-08-10;
Fable usage was the bottleneck). CLAUDE.md now reads: the MAIN
SESSION MODEL owns code and docs and is accountable for verifying
its own work. Reviewers still MENTION-never-edit, cap 5, ask
first; model choice for reviewer seats is now open. Memory file
renamed no-subagents-fable-only -> subagent-and-authorship-policy
(links repointed). This session ran as Opus 5.
Also new memory: readable-over-dense-prose (Artin 2026-08-10 —
chat prose was too dense; ledger register unchanged).

## THE ARC: exact training, riff to deleted wall in ~24h
1. VERDICT EXACT1-SMALL (23852): 3/3 bars fire on BOTH cells.
   d8 anchor step 3 killed at 19.34h, dumps pulled, divergence
   recomputed house-side. Ring grain ABSORBED at p=32 both dims;
   carry floor grain-independent EXACTLY; grain growth
   super-diffusive. NEW: carry floor is p-independent but scales
   with WIDTH (19.0/step d8, ~42.5 d16, ~528 d64) — ENGINE-EXACT-2's
   carry-ladder arm has a measured curve to beat.
2. AMENDMENT -EXPONENT (23910) then -EXPONENT-2 (23948, Artin's
   double-check): the two quoted exponents used DIFFERENT
   estimators, then the correction itself was fit on 2-decimal
   PRINTED means. Matched and artifact-fit: d8 0.800/0.714,
   d16 0.817/0.729 — the cells AGREE.
3. COUNTER-BOOK ANCHOR-V2 (23990): P-DIGEST-EQUAL FIRES —
   anchor2 d64 step-1 weights BYTE-IDENTICAL to the exact anchor
   (verified house-side, cmp clean, sha 7c9b8f0b...). ~160 s/step
   FLAT. P-HORIZON MISSES (8/12). THE GCD WALL IS DELETED.
4. AMENDMENTs -SITE-ATTRIBUTION (24088), -THROW-ATTRIBUTION
   (24153), -PATH-CLASSES (24224): five-throw table verified 5/5
   by recomputing each precision from its own commit's diff;
   step-7 sites PATH-CURABLE, step-9 site PATH-RESISTANT; the
   three step-9 runs are a PARTIAL ORDER so the site defeats
   more-precision-late AND more-precision-throughout.
5. PRE-REG COFACTOR-WITNESS (24298), Artin GO, relay -7 sent.

## FOUR STANDING RULES adopted today (both labs)
name the estimator; fit the artifact not a printed summary; an
explanation offered for a measured constant books at its own
evidence level; report every event in a class, then explain.
Origin: five inference-as-measurement defects on 2026-08-10 —
three house (mixed estimator, rounded-input fit, site-identity
assertion), two axiom (phantom deepening rate, dropped w=11
throw).

## LIVE — crown battery (Mac, rjob rev3crown, pid tracked)
3 of 6 cells COMPLETE (c_s2, m_s2, c_s3); m_s3 running at
0.6 it/s; c_s4 + m_s4 remain. Rough ~13h to full battery, so the
s4 pair lands late 08-10 / early 08-11.
TWO GOTCHAS FOR WHOEVER READS THE LOGS NEXT:
- The driver only PRODUCES checkpoints. The final gate evaluation
  is a SEPARATE step run after all six exist. There are no
  results in the logs yet.
- The gate lines already present (c_s2 60/60, c_s3 65/65, dicts
  IDENTICAL between birth and grown) are NOT results. They are
  scratch/rev3_crown.py's zero-tolerance IDENTITY PRE-CHECK: fp32
  growth adds +256 FFN/layer with zeroed down-columns, which is
  exactly function-preserving, so gate(grown) MUST equal
  gate(birth) or the script aborts. Passing 2/2 = growth is
  correct. Do not raise this as a caching bug (this session
  nearly did; reading the driver resolved it).
Booking notes carried from handoff 2026-08-09-0: tie language
scoped to +-2.3 solves/seed; n=2 fallback re-derives bars
per-seed; note single-seed dominance if signs ride one seed.

## WAITING ON AXIOM
Co-factor witness build against PRE-REG 24298. Four bars:
P-DIGEST-INVARIANT (safety, reads FIRST — steps 1-8 bit-identical
+ d64 step-1 byte-identical to committed ref 7c9b8f0b),
P-WITNESS-DECIDES, P-HORIZON-2 (12-step d64 <= 4h),
P-PATH-INVARIANT (house-registered, never run by either lab).
Registered observable |r| per site, with a MECHANISM-refutation
clause if |r| tracks the denominator. NOT-APPLICABLE clause if the
blocking site is not a de-grain seam. Their last commit is
d982b5a; llmopt relay -7 delivered by Artin.
House counter-books on receipt; verification standard set this
week is re-derivation from the other lab's own commits, not
acceptance of tables.

## QUEUED / DEFERRED
EX4-UNIF fires on the freed Mac window after crown (riders
booked). ENGINE-EXACT-2 pre-reg (carry ladder x normalizer) still
unwritten — now has the d-scaling curve AND tie depth as budget
axes. Seat tabulation (Artin's "more minds up to a point"):
catches per seat type x error class over ~26 review-adoption
entries — a DESK pass, banked in RIFF-LEDGER 08-10, not an
experiment. Frozen-driver bug list unchanged (registered re-run
only). 12 [UNVERIFIED] specs-INDEX rows. star-profile +
small-specialists pre-regs.

## MACHINES / STATE
Mac: crown battery owns it. 3080/WSL: IDLE, stays ONE worker
until Artin frees it. axiom: Mac CPU, one worker, 3080 untouched.
FINDINGS backlog 318 v ratchet 320 (two slots). 483 tests green.
Everything pushed. Only untracked paths are scratch/engine_exact1/
and scratch/engine_scale1/ — axiom artifact drops, intentionally
untracked per file-handoff convention.
Hygiene closed: axiom rewrote out 76 Claude-Session commits (now
0; llmopt always 0) and correctly DECLINED to rewrite a
wrong-claim comment — rewrite for disclosure, retract for error.
