# Handoff 2026-10-16-3: phase 2F preparation landed (dynamic-site resolver, file-by-file plan, census corrections); 3080 check_source diagnosed and fixed; nothing moved or deleted

Seat: Fable 5.1 on the Mac. Follows handoff 2026-10-16-2. No live
registered run at any commit. Nothing trained, downloaded, deleted,
moved, cloned or relinked; the checkpoint baseline manifest is
identical across all six census runs. Artin's standing instruction
for 2F: analysis and preparation only; no GO on anything destructive
until he has read a file-by-file execution plan. That plan now exists
at logs/rehab-2e/run6/PLAN.md (force-added) and is booked as
OBSERVATION CHECKPOINT-PLAN-0 (RESULTS L78558).

## What landed (commits after 81442b6a)

1. 2661ecdd: AMENDMENT CHECKPOINT-CENSUS-0-FREE-SPACE-CAUSE (the 57.2
   GB free came from Artin deleting ~/qwen, the 27B model parts); the
   QWEN-27B REVISIT riff banked in RIFF-LEDGER.
2. 8b8fa95d, 0ffd424c: `scripts/ckpt_plan.py` + tests (13): anchored
   patterns from every dynamic path site, roles, exact expansion from
   same-file literals, generic sites exact-only, buckets with
   approval boundaries.
3. a9ffbfc3: census and plan read `f"{CKPT}/name.pt"` and
   `os.path.join(X, "name.pt")` as path references (the three largest
   mathnative files were missed by both tools).
4. df8ca8fd: three 3080-only test failures fixed (determinism flag
   leak into the CUDA codec roundtrip; meter pins now a 1e-5 relative
   tolerance instead of Mac float bits; ex5-layermatch-0's Mac-only
   operand provider skips the pair with the reason on other hosts);
   census cuts CHECKPOINT-CENSUS-* / CHECKPOINT-PLAN-* ledger entries
   out of the citation scan (run5 had frozen five files because the
   census booking named them).
5. This commit: OBSERVATION CHECKPOINT-PLAN-0, run6 receipts
   force-added, spec 2F preparation note, BOARD.

## Plan buckets (run6; details and every row in PLAN.md)

| bucket | files | GB |
|---|---:|---:|
| no site reaches the file after resolution (CANDIDATE queue) | 121 | 10.43 |
| unresolved consumer pattern (metabolic TAG) | 18 | 1.99 |
| producer patterns that did not resolve | 67 | 1.60 |
| census CANDIDATE | 43 | 0.85 |
| exact consumer only in uncited scripts | 8 | 0.61 |
| exact consumer in a results-cited script (keep) | 11 | 0.42 |
| produced by a known script, read by nothing exact | 4 | 0.11 |
| duplicate of a retained copy (relink, then remove) | 6 | 0.12 |

Census run6: FROZEN 547 / 46.55 GB, ESSENTIAL 334 / 7.82, DUPLICATE
6 / 0.12, CANDIDATE 43 / 0.85, UNKNOWN 229 / 15.16; 29 frozen pairs
2.48 GB on distinct blocks.

## 3080

matplotlib installed in its venv on Artin's instruction. check_source
at 81442b6a there: generated docs, lock, figures (raster NOT verified,
no Chrome) all passed; pytest 12 failed / 2270 passed / 53 skipped,
all 12 host-sensitivity, fixed in df8ca8fd. Needs a pull and a rerun
after this push to confirm green.

## Next

- Artin reads PLAN.md and gives per-step GOs in the booked order:
  duplicates (relink, remove, 0.12 GB), clone 29 frozen pairs (2.48
  GB), CANDIDATE 43 rows per family, the 121 no-site rows per family,
  metabolic after its arm list is resolved, then the script decisions.
- Resolve the metabolic TAG name set from jobs/revmetab.cmd and
  re-plan (read-only).
- Adapter-manifest follow-up (train_calculus.py) and the 27B revisit
  stay banked.
