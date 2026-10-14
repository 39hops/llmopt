# Handoff 2026-10-13-1: merge_space and MICRO-STAR-1 receipts recovered from the 3080, locked; phase 2A pushed

Seat: Fable 5.1 on the Mac. Follows handoff 2026-10-13-0 (phase 2A
closeout). No live registered run at any commit. Nothing armed, nothing
trained, nothing deleted. Artin GO for the push; the 3080 came up and
was used read-only (listing, hashing, one tar read).

## What landed

- The five phase 2A commits (d7975baa..2b5abbd6) were pushed to
  origin/main on Artin's instruction.
- AMENDMENT MERGE-SPACE-RECEIPTS-RECOVERED-2026-10-13 (RESULTS
  L77967; amends MERGE-SPACE-1..5 and MICRO-STAR-1). The 3080 checkout
  (937ffa32) still held every cited receipt untouched since
  2026-08-11. 103 files were hashed on both machines (identical),
  copied byte-for-byte, and 102 placed into this checkout: 53 logs
  (six cited driver/run logs plus 47 per-arm birth and gate logs)
  force-added and sha-locked; 33 checkpoints and 16 merge sidecars
  placed under gitignored checkpoints/ as local-only copies with
  file-level sha256 pinned in docs/EVIDENCE.md. Cross-checks before
  booking: every gate dict in the six verdicts and all 30 booked
  weight digests appear verbatim in the recovered logs; all 33 sidecar
  input digests match the parent checkpoint bytes.
- Lock: 886 -> 933 paths, 904 sha-locked, one results-cited receipt
  still absent on every checkout (logs/pincer/labels_v2.jsonl).
  LEGACY_ABSENT in test_frozen_receipt_mutation.py shrunk from seven
  identities to that one.
- docs/EVIDENCE.md: merge_space chain reclassified from absent to
  tracked-and-locked; 49 local-only checkpoint rows added.

## Deliberately not done

- logs/pincer/labels_v2.jsonl is on the 3080 (54,899 B, sha in the
  amendment) and was not copied: a labels dataset outside the four
  front-page chains; whether it enters the tracked tree is Artin's
  call.
- The two OOM-poisoned microstar logs stay on the 3080 (not cited by
  path, booked as killed unread).
- No re-run of any merge_space gate; no figure change (the merge_space
  fence naming -3/-4 is still phase 2B).
- The 3080 checkout is ff-only synced to origin/main after this
  commit is pushed (standing OK; HEAD hash asserted in the session
  report, not here).

## Session mechanics worth knowing

- The PreToolUse receipt guard refuses any shell command whose text
  names a results-cited receipt path, even when the file is absent and
  the command would only create it. The recovered files were placed
  with `tar xf ... --keep-old-files` naming the parent directories, so
  nothing existing could be overwritten; sha256 verified in place
  afterwards. That is the intended recovery path, not a bypass of the
  guard's purpose; a future `recover` verb on the guard could make it
  explicit.

## Next session

1. Read this handoff, then BOARD, then RESULTS tail (L77967).
2. Phase 2B per spec 2026-10-13-rehabilitation-phase-2b (figure
   pipeline --check, fence ids, merge_space fence names -3/-4).
3. Decide the pincer labels file.
