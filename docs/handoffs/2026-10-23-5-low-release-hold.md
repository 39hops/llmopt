# Handoff 2026-10-23-5: QWEN-LOW-RELEASE-0 (HOLD on bar 9; both gates met in substance) and the DeepInfer read-only review

Session: Fable 5.1 on the Mac. Opus 5.5 reviewers: a pre-freeze review of the v3 collection, validators, sandbox and registration (applied before the freeze); a receipt audit before booking; three DeepInfer readers.
Previous handoff: 2026-10-23-4. Read BOARD first.

## What happened

1. Artin's conditional GO QWEN-LOW-RELEASE-0: confirm low effort on a
   fresh collection against the no-thinking arms at 384 and 768, test
   deeper conversations, flip the default only if nine registered bars
   all FIRE.
2. Phase 1: scratch/evalsets/qwen_battery_v3.json (60 tasks in new
   families; llmopt/lab/qbattery3.py; v3 validator branches in
   qbattery.py; reference solutions asserted at build; the overlap report
   names eleven near relatives of v2 families). Pre-freeze Opus review
   applied (bar 7 fence, validators, conversation checks, bar 9's
   budget-cut coupling, adapter validity). OBSERVATION -FREEZE and
   PRE-REG booked at 59097d0f before any candidate output.
3. Phases 2 to 4: chain qwen-low-release-0 at f235306c (12 template and
   live tests; arms A, B, C; four eight-turn conversations under NOTHINK
   and LOW; 76 min).
4. VERDICT QWEN-LOW-RELEASE-0 (RESULTS L85195): 56 / 45 / 44 of 60 for
   low / no-thinking 768 / no-thinking 384; arithmetic plus structured
   22 / 12 / 11 of 24; code 11 / 10 / 10; conversations under low 32 of
   32 addressed, no empty turn, travel 8 of 8; eight bars FIRE; bar 9
   NO-FIRE on one later turn (planning 7) that fell back after a closed
   turn because the tokenizer merged two emitted tokens on re-rendering
   (a safe fallback, replay token-equal). The registered rule holds the
   promotion. Defaults unchanged. The held flip patch (Session thinking
   low 768 by default, CLI --max-tokens 768, subprocess default tests)
   was NOT applied and lives only in the session's tmp.
5. DeepInfer (deepseek-ai, pin 9bd9cf24) reviewed read-only for three
   mechanisms; ranked assessment in
   docs/superpowers/specs/2026-10-23-deepinfer-review.md and the RIFF
   bank; nothing implemented.

## Next (held for Artin's GO)

- The flip GO: re-register bar 9 as the correctness property (no reuse
  of an incompatible cache, replay token-equal, offsets equal; fallbacks
  reported) and read bars 1 to 8 from these receipts or re-run; then
  apply the flip with the subprocess default tests and book the release
  verdict.
- A README limitations note on the thinking-off collapse at depth (two
  receipts now).
- After the release: deterministic sampling design, then the multi-row
  fused kernel as the enabler of speculation; grammar only as a serving
  property.

## Security side note (evidence)

Mac check for the litellm 1.107.0 supply-chain incident of 2026-10-21:
no litellm in any Python environment, pipx or uv cache, no indicator
files, no daemon lock, Downloads clean (checked by Artin with the same
find), Docker daemon not running. Node was 26.10.0 (vulnerable to the
Brotli CVE) and Artin upgraded it; `node --version` now prints
v26.12.1. PHP 8.5.11 CLI only, no server. No GitLab tooling. No further
CVE investigation authorised.

## Hygiene

Receipts force-added by file name: the three v3 battery receipts, the
conversation receipt, the adjudication pair, the chain log, the liverun
record, the two smoke receipts. Both checkouts at origin/main.
