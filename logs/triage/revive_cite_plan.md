# Revive-and-cite plan (checkpoint triage, 2026-08-08)

Policy (Artin, 2026-08-08): UNCITED artifacts delete only after a
rebuild/verify/cite path is listed per family — revive-and-cite, not
disk wipe. Mac = home for heavy untracked data; WSL = clean
clone-plus-active-runs. Nothing below deletes without sign-off on
this document.

## Executed already (mechanical, per the by-class table)

- WSL: 55 files removed (~6.1 GiB), every one sha-verified first —
  47 untracked byte-twins of Mac copies + 8 cited orphans AFTER
  pulling them to the Mac and verifying shas (metab_d2_fp64/dd,
  exchange_p1, metab_v5_s1, metab_v4, union_45m, step_lora_grpo
  pair). Name collision handled: WSL step_lora_grpo.pt differs
  byte-wise from the Mac file of the same name — kept as
  checkpoints/step_lora_grpo.pt.wsl.
- Classifier corrections found in review: (a) glob-form citations
  (RESULTS 4224 cites "checkpoints/tourn_B*.") were missed by the
  basename matcher; (b) 141 "UNCITED" files are CONSUMED by
  committed scratch/scripts code (logs/triage/uncited_but_consumed
  .json) — reclassified, never delete while the consuming driver
  is committed (boundary_or_bulk.py reads param counts from the
  actual files: "never trust a label").

## Load-bearing set (do NOT touch, any class)

mathnative_gen6_grown.pt, mathnative_gen6_ternary(.latent).pt,
merged_grown(.latent).pt, mathnative_19m.pt,
mathnative_19m_infixtwin.pt, metab_v4.pt, metab_v5_s1.pt,
seedvar_1.pt, mathnative_grpo.pt, mathnative_grpo_c010.pt, the
boundary_or_bulk.py RISING/UNDERFED set, everything in
uncited_but_consumed.json.

## UNCITED families — status calls (Fable-verified reviewer map)

DELETE-AFTER-SIGNOFF (regenerable, era closed, nothing honest to
revive; rebuild path named):

| family | GiB | rebuild path | why dead |
|---|---|---|---|
| grpo _cand per-cycle snapshots (mathnative_grpo_c*/cand*, gen5mine*_cand*, grpo_shaped_cand*) | ~8.5 | scripts/step_grpo_micro.py re-emits cand%03d per cycle | GRPO era CLOSED (BOARD); the two evidence files are kept above |
| snapm_q* | 0.4 | scratch/rational_snap.py <in> <Q> <out> — deterministic, seconds | Q-knee already booked (Q16/Q24/Q48 verdicts) |
| u45_fq*/u45_rat* (WSL) | 0.8 | same snap/fake-quant transforms of union_45m.pt (now on Mac) | deterministic PTQ derivatives |
| checkpoints/v4flash_sample/*.bin (90 files) | 0.35 | sha-verified byte-range refetch (v4flash_f1c.py) | download cache; costs bandwidth only (v4flash_rungA then needs network) |
| gen4/gen5 intermediates incl. 110m/200m/400m ep snapshots, mathnative_110m_v21_lr25/grpo (bf16-tainted window) | ~7 | scripts/train_mathnative.py with logged args | superseded by gen6 crown; 113M/200M/400M excluded from every fit |
| cplx_none_zx s1-s3 (19M, WSL) | 0.2 | scratch/gate_zx.py chain | 19M ZX column measured UNREADABLE (seed sd ~4.2 drowns cells) |
| pred_syndromes_5b.pt | 0.34 | scripts/bench_pred_syndromes.py | syndrome head CLOSED (payoff NULL); _5b variant cited nowhere |
| tourn/grid derivatives not glob-covered by RESULTS 4224 | ~1 | tournament_birth.py / night2_mac.sh lines | numbers booked; files replaceable |

REVIVE-CANDIDATES (could become honest FINDINGS lines; each needs
n>=3 births per the resolution law — a single re-run would just
mint another unresolved line):

1. metabolic_late / hot_snap — fully specified by committed
   scratch/hot_chain.sh + metabolic_hot.py; no verdict ever booked.
   Cheapest revive: 3 paired births, book positive or null.
2. merged_grown crown TIE — the thread (not the file): BOARD
   already fences it as n=1-births; 2-3 fresh birth pairs resolve
   the 75-v-76 tie. merged_grown_identity.pt (WSL, 89.7M) is the
   identity-gate transient — delete after the tie thread decides.
3. grid width x bits crossover — the "ternary beats fp32 by 7 at
   d768" headline is single-seed at sigma~5; a 3-seed re-run either
   mints a real law or a clean null (Phase-3-shaped, needs new GO —
   not in the frozen 7-row list).

KEEP-AS-IS: everything CITED/CONSUMED; seedvar set (house seed-
fence baseline); metab v3 arms until the metabolic thread's
Phase-4 slot decides.

## Machine layout after this pass

- WSL checkpoints: tracked files + 304 UNCITED (16.4 GiB) awaiting
  the sign-off above; nothing new lands there (consolidation rule).
- Mac gains ~1.9 GiB of formerly-WSL-only cited weight; all cited
  weight now has a Mac home. NOTE against the policy assumption:
  only 43 checkpoint files are tracked in git — nearly all cited
  weight is UNCOMMITTED, so "a pull restores it" does NOT hold for
  checkpoints; the Mac copy is the only durable home.

## Grok audit adopted (2026-08-08, repo-view review)

- Mac sha audit of all 55 manifest keys: 55/55 accounted — 54
  direct matches + step_lora_grpo via the .wsl copy (known byte
  divergence; BOTH Mac files stay until the pair is reconciled).
  Zero data loss confirmed.
- All DELETE-row era claims verified against the ledger (BOARD ~85
  GRPO closed, KNEE ~7655, gen6 crown ~2861, 113M excluded ~4576,
  ZX sd ~7595, syndrome NULL BOARD ~122). No fabricated reasons.
- Classifier law fixed as two-stage: evidence ledger + jobs +
  SCRIPT-CONSUMPTION (required stage, not optional). handoffs/ and
  specs/ stay a SOFT citation channel — report-only ("named in
  handoff but uncited in evidence"), never auto-CITED, to avoid
  chatty-path inflation.
- uncited_but_consumed.json (141 paths) is FROZEN — never expands
  into DELETE without clearing the consumer first.
- NULL-recreate policy: modern n>=3 re-runs only for thin/unbooked/
  suspect cells; closed regenerable eras on the DELETE table are
  deleted, not re-run.

## DELETE pass EXECUTED (Artin GO 2026-08-08)

- Interlock as mandated: executor re-hashed EVERY file at rm time
  against the inventory sha; mismatch = stop row + report. Zero
  mismatches fired.
- Mac: 196 removed, 3.86 GiB freed (delete_pass_mac.json).
- WSL: 39 removed, 7.16 GiB freed, 1 already absent —
  consistent with the earlier twin pass (delete_pass_wsl.json).
- Freeze rules held mechanically: consumed/CITED/tracked/named-set
  candidates auto-excluded (v4flash_f1/ CITED cache, u45_* and
  gen4/tourn cells consumed by committed code all KEPT).
- merged_grown_identity.pt HELD pending the crown-tie revive.
- Revive queue (Artin GO, pre-reg before each fire, n>=3):
  (1) metabolic_late/hot, (2) merged_grown crown-tie births,
  (3) d768 ternary-v-fp32 crossover — outside the frozen Phase-3
  seven, run as the revive track.

Sign-off asks: (1) GO/no-GO per DELETE row above; (2) whether
revive-candidates 1-3 get queued (each priced at 3 births);
(3) confirm the load-bearing set stays frozen.
