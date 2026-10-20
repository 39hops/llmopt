# Handoff 2026-10-20-0: QWEN-LAYER-RESTORE-0 registered and smoke-qualified, the shard fetch on hold

Session: Fable 5.1 on the Mac. Reviewer on Opus 5.5.
Previous handoff: 2026-10-19-1 (the head restoration, NULL). Read BOARD first.

## What landed

1. Plan: docs/superpowers/specs/2026-10-20-layer-restore-pilot.md. The
   vendor index at revision 1d4bf0f2 was mapped layer by layer: 64
   layers over 17 shards, eight straddling a boundary (seven DeltaNet,
   one full attention). The cheapest shard tier (2,108,759,344 bytes;
   shards 7, 9, 11, 13, 15, 17) holds one DeltaNet and one full-attention
   layer complete each. Candidate: layers 22 and 23 from shard 7, chosen
   on architecture (both families, contiguous, mid-stack with 40 layers
   downstream for the sensitivity instrument) and cost, never on E0.

2. Code (commit 403c7cdb): scratch/qwen_layer_restore.py (preflight,
   capture, oracle_mlx, fetch, score x 5 arms, summarize, sensitivity),
   scratch/qwen_layer_restore_torch.py (the vendor PyTorch layer in
   .venv_teacher, transformers 5.15.0), scratch/qwen_layer_restore_study.sh,
   llmopt/lab/layer_restore.py with tests. The one-layer oracle on the
   decoded arm C tensors (no download) agrees torch against mlx at
   relative L2 5e-7 and max abs over row RMS 4e-5 on both layers
   (tolerance 1e-4 and 1e-3). Full smoke chain green at 403c7cdb, clean
   tree, stand-in labelled and read as inadmissible by the adjudicator.

3. PRE-REG QWEN-LAYER-RESTORE-0 (RESULTS L81373; JSON
   docs/preregs/qwen-layer-restore-0.json, digest ca108cd6 pinned in
   the driver). Bars: E0 +1.0 point (diagnostic), validation +1.0 point
   with no subset negative (VALIDATED only on a bootstrap interval
   excluding zero), validation KL -10 percent. The refutation counts a
   low-resolution fire as no fire. Preconditions: oracle on the vendor
   weights, DENSE_C parity, CONTROL against the census dump, shard sha.
   Opus review applied before registration (seven should-fix items:
   refutation fold, verdict classes, resolution derivation, census
   check, smoke at a code commit, spec layer family, prior citation).

## On hold (Artin's separate GO, section 6 of the GO message)

The shard fetch: model-00007-of-00018.safetensors, 2,108,759,344 bytes,
sha256 80b0c490..., to ~/qwen_vendor_shards/1d4bf0f2/ beside the head
shard (which stays). Then under liverun at a clean commit:

    .venv/bin/python scripts/liverun.py run qwen-layer-restore-0 -- env FETCH=1 SCORE=1 bash scratch/qwen_layer_restore_study.sh

(the chain runs preflight, capture, both oracle halves on arm C tensors,
fetch, both oracle halves on the vendor weights, five score arms,
summarize, sensitivity, adjudicate). Expected: about 20 to 25 minutes
for the arms, peak memory about 17 GB. Then receipt, prereg and
wording audits, booking, FINDINGS bullet, index links, scoreboard and
figures regen, handoff.

## Not done

- No vendor byte beyond the head shard retrieved; no real receipt under
  logs/qwen_layer_restore/ exists.
- The 3080 mirror: attempt after the push; it was unreachable this
  morning.
