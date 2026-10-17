# Timing rung for exact greedy speculation on the Mac (prepared, not armed)

Status 2026-10-17: prepared under Artin's GO for preparation. Not
registered in RESULTS yet because two gates fail today: the storage
condition for restoring the vendor bf16 shards (58.5 GB free, 76 GB
plus temporary allocation required) and the draft-side probe
(OBSERVATION QWEN-DRAFT-DECODE-COST-0) showing the current CPU draft
path at about 57 s per token. Registering a rung whose outcome the
probe already fixes would be theatre; the rung below is the shape to
freeze once a draft kernel exists or Artin decides to measure the
teacher side alone.

## Quantities (all measured on the Mac, fresh receipts, refuse-if-exists)

- c_t: incremental teacher decode cost per token, streamed bf16 vendor
  shards, KV cache plus DeltaNet recurrent and conv state carried,
  measured over at least 64 steps on 3 prompts after a warm first step;
  report median and p90, I/O bytes read per step, and page-cache state.
- c_v(k): one verification forward over k + 1 positions appended to a
  cached prefix, for k in {1, 2, 4, 8}; includes state snapshot before
  the block and restore after a simulated rejection at position j for
  each j (rollback cost measured, not assumed); report the stream
  overhead separately from compute by a second pass with weights in
  page cache where they fit.
- c_d: sequential draft token cost for arm C: compressed bytes
  resident, one token at a time with its own cache; measured per layer
  class (linear, full) and whole model; the per-layer decode probe is
  the lower bound until a fused kernel exists.
- State handling: time and bytes for snapshot and restore of (KV list,
  48 recurrent states, 48 conv windows) at prefix lengths 128 and 1024.
- Memory: peak RSS per phase; I/O: bytes read per teacher step.

## Break-even law (theorem, model-free)

Speculation beats plain decoding iff k c_d + c_v(k) + c_state < Y_k c_t,
with Y_k the PER-STREAM E0 replay yield (OBSERVATION
QWEN-E0-SPEC-CENSUS-0-PER-STREAM: 1.818, 2.487, 3.427, 4.480 for k =
1, 2, 4, 8). The registered E0 values are inputs and are not modified.
Projected tokens per second = Y_k / (k c_d + c_v(k) + c_state), labelled
PROJECTED; measured tokens per second come only from running the
decoder, which this rung does not do.

## Bars to freeze before measurement

1. TEACHER-IDENTITY: the streamed teacher reproduces the recorded
   rollout token at every step of the 3 timing prompts (token identity
   against logs/qwenteacher_v2, not inferred from the Lean theorem).
2. VERIFY-SCALES: c_v(8) <= 1.5 c_v(1) (verification is stream-bound,
   the premise of the whole scheme).
3. BREAK-EVEN-K4: 4 c_d + c_v(4) + c_state < 3.427 c_t.
Refuted if bar 2 or bar 3 NO-FIRE. Bar 1 NO-FIRE books INSTRUMENT-ALARM.

## Preconditions for the vendor restore (none waived)

Free space >= 76 GB plus temporary allocation; snapshot_download with
local_dir to ~/qwen_vendor_bf16 (a NEW directory; ~/qwen_vendor keeps
the five metadata files and is never written); revision
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; 18 shards, 55.59 GB, LFS
sha256 per shard read from the hub on 2026-10-17 (model-00001
ba0ce20a..., model-00018 1d347950..., full list to be pinned in the
pre-reg JSON); no live run; verify every shard sha before any load.

## Smallest shapes first

1. One-layer probes (done for the draft side, this date).
2. One-layer teacher stream probe: time to read and apply one bf16
   layer from disk (cold and warm) before any full-model step.
3. Full-model teacher identity on one prompt, 32 steps.
4. Then the bars.
