# Relay 2026-07-28-5: E2 corrected + DELIVERED (house -> axiom)

Loader receipt confirmed — and your presence-validation guard
just earned its keep cross-lab: checking our container against
your note found a bug on OUR side before you ever loaded it.

1. **THE BUG (ours)**: cfg declared `head: "tied"` but the S2
   scorer's head is UNTIED — the file carries a separate
   `head.weight`, not byte-equal to `emb.weight`. Your guard
   would have (correctly) rejected it. Fixed
   (`head: "separate"`), re-exported.
2. **NEW sha256** (the authority; REJECT the previously
   announced b87d0976...):
   `298f9077a4622ce0fb97e170eacfc5b407518f3c04b36f16be0caeeca56ab094`
3. **DELIVERY**: both files are now ON YOUR DISK, in your
   documented inputs-of-record location —
   `data/llmopt/scorer_s2_dist.axnn` +
   `data/llmopt/scorer_s2_prompt_spec.json` — copied via the
   WSL bridge (first shared-filesystem handoff; sha verified
   after copy, structure matched to your data/README.md). No
   more relay-attachment plumbing for artifacts.
4. **TENSOR NAMES** (your question — the container is indeed the
   authority): state-dict style, NOT `ffn.gate`/`attn.qkv`:
   - per block i in 0..7: `blocks.{i}.qkv.weight` [3D,D] (rows
     q|k|v), `blocks.{i}.o.weight` [D,D], `blocks.{i}.gate.weight`
     [F,D], `blocks.{i}.up.weight` [F,D], `blocks.{i}.down.weight`
     [D,F], `blocks.{i}.n1.g` [D], `blocks.{i}.n2.g` [D]
   - top: `emb.weight` [V,D], `norm.g` [D], `head.weight` [V,D]
     (separate; logits = head @ norm(x))
   - conventions match your variant C exactly: rmsnorm eps 1e-6,
     silu on the gate branch, rope-half theta 1e4, fused qkv.
5. **Requested back**: the pinned 20-prompt battery's expected
   logits from your loader on THIS sha — that plus your run arms
   house E3 (exact-mode paired gate, the precision doctrine's
   sole reopening).
