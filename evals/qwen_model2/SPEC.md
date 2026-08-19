# QWEN-MODEL-2 held-out evaluation surface — FROZEN PRE-COMPOSE

Committed 2026-08-18, BEFORE the P_X/P_K arms exist. The MODEL-1
surface (evals/qwen_model1/) selected the MODEL-2 allocation policy
(every marginal in the design spec was measured on it), so it is
the DEVELOPMENT surface; this directory is the disjoint evaluation
surface. Disjointness is asserted mechanically (no shared corpus
line, no shared prefix) at freeze.

Same instrument contract as MODEL-1: teacher-forced X (mean excess
CE over the locked teacher) on corpus.txt positions, forward
KL(teacher||arm) + top-1 agreement on every prefixes.jsonl
position, fp16 records, sensitivity floors, small-n fence 30.
Teacher: the SAME locked vendor pass (commit pin 0ca4151 class,
revision 1d4bf0f2) run once over this payload; record shas pinned
in its manifest before any P_X/P_K compose.

prompts.jsonl rollouts are OPTIONAL color for this surface (the
registered MODEL-2 bars are teacher-forced only).
