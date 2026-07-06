# Move-proposer rung: policy model in front of the searcher — design

Date: 2026-07-07. Status: approved design (standing autonomous
authorization; user sick/WFH, minimal interruption). Parent: roadmap #1
final piece; CLAUDE.md expert-iteration thread ("step-level search
version — model scores candidate rewrites — is the long-term goal").

## Question

Does a fine-tuned 0.5B ranking legal moves let the beam trade branching
for depth — higher solve rate at fixed node budget than full
enumeration, under both HCE and NNUE evals?

## Decisions

1. **Proposer ranks, never generates.** `successors()` keeps
   enumerating legal moves (legality by construction). The model ranks
   the enumerated list; beam expands top-k (default k=3). Rejected:
   free-form rewrite generation (needs repair; legal set is enumerable).
2. **Ranking = answer-token likelihood over a numbered-choice prompt.**
   Prompt renders the state (sp.sstr) and the numbered legal-move
   labels; candidates scored by the logprob of their number tokens,
   batched. 1-2 answer tokens per candidate keeps inference cheap.
3. **Training data = verifier-approved winning paths.** For each solved
   search (both kinds, L1-3), every ply contributes
   (state, legal-move labels, on-path label index). Oracle guarantees
   every label is a winning move — the imitation half of expert
   iteration. JSONL rows: {"state": sstr, "moves": [labels...],
   "answer": index}. Target ~500 solved problems -> ~3k rows.
4. **Hygiene:** seeds `proposer-train/-eval/-race-{kind}-{level}-{i}`;
   root-srepr exclude guard between splits; training recipe verbatim
   from CLAUDE.md: LoRA r=16 all proj linears (`train/lora.py`), loss
   on answer tokens only, length-bucketed batches with per-epoch order
   shuffling. Model: Qwen2.5-0.5B-Instruct.
5. **Engine interface:** `beam_search(..., proposer=None, propose_k=None)`
   where `proposer: Callable[[State, list[tuple[str, State]]], list[tuple[str, State]]]`
   reranks the successor list before truncation to propose_k. Engine
   stays model-free; the model-backed callable lives in
   `llmopt/search/proposer.py` (prompt building + HF scoring, device
   agnostic) and is constructed by scripts.
6. **Machines:** data gen + proposer module + bench scaffolding on the
   Mac (CPU); LoRA training on the Windows 3080 over SSH (torch+PEFT);
   the race back on the Mac (transformers MPS inference — one scoring
   code path everywhere).
7. **Heats:** (0) held-out top-1/top-3 move accuracy;
   (1) solve rate at budgets 25/50/100/200 on `proposer-race-*`
   problems, configs: full+HCE (baseline), full+NNUE, proposed-k3+HCE,
   proposed-k3+NNUE (headline). Random-k3 truncation as a control —
   if random pruning matches the model, the model adds nothing (honest
   null pre-registered).
8. **Node accounting:** nodes = children actually generated (unchanged),
   so proposer configs generate fewer children per state and go deeper
   at equal budget — that IS the mechanism being measured. Proposer
   model inference time is reported separately (wall clock), not
   counted as nodes.

## Out of scope

Iterated expert iteration (retrain on new search outputs), difficulty
frontier curriculum, value-head multi-task training, GPU inference in
the race, tree search variants (MCTS).
