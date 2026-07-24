# Repo taxonomy reorganization (APPROVED by Artin 2026-07-24; execute at tonight's freeze)

Scheme: BY PROGRAM (what the artifact is). Solo execution (Fable,
no-subagents policy). Freeze point: after gen-9 A/B + poly5 land
and before the 45M re-ask launches (or after it completes).

## Target layout
checkpoints/{engine, lora05b, micromodel, ternary, metabolic,
             continents, instruments}
data/{chains, continents, diets, exchange, qual, archive}

Filing rules: engine = search-era brains (nnue, syndrome, dispatch,
magic, value/policy heads); micromodel = mathnative fp32 lineage
incl. GRPO/consol; ternary = discrete substrate incl. merged/grown/
tournament + latents; metabolic = metab_*/exchange_*/d2/v5 sessions;
continents = series/phys/poly/vmasm/dual births + their chains/
probes (data side); diets = COMPOSED diet files only; exchange =
stuck/practice/walls/duo shards; qual = axiom reference sets;
instruments = seedvar/warmbirth/template/grid/probe-support ckpts.

## The gate (all must pass, same pass)
1. Update every path reference: scripts + scratch hardcodes (grep
   data/ and checkpoints/ across tree), watcher scripts, pipeline
   templates. scripts/INDEX.md regen.
2. **gate_rarity census**: glob data/*.jsonl -> recursive
   data/**/*.jsonl AND re-freeze bins; book a one-line RESULTS note
   that census composition is unchanged (assert same skeleton count
   pre/post move).
3. pytest green + smoke-launch every entry-point script.
4. WSL mirror: apply the same moves on the 3080 checkout in the
   same session (hash-verified sync after).
5. Nothing DELETED — moves only; data/archive/ for superseded.
