"""llmopt.common — shared utilities (spec 2026-08-12 §Phase 4).

Not instruments (those live in llmopt.lab): device resolution,
string-seeded RNG, checkpoint IO. torch imports stay lazy so
`import llmopt` never drags torch in.
"""
