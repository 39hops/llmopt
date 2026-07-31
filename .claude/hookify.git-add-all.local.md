---
name: git-add-all-warning
enabled: true
event: bash
pattern: git add (-A|--all|\.)($| )
action: warn
---
**Blanket git add in llmopt**: prefer explicit paths. This repo
carries huge untracked artifact trees; blanket adds have timed
out (51GB checkpoints/ incident, 2026-07-31) and risk committing
data files. checkpoints/ is gitignored — curated ones go through
`git add -f checkpoints/confirmed/...` + the manifest.
