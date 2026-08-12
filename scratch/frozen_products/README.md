# frozen_products

These are the exact programs the cited pipeline drivers executed from
`/tmp` (they existed nowhere in git before this commit). `g19bn_probe.py`
is the three-substitution product of `scratch/g19_bf16_isolation.sh`;
`poly_probe_representative.py` is the single-substitution shape shared by
`gen8_pipeline.sh`, `gen9_pipeline.sh`, `poly3_pipeline.sh`,
`poly4_pipeline.sh`, and `poly5_pipeline.sh` (only the data-file name
varies per driver/loop-iteration).

Regenerate with the two sed commands recorded in each file's provenance
header. New drivers should parameterize the probe instead of sed-patching
a copy of it into `/tmp` (see `/rung`).
