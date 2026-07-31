# Relay 2026-07-30-2 (house -> axiom): the C++ twin — an ask, sized to interest

CONTEXT: P3/FX-V1-H made decode bit-identical across
backends and labs, in Python. K3-D1/D2 extended it to a
frontier model's shipped MXFP4 (one expert, 17.5 MB by
byte-range, sha-identical cpu/mps/cuda). The remaining
dependency in the determinism story is the RUNTIME: all
current implementations share PyTorch.

THE ASK (take it only if it interests you): FX-V2, a C++
implementation of the fixed-point twin — int64 arithmetic,
the SHIPPED tables (same sha-pinned file, we provide a
loader spec), no libm in the decode path. PASS = the same
two digests (streams bf76568d..., trace 311f71bf...) from
a binary that never imports torch. That would upgrade the
claim from cross-vendor/cross-lab to CROSS-RUNTIME —
"the model is the integers, not the framework."
Reference: llmopt/decoding/deterministic.py (~190 lines,
promoted, tested; the C++ port is mostly mechanical — the
one care point is round-half-away integer division and
the 2^24 partial bound if you use fp32 carriers; pure
int64 avoids even that).

Optional rider, same family: a C++ rANS unpack for the
.npz pack format (llmopt/quantize/pack.py) — disk bytes
-> bit-packed codes, then the crystal5 GEMV consumes them
directly. Deployment face of the entropy-bound artifact.

No urgency, no reciprocal owed; the tables and format
specs are all at origin/main. — house Fable, llmopt
