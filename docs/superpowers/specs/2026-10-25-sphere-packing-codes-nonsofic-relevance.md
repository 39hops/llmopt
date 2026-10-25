# Read-only study: the August 2026 packing, code and non-sofic results against the house's compression, geometry, verification and transfer questions (2026-10-25)

Artin's ask (2026-10-25): review the August 2026 mathematical advances
in high-dimensional sphere packing, binary and spherical codes, and
non-sofic group constructions, and say whether their ACTUAL techniques,
not analogies, give an actionable path toward better neural-network
compression, representation geometry, formal verification, or knowledge
transfer. No implementation, no Qwen experiment. The goal is to find
genuine leverage or to say plainly that there is none.

Sources were read at the abstract and metadata level only (the cite
skill: arXiv metadata through scripts/cite_lookup.py, the OpenAI GitHub
README, web summaries). The 249-page OpenAI manuscript is a PDF and was
NOT opened; every statement about its chapters below is what the
abstracts of follow-up papers, the repository README and secondary
summaries say, and is marked as such. Numbers quoted from a social-media
post are marked as unverified.

## 1. What the August 2026 results are (as far as metadata carries)

OpenAI, "Ten Advances in Mathematics and Theoretical Computer Science"
(1 August 2026; Lean 4.32.0 certificates at github.com/openai/ten-proofs,
one file per chapter; the README states no caveats). The three chapters
Artin named, plus two neighbours that matter here:

- Chapter 1, high-dimensional sphere packing (SpherePacking.lean). The
  exact exponential rate of the Cohn-Elkies linear program is determined:
  per MathWorld's statement of the result, lim LP_d^(1/d) = sqrt(e/(2 pi)),
  so the LP's density bound decays as 2^(-(alpha* + o(1)) d) with alpha* =
  (1/2) log2(2 pi / e) = 0.6044..., against the Kabatiansky-Levenshtein
  exponent 0.5990 that had stood since 1978. Salmon's abstract (arXiv
  2609.03736) records the mechanism: both Fourier sign-uncertainty radii
  are (1/pi + o(1)) sqrt(d), which proves the Afkhami-Jeddi, Cohn,
  Hartman, de Laat and Tajdini conjecture. This is an UPPER bound on
  packing density, and a limitation theorem: the LP cannot do better than
  this exponent. Lower bounds moved separately and earlier: Klartag's
  lattice packings of density at least c n^2 2^(-n) (arXiv 2504.05042,
  April 2025, a stochastically evolving ellipsoid; expository account
  Aubrun, arXiv 2606.13313) and, in October 2026, Gargava, Groebner and
  Simkin's d^2 log log d 2^(-d) in every large dimension (arXiv
  2610.12619, a multi-cyclotomic gluing of Klartag's method with
  Venkatesh's symmetries). Vanoni, Guo and Torquato (arXiv 2608.00920)
  analyse CONJECTURAL exponential improvements of Minkowski's lower bound
  (rate 2^(-(0.6226 + o(1)) d) from an explicit hyperuniform
  construction), conditional on a realisability conjecture, and prove
  strong duality between the pair-correlation program and the
  Cohn-Elkies dual. The gap between the best unconditional lower bound
  (polynomial times 2^(-d)) and the upper bound (2^(-0.6044 d)) stays
  exponential.
- Chapter 2, binary and spherical codes (MetricCodes.lean). The first
  improvement of the McEliece-Rodemich-Rumsey-Welch (MRRW) upper bounds
  on the rate of binary codes since 1977, by analytic two-point
  Delsarte certificates, with a matching improvement of the
  Kabatiansky-Levenshtein bound for spherical codes (Salmon's abstract,
  arXiv 2610.21263: "OpenAI's Chapter 2 improved both bounds"). Concurrent
  and independent: Alrabiah and Guruswami (arXiv 2608.09347) reach the
  same bound through classical-quantum channels and the pretty good
  measurement; Barg (arXiv 2609.01860) explains why the two proofs agree
  (a subspace attached to every codeword and moved with it; the bound
  counts how many such subspaces fit). Size of the move, UNVERIFIED (a
  social-media post by a coding theorist, not checked against the
  manuscript): at one relative distance GV 0.18059, MRRW 0.34271, new
  0.33934, about 2.1 percent of the GV-to-MRRW gap. Salmon's table
  (abstract-level): about 0.35038 against 0.35371 at delta = 0.25. The
  GV LOWER bound for binary and spherical codes did not move in 2026
  (the only GV improvements found are q-ary, arXiv 2601.18590, and an
  alphabet-42 algebraic-geometry result, ECCC TR26-272). Follow-ups:
  Gay, Jeronimo and Liu's honeycomb hierarchy (arXiv 2608.20287,
  representation-theoretic, monotone, eventually recovering the exact
  A_2(n, d)); Salmon's half-rate theorem R_D(1/2 - 1/pi) = 1/2 (the
  Delsarte LP's exact rate at that distance, arXiv 2609.03736) and the
  full Delsarte rate for binary and spherical codes in closed form
  (elliptic integrals, arXiv 2610.21263). Every one of these is an upper
  bound or an exact LP exponent; none is a code construction.
- Chapter 3, non-sofic groups (NonSoficGroup.lean, about 34,000 lines
  per secondary reports). The unit group of the binary Leavitt algebra
  L(1,2) is non-sofic, settling Gromov's 1999 question negatively;
  secondary summaries name Property (T), Kun's theorem and Thompson's
  group V in the argument. Fournier-Facio (arXiv 2608.02025, August 2026)
  takes the result as established and gives a different source of
  examples from the same technical criterion, including finitely
  presented torsion-free ones (the Leavitt unit group always has
  torsion). Ersoy (arXiv 2604.19174) had already derived structural
  consequences conditional on existence. A September 2026 preprint
  (arXiv 2609.27795) gives a conditional non-hyperlinear construction.
  The human audit of the ten results (Sienicki and Sienicki, arXiv
  2608.14673, v3 September 2026) reports no confirmed substantive error
  in a principal result, uneven review depth, and specialists' disputes
  about novelty and attribution for the sphere-packing and non-sofic
  chapters; the OpenAI README itself states no caveat.
- Chapter 7, closest vector problem (GapCVP.lean): polynomial-factor
  hardness of approximation for CVP, "with consequences for decoding and
  lattice problems" (README). Relevant below because nearest-lattice-point
  decoding IS vector quantization.
- Chapter 8, Ehrhart's volume conjecture: the sharp maximal volume of a
  convex body whose centroid is its only interior lattice point.
  Tangential; listed for completeness.

## 2. The house facts these results have to meet

- Arm C's codecs (llmopt/lab/qcodec.py): W4 is a FIXED-RATE vector
  quantizer in dimension 4 with 256 codewords (8 bits per 4 weights, a
  per-matrix fp16 codebook, one E8M0 exponent byte per block of 128),
  about 2.06 bits per weight; S16 is a 16-level scalar quantizer (4 bits
  per weight plus the block exponent), about 4.06. Both are
  entropy-unconstrained (no entropy coding of indices).
- Measured: W4 beats the dynamic-programming-optimal scalar quantizer by
  14.3 percent pooled operator error on a dense frontier FFN
  (OBSERVATION QWEN-STREAM-PROBE-0, RESULTS L32578); on layer 22 with
  real activations, an orthogonal input-basis rotation and a HIGGS-style
  data-free grid cut the activation-weighted output error of the W4
  matrices by 4 to 11 percent, a refit codebook by at most 1.5 percent,
  the development-weighted assignment by 26 percent on gate_proj and not
  at all on down_proj; every bar at 15 percent NO-FIRE (VERDICT
  QWEN-LOSSY-GEOMETRY-0, L83603); the complete gated MLP output moves 2.4
  percent under the best candidate (VERDICT QWEN-MLP-COUPLED-0, L84000);
  the original BF16 weights carry 10.47 to 10.54 bits of zeroth-order
  entropy per weight, so lossless storage under 8 bits is refuted
  (VERDICT LOSSLESS-BF16-FEASIBILITY-0, L83153). Quantization and
  representation research is CLOSED for now by those verdicts.
- Exact symmetries already proved and measured: the SwiGLU
  intermediate-neuron permutation (scratch/leancheck/MlpPermutation.lean,
  OBSERVATION SWIGLU-PERMUTATION-0, L80758), the orthogonal input-basis
  rotation (InputRotation.lean, exact to float32 1e-6 and NULL for error
  reduction), the BF16 exponent split (BF16Split.lean); the weight-space
  reader where permutation AUGMENTATION (88.4 percent) beat canonical
  sorting (82.4) over raw weights (80.8) (VERDICT WEIGHT-READER-0,
  L20747); merges across independent lineages crater to zero (MERGE-SPACE-1
  and -2, L26770, L26866). The DeltaNet block's value-head permutation and
  the residual rotations are the remaining exact transformations named in
  the 2026-10-24 transfer assessment (banked in RIFF-LEDGER this session).
- Dimensions: hidden 5,120; a token-embedding table of 248,320 rows; W4
  codebooks live in dimension 4; S16 in dimension 1.

## 3. The distinctions Artin asked for

PACKING against COVERING and rate-distortion. A quantizer is a COVERING
problem (every source point within distance r of a codeword; distortion
is the second moment of the cell), not a packing problem (codewords at
mutual distance 2r). The August results are packing UPPER bounds (and an
exact LP exponent) in the d -> infinity regime. The quantities that govern
a fixed-rate quantizer at a fixed small dimension are the normalized
second moment G of the lattice or codebook (Z^n 1/12 = 0.0833, D4 0.0766,
E8 0.0717, the asymptotic Zador limit 1/(2 pi e) = 0.0585; Conway and
Sloane; Agrell and Allen, arXiv 2202.09605, prove the optimal lattice
quantizer's error is white; Agrell, Pook-Kolb and Allen, arXiv
2401.01799, search such lattices numerically) and the shape gain of the
source. Translated: the GRANULAR gain still available over our d = 4
codebook is at most about 0.4 dB to a D4-class cell and about 0.65 dB to
E8 at d = 8, and the whole asymptotic granular gain over a cube is 1.53 dB
(0.25 bit per dimension). An asymptotic packing exponent moving by 2
percent of a gap says nothing at d = 4 or d = 8 and nothing about
covering at any d. Upper bounds are upper bounds: none of the three
chapters, nor Klartag's existence proof (a stochastic ellipsoid, not an
explicit lattice), nor the Gargava-Groebner-Simkin gluing, hands anyone
a quantizer. The constructive side of this literature is elsewhere:
nested lattice quantizers on the Gosset lattice E8 (NestQuant, Savkin,
Ordentlich, Polyanskiy and Porat, arXiv 2502.09720, "information-
theoretically optimal for low-precision matrix multiplication" per its
abstract) and data-oblivious online vector quantization with
near-optimal distortion (TurboQuant, arXiv 2504.19874). Those two are
the papers a reopened codebook study would read, and the house's HIGGS,
QuIP# and QTIP grounding already sits beside them.

CODEBOOK CAPACITY at our dimensions and bitrates. The question "do the
new bounds constrain how good a 256-codeword d = 4 codebook can be" has a
clean answer: no, and the constraint that does exist is older and
already compatible with the measurements. At fixed rate 2 bits per
dimension in d = 4, the error floor is set by G(d = 4 best) and the
source's shape; the house's W4 is 14.3 percent (about 0.7 to 1.3 dB
depending on whether the operator error is read as RMS or MSE) better
than the optimal scalar at the same bits, inside the 1.53 dB envelope,
and the geometry study's 4 to 11 percent (0.2 to 0.5 dB) from rotation
and grids is what a d = 4 to d = 8 granular and shape gain predicts; the
26 percent on gate_proj from activation-weighted assignment is a
source-weighting effect, not a codebook-geometry effect, and did not
transfer. For the embedding table and the normalized residual stream,
the spherical-code bounds are vacuous by orders of magnitude: 248,320 =
2^17.9 rows in d = 5,120 is a rate of 0.0035 bits per dimension, where
every bound in this literature reads "essentially unconstrained"; the
Kabatiansky-Levenshtein regime is rates of order 0.1 to 0.5. Binary codes
enter through sign quantization of normalized vectors (1 bit per
dimension), and there the operative quantity for retrieval or hashing is
a collision probability, not the size of a distance-d code; the MRRW
move does not touch it.

NEURAL PARAMETER SYMMETRIES against group constructions. The symmetry
groups of the towers the house measures are: the finite symmetric
group S_m on MLP neurons (and on DeltaNet value heads with their norm
and out_proj), finite sign groups where an activation is odd (not
SwiGLU), and the compact orthogonal group O(n) on the RMSNorm-bounded
residual basis; positive rescaling is NOT a symmetry here (SiLU is not
homogeneous). Finite groups and compact groups are sofic for trivial
reasons (finite groups by definition, amenable groups by Gromov and
Weiss). A non-sofic group is a countable discrete group with NO
sequence of almost-actions on finite sets; it is the opposite of
everything a weight-space symmetry is. The only honest contact point is
STABILITY: whether weights that ALMOST satisfy a symmetry (two
networks nearly permutation-equivalent, the Git Re-Basin setting,
Ainsworth, Hayase and Srinivasa, arXiv 2209.04836) are close to weights
that exactly satisfy it. That is the matricial-stability question of
Hadwin and Shulman (arXiv 1706.08405) and its Ulam-stability relatives,
and for FINITE groups it is classical and true (averaging over the group
is a retraction); the house already uses the exact group in Lean and the
augmentation result says the right move is to teach the invariance, not
impose it. Non-soficity adds nothing here, and this study does not
assume it applies anywhere in the lab's objects.

FORMAL VERIFICATION. The ten certificates are a precedent, not a tool:
large Lean developments for asymptotic exponents (MetricCodes.lean,
SpherePacking.lean) formalize limits in d, which the house never uses.
The house's own practice (model-free theorems about exactness of a
transformation, scratch/leancheck/) is the right register for what the
lab can verify, and the two verifiable statements that WOULD matter if
a codebook study ever reopened are finite-dimensional and elementary:
the coset arithmetic of a nested lattice quantizer (encode, decode, the
modulo-shift identity NestQuant relies on) and the exactness of a block
exponent plus lattice point as a pair of maps. Chapter 7's CVP hardness
is the formal reason such a quantizer must use a STRUCTURED lattice with
a fast decoder (D4, E8, a trellis): nearest-point decoding on a generic
lattice is hard to approximate to a polynomial factor, so "learn an
arbitrary lattice codebook and decode optimally" is not a plan at any
dimension that matters.

KNOWLEDGE TRANSFER. Nothing in the three chapters touches function
transfer between networks. Barg's reading of the code bound (a subspace
attached to each codeword) is a proof technique for counting, not a
representation of knowledge. The transfer candidates stand where the
2026-10-24 assessment left them (RIFF-LEDGER, this session).

COMPATIBILITY with the house's NULLs. Everything above is consistent
with QWEN-LOSSY-GEOMETRY-0 and QWEN-MLP-COUPLED-0: the quantizer theory
at d = 4 to 8 predicted gains of the size the study measured and no
larger, the asymptotic bounds predict nothing at these dimensions, and
the lossless entropy census (10.5 bits per weight) is a statement about
the SOURCE, which no code bound changes.

## 4. Ranked theorem-to-experiment relevance matrix

Rank 1 is the most relevant; "none" means the honest answer is that no
experiment follows.

| rank | theorem or result | precise assumption under which it bears on the house | expected benefit | mathematical obstacle | falsifiable prediction (not authorized to run) |
|---|---|---|---|---|---|
| 1 | GapCVP polynomial-factor hardness (OpenAI Ch. 7) | a future codebook that quantizes by nearest lattice point in an unstructured lattice of dimension above about 8 | a stop sign: codebooks must be D4, E8 or trellis with fast decoders (the NestQuant and QTIP route), never a learned generic lattice | none; it is a limitation theorem | any "learned lattice codebook" paper that claims optimal decoding above d = 8 at interactive speed is using a structured lattice or an approximate decoder |
| 2 | finite-dimensional lattice quantizer theory (Zador; Agrell-Allen white-error theorem; G of D4, E8; the 1.53 dB envelope) | fixed-rate, entropy-unconstrained VQ at 2 bits per dimension on roughly Gaussian blocks, as W4 is | an explicit CAP on what any codebook geometry can still buy over W4: about 0.4 dB at d = 4, about 0.65 dB at d = 8; compatible with the 4 to 11 percent measured | the source is not Gaussian and not white; activation weighting dominates (the 26 percent on gate_proj) | an E8 nested-lattice codebook at 2.06 bits per weight on layer-22 gate_proj with real activations reads an output-error ratio of at least 0.85 against W4 (the registered NULL bar does not fire); if it reads under 0.80 this study is wrong about the regime |
| 3 | spherical-code and binary-code upper bounds (OpenAI Ch. 2; Alrabiah-Guruswami; Salmon's Delsarte rates; honeycomb hierarchy) | a representation at rate 0.1 to 0.5 bits per dimension with a hard minimum angular separation, in d -> infinity | none at the house's rates (0.0035 bits per dimension for the embedding table; sign codes read by collision probability) | the bounds are asymptotic in d and move by 2 percent of a gap | the maximal pairwise cosine among the 248,320 S16 embedding rows is far above what a Kabatiansky-Levenshtein-optimal spherical code of that size in d = 5,120 would need (which is near zero): the table is nowhere near a packing, so no code bound binds; a CPU desk census on the stored embedding could read it, not requested |
| 4 | the exact Cohn-Elkies exponent sqrt(e / 2 pi) (OpenAI Ch. 1) and the polynomial lower bounds (Klartag; Gargava-Groebner-Simkin) | sphere PACKING density as d -> infinity | none for quantization (covering, finite d) | upper bound and existence proofs; no construction; packing is not covering | none |
| 5 | non-sofic groups (OpenAI Ch. 3; Fournier-Facio torsion-free examples) | an infinite, non-amenable, finitely presented symmetry group acting on parameters | none; every parameter symmetry group in the lab is finite or compact, hence sofic; the stability question that does arise (almost-symmetric weights near exactly symmetric ones) is classical for finite groups | the objects do not meet | none; stated as a non-connection, not forced |
| 6 | the Lean certificate corpus (ten-proofs) | the lab wants precedent for large formal developments | methodological: independent checking (Comparator) as a pattern; the house already verifies exactness theorems in Lean | the certified objects are asymptotic exponents the lab never uses | none |

## 5. Verdict of the study

No genuinely new mathematical leverage for the lab's compression,
representation, verification or transfer questions comes out of the
August 2026 results. The code and packing chapters are upper bounds and
exact LP exponents in the d -> infinity regime; the lab's codebooks live
at d = 4 and its representations at rates where every such bound is
vacuous; the relevant finite-dimensional theory is older and already
agrees with the house's NULLs. The one item worth remembering is
negative and useful: CVP hardness fixes the shape any future codebook
must take. The non-sofic construction is a landmark with no object in
common with weight-space symmetry. Nothing here reopens the closed
quantization thread, and nothing here authorizes a run.

## Sources (metadata level)

arXiv 2504.05042 (Klartag), 2606.13313 (Aubrun), 2610.12619 (Gargava,
Groebner, Simkin), 2608.00920 (Vanoni, Guo, Torquato), 2608.09347
(Alrabiah, Guruswami), 2609.01860 (Barg), 2608.20287 (Gay, Jeronimo,
Liu), 2609.03736 and 2610.21263 (Salmon), 2608.02025 (Fournier-Facio),
2604.19174 (Ersoy), 2608.14673 (Sienicki, Sienicki), 2512.04829
(Tutunov et al., AI-assisted SDP bounds in dimensions 4 to 16, 2025),
2202.09605 and 2401.01799 (Agrell et al.), 2502.09720 (NestQuant),
2504.19874 (TurboQuant), 2209.04836 (Git Re-Basin), 1706.08405 (Hadwin,
Shulman); github.com/openai/ten-proofs README; MathWorld's statement of
the Chapter 1 limit. Conway and Sloane for the lattice second moments.
