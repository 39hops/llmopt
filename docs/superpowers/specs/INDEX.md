# Specs index — design docs with their outcomes

Specs are provenance, kept verbatim after execution. Status here, one
line each; verdicts live in docs/RESULTS.md, current queue in
docs/BOARD.md. Regenerated 2026-08-09 (reviewer-fleet draft,
Fable-adopted; rows tagged [UNVERIFIED] carry a derived-not-confirmed
status — resolve at next housekeeping).

| Spec | Purpose (spec's own heading) | Status |
|---|---|---|
| 2026-07-05-metal-splitk-decode-design | Split-K attention decode for Metal (MLX) | COMPLETE-BOOKED — ties mx.fast SDPA at T=32k (kernels/metal.py) |
| 2026-07-06-hce-rung1-primitive-moves-design | HCE rung 1: primitive differentiation move set | COMPLETE-BOOKED — rung 1 shipped, doit demoted to verifier |
| 2026-07-06-hce-calibration-design | HCE calibration harness | COMPLETE-BOOKED — rho=+0.685/+0.712, motivated NNUE |
| 2026-07-06-next-sessions-roadmap | Next-sessions roadmap (drafted 2026-07-06) | SUPERSEDED-BY-docs/BOARD.md |
| 2026-07-06-rotate-quantize-design | Rotate-then-quantize | COMPLETE-BOOKED — 15-20% RTN error cut at 4/3 bits; 2-bit inversion recorded |
| 2026-07-06-task-arithmetic-design | Task arithmetic on LoRA task vectors | COMPLETE-BOOKED — transfer complete, addition annihilates |
| 2026-07-06-weightspace-reader-design | Weight-space reader | COMPLETE-BOOKED — 80.8/82.4/88.4 (WEIGHT-READER-0, TENET D0) |
| 2026-07-07-adaptive-k-design | Adaptive branching: confidence-gated depth vs breadth | COMPLETE-BOOKED — T=1.0 null -> T=0.1 champion (300/360) |
| 2026-07-07-engine-optimizations-design | Engine optimization trio | COMPLETE-BOOKED — sampled verification 1.65x, byte-identical |
| 2026-07-07-expert-iteration-r2-design | Expert iteration round 2 | COMPLETE-BOOKED — step-function-to-ceiling |
| 2026-07-07-mathgen-expansion-design | mathgen expansion: complex calculus + limit rules | COMPLETE-BOOKED — ten kinds in a day |
| 2026-07-07-mathgen-series-inequalities-design | mathgen: series convergence + inequalities | PARKED — series landed; inequalities unconsumed |
| 2026-07-07-move-proposer-design | Move-proposer rung | COMPLETE-BOOKED — 99.7% top-3; later ambushed by the markov dictionary |
| 2026-07-07-nnue-eval-design | NNUE rung: learned eval | COMPLETE-BOOKED — rho=+0.937, wins/ties all 24 cells |
| 2026-07-07-proofs-rung-design | Proofs rung: induction v0 -> Lean vision | PARKED [UNVERIFIED] — Lean vision realized via the 08-04..08-07 LEAN-TIER arc (separate relays) |
| 2026-07-07-rung2-integration-moves-design | Rung 2: integration move set | COMPLETE-BOOKED — u-sub/by-parts/linearity |
| 2026-07-07-tabula-rasa-design | Tabula rasa: the AlphaZero-way ablation | COMPLETE-BOOKED — self-teaching is a step function to the reachable-set ceiling |
| 2026-07-08-tcount-engine-design | T-count engine (quantum circuits) | COMPLETE-BOOKED — ZX rungs 0-6 (RESULTS:448) |
| 2026-07-12-step-expert-iteration-design | Step-Level Expert Iteration Loop | SUPERSEDED [UNVERIFIED] — GRPO era closed (BOARD); driver scripts/expert_loop.py still exists |
| 2026-07-12-variational-ground-engine-design | Variational Ground-State Engine (physics rung) | COMPLETE-BOOKED — rung 1 shipped; structure search closed |
| 2026-07-13-syndrome-head-design | Syndrome head on the step model | COMPLETE-BOOKED — CLOSED, payoff-3 NULL; rungs 2/3 banked |
| 2026-07-14-step-grpo-design | GRPO at the frontier band | SUPERSEDED-BY-metabolic/exchange loop |
| 2026-07-14-grpo-v2-and-unified-climb | GRPO v2 + the unified climb | SUPERSEDED-BY-metabolic/exchange loop; unified-climb fold failed its gate |
| 2026-07-15-post-climb-strategy | Post-climb strategy | SUPERSEDED [UNVERIFIED] — RL-climb premise closed with the GRPO era |
| 2026-07-15-math-native-micromodel | Math-native micro-model (hyphenated twin) | SUPERSEDED-BY-2026-07-15-mathnative-micromodel.md [UNVERIFIED — twin resolution by mtime inference] |
| 2026-07-15-mathnative-micromodel | Math-native micro-model | COMPLETE-BOOKED — the line became production (gen-6..gen-9, the CROWN row) |
| 2026-07-17-run3-small-cycles | Run 3: small cycles, dense readings | COMPLETE-BOOKED — recipe landed; GRPO era since closed |
| 2026-07-17-rabbit-hole | Five linear-algebra tunnels | PARTIAL [UNVERIFIED] — tunnel 1 booked (GPTQ-int3); tunnels 2-5 unlocated |
| 2026-07-18-alphabet-tournament | The Alphabet Tournament | PARKED — BANKED row; partially answered by the ZX column |
| 2026-07-18-axiom-backend | Axiom as the engine backend | LIVE [UNVERIFIED scope] — AXIOM LOOP thread; generate/solve rungs unconfirmed |
| 2026-07-19-gen6 | Gen-6: first territory birth, grow-vs-rebirth | COMPLETE-BOOKED — gen6_grown is the standing champion |
| 2026-07-20-mass-targeted-training | Mass-targeted training | COMPLETE-BOOKED — "PROVEN, knob broken", -38% wall |
| 2026-07-20-l9-territory | L9 territory design (ODE continent) | PARKED [UNVERIFIED] — self-marked DRAFT; L9a shard farmed, continent BANKED |
| 2026-07-21-training-attacks | Training attacks (lossless-speed program) | STANDING-DOCTRINE — its levers are the speed defaults |
| 2026-07-22-schedule-law-queue | The schedule-law queue | COMPLETE-BOOKED — merged run produced merged_grown |
| 2026-07-22-duo-substrate | Duo-substrate wave | COMPLETE-BOOKED — rarity instrument ADOPTED, duo shard in gen-9 diet |
| 2026-07-22-code-continent-rung1 | Code continent, rung 1 | COMPLETE-BOOKED — rung 1 PASSED (89.2% valid-rewrite) |
| 2026-07-23-metabolic-v4 | Metabolic v4 | SUPERSEDED-BY-2026-07-23-metabolic-v5.md |
| 2026-07-23-metabolic-v5 | Metabolic v5 — the complete practice loop | COMPLETE-BOOKED / STANDING — the exchange CONVERTS |
| 2026-07-24-repo-taxonomy | Repo taxonomy reorganization | PARKED — approved, waits on a natural freeze point |
| 2026-07-24-evening-queue | Evening queue spec | PARKED [UNVERIFIED drain-state] — crown tiebreak since fired as REVIVE-CROWN-TIE-BIRTHS (2026-08-09) |
| 2026-07-25-native-transformer | Closed-system-native transformer rungs 1-3 | COMPLETE-BOOKED — CLOSED with a mechanism; rung-3 attention-init banked |
| 2026-07-25-day-spec | Day spec: sigma-recal + determinability + speed | COMPLETE-BOOKED |
| 2026-07-26-complex-zx-program | The complex/ZX program | PARTIAL [UNVERIFIED Leg A] — Leg B CLOSED at n=3 (HARDENING-P4-2); Leg A complex-bracket cells unconfirmed |
| 2026-07-26-next-session | Next-session plan (07-26) | SUPERSEDED-BY-2026-07-26-next-session-2.md |
| 2026-07-26-next-session-2 | Next-session plan v2 | COMPLETE-BOOKED as session artifact |
| 2026-07-26-format-ladder | The format ladder | COMPLETE-BOOKED + THEORY row |
| 2026-07-26-results-index | RESULTS index + query | COMPLETE-BOOKED — implemented (results_query; index live) |
| 2026-07-26-reverse-llmue-pincer | Reverse LLMUE / temporal pincer (living spec) | PARKED — 7 cells booked; residue awaits pincer-revival GO |
| 2026-07-27-calibrated-scorer | The calibrated scorer (B-b v1) | PARTIAL [UNVERIFIED] — S1/S2 booked; S-tier export BANKED |
| 2026-07-27-exact-stack | The exact stack (FX-V1 era) | COMPLETE-BOOKED — E-series closed |
| 2026-07-27-exact-representations | Exact representations | PARKED — rungs await revival GO |
| 2026-07-28-calibration-program | The Calibration Program | COMPLETE-BOOKED — instrument shipped |
| 2026-07-28-next-session | Next session (07-28, "all") | COMPLETE-BOOKED as session artifact |
| 2026-07-28-symmetry-ladder | THE SYMMETRY LADDER | COMPLETE-BOOKED — self-marked COMPLETE same day |
| 2026-07-29-minimal-crystal | THE MINIMAL CRYSTAL | COMPLETE-BOOKED — width floor d56, n=3 |
| 2026-07-29-attention-core | THE ATTENTION CORE | COMPLETE-BOOKED — core = count x geometry |
| 2026-07-29-slack-restoration | SLACK RESTORATION + anatomy kit | COMPLETE-BOOKED — anatomy.py shipped |
| 2026-07-30-escalation-engine | THE ESCALATION ENGINE | COMPLETE-BOOKED — policy 62/120 beats its dense tier |
| 2026-07-30-packed-crystal | THE PACKED CRYSTAL | COMPLETE-BOOKED — C-series through C6c |
| 2026-07-30-capacity-program | THE CAPACITY PROGRAM | COMPLETE-BOOKED — capacity meter becomes a dial |
| 2026-07-30-blackhole-moes | BLACK HOLE MoEs | COMPLETE-BOOKED — B0-B2 atlas + expert-size law |
| 2026-07-31-scaffold-program | The scaffold program | COMPLETE-BOOKED — CLOSED, transport NO; merge-free promoted |
| 2026-07-31-revival-sweep | Revival sweep | LIVE — folded into 2026-08-07-results-hardening.md |
| 2026-08-01-deterministic-birth | Deterministic Birth | LIVE — flagship, integer-closed full-block cross-lab |
| 2026-08-01-deterministic-gravmoe | Deterministic gravmoe pair | PARKED — design pass first |
| 2026-08-01-sol-answer-only-gate-design | Sol answer-only gate | PARKED — no experiment before the Sol ledger pre-reg |
| 2026-08-02-external-reader-presentation | External-reader presentation | PARKED — implementation pending |
| 2026-08-02-v4flash-lossless-recode | DeepSeek-V4-Flash exact re-coding (v3) | COMPLETE-BOOKED — F1 programme end to end; v3 claims struck in-file by RECEIPT V4-CENSUS |
| 2026-08-03-next-session | Next-session (08-03) | COMPLETE-BOOKED as session artifact |
| 2026-08-04-next-session | Post-R6 coalition program | PARTIAL — rung 1 executed; 2/3/4 unrun (self-marked) |
| 2026-08-05-recall-shoulder-and-queue | Post-branch-day queue | PARTIAL — item 1 ran (MOE-GT-6); items 2-5 unrun (self-marked) |
| 2026-08-05-llmopt-lab-extraction | llmopt.lab — instruments in the library | LIVE — verify_wave/oracle/traj done; gate, sink, timebox remain |
| 2026-08-05-tenet-battery | The TENET battery | COMPLETE-BOOKED — self-marked CLOSED 2026-08-06 |
| 2026-08-06-lab-traj-session | lab/traj (module 4) | COMPLETE-BOOKED — VERDICT LAB-TRAJ |
| 2026-08-06-3080-lockstep-window | The 3080 LOCKSTEP window | COMPLETE-BOOKED — A1/A2 PASS, device-free; C2 Mac-precision addendum queued |
| 2026-08-06-identity-battery | The IDENTITY battery (crest program) | LIVE — EX-ANAT arc + EX-FRESH booked; EX4-UNIF pre-registered 08-09; 30B arms on Artin GO |
| 2026-08-07-morning-specs | Three grounded pre-reg designs | PARTIAL [UNVERIFIED] — design 1 (EX-ANAT-3) fired; designs 2-3 unchecked |
| 2026-08-07-engine-scale | ENGINE-SCALE joint-scaling sweep | COMPLETE-BOOKED 08-09 — VERDICT ENGINE-SCALE-1: neither bar; the schedule is the binder |
| 2026-08-07-results-hardening | RESULTS HARDENING | LIVE — P0-P4 booked through 08-08; source of truth for the live track |

Non-spec artifacts here: 2026-08-07-engine-scale-cells.jsonl (30
pre-registered cells, RESULTS-cited) and engine_scale_cells/ (9 .bin
cell artifacts + manifest.jsonl, the ENGINE-SCALE-1 shipment).
