# Checkpoint execution plan (draft; executes nothing)

Rows: 278 files, 16.14 GB allocated. Named dynamic sites: 346; exhaustive name sets 258, default-bound (an override reaches other names) 16, unresolved 72. Generic loaders (open any name under checkpoints/ from an argument): 14: scratch/ex1_swap.py, scratch/ex2_build.py, scratch/ex3_build.py, scratch/ex5_manifest.py, scratch/ex5_traj_census.py, scratch/ex5_traj_rider.py, scratch/ex5_traj_rider2.py, scratch/gen9_45m_probes.sh, scratch/gt6_recall_ladder.py, scratch/gt7_coverage_rederive.py, scratch/gt7_draw.py, scratch/mathworld1_birth.py, scratch/mathworld1_liveness.py, scripts/gen_catalog.py.

Every action below waits for an explicit GO; this document executes nothing.

| reach | files | GB | meaning |
|---|---:|---:|---|
| unreached-with-generic-loaders | 156 | 11.39 | no named site; a generic loader could open it by argument |
| pattern | 88 | 3.60 | matched by an unresolved site, or by an overridable default |
| exhaustive | 30 | 1.14 | a site expands to exactly this name from literal-only bindings |
| default | 4 | 0.01 | reached only under default configuration (env or CLI default) |

| proposed action | files | GB |
|---|---:|---:|
| REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO | 120 | 10.42 |
| REVIEW: reached only by an unresolved consumer pattern; resolve or decide script | 20 | 2.00 |
| REVIEW: reached only by producer patterns that did not resolve | 66 | 1.59 |
| REVIEW: no reference found (CANDIDATE); per-family GO | 43 | 0.85 |
| REVIEW: exact consumer only in uncited script(s); decide script first | 8 | 0.61 |
| KEEP: exact consumer in results-cited script | 11 | 0.42 |
| REVIEW: produced by a known script, read by nothing exact; regenerable if the script and its inputs stand | 4 | 0.11 |
| REMOVE after confirming checkpoints/atomtraj1_smoke/stock_s11_probe_emit1/step_00300.pt is the cited copy (per-row GO) | 1 | 0.08 |
| REMOVE after confirming checkpoints/step_lora_grpo.pt.wsl is the cited copy (per-row GO) | 1 | 0.04 |
| RELINK scratch/scaffold_review.py, scratch/umoe_conserve.py to checkpoints/confirmed/scaffold-moe/umoe_gravmoe_s2.pt, then REMOVE (per-row GO) | 1 | 0.00 |
| RELINK scratch/cal_dilute.py to checkpoints/confirmed/calibration/cal_dilute_30_s1.pt, then REMOVE (per-row GO) | 1 | 0.00 |
| RELINK scratch/grav2_spacetime.py to checkpoints/confirmed/grav-spacetime/grav2_contract_s1.pt, then REMOVE (per-row GO) | 1 | 0.00 |
| RELINK scratch/grav2_spacetime.py to checkpoints/confirmed/grav-spacetime/grav2_ctl_s1.pt, then REMOVE (per-row GO) | 1 | 0.00 |

| GB | path | class | reach | consumers (exact or default) | producers (exact or default) | consumer patterns | copies | proposed |
|---:|---|---|---|---|---|---|---|---|
| 0.36 | checkpoints/metab_v3_fp32arm.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.22 | checkpoints/mathnative_gen6_ternary_grown_latent.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.22 | checkpoints/metabolic_hot_snap.pt | UNKNOWN | pattern | - | - | scratch/metabolic_hot.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.22 | checkpoints/metabolic_late_b95.pt | UNKNOWN | pattern | - | - | scratch/metabolic_hot.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.22 | checkpoints/metabolic_late_b95_snap.pt | UNKNOWN | pattern | - | - | scratch/metabolic_hot.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.22 | checkpoints/metabolic_late_b96.pt | UNKNOWN | pattern | - | - | scratch/metabolic_hot.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.22 | checkpoints/metabolic_late_b96_snap.pt | UNKNOWN | pattern | - | - | scratch/metabolic_hot.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.22 | checkpoints/metabolic_late_b97.pt | UNKNOWN | pattern | - | - | scratch/metabolic_hot.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.22 | checkpoints/metabolic_late_b97_snap.pt | UNKNOWN | pattern | - | - | scratch/metabolic_hot.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.22 | checkpoints/metabolic_late_snap.pt | UNKNOWN | pattern | - | - | scratch/metabolic_hot.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.22 | checkpoints/mathnative_gen6_ternary_grown.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.22 | checkpoints/metabolic_snap.pt | UNKNOWN | pattern | - | - | scratch/metabolic_hot.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.20 | checkpoints/mathnative_45m_gen6_reborn.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo2_c004.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo2_c006.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo2_c010.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo2_c012.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_c004.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_c006.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_c008.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_c010.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_c012.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_c014.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_c018.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_c020.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_c022.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_c024.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand002.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand004.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand006.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand008.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand010.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand012.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand014.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand016.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand018.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand020.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand022.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo3_cand024.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo_c002.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo_c006.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo_c010.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo_c012.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo_run1_c002.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo_run1_c004.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo_run1_c006.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_grpo_run1_c010.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_ternary_latent_3ep.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.20 | checkpoints/mathnative_45m_soup.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_mw1_active_final.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/smoke4_mathnative_19m_mw1_active_final.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.08 | checkpoints/smoke_mathnative_19m_mw1_active_final.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.08 | checkpoints/vmasm2_19m.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.08 | checkpoints/vmasm_19m.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.08 | checkpoints/dual_19m.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.08 | checkpoints/mathnative_19m_gen9A.pt | UNKNOWN | exhaustive | - | scratch/gen9_pipeline.sh [results-cited] | - | - | REVIEW: produced by a known script, read by nothing exact; regenerable if the script and its inputs stand |
| 0.08 | checkpoints/mathnative_19m_gen9B_mps_s2.pt | UNKNOWN | exhaustive | scratch/mps_sigma_gates.sh [UNCITED], scratch/night_gates.sh [UNCITED] | - | - | - | REVIEW: exact consumer only in uncited script(s); decide script first |
| 0.08 | checkpoints/phys_energy_19m.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.08 | checkpoints/cc_chains.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/cc_oneshot.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/gallery19m_atoms_s4.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/gallery19m_comp30_s2.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/gallery19m_ctrl3218_s3.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/gallery19m_curric_cap_s2.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/gallery19m_noheur_s3.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/gallery19m_softspeed_soft_s3.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/gallery19m_stock_s4.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/mathnative_19m_infixtwin_b1.pt | UNKNOWN | exhaustive | scratch/z1_gate.sh [UNCITED] | - | - | - | REVIEW: exact consumer only in uncited script(s); decide script first |
| 0.08 | checkpoints/mathnative_19m_infixtwin_gflip_m4.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_gflip_m4x.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_gflip_s4.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_gflip_tern.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_int3.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_lm16z.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_lm2.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_lm2z.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_lm3.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_lm3z.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_lmt.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_m4.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_m4x.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_nf4.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_p4.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_s4.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_sparse.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_tern.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_infixtwin_z1c.pt | UNKNOWN | exhaustive | scratch/z1_gate.sh [UNCITED] | - | - | - | REVIEW: exact consumer only in uncited script(s); decide script first |
| 0.08 | checkpoints/mathnative_19m_poly.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_poly2.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_prefix_int3.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_series.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_series1b.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_series1c.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_series1d.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_19m_series1e.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_v2_grpo_c002.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_v2_grpo_c004.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_v2_grpo_c006.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_v2_grpo_c008.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_v2_grpo_c010.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/mathnative_v2_grpo_c012.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.08 | checkpoints/port_control.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.08 | checkpoints/seedvar_1.pt | UNKNOWN | exhaustive | scratch/night2_mac_shift2.sh [UNCITED], scratch/night_calib.sh [UNCITED] | scratch/night_calib.sh [UNCITED] | - | - | REVIEW: exact consumer only in uncited script(s); decide script first |
| 0.08 | checkpoints/seedvar_2.pt | UNKNOWN | exhaustive | scratch/night2_mac_shift2.sh [UNCITED], scratch/night_calib.sh [UNCITED] | scratch/night_calib.sh [UNCITED] | - | - | REVIEW: exact consumer only in uncited script(s); decide script first |
| 0.08 | checkpoints/seedvar_3.pt | UNKNOWN | exhaustive | scratch/night2_mac_shift2.sh [UNCITED], scratch/night_calib.sh [UNCITED] | scratch/night_calib.sh [UNCITED] | - | - | REVIEW: exact consumer only in uncited script(s); decide script first |
| 0.08 | checkpoints/snap19m_q16.pt | UNKNOWN | exhaustive | scratch/run_snap_gates.sh [spec-cited] | - | scratch/distortion_collapse.py [results-cited] | - | KEEP: exact consumer in results-cited script |
| 0.08 | checkpoints/snap19m_q24.pt | UNKNOWN | exhaustive | scratch/run_snap_knee.sh [UNCITED] | - | scratch/distortion_collapse.py [results-cited] | - | REVIEW: exact consumer only in uncited script(s); decide script first |
| 0.08 | checkpoints/snap19m_q48.pt | UNKNOWN | exhaustive | scratch/run_snap_knee.sh [UNCITED] | - | scratch/distortion_collapse.py [results-cited] | - | REVIEW: exact consumer only in uncited script(s); decide script first |
| 0.08 | checkpoints/snap19m_q64.pt | UNKNOWN | exhaustive | scratch/run_snap_gates.sh [spec-cited] | - | scratch/distortion_collapse.py [results-cited] | - | KEEP: exact consumer in results-cited script |
| 0.08 | checkpoints/template_zero.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.08 | checkpoints/vrm_routed.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/vrm_uniform.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.08 | checkpoints/_attempt3_QUALIFIED_prefold_atomtraj1_smoke/stock_s11/step_01028.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.08 | checkpoints/_attempt3_QUALIFIED_prefold_atomtraj1_smoke/stock_s11/step_01100.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.08 | checkpoints/_attempt3_QUALIFIED_prefold_atomtraj1_smoke/stock_s11_probe_emit1/step_00300.pt | DUPLICATE | unreached-with-generic-loaders | - | - | - | - | REMOVE after confirming checkpoints/atomtraj1_smoke/stock_s11_probe_emit1/step_00300.pt is the cited copy (per-row GO) |
| 0.08 | checkpoints/fp64_birth_f32.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.04 | checkpoints/step_lora_grpo_c010.pt | DUPLICATE | unreached-with-generic-loaders | - | - | - | - | REMOVE after confirming checkpoints/step_lora_grpo.pt.wsl is the cited copy (per-row GO) |
| 0.03 | checkpoints/fmt_dechain_1p.pt | UNKNOWN | exhaustive | scratch/fmt_chain.sh [results-cited], scratch/fmt_pp_watcher.sh [results-cited] | - | - | - | KEEP: exact consumer in results-cited script |
| 0.03 | checkpoints/fmt_delta_1p.pt | UNKNOWN | exhaustive | scratch/fmt_chain.sh [results-cited], scratch/fmt_pp_watcher.sh [results-cited] | - | - | - | KEEP: exact consumer in results-cited script |
| 0.03 | checkpoints/fmt_delta_3e.pt | UNKNOWN | exhaustive | scratch/fmt_chain2.sh [results-cited] | - | - | - | KEEP: exact consumer in results-cited script |
| 0.03 | checkpoints/fmt_randpack_1p.pt | UNKNOWN | exhaustive | scratch/fmt_chain.sh [results-cited], scratch/fmt_pp_watcher.sh [results-cited] | - | - | - | KEEP: exact consumer in results-cited script |
| 0.03 | checkpoints/fmt_randpack_3e.pt | UNKNOWN | exhaustive | scratch/fmt_chain2.sh [results-cited] | - | - | - | KEEP: exact consumer in results-cited script |
| 0.03 | checkpoints/fmt_skip_1p.pt | UNKNOWN | exhaustive | scratch/fmt_chain.sh [results-cited], scratch/fmt_pp_watcher.sh [results-cited] | - | - | - | KEEP: exact consumer in results-cited script |
| 0.03 | checkpoints/fmt_traces_1p.pt | UNKNOWN | exhaustive | scratch/fmt_chain.sh [results-cited], scratch/fmt_pp_watcher.sh [results-cited] | - | - | - | KEEP: exact consumer in results-cited script |
| 0.03 | checkpoints/fmt_traces_3e.pt | UNKNOWN | exhaustive | scratch/fmt_chain2.sh [results-cited] | - | - | - | KEEP: exact consumer in results-cited script |
| 0.03 | checkpoints/scorer_s2_imit.pt | UNKNOWN | exhaustive | - | scratch/scorer_s2_train.py [spec-cited] | - | - | REVIEW: produced by a known script, read by nothing exact; regenerable if the script and its inputs stand |
| 0.03 | checkpoints/quat_convert_a.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.03 | checkpoints/rot_convert_a.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.03 | checkpoints/sym_birth_dense_a0sched.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.03 | checkpoints/sym_birth_dense_a0sched_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.03 | checkpoints/sym_birth_dense_d256ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.03 | checkpoints/sym_birth_dense_d256ema_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.03 | checkpoints/sym_circ8_a.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.03 | checkpoints/sym_z2_a.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.03 | checkpoints/sym_z2_b.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.02 | checkpoints/qwen05b_instruct_l14gate.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.01 | checkpoints/umoe_tree_s1.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.01 | checkpoints/umoe_lb_f224_s1.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.01 | checkpoints/k3_expert_l70_e512/w1.weight_packed.bin | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.01 | checkpoints/k3_expert_l70_e512/w2.weight_packed.bin | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.01 | checkpoints/k3_expert_l70_e512/w3.weight_packed.bin | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/umoe_tied_s1.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/umoe_tied_s2.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/umoe_channel_s1.pt | UNKNOWN | pattern | - | - | scratch/scaffold_review.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/umoe_channel_s2.pt | UNKNOWN | default | scratch/scaffold_review.py [results-cited] | - | - | - | KEEP: exact consumer in results-cited script |
| 0.00 | checkpoints/umoe_free_s1.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/umoe_free_s2.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/umoe_gravmoe_s2.pt | DUPLICATE | default | scratch/scaffold_review.py [results-cited] | - | - | - | RELINK scratch/scaffold_review.py, scratch/umoe_conserve.py to checkpoints/confirmed/scaffold-moe/umoe_gravmoe_s2.pt, then REMOVE (per-row GO) |
| 0.00 | checkpoints/umoe_gravmoe_s3.pt | UNKNOWN | pattern | - | - | scratch/scaffold_review.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/umoe_soft_s1.pt | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/sym_birth_dense_mps_L16.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_L16_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_L12.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_L12_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/fourier_g9_G9.pt | UNKNOWN | default | - | scratch/fourier_g9.py [results-cited] | - | - | REVIEW: produced by a known script, read by nothing exact; regenerable if the script and its inputs stand |
| 0.00 | checkpoints/fourier_g9_none.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/fourier3_algdiet_v1.pt | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/cal_dilute_30_s1.pt | DUPLICATE | unreached-with-generic-loaders | - | - | - | - | RELINK scratch/cal_dilute.py to checkpoints/confirmed/calibration/cal_dilute_30_s1.pt, then REMOVE (per-row GO) |
| 0.00 | checkpoints/grav2_contract_s1.pt | DUPLICATE | pattern | - | - | - | - | RELINK scratch/grav2_spacetime.py to checkpoints/confirmed/grav-spacetime/grav2_contract_s1.pt, then REMOVE (per-row GO) |
| 0.00 | checkpoints/grav2_contract_s2.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/grav2_contract_s3.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/grav2_contract_s4.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/grav2_ctl_s1.pt | DUPLICATE | pattern | - | - | - | - | RELINK scratch/grav2_spacetime.py to checkpoints/confirmed/grav-spacetime/grav2_ctl_s1.pt, then REMOVE (per-row GO) |
| 0.00 | checkpoints/grav2_ctl_s2.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/grav2_ctl_s3.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/grav2_ctl_s4.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/umoe_dense_s2.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense.pt | UNKNOWN | default | - | scratch/p3_ffnslack.py [results-cited] | - | - | REVIEW: produced by a known script, read by nothing exact; regenerable if the script and its inputs stand |
| 0.00 | checkpoints/sym_birth_dense_mps_h8.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_h8_s2.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_h8_s2_ema.pt | UNKNOWN | pattern | - | - | scratch/p3_autopsy.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/sym_birth_dense_mps_h8_s3.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_h8_s3_ema.pt | UNKNOWN | pattern | - | - | scratch/p3_autopsy.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/sym_birth_dense_mps_h8_s4.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_h8_s4_ema.pt | UNKNOWN | pattern | - | - | scratch/p3_autopsy.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/sym_birth_dense_revdepth_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_revfarm.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_revpairs.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_revpairs_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_s2_h8.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_s2_h8_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_s3_h8.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_s3_h8_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_L4d80.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_L4d80_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_h2.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_h2_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_L8.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_L8_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_w56.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_r32.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_r32_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_w48.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_w48_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_mps_L4.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_s2_L4.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_s2_L4_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_s3_L4.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/sym_birth_dense_s3_L4_ema.pt | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/k3_expert_l70_e512/w1.weight_scale.bin | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/k3_expert_l70_e512/w2.weight_scale.bin | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/k3_expert_l70_e512/w3.weight_scale.bin | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/packed_s3_h8.npz | UNKNOWN | pattern | - | - | scratch/pack_rans.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/packed_d64h8.npz | UNKNOWN | pattern | - | - | scratch/pack_rans.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/packed_s2_h8.npz | UNKNOWN | pattern | - | - | scratch/pack_rans.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/packed_L4d56.npz | UNKNOWN | pattern | - | - | scratch/pack_rans.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/packed_s2_L4.npz | UNKNOWN | pattern | - | - | scratch/pack_rans.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/packed_s3_L4.npz | UNKNOWN | pattern | - | - | scratch/pack_rans.py [results-cited] | - | REVIEW: reached only by an unresolved consumer pattern; resolve or decide script |
| 0.00 | checkpoints/gt2_code_ladder_s99.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex5_del_layer0.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex5_del_layer1.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex5_del_layer2.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex5_del_rank0.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex5_del_rank1.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex5_del_rank2.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/gt7_anom_r80_0.json | CANDIDATE | exhaustive | scratch/gt7_run.py [results-cited] | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/gt7_anom_r80_1.json | CANDIDATE | exhaustive | scratch/gt7_run.py [results-cited] | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1_c15_los.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex1_c30_los.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex1_c45_his.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex1_c45_los.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex1b_k1_c15_his.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k1_c15_los.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k1_c30_his.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k1_c30_los.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k1_c45_his.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k1_c45_los.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k2_c15_his.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k2_c15_los.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k2_c30_his.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k2_c30_los.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k2_c45_his.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex1b_k2_c45_los.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/ex2_c15_hi_add.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c15_hi_addc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c15_hi_rem.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c15_hi_remc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c15_lo_add.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c15_lo_addc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c15_lo_rem.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c15_lo_remc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c30_hi_add.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c30_hi_addc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c30_hi_rem.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c30_hi_remc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c30_lo_add.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c30_lo_addc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c30_lo_rem.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c30_lo_remc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c45_hi_add.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c45_hi_addc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c45_hi_rem.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c45_hi_remc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c45_lo_add.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c45_lo_addc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c45_lo_rem.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/ex2_c45_lo_remc.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/gt5c_keep_r0.json | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/gt5c_keep_r1.json | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/gt5c_keep_r2.json | UNKNOWN | pattern | - | - | - | - | REVIEW: reached only by producer patterns that did not resolve |
| 0.00 | checkpoints/gt6_ladder_r70_d0.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/gt6_ladder_r70_d1.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/gt7_anom_r75_0.json | CANDIDATE | exhaustive | scratch/gt7_run.py [results-cited] | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/gt7_anom_r75_1.json | CANDIDATE | exhaustive | scratch/gt7_run.py [results-cited] | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/gt7_ladder_c15_d0.json | CANDIDATE | exhaustive | scratch/gt7_run.py [results-cited] | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/gt7_ladder_c15_d1.json | CANDIDATE | exhaustive | scratch/gt7_run.py [results-cited] | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/gt7_ladder_c30_d0.json | CANDIDATE | exhaustive | scratch/gt7_run.py [results-cited] | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/gt7_ladder_c30_d1.json | CANDIDATE | exhaustive | scratch/gt7_run.py [results-cited] | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/gt7_ladder_c45_d0.json | CANDIDATE | exhaustive | scratch/gt7_run.py [results-cited] | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/gt7_ladder_c45_d1.json | CANDIDATE | exhaustive | scratch/gt7_run.py [results-cited] | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/gt6_ladder_r60_d0.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/gt6_ladder_r60_d1.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/gt6_ladder_r65_d0.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/gt6_ladder_r65_d1.json | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/gt3_smoke_prompts.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/k3_expert_l70_e512/meta.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
| 0.00 | checkpoints/mathnative_gen6_ternary_grown_latent.pt.ep.bak | UNKNOWN | unreached-with-generic-loaders | - | - | - | - | REVIEW: no named site reaches this file after pattern resolution; 14 generic loader(s) could open it by argument; CANDIDATE queue; per-family GO |
| 0.00 | checkpoints/smoke_mathnative_19m_mw1_theta0.json | CANDIDATE | unreached-with-generic-loaders | - | - | - | - | REVIEW: no reference found (CANDIDATE); per-family GO |
