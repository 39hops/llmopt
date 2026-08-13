# CODEMAP — the move-gate inventory (generated, do not hand-edit)

Regenerate: `.venv/bin/python scripts/gen_codemap.py`. One row per
top-level file in scratch/ and scripts/. Class ladder (mechanical):
library > reproduce-pinned > results-cited > spec-cited > UNCITED.
House law: cited files are the evidence record — extraction means
adoption-with-reverification, never a silent move. `imports`
counts code files with a real `import`/`from` statement on the
module (drives `library`); `mentions` counts files that only
embed its literal filename (path strings, shell invocations —
catches couplings like llmopt/reproduce.py → detbwd_gravmoe, but
does not by itself make a file `library`).

Census: UNCITED 202, library 62, reproduce-pinned 10, results-cited 167, spec-cited 85, cited-but-library 51

## scratch/

| family | file | class | cited by | doc citations | imports | mentions |
|---|---|---|---|---|---|---|
| absorb | absorb_1e5.py | UNCITED | — | — | — | — |
| adjudicate | adjudicate_zx.py | library | — | — | 1 | — |
| anatomy | anatomy.py | results-cited | RESULTS, specs | RESULTS×5, specs×7 | — | 3 |
| assets | assets_classify.py | spec-cited | specs | specs×1 | — | — |
| atlas | atlas_precompute.py | UNCITED | — | — | — | 1 |
| attractor | attractor_census.py | results-cited | RESULTS | RESULTS×1 | — | 1 |
| attractor | attractor_census2.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — |
| b768 | b768_after_v5.sh | UNCITED | — | — | — | — |
| basin | basin_probe.py | UNCITED | — | — | — | — |
| birth19m | birth19m_snaps.py | UNCITED | — | — | — | — |
| blackhole | blackhole_b0.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — |
| boundary | boundary_or_bulk.py | results-cited | RESULTS | RESULTS×3 | — | — |
| brute | brute_arms_0801.sh | UNCITED | — | — | — | — |
| brute | brute_b_arms_0801.sh | UNCITED | — | — | — | — |
| brute | brute_c_arm_0801.sh | UNCITED | — | — | — | — |
| build | build_dist_diets.py | UNCITED | — | — | — | — |
| build | build_merged_diet.py | UNCITED | — | — | — | — |
| cal | cal_dilute.py | UNCITED | — | — | — | — |
| cal | cal_dk_probe.py | UNCITED | — | — | — | 1 |
| calib | calib_dist_birth.sh | spec-cited | specs | specs×4 | — | 1 |
| calib | calib_probe.py | library | RESULTS, specs | RESULTS×1, specs×11 | 1 | 2 |
| calib | calib_snap_gates.sh | spec-cited | specs | specs×5 | — | — |
| callspan | callspan_arms.py | UNCITED | — | — | — | — |
| capacity | capacity_meter.py | library | RESULTS, specs | RESULTS×1, specs×2 | 3 | 1 |
| ce | ce_gate_study.py | spec-cited | specs | specs×2 | — | — |
| ce400 | ce400.py | UNCITED | — | — | — | 2 |
| ceiling | ceiling_probe_cuda.py | UNCITED | — | — | — | — |
| chain | chain_carry.py | UNCITED | — | — | — | 1 |
| champ | champ_cuda_probe.py | UNCITED | — | — | — | — |
| churn | churn_judge_eval.py | reproduce-pinned | REPRODUCE | REPRODUCE×2 | — | — |
| ckpt | ckpt_delete_pass.py | UNCITED | — | — | — | — |
| ckpt | ckpt_inventory.py | spec-cited | specs | specs×2 | — | 1 |
| ckpt | ckpt_triage_table.py | UNCITED | — | — | — | — |
| clade | clade_stream_d256.py | results-cited | RESULTS | RESULTS×1 | — | — |
| closers | closers_chain.sh | UNCITED | — | — | — | — |
| complex | complex_birth.py | UNCITED | — | — | — | 8 |
| complex | complex_model.py | library | specs | specs×1 | 6 | 1 |
| complex | complex_nnue.py | UNCITED | — | — | — | — |
| complexify | complexify_control.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| confluence | confluence.py | UNCITED | — | — | — | — |
| corner | corner_snap.py | UNCITED | — | — | — | — |
| cplx | cplx_chain.sh | UNCITED | — | — | — | — |
| crystal | crystal_recreate_test.py | spec-cited | specs | specs×3 | — | 1 |
| d2 | d2_verify.py | UNCITED | — | — | — | 1 |
| day | day_chain.sh | UNCITED | — | — | — | — |
| desert | desert_v2.py | spec-cited | specs | specs×1 | — | — |
| detbwd | detbwd_diet.py | library | RESULTS, specs | RESULTS×6, specs×1 | 2 | — |
| detbwd | detbwd_gravmoe.py | library | RESULTS, specs | RESULTS×2, specs×27 | 1 | 8 |
| detbwd | detbwd_mb.py | library | RESULTS, specs | RESULTS×4, specs×7 | 4 | — |
| detbwd | detbwd_plateau.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| detbwd | detbwd_r1.py | library | RESULTS | RESULTS×3 | 9 | — |
| detbwd | detbwd_r1b.py | library | RESULTS | RESULTS×1 | 1 | — |
| detbwd | detbwd_r2_adamw.py | library | RESULTS | RESULTS×4 | 2 | — |
| detbwd | detbwd_r2b.py | library | RESULTS, specs | RESULTS×2, specs×2 | 6 | — |
| detbwd | detbwd_r3_qw.py | library | RESULTS | RESULTS×1 | 5 | — |
| determinability | determinability_census.py | results-cited | RESULTS | RESULTS×1 | — | — |
| distortion | distortion_collapse.py | results-cited | RESULTS | RESULTS×2 | — | — |
| dual | dual_probe.py | UNCITED | — | — | — | — |
| duo | duo_mine.py | UNCITED | — | — | — | — |
| duo | duo_wave.py | spec-cited | specs | specs×1 | — | — |
| e2 | e2_logit_check.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| e3 | e3_battery.py | spec-cited | specs | specs×1 | — | — |
| emission | emission_wall_pair.py | UNCITED | — | — | — | — |
| engine | engine_scale_export.py | spec-cited | specs | specs×1 | — | — |
| ex1 | ex1_swap.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | 1 |
| ex2 | ex2_build.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| ex3 | ex3_build.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | 2 |
| exact | exact_twin_d56.py | spec-cited | specs | specs×1 | — | — |
| exact1 | exact1_small_cells.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — |
| exchange | exchange_test.py | results-cited | RESULTS | RESULTS×1 | — | — |
| export | export_axnn.py | results-cited | RESULTS | RESULTS×1 | — | — |
| export | export_mb_ref.py | UNCITED | — | — | — | — |
| export | export_r2b_ref.py | results-cited | RESULTS | RESULTS×1 | — | — |
| farm | farm_dist_rows.py | results-cited | RESULTS, specs | RESULTS×1, specs×4 | — | 1 |
| farmer | farmer_probe.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — |
| fig | fig_magic_scatter.py | spec-cited | specs | specs×1 | — | — |
| fixed | fixed_q_snap.py | UNCITED | — | — | — | 1 |
| floor | floor_hk1.sh | UNCITED | — | — | — | 1 |
| floor | floor_hk1_d256.sh | UNCITED | — | — | — | — |
| fmt | fmt_chain.sh | UNCITED | — | — | — | — |
| fmt | fmt_chain2.sh | UNCITED | — | — | — | — |
| fmt | fmt_pp_watcher.sh | UNCITED | — | — | — | — |
| format | format_delta_prep.py | UNCITED | — | — | — | 2 |
| format | format_ladder.py | UNCITED | — | — | — | 3 |
| fourier | fourier_g9.py | results-cited | RESULTS | RESULTS×1 | — | — |
| fourier | fourier_probe.py | UNCITED | — | — | — | — |
| fourier2 | fourier2_modbirth.py | UNCITED | — | — | — | — |
| fourier2b | fourier2b_widemod.py | library | RESULTS | RESULTS×1 | 2 | — |
| fourier3 | fourier3_algdiet.py | results-cited | RESULTS | RESULTS×1 | — | — |
| fourier4a | fourier4a_dynamics.py | results-cited | RESULTS | RESULTS×1 | — | — |
| fp64 | fp64_paired.py | spec-cited | specs | specs×1 | — | — |
| fx3 | fx3_house.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — |
| g19 | g19_bf16_isolation.sh | spec-cited | specs | specs×2 | — | 1 |
| g19 | g19_fp32_cell.sh | UNCITED | — | — | — | — |
| g19 | g19_probes_fix.sh | UNCITED | — | — | — | — |
| g19 | g19_sigma_cuda.sh | UNCITED | — | — | — | — |
| g5 | g5_polar.py | UNCITED | — | — | — | — |
| gate | gate_batched.py | UNCITED | — | — | — | 3 |
| gate | gate_ckpt.py | results-cited | RESULTS, specs | RESULTS×3, specs×12 | — | 38 |
| gate | gate_ckpt_cuda.py | UNCITED | — | — | — | 12 |
| gate | gate_cplx.py | spec-cited | specs | specs×1 | — | 1 |
| gate | gate_pp.py | UNCITED | — | — | — | 1 |
| gate | gate_prefix.py | results-cited | RESULTS | RESULTS×1 | — | 1 |
| gate | gate_rarity.py | results-cited | RESULTS | RESULTS×1 | — | 10 |
| gate | gate_regate.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| gate | gate_transcripts.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| gate | gate_v2_bench.sh | UNCITED | — | — | — | — |
| gate | gate_zx.py | UNCITED | — | — | — | 10 |
| gatepins | gatepins_freeze.py | spec-cited | specs | specs×5 | — | 1 |
| gauge | gauge_distance_d256.py | UNCITED | — | — | — | — |
| gauge | gauge_m4x.py | UNCITED | — | — | — | — |
| gauge | gauge_slack_rat.py | results-cited | RESULTS | RESULTS×1 | — | — |
| gen | gen_lab_overview_pdf.py | UNCITED | — | — | — | 1 |
| gen | gen_lean_corpus.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| gen8 | gen8_pipeline.sh | spec-cited | specs | specs×2 | — | 1 |
| gen9 | gen9_19m_cuda_control.sh | UNCITED | — | — | — | — |
| gen9 | gen9_45m_fp32_control.sh | UNCITED | — | — | — | — |
| gen9 | gen9_45m_probes.sh | UNCITED | — | — | — | — |
| gen9 | gen9_pipeline.sh | spec-cited | specs | specs×2 | — | 1 |
| genpins | genpins_freeze.py | UNCITED | — | — | — | 1 |
| graph | graph_mod_sigma.py | results-cited | RESULTS | RESULTS×1 | — | — |
| graph | graph_modularity_gen8.py | library | — | — | 1 | — |
| grav | grav_posthoc.py | UNCITED | — | — | — | — |
| grav | grav_probe.py | library | RESULTS | RESULTS×1 | 2 | — |
| grav1b | grav1b_distance.py | UNCITED | — | — | — | — |
| grav2 | grav2_spacetime.py | library | — | — | 1 | — |
| greedy | greedy_first_gate.py | UNCITED | — | — | — | 1 |
| grow | grow_decomp1.sh | UNCITED | — | — | — | — |
| grpo | grpo_shaped.py | UNCITED | — | — | — | 1 |
| gt2 | gt2_code_arm0.py | reproduce-pinned | REPRODUCE, RESULTS, specs | REPRODUCE×1, RESULTS×1, specs×2 | — | — |
| gt2 | gt2_jaccard.py | library | REPRODUCE, RESULTS, specs | REPRODUCE×1, RESULTS×3, specs×15 | 8 | 4 |
| gt3 | gt3_probe_arm0.py | reproduce-pinned | REPRODUCE, RESULTS | REPRODUCE×1, RESULTS×2 | — | — |
| gt4 | gt4_dialog_prompts.py | reproduce-pinned | REPRODUCE, RESULTS | REPRODUCE×1, RESULTS×2 | — | — |
| gt4 | gt4_verbal_core.py | reproduce-pinned | REPRODUCE, RESULTS | REPRODUCE×1, RESULTS×3 | — | — |
| gt5 | gt5_union_keep.py | reproduce-pinned | REPRODUCE, RESULTS, specs | REPRODUCE×1, RESULTS×1, specs×1 | — | — |
| gt5c | gt5c_randfill_keep.py | reproduce-pinned | REPRODUCE, RESULTS | REPRODUCE×1, RESULTS×1 | — | — |
| gt6 | gt6_recall_ladder.py | reproduce-pinned | REPRODUCE, RESULTS | REPRODUCE×1, RESULTS×1 | — | — |
| gt6 | gt6_resume_arms.sh | UNCITED | — | — | — | — |
| gt7 | gt7_coverage_rederive.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | 1 |
| gt7 | gt7_draw.py | spec-cited | specs | specs×3 | — | 1 |
| gt7 | gt7_run.py | results-cited | RESULTS, specs | RESULTS×7, specs×2 | — | — |
| head | head_autopsy.py | library | — | — | 1 | — |
| head | head_census.py | spec-cited | specs | specs×2 | — | 1 |
| holdout | holdout_gate.py | spec-cited | specs | specs×1 | — | 1 |
| holdout | holdout_v2.py | UNCITED | — | — | — | 2 |
| hot | hot_chain.sh | results-cited | RESULTS | RESULTS×2 | — | 1 |
| int2 | int2_regate.sh | UNCITED | — | — | — | — |
| int3 | int3_rider.py | UNCITED | — | — | — | — |
| jointperm | jointperm_distance.py | UNCITED | — | — | — | — |
| judge | judge_decode.py | spec-cited | specs | specs×3 | — | — |
| k3 | k3_expert_demo.py | results-cited | RESULTS, specs | RESULTS×4, specs×3 | — | 4 |
| keff | keff_probe.py | results-cited | RESULTS | RESULTS×1 | — | — |
| kv | kv_after_night.sh | UNCITED | — | — | — | — |
| kv | kv_equiv.py | UNCITED | — | — | — | 2 |
| l9 | l9_probe.py | UNCITED | — | — | — | 3 |
| lam | lam_merge_review.py | UNCITED | — | — | — | 1 |
| lean | lean_check.py | reproduce-pinned | REPRODUCE, RESULTS, specs | REPRODUCE×1, RESULTS×7, specs×3 | — | — |
| lean | lean_sample_build.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| legacy | legacy_diet_audit.py | results-cited | RESULTS | RESULTS×1 | — | — |
| lloydmax | lloydmax_race.py | library | — | — | 1 | — |
| loss | loss_floor_census.py | results-cited | RESULTS | RESULTS×2 | — | — |
| lyap | lyap_compare.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 2 |
| lyapunov | lyapunov_birth.sh | UNCITED | — | — | — | — |
| mac | mac_day_chain.sh | UNCITED | — | — | — | — |
| make | make_altpairs.py | results-cited | RESULTS | RESULTS×1 | — | 1 |
| make | make_union_diet.py | UNCITED | — | — | — | 1 |
| margin | margin_by_level.py | results-cited | RESULTS | RESULTS×1 | — | — |
| margin | margin_by_ply.py | results-cited | RESULTS | RESULTS×1 | — | — |
| margin | margin_census.py | UNCITED | — | — | — | — |
| margin | margin_vs_branching.py | results-cited | RESULTS | RESULTS×1 | — | — |
| mass | mass_on_valid.py | spec-cited | specs | specs×4 | — | — |
| matryoshka | matryoshka_r1.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — |
| matryoshka | matryoshka_r2.py | spec-cited | specs | specs×1 | — | — |
| merge | merge_space1.sh | UNCITED | — | — | — | — |
| merge | merge_space2.sh | UNCITED | — | — | — | — |
| merge | merge_space3.sh | UNCITED | — | — | — | — |
| merge | merge_space4.sh | UNCITED | — | — | — | — |
| merge | merge_space5.sh | UNCITED | — | — | — | — |
| metabolic | metabolic_d2.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| metabolic | metabolic_hot.py | results-cited | RESULTS | RESULTS×1 | — | 1 |
| metabolic | metabolic_v3.py | spec-cited | specs | specs×1 | — | — |
| metabolic | metabolic_v4.py | UNCITED | — | — | — | — |
| metabolic | metabolic_v5.py | UNCITED | — | — | — | 2 |
| metallicity | metallicity_diets.py | results-cited | RESULTS | RESULTS×2 | — | 1 |
| metallicity1 | metallicity1.sh | results-cited | RESULTS | RESULTS×2 | — | — |
| moe | moe_gt1.py | library | REPRODUCE, RESULTS, specs | REPRODUCE×4, RESULTS×4, specs×4 | 2 | 3 |
| moe | moe_gt1_arm2.py | library | REPRODUCE, RESULTS, specs | REPRODUCE×6, RESULTS×6, specs×4 | 1 | 4 |
| morning | morning_run.sh | UNCITED | — | — | — | — |
| mps | mps_sigma_gates.sh | UNCITED | — | — | — | — |
| muon | muon_3ep_d256.py | UNCITED | — | — | — | 1 |
| night | night_28.sh | results-cited | RESULTS | RESULTS×1 | — | — |
| night | night_28_mac.sh | results-cited | RESULTS | RESULTS×1 | — | — |
| night | night_45m_union.sh | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | 2 |
| night | night_calib.sh | UNCITED | — | — | — | — |
| night | night_g9.sh | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| night | night_gates.sh | UNCITED | — | — | — | — |
| night | night_rat.sh | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| night | night_rat_s2.sh | spec-cited | specs | specs×1 | — | — |
| night | night_sr.sh | results-cited | RESULTS | RESULTS×1 | — | — |
| night | night_zx.sh | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — |
| night | night_zx2.sh | results-cited | RESULTS | RESULTS×1 | — | 1 |
| night | night_zx3.sh | results-cited | RESULTS | RESULTS×1 | — | — |
| night | night_zx45_x2.sh | results-cited | RESULTS | RESULTS×1 | — | 1 |
| night2 | night2_mac.sh | results-cited | RESULTS | RESULTS×1 | — | 1 |
| night2 | night2_mac_shift2.sh | UNCITED | — | — | — | — |
| night28b | night28b.sh | UNCITED | — | — | — | — |
| night29 | night29.sh | spec-cited | specs | specs×1 | — | — |
| night29b | night29b.sh | spec-cited | specs | specs×1 | — | — |
| night30 | night30.sh | spec-cited | specs | specs×1 | — | — |
| night30 | night30_mac.py | UNCITED | — | — | — | — |
| night30b | night30b.sh | UNCITED | — | — | — | — |
| night31 | night31.sh | results-cited | RESULTS | RESULTS×1 | — | — |
| night31 | night31_cuda.sh | results-cited | RESULTS | RESULTS×1 | — | — |
| night31 | night31_mac.sh | UNCITED | — | — | — | — |
| night31b | night31b_cuda.sh | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| nineteen | nineteen_m_displace.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — |
| oracle | oracle_worker.py | reproduce-pinned | REPRODUCE, RESULTS, specs | REPRODUCE×1, RESULTS×2, specs×1 | — | 6 |
| ozaki | ozaki_2b_bisect.py | UNCITED | — | — | — | — |
| ozaki | ozaki_2b_check.py | results-cited | RESULTS | RESULTS×1 | — | — |
| ozaki | ozaki_2b_debug.py | UNCITED | — | — | — | — |
| ozaki | ozaki_2b_ident.py | UNCITED | — | — | — | — |
| ozaki | ozaki_cuda.py | UNCITED | — | — | — | — |
| ozaki | ozaki_cuda2.py | results-cited | RESULTS | RESULTS×1 | — | — |
| ozaki | ozaki_cuda3.py | results-cited | RESULTS | RESULTS×1 | — | — |
| ozaki | ozaki_cuda4.py | results-cited | RESULTS | RESULTS×1 | — | — |
| ozaki | ozaki_cuda5.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| ozaki | ozaki_cuda6.py | results-cited | RESULTS | RESULTS×1 | — | — |
| ozaki | ozaki_fused.py | results-cited | RESULTS | RESULTS×2 | — | — |
| ozaki | ozaki_rung1.py | results-cited | RESULTS | RESULTS×1 | — | — |
| ozaki | ozaki_rung1b.py | results-cited | RESULTS | RESULTS×2 | — | — |
| ozaki | ozaki_rung2bc.py | results-cited | RESULTS | RESULTS×1 | — | 4 |
| p2 | p2_crown_draws.py | UNCITED | — | — | — | — |
| p3 | p3_autopsy.py | results-cited | RESULTS | RESULTS×1 | — | — |
| p3 | p3_bits.py | results-cited | RESULTS | RESULTS×2 | — | — |
| p3 | p3_ffnslack.py | results-cited | RESULTS | RESULTS×3 | — | — |
| p3 | p3_grav2.py | results-cited | RESULTS | RESULTS×2 | — | — |
| p3 | p3_quat.py | results-cited | RESULTS | RESULTS×1 | — | — |
| p3 | p3_stream2x2.py | results-cited | RESULTS | RESULTS×2 | — | — |
| p3 | p3_umoe_soft.py | results-cited | RESULTS | RESULTS×1 | — | — |
| p4 | p4_arms_0801.sh | results-cited | RESULTS, specs | RESULTS×3, specs×11 | — | 2 |
| pack | pack_baselines.py | spec-cited | specs | specs×1 | — | — |
| pack | pack_c6.py | library | RESULTS, specs | RESULTS×1, specs×1 | 4 | — |
| pack | pack_c7.py | results-cited | RESULTS | RESULTS×1 | — | — |
| pack | pack_crystal.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — |
| pack | pack_decode.py | library | RESULTS | RESULTS×4 | 1 | 1 |
| pack | pack_determinism.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| pack | pack_gemv.py | spec-cited | specs | specs×1 | — | — |
| pack | pack_p2a.py | UNCITED | — | — | — | — |
| pack | pack_rans.py | results-cited | RESULTS, specs | RESULTS×3, specs×2 | — | — |
| pack | pack_tiered.py | spec-cited | specs | specs×1 | — | — |
| paper | paper_figs.py | UNCITED | — | — | — | — |
| phase4 | phase4_rewrite.py | spec-cited | specs | specs×1 | — | — |
| phase4 | phase4_sites.py | spec-cited | specs | specs×3 | — | 1 |
| phase4 | phase4_unboot.py | UNCITED | — | — | — | — |
| phase5 | phase5_deadcode.py | spec-cited | specs | specs×1 | — | — |
| phys | phys_probe.py | UNCITED | — | — | — | — |
| pincer | pincer_dist_probe.py | results-cited | RESULTS | RESULTS×1 | — | — |
| pincer | pincer_dist_report.py | UNCITED | — | — | — | 1 |
| pincer | pincer_labels_v2.py | library | RESULTS | RESULTS×1 | 1 | — |
| pincer | pincer_r0.py | results-cited | RESULTS | RESULTS×2 | — | — |
| pincer | pincer_r0b.py | results-cited | RESULTS | RESULTS×1 | — | — |
| pincer | pincer_r1_indist.py | results-cited | RESULTS | RESULTS×1 | — | — |
| pincer | pincer_r1_probe.py | results-cited | RESULTS | RESULTS×1 | — | 1 |
| pincer | pincer_r1b_labels.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 2 |
| pincer | pincer_r8.py | results-cited | RESULTS | RESULTS×1 | — | — |
| place1 | place1_gravity.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — |
| polar | polar_snap.py | results-cited | RESULTS | RESULTS×1 | — | — |
| poly3 | poly3_pipeline.sh | spec-cited | specs | specs×2 | — | 1 |
| poly4 | poly4_pipeline.sh | spec-cited | specs | specs×1 | — | 2 |
| poly4 | poly4_watcher.sh | UNCITED | — | — | — | — |
| poly5 | poly5_pipeline.sh | spec-cited | specs | specs×1 | — | 2 |
| poly5 | poly5_watcher.sh | UNCITED | — | — | — | — |
| practice | practice_mine.py | spec-cited | specs | specs×1 | — | — |
| prefix | prefix_pair.sh | UNCITED | — | — | — | — |
| probe | probe_int_device_parity.py | results-cited | RESULTS | RESULTS×1 | — | — |
| prologue | prologue_arms.py | library | — | — | 1 | — |
| prologue | prologue_gates.sh | UNCITED | — | — | — | — |
| ptq4 | ptq4_arms.py | UNCITED | — | — | — | — |
| ptq4 | ptq4_gates.sh | UNCITED | — | — | — | 1 |
| quat | quat_commutant.py | library | RESULTS, specs | RESULTS×1, specs×1 | 1 | — |
| quat | quat_convert.py | library | RESULTS, specs | RESULTS×1, specs×1 | 1 | — |
| quick | quick_exact_3080.sh | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| qwen | qwen_displace_extract.py | UNCITED | — | — | — | 1 |
| rank | rank_read.py | spec-cited | specs | specs×2 | — | 1 |
| rat | rat_deploy.py | results-cited | RESULTS, specs | RESULTS×1, specs×5 | — | 3 |
| rat | rat_repair.py | UNCITED | — | — | — | 1 |
| rational | rational_snap.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 6 |
| retention | retention_watcher.sh | UNCITED | — | — | — | — |
| rev2 | rev2_d768.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| rev3 | rev3_crown.py | results-cited | RESULTS, specs | RESULTS×3, specs×3 | — | — |
| rev4 | rev4_zx45.py | results-cited | RESULTS | RESULTS×1 | — | 1 |
| rot | rot_commutant.py | library | RESULTS, specs | RESULTS×1, specs×1 | 3 | — |
| rot | rot_convert.py | spec-cited | specs | specs×1 | — | — |
| rot | rot_snap_anatomy.py | UNCITED | — | — | — | — |
| rotinstr | rotinstr_control.py | results-cited | RESULTS | RESULTS×1 | — | — |
| run | run_snap_gates.sh | spec-cited | specs | specs×1 | — | — |
| run | run_snap_knee.sh | UNCITED | — | — | — | — |
| saturation | saturation_s2.py | results-cited | RESULTS | RESULTS×2 | — | — |
| saturation | saturation_s2b.py | spec-cited | specs | specs×1 | — | — |
| scaffold | scaffold_review.py | results-cited | RESULTS | RESULTS×1 | — | — |
| scorer | scorer_s1_battery.py | results-cited | RESULTS | RESULTS×1 | — | — |
| scorer | scorer_s2_data.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| scorer | scorer_s2_train.py | spec-cited | specs | specs×1 | — | — |
| seed | seed_audit.py | spec-cited | specs | specs×2 | — | — |
| seeds | seeds_ladder_0804.sh | UNCITED | — | — | — | — |
| series | series_probe.py | spec-cited | specs | specs×5 | — | 15 |
| snap | snap_alloc.py | spec-cited | specs | specs×2 | — | 1 |
| snap | snap_anatomy.py | results-cited | RESULTS | RESULTS×1 | — | — |
| softprompt | softprompt_sampler_probe.py | spec-cited | specs | specs×1 | — | — |
| softprompt1 | softprompt1.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | 3 |
| softprompt1 | softprompt1.sh | results-cited | RESULTS | RESULTS×1 | — | — |
| soup | soup_gate.py | spec-cited | specs | specs×1 | — | 1 |
| ssm | ssm_star.py | library | RESULTS | RESULTS×1 | 1 | 1 |
| ssm | ssm_star1.sh | UNCITED | — | — | — | — |
| stability | stability_atlas.sh | UNCITED | — | — | — | — |
| star | star_profile.py | results-cited | RESULTS | RESULTS×3 | — | — |
| streaming | streaming_birth_d256.py | library | RESULTS | RESULTS×3 | 1 | — |
| successors | successors_acceptance.py | results-cited | RESULTS | RESULTS×1 | — | — |
| sym | sym_birth.py | library | RESULTS, specs | RESULTS×1, specs×3 | 1 | 8 |
| sym | sym_convert.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — |
| sym | sym_spectrum.py | spec-cited | specs | specs×2 | — | — |
| sym45 | sym45.py | spec-cited | specs | specs×1 | — | 1 |
| sym45 | sym45_run.sh | UNCITED | — | — | — | — |
| synonym | synonym_test.py | UNCITED | — | — | — | 1 |
| tenet | tenet_d1_revgate.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | 1 |
| tenet | tenet_d2_revdiet.py | library | RESULTS, specs | RESULTS×1, specs×1 | 15 | — |
| tenet | tenet_d3_budget.py | library | specs | specs×1 | 2 | — |
| tenet | tenet_mult_b32.py | results-cited | RESULTS | RESULTS×1 | — | — |
| tenet | tenet_mult_census.py | library | RESULTS, specs | RESULTS×1, specs×1 | 1 | — |
| tenet | tenet_r1b_micro.py | UNCITED | — | — | — | — |
| tenet | tenet_w0.py | results-cited | RESULTS | RESULTS×1 | — | — |
| tenet | tenet_w1_bridge.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 1 |
| tenet | tenet_w1_population.py | UNCITED | — | — | — | — |
| tenet | tenet_w1_relational.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| tenet | tenet_w1_surfaces.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| ternary | ternary_control.py | UNCITED | — | — | — | — |
| ternary | ternary_gate.py | UNCITED | — | — | — | — |
| ternary | ternary_session2.py | UNCITED | — | — | — | 1 |
| tier | tier_escalate.py | UNCITED | — | — | — | — |
| tier | tier_retry.py | spec-cited | specs | specs×1 | — | — |
| train | train_fp64.py | UNCITED | — | — | — | — |
| traj | traj_accept.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| tuesday | tuesday_night.sh | UNCITED | — | — | — | — |
| umoe | umoe_conserve.py | library | RESULTS | RESULTS×2 | 9 | 3 |
| v4flash | v4flash_anatomy.py | UNCITED | — | — | — | — |
| v4flash | v4flash_census.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| v4flash | v4flash_f1b.py | library | RESULTS | RESULTS×1 | 2 | — |
| v4flash | v4flash_f1c.py | library | RESULTS | RESULTS×1 | 1 | — |
| v4flash | v4flash_f1d.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — |
| v4flash | v4flash_header.py | results-cited | RESULTS | RESULTS×1 | — | — |
| v4flash | v4flash_router.py | library | RESULTS, specs | RESULTS×1, specs×1 | 3 | — |
| v4flash | v4flash_rung0.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| v4flash | v4flash_rung2b.py | library | RESULTS, specs | RESULTS×1, specs×2 | 1 | — |
| v4flash | v4flash_rung2b_router.py | results-cited | RESULTS | RESULTS×1 | — | — |
| v4flash | v4flash_rungA.py | library | RESULTS, specs | RESULTS×2, specs×3 | 8 | — |
| v4flash | v4flash_rungd.py | library | RESULTS | RESULTS×3 | 1 | — |
| v4flash | v4flash_rungd2.py | results-cited | RESULTS | RESULTS×1 | — | — |
| v4flash | v4flash_s0.py | results-cited | RESULTS | RESULTS×3 | — | — |
| v4flash | v4flash_twin.py | library | RESULTS | RESULTS×4 | 3 | — |
| verify | verify_intbirth_prims.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — |
| vmasm | vmasm.py | library | — | — | 1 | — |
| vmasm | vmasm_probe.py | UNCITED | — | — | — | — |
| vrm | vrm_ab.py | UNCITED | — | — | — | 1 |
| weight | weight_fft_euler.py | UNCITED | — | — | — | — |
| wfloor | wfloor_ladder.sh | results-cited | RESULTS | RESULTS×1 | — | — |
| wsl | wsl.sh | results-cited | RESULTS, specs | RESULTS×10, specs×20 | — | 4 |
| z1 | z1_gate.sh | UNCITED | — | — | — | — |
| z1s | z1s_hot_watcher.sh | UNCITED | — | — | — | — |
| zx | zx_chain.sh | UNCITED | — | — | — | — |
| zx | zx_chain_cuda.sh | UNCITED | — | — | — | — |
| zx | zx_gate_watcher.sh | UNCITED | — | — | — | — |

## scripts/

| family | file | class | cited by | doc citations | imports | mentions |
|---|---|---|---|---|---|---|
|  | __init__.py | results-cited | RESULTS, specs | RESULTS×1, specs×12 | — | 1 |
| anim | anim_precompute.py | spec-cited | specs | specs×4 | — | 3 |
| arena | arena.py | UNCITED | — | — | — | — |
| autopsy | autopsy_int.py | UNCITED | — | — | — | — |
| backfill | backfill_code_commit.py | spec-cited | specs | specs×3 | — | — |
| bench | bench_adaptive.py | spec-cited | specs | specs×1 | — | — |
| bench | bench_adaptive_draft.py | UNCITED | — | — | — | — |
| bench | bench_anneal.py | UNCITED | — | — | — | — |
| bench | bench_ansatz_search.py | UNCITED | — | — | — | — |
| bench | bench_ansatz_search_2b.py | UNCITED | — | — | — | — |
| bench | bench_bandit.py | UNCITED | — | — | — | — |
| bench | bench_bestfirst.py | UNCITED | — | — | — | 1 |
| bench | bench_bestfirst_llm.py | UNCITED | — | — | — | — |
| bench | bench_bestfirst_nnue.py | UNCITED | — | — | — | 1 |
| bench | bench_budget_alloc.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_commute.py | UNCITED | — | — | — | — |
| bench | bench_compile.py | UNCITED | — | — | — | 4 |
| bench | bench_control.py | UNCITED | — | — | — | — |
| bench | bench_decoding.py | UNCITED | — | — | — | 1 |
| bench | bench_derivation.py | spec-cited | specs | specs×14 | — | — |
| bench | bench_dispatch_race_v4.py | UNCITED | — | — | — | — |
| bench | bench_distilled_draft.py | UNCITED | — | — | — | — |
| bench | bench_engine_regret.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_entropy_beam.py | UNCITED | — | — | — | 1 |
| bench | bench_fib_restarts.py | UNCITED | — | — | — | — |
| bench | bench_flash_prefill.py | UNCITED | — | — | — | — |
| bench | bench_frontier.py | spec-cited | specs | specs×1 | — | — |
| bench | bench_fused.py | UNCITED | — | — | — | — |
| bench | bench_fused_ce.py | results-cited | RESULTS | RESULTS×1 | — | 1 |
| bench | bench_gated.py | UNCITED | — | — | — | — |
| bench | bench_gweight.py | UNCITED | — | — | — | — |
| bench | bench_hints_ab.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_hybrid.py | library | — | — | 1 | — |
| bench | bench_int4_config_sweep.py | UNCITED | — | — | — | — |
| bench | bench_int4_gemv.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_interference.py | UNCITED | — | — | — | — |
| bench | bench_ksweep.py | UNCITED | — | — | — | — |
| bench | bench_kv_quant_decode.py | UNCITED | — | — | — | 1 |
| bench | bench_ladder.py | spec-cited | specs | specs×2 | — | — |
| bench | bench_lazy.py | library | RESULTS | RESULTS×1 | 1 | — |
| bench | bench_llm_gating.py | UNCITED | — | — | — | — |
| bench | bench_lookup_static.py | UNCITED | — | — | — | — |
| bench | bench_luby.py | UNCITED | — | — | — | — |
| bench | bench_magic.py | library | RESULTS | RESULTS×1 | 1 | — |
| bench | bench_markov.py | UNCITED | — | — | — | — |
| bench | bench_markov_adaptive.py | UNCITED | — | — | — | — |
| bench | bench_metal_kernels.py | spec-cited | specs | specs×8 | — | 2 |
| bench | bench_mlx_integration.py | UNCITED | — | — | — | — |
| bench | bench_nnue.py | spec-cited | specs | specs×8 | — | 1 |
| bench | bench_ode_engine.py | UNCITED | — | — | — | — |
| bench | bench_opcap.py | UNCITED | — | — | — | — |
| bench | bench_population.py | UNCITED | — | — | — | 1 |
| bench | bench_pred_syndromes.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_prefix_reuse.py | UNCITED | — | — | — | — |
| bench | bench_proposer.py | spec-cited | specs | specs×9 | — | 1 |
| bench | bench_quant_schemes.py | results-cited | RESULTS | RESULTS×1 | — | 1 |
| bench | bench_record.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_regret_resample.py | UNCITED | — | — | — | — |
| bench | bench_rotate_quantize.py | spec-cited | specs | specs×2 | — | 1 |
| bench | bench_rule_basis.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_stack_winners.py | UNCITED | — | — | — | — |
| bench | bench_stacked.py | UNCITED | — | — | — | — |
| bench | bench_static.py | UNCITED | — | — | — | — |
| bench | bench_step_diversity.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_step_tokens.py | library | RESULTS, specs | RESULTS×2, specs×7 | 67 | 2 |
| bench | bench_stitch_poc.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_syndrome_head.py | UNCITED | — | — | — | — |
| bench | bench_syndrome_policy.py | library | — | — | 1 | — |
| bench | bench_temp_race.py | UNCITED | — | — | — | — |
| bench | bench_tree_verify.py | UNCITED | — | — | — | — |
| bench | bench_triton_kernels.py | UNCITED | — | — | — | 1 |
| bench | bench_verify_fast.py | library | RESULTS, specs | RESULTS×1, specs×4 | 46 | 2 |
| bench | bench_vge.py | spec-cited | specs | specs×1 | — | — |
| bench | bench_weight_anatomy.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_zx.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_zx_r3.py | library | — | — | 3 | — |
| bench | bench_zx_r5.py | library | RESULTS | RESULTS×1 | 2 | 1 |
| bench | bench_zx_r6.py | results-cited | RESULTS | RESULTS×1 | — | — |
| bench | bench_zx_r7.py | results-cited | RESULTS | RESULTS×1 | — | — |
| book | book.py | library | RESULTS, specs | RESULTS×4, specs×3 | 1 | — |
| build | build_gen7_diet.py | UNCITED | — | — | — | — |
| calibrate | calibrate_hce.py | spec-cited | specs | specs×8 | — | — |
| ckpt | ckpt_manifest.py | results-cited | RESULTS, specs | RESULTS×3, specs×1 | — | 2 |
| consolidate | consolidate_mathnative.py | results-cited | RESULTS | RESULTS×1 | — | — |
| control | control_round.py | UNCITED | — | — | — | — |
| convert | convert_diet_prefix.py | spec-cited | specs | specs×1 | — | — |
| eval | eval_mathnative.py | UNCITED | — | — | — | — |
| eval | eval_pruned_moe.py | spec-cited | specs | specs×2 | — | — |
| eval | eval_ruler.py | spec-cited | specs | specs×1 | — | — |
| expert | expert_iter_steps.py | library | specs | specs×4 | 4 | — |
| expert | expert_loop.py | library | specs | specs×17 | 3 | — |
| farm | farm_algebra.py | results-cited | RESULTS | RESULTS×1 | — | — |
| farm | farm_l4_calc.py | UNCITED | — | — | — | — |
| farm | farm_v22.py | UNCITED | — | — | — | 1 |
| figlib | figlib.py | spec-cited | specs | specs×1 | — | — |
| gen | gen_catalog.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 1 |
| gen | gen_codemap.py | results-cited | RESULTS, specs | RESULTS×1, specs×21 | — | 1 |
| gen | gen_dispatch_labels.py | UNCITED | — | — | — | 2 |
| gen | gen_dispatch_labels_v2.py | UNCITED | — | — | — | — |
| gen | gen_figures_web.py | spec-cited | specs | specs×1 | — | — |
| gen | gen_frontier.py | UNCITED | — | — | — | — |
| gen | gen_index.py | spec-cited | specs | specs×10 | — | 1 |
| gen | gen_lake.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 1 |
| gen | gen_magic_labels.py | results-cited | RESULTS | RESULTS×1 | — | — |
| gen | gen_policy_labels.py | UNCITED | — | — | — | — |
| gen | gen_proposer_data.py | spec-cited | specs | specs×7 | — | 1 |
| gen | gen_readme.py | spec-cited | specs | specs×24 | — | 1 |
| gen | gen_regret_labels.py | UNCITED | — | — | — | — |
| gen | gen_results_index.py | results-cited | RESULTS, specs | RESULTS×2, specs×12 | — | 4 |
| gen | gen_scoreboard.py | UNCITED | — | — | — | — |
| gen | gen_syndrome_labels.py | results-cited | RESULTS | RESULTS×1 | — | 1 |
| grow | grow_mathnative.py | library | specs | specs×1 | 1 | 1 |
| harvest | harvest_champion.py | UNCITED | — | — | — | — |
| harvest | harvest_frontier.py | spec-cited | specs | specs×1 | — | — |
| list | list_uncurated.py | spec-cited | specs | specs×6 | — | — |
| log | log_hygiene.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | 1 |
| markov | markov_eval.py | UNCITED | — | — | — | — |
| markov | markov_prior.py | UNCITED | — | — | — | — |
| mine | mine_highways.py | UNCITED | — | — | — | 1 |
| mine | mine_prior_update.py | results-cited | RESULTS | RESULTS×1 | — | — |
| moe | moe_router_stats.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | 4 |
| plot | plot_gt1_crest.py | spec-cited | specs | specs×2 | — | — |
| plot | plot_identity_crest.py | spec-cited | specs | specs×2 | — | — |
| plot | plot_neurons.py | spec-cited | specs | specs×2 | — | 1 |
| probe | probe_depth.py | results-cited | RESULTS | RESULTS×1 | — | — |
| render | render_gallery.py | spec-cited | specs | specs×6 | — | — |
| render | render_hero_neurons.py | spec-cited | specs | specs×3 | — | 1 |
| results | results_query.py | spec-cited | specs | specs×15 | — | 2 |
| rjob | rjob.py | results-cited | RESULTS, specs | RESULTS×3, specs×2 | — | — |
| sol | sol_enrich_results.py | library | specs | specs×4 | 1 | — |
| sol | sol_generate_tables.py | spec-cited | specs | specs×3 | — | — |
| step | step_grpo.py | spec-cited | specs | specs×1 | — | — |
| step | step_grpo_micro.py | library | RESULTS, specs | RESULTS×5, specs×19 | 95 | 5 |
| sweep | sweep_lookup.py | UNCITED | — | — | — | 1 |
| sweep | sweep_lookup_mlx.py | UNCITED | — | — | — | — |
| tabula | tabula_rasa_r0.py | UNCITED | — | — | — | — |
| tabula | tabula_rasa_r1.py | UNCITED | — | — | — | — |
| tabula | tabula_rasa_r2.py | UNCITED | — | — | — | — |
| task | task_arithmetic.py | spec-cited | specs | specs×1 | — | — |
| task | task_composition.py | UNCITED | — | — | — | — |
| tournament | tournament_birth.py | library | — | — | 3 | 6 |
| train | train_calculus.py | library | specs | specs×6 | 1 | 4 |
| train | train_dispatcher.py | UNCITED | — | — | — | — |
| train | train_magic_estimator.py | library | RESULTS | RESULTS×1 | 7 | 1 |
| train | train_magic_llm.py | results-cited | RESULTS | RESULTS×1 | — | — |
| train | train_mathnative.py | library | RESULTS, specs | RESULTS×4, specs×7 | 53 | 31 |
| train | train_nnue.py | library | specs | specs×8 | 1 | 12 |
| train | train_proposer.py | spec-cited | specs | specs×5 | — | — |
| train | train_syndrome_decoder.py | results-cited | RESULTS | RESULTS×1 | — | 1 |
| train | train_syndrome_policy.py | results-cited | RESULTS | RESULTS×1 | — | — |
| train | train_ternary.py | results-cited | RESULTS, specs | RESULTS×1, specs×4 | — | 2 |
| train | train_tf32x3.py | UNCITED | — | — | — | — |
| train | train_value_head.py | results-cited | RESULTS | RESULTS×1 | — | — |
| train | train_weight_reader.py | spec-cited | specs | specs×4 | — | — |
| validity | validity_autopsy.py | results-cited | RESULTS | RESULTS×1 | — | — |
