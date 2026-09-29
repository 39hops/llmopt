# CODEMAP — the move-gate inventory (generated, do not hand-edit)

Regenerate: `.venv/bin/python scripts/gen_codemap.py`. One row per
top-level file in scratch/ and scripts/. Class ladder (mechanical):
library > reproduce-pinned > results-cited > spec-cited >
tool-referenced > UNCITED. House law: cited files are the evidence
record — extraction means adoption-with-reverification, never a
silent move. Columns: `cited by` lists the file's OWN citation
groups plus `via:<group>` for groups inherited through a cited
caller; `doc citations` counts the own ones; `imports` counts code
files that import it
(bare, dotted, or dynamic loader; drives `library`); `invoked by`
counts files that run it by path or shell (from llmopt/ or tests/
this also drives `library`; from lab tooling it drives
`tool-referenced`); `mentions` counts files that only name it in
comments or plain strings; `via` names the cited caller a file
inherits its class from when no document names the file itself.

Census: library 190, reproduce-pinned 9, results-cited 496, spec-cited 80, tool-referenced 5, UNCITED 124 (of 904; library rows that are also doc-cited: 185)

## scratch/

| family | file | class | cited by | doc citations | imports | invoked by | mentions | via |
|---|---|---|---|---|---|---|---|---|
| absorb | absorb_1e5.py | UNCITED | — | — | — | — | — | — |
| adjudicate | adjudicate_zx.py | library | specs, via:RESULTS | specs×1 | 1 | — | — | gate_zx.py |
| anatomy | anatomy.py | results-cited | RESULTS, specs | RESULTS×5, specs×5 | — | — | 1 | — |
| answerform0 | answerform0_censor0.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — | — | — |
| assets | assets_classify.py | spec-cited | specs | specs×1 | — | — | — | — |
| atlas | atlas_precompute.py | spec-cited | specs | specs×1 | — | — | — | — |
| atomdose1 | atomdose1_driver.sh | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| atomladder1 | atomladder1_driver.sh | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| atomtraj | atomtraj_census.py | results-cited | RESULTS | RESULTS×10 | — | 4 | — | — |
| atomtraj | atomtraj_pins.py | library | RESULTS, via:specs | RESULTS×53 | 42 | 5 | — | — |
| atomtraj | atomtraj_qual.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| atomtraj | atomtraj_verify.py | results-cited | RESULTS | RESULTS×8 | — | 3 | — | — |
| atomtraj1 | atomtraj1_driver.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| atomtraj1 | atomtraj1_repair_driver.sh | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| attractor | attractor_census.py | results-cited | RESULTS | RESULTS×2 | — | — | 1 | — |
| attractor | attractor_census2.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — | — | — |
| b768 | b768_after_v5.sh | UNCITED | — | — | — | — | — | — |
| basics | basics_census0.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| basics | basics_probe0.py | library | RESULTS, specs, via:TOOLING | RESULTS×9, specs×1 | 2 | — | — | — |
| basicsdiet1 | basicsdiet1_driver.sh | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| basin | basin_probe.py | UNCITED | — | — | — | — | — | — |
| birth19m | birth19m_arith.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | 1 | 1 | — |
| birth19m | birth19m_atoms.py | results-cited | RESULTS, specs | RESULTS×5, specs×2 | — | — | 1 | — |
| birth19m | birth19m_atoms_dose.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | 1 | 1 | — |
| birth19m | birth19m_atoms_ladder.py | library | RESULTS | RESULTS×9 | — | 3 | 3 | — |
| birth19m | birth19m_atoms_rule.py | results-cited | RESULTS, specs | RESULTS×6, specs×1 | — | 1 | 2 | — |
| birth19m | birth19m_atoms_traj.py | library | RESULTS | RESULTS×12 | — | 10 | 2 | — |
| birth19m | birth19m_atoms_trajgate.py | results-cited | RESULTS | RESULTS×10 | — | 4 | 1 | — |
| birth19m | birth19m_backsched.py | results-cited | RESULTS, specs | RESULTS×6, specs×2 | — | 1 | — | — |
| birth19m | birth19m_caf.py | library | RESULTS, specs | RESULTS×5, specs×1 | — | 5 | 1 | — |
| birth19m | birth19m_curric.py | library | RESULTS, specs, via:TOOLING | RESULTS×42, specs×2 | 34 | 2 | — | — |
| birth19m | birth19m_curric_rev.py | results-cited | RESULTS, specs | RESULTS×5, specs×1 | — | — | 1 | — |
| birth19m | birth19m_curric_swap.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — | — | — |
| birth19m | birth19m_dfa.py | library | RESULTS, specs | RESULTS×6, specs×1 | — | 7 | 2 | — |
| birth19m | birth19m_fb.py | library | RESULTS, specs | RESULTS×3, specs×2 | — | 4 | 1 | — |
| birth19m | birth19m_phase.py | results-cited | RESULTS, specs | RESULTS×6, specs×1 | — | 2 | 1 | — |
| birth19m | birth19m_sg.py | library | RESULTS, specs | RESULTS×4, specs×1 | — | 3 | 1 | — |
| birth19m | birth19m_sg7.py | library | RESULTS, specs | RESULTS×3, specs×1 | — | 2 | — | — |
| birth19m | birth19m_snaps.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | 1 | — |
| birth19m | birth19m_softnext.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | 1 | — | — |
| birth19m | birth19m_softspeed.py | results-cited | RESULTS, specs, TOOLING | RESULTS×4, specs×2, TOOLING×2 | — | 2 | — | — |
| birth19m | birth19m_xterm.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | 1 | — | — |
| blackhole | blackhole_b0.py | results-cited | RESULTS, specs | RESULTS×2, specs×2 | — | — | — | — |
| boundary | boundary_or_bulk.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | — | — | — |
| brute | brute_arms_0801.sh | tool-referenced | TOOLING | TOOLING×2 | — | 1 | — | — |
| brute | brute_b_arms_0801.sh | tool-referenced | TOOLING | TOOLING×2 | — | 1 | — | — |
| brute | brute_c_arm_0801.sh | results-cited | RESULTS, TOOLING | RESULTS×1, TOOLING×2 | — | 1 | — | — |
| build | build_dist_diets.py | UNCITED | — | — | — | — | — | — |
| build | build_merged_diet.py | spec-cited | specs | specs×1 | — | — | — | — |
| caf | caf_actpost.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | 1 | — | — |
| caf | caf_leakage_smoke.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| caf | caf_prune.py | UNCITED | — | — | — | — | — | — |
| caf | caf_qualgate.py | library | RESULTS, specs | RESULTS×3, specs×1 | — | 2 | — | — |
| cal | cal_dilute.py | UNCITED | — | — | — | — | — | — |
| cal | cal_dk_probe.py | UNCITED | — | — | — | — | 1 | — |
| calib | calib_dist_birth.sh | spec-cited | specs | specs×4 | — | — | 1 | — |
| calib | calib_probe.py | library | RESULTS, specs | RESULTS×3, specs×14 | 1 | 2 | — | — |
| calib | calib_snap_gates.sh | spec-cited | specs | specs×5 | — | — | — | — |
| callspan | callspan_arms.py | UNCITED | — | — | — | — | — | — |
| capacity | capacity_meter.py | library | RESULTS, specs | RESULTS×6, specs×4 | 4 | 1 | 1 | — |
| ce | ce_gate_study.py | spec-cited | specs | specs×3 | — | — | — | — |
| ce400 | ce400.py | results-cited | via:RESULTS | — | — | 2 | 1 | fmt_chain.sh |
| ceiling | ceiling_probe_cuda.py | UNCITED | — | — | — | — | — | — |
| census | census_night.sh | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| chain | chain_carry.py | UNCITED | — | — | — | 1 | — | — |
| champ | champ_cuda_probe.py | UNCITED | — | — | — | — | — | — |
| checkers0 | checkers0.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — | — | — |
| churn | churn_judge_eval.py | reproduce-pinned | REPRODUCE | REPRODUCE×2 | — | — | — | — |
| ckpt | ckpt_delete_pass.py | UNCITED | — | — | — | — | — | — |
| ckpt | ckpt_inventory.py | spec-cited | specs, TOOLING | specs×2, TOOLING×2 | — | 1 | 1 | — |
| ckpt | ckpt_triage_table.py | UNCITED | — | — | — | — | — | — |
| clade | clade_stream_d256.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| closers | closers_chain.sh | UNCITED | — | — | — | — | — | — |
| comp | comp_ladder.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | — | — | — |
| complex | complex_birth.py | results-cited | specs, via:RESULTS | specs×2 | — | 8 | 1 | cplx_chain.sh |
| complex | complex_model.py | library | RESULTS, specs | RESULTS×1, specs×3 | 6 | — | 1 | — |
| complex | complex_nnue.py | UNCITED | — | — | — | — | — | — |
| complexify | complexify_control.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| confluence | confluence.py | UNCITED | — | — | — | — | — | — |
| corner | corner_snap.py | UNCITED | — | — | — | — | — | — |
| cplx | cplx_chain.sh | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| crossfoster | crossfoster_chain.py | library | RESULTS, specs | RESULTS×5, specs×1 | 1 | 2 | — | — |
| crossfoster | crossfoster_chain_determinism_probe.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| crossfoster | crossfoster_donor.py | library | RESULTS, specs | RESULTS×5, specs×1 | 3 | 2 | — | — |
| crossfoster1c | crossfoster1c_launch.sh | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| crossfoster1d | crossfoster1d_launch.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| crystal | crystal_recreate_test.py | spec-cited | specs, TOOLING | specs×3, TOOLING×1 | — | — | 1 | — |
| d2 | d2_verify.py | results-cited | via:RESULTS | — | — | 1 | — | night_28.sh |
| day | day_chain.sh | UNCITED | — | — | — | — | — | — |
| desert | desert_v2.py | spec-cited | specs | specs×1 | — | — | — | — |
| detbwd | detbwd_diet.py | library | RESULTS, specs, via:TOOLING | RESULTS×9, specs×3 | 2 | — | — | — |
| detbwd | detbwd_gravmoe.py | library | RESULTS, specs, TOOLING | RESULTS×6, specs×32, TOOLING×2 | 1 | 8 | 1 | — |
| detbwd | detbwd_mb.py | library | RESULTS, specs, via:TOOLING | RESULTS×11, specs×8 | 4 | — | — | — |
| detbwd | detbwd_plateau.py | results-cited | RESULTS, specs | RESULTS×2, specs×3 | — | — | — | — |
| detbwd | detbwd_r1.py | library | RESULTS, via:specs, via:TOOLING | RESULTS×9 | 9 | — | — | — |
| detbwd | detbwd_r1b.py | library | RESULTS, specs, via:TOOLING | RESULTS×4, specs×2 | 1 | — | — | — |
| detbwd | detbwd_r2_adamw.py | library | RESULTS, specs, via:TOOLING | RESULTS×6, specs×1 | 2 | — | — | — |
| detbwd | detbwd_r2b.py | library | RESULTS, specs, via:TOOLING | RESULTS×8, specs×5 | 6 | — | — | — |
| detbwd | detbwd_r3_qw.py | library | RESULTS, specs, via:TOOLING | RESULTS×2, specs×1 | 5 | — | — | — |
| determinability | determinability_census.py | results-cited | RESULTS | RESULTS×2 | — | — | 1 | — |
| dfa | dfa_act.py | library | RESULTS, specs | RESULTS×9, specs×1 | 4 | 2 | — | — |
| dfa | dfa_align.py | library | RESULTS, specs | RESULTS×5, specs×1 | 2 | 2 | 1 | — |
| dfa | dfa_credit.py | library | RESULTS, specs | RESULTS×6, specs×3 | 23 | 4 | — | — |
| dfa | dfa_depthclass.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 2 | — | — |
| dfa | dfa_harm_desk.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | 1 | — | — |
| dfa | dfa_leakage_smoke.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | 1 | — |
| dfa | dfa_postmortem.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | 1 | — | — |
| dfa | dfa_probe.py | library | RESULTS, specs | RESULTS×2, specs×1 | 15 | 2 | — | — |
| dfa | dfa_prune.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| dfa | dfa_qualgate.py | results-cited | RESULTS, specs | RESULTS×3, specs×1 | — | 2 | — | — |
| dfa | dfa_trajcensus.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 2 | — | — |
| dfa | dfa_verify.py | results-cited | RESULTS, specs | RESULTS×5, specs×1 | — | 1 | — | — |
| dfaharm0 | dfaharm0_driver.sh | UNCITED | — | — | — | — | — | — |
| dfapost0 | dfapost0_driver.sh | UNCITED | — | — | — | — | — | — |
| distortion | distortion_collapse.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| dual | dual_probe.py | UNCITED | — | — | — | — | — | — |
| duo | duo_mine.py | spec-cited | specs | specs×1 | — | — | — | — |
| duo | duo_wave.py | spec-cited | specs | specs×1 | — | — | — | — |
| e2 | e2_logit_check.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| e3 | e3_battery.py | spec-cited | specs | specs×1 | — | — | — | — |
| emission | emission_wall_pair.py | spec-cited | specs | specs×2 | — | — | — | — |
| engine | engine_scale_export.py | spec-cited | specs | specs×2 | — | — | — | — |
| ex1 | ex1_swap.py | results-cited | RESULTS, specs | RESULTS×8, specs×2 | — | — | 1 | — |
| ex2 | ex2_build.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| ex3 | ex3_build.py | library | RESULTS, specs | RESULTS×8, specs×4 | 2 | 3 | 1 | — |
| ex4 | ex4_build.py | library | RESULTS, specs | RESULTS×6, specs×1 | 1 | 2 | — | — |
| ex4 | ex4_mask_census.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| ex5 | ex5_build.py | results-cited | RESULTS | RESULTS×7 | — | 3 | — | — |
| ex5 | ex5_manifest.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| ex5 | ex5_observe.py | results-cited | RESULTS | RESULTS×5 | — | — | — | — |
| ex5 | ex5_run.sh | results-cited | RESULTS | RESULTS×4 | — | 1 | — | — |
| ex5 | ex5_traj_census.py | results-cited | RESULTS | RESULTS×6 | — | 1 | — | — |
| ex5 | ex5_traj_rider.py | results-cited | RESULTS | RESULTS×2 | — | 1 | — | — |
| ex5 | ex5_traj_rider2.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| ex6 | ex6_observe.py | results-cited | RESULTS | RESULTS×5 | — | — | — | — |
| ex6 | ex6_phase.py | results-cited | RESULTS, specs | RESULTS×11, specs×3 | — | 3 | — | — |
| ex6 | ex6_run.sh | results-cited | RESULTS | RESULTS×5 | — | 1 | — | — |
| ex6b43 | ex6b43_decomp.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| ex6b43 | ex6b43_idcensus.py | results-cited | RESULTS | RESULTS×6 | — | — | — | — |
| ex6b43 | ex6b43_knife.py | results-cited | RESULTS | RESULTS×6 | — | — | — | — |
| ex6depth | ex6depth.py | results-cited | RESULTS | RESULTS×9 | — | — | 1 | — |
| ex6depth1 | ex6depth1.py | results-cited | RESULTS | RESULTS×9 | — | — | — | — |
| ex6loc | ex6loc.py | results-cited | RESULTS | RESULTS×4 | — | 1 | — | — |
| ex6loc | ex6loc_rider.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| ex6med | ex6med.py | library | RESULTS, specs | RESULTS×1, specs×1 | 4 | 5 | — | — |
| ex6med | ex6med_idaudit.py | UNCITED | — | — | — | — | 1 | — |
| ex6med | ex6med_idaudit2.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| ex6med | ex6med_run.sh | UNCITED | — | — | — | — | — | — |
| ex6med2 | ex6med2.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — | — | — |
| ex6temporal | ex6temporal.py | results-cited | RESULTS | RESULTS×10 | — | — | 3 | — |
| exact | exact_twin_d56.py | spec-cited | specs | specs×1 | — | — | — | — |
| exact1 | exact1_small_cells.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | — | — | — |
| exchange | exchange_test.py | results-cited | RESULTS, specs, TOOLING | RESULTS×2, specs×1, TOOLING×1 | — | — | — | — |
| export | export_axnn.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| export | export_mb_ref.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| export | export_r2b_ref.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| farm | farm_arith.py | results-cited | RESULTS, specs, TOOLING | RESULTS×5, specs×1, TOOLING×2 | — | — | 1 | — |
| farm | farm_atoms.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | — | 1 | — |
| farm | farm_atoms_axiom.py | results-cited | RESULTS, specs | RESULTS×5, specs×1 | — | 1 | 1 | — |
| farm | farm_dist_rows.py | results-cited | RESULTS, specs | RESULTS×2, specs×7 | — | — | 1 | — |
| farm | farm_xterm.py | results-cited | RESULTS, specs, TOOLING | RESULTS×5, specs×1, TOOLING×1 | — | — | — | — |
| farmer | farmer_probe.py | results-cited | RESULTS, specs | RESULTS×2, specs×2 | — | — | — | — |
| fb | fb_gate.py | library | RESULTS, specs | RESULTS×11, specs×2 | 1 | 3 | 1 | — |
| fb | fb_prune.py | spec-cited | specs | specs×1 | — | — | — | — |
| fig | fig_magic_scatter.py | spec-cited | specs | specs×1 | — | — | — | — |
| first | first_moment_erasure.py | library | RESULTS, specs | RESULTS×37, specs×5 | 2 | 5 | — | — |
| first | first_moment_erasure_ladder.py | library | RESULTS, specs | RESULTS×34, specs×5 | 2 | 5 | — | — |
| first | first_moment_erasure_ladder2.py | library | RESULTS, specs | RESULTS×22, specs×4 | 2 | 4 | — | — |
| fixed | fixed_q_snap.py | results-cited | via:RESULTS, via:specs | — | — | 1 | — | quick_exact_3080.sh |
| floor | floor_hk1.sh | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 1 | — | — |
| floor | floor_hk1_d256.sh | UNCITED | — | — | — | — | — | — |
| fme1 | fme1_launch.sh | results-cited | RESULTS, specs | RESULTS×8, specs×2 | — | — | — | — |
| fmel1 | fmel1_launch.sh | results-cited | RESULTS, specs | RESULTS×11, specs×3 | — | — | — | — |
| fmel2 | fmel2_launch.sh | results-cited | RESULTS, specs | RESULTS×14, specs×3 | — | — | — | — |
| fmt | fmt_chain.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| fmt | fmt_chain2.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| fmt | fmt_pp_watcher.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| format | format_delta_prep.py | results-cited | via:RESULTS | — | — | 1 | 1 | fmt_chain.sh |
| format | format_ladder.py | results-cited | specs, via:RESULTS | specs×1 | — | 3 | — | fmt_chain.sh |
| fourier | fourier_g9.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| fourier | fourier_probe.py | UNCITED | — | — | — | — | — | — |
| fourier2 | fourier2_modbirth.py | UNCITED | — | — | — | — | — | — |
| fourier2b | fourier2b_widemod.py | library | RESULTS, specs | RESULTS×4, specs×1 | 2 | — | — | — |
| fourier3 | fourier3_algdiet.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| fourier4a | fourier4a_dynamics.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| fp64 | fp64_paired.py | spec-cited | specs | specs×1 | — | — | — | — |
| frozenbb1 | frozenbb1_driver.sh | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| fx3 | fx3_house.py | results-cited | RESULTS, specs | RESULTS×2, specs×3 | — | — | — | — |
| g19 | g19_bf16_isolation.sh | spec-cited | specs | specs×2 | — | — | 1 | — |
| g19 | g19_fp32_cell.sh | UNCITED | — | — | — | — | — | — |
| g19 | g19_probes_fix.sh | UNCITED | — | — | — | — | — | — |
| g19 | g19_sigma_cuda.sh | UNCITED | — | — | — | — | — | — |
| g5 | g5_polar.py | UNCITED | — | — | — | — | — | — |
| gate | gate_backsched.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | — | — | — |
| gate | gate_batched.py | UNCITED | — | — | — | 2 | 1 | — |
| gate | gate_ckpt.py | library | RESULTS, specs, TOOLING | RESULTS×9, specs×18, TOOLING×2 | — | 37 | 2 | — |
| gate | gate_ckpt_cuda.py | library | RESULTS, specs | RESULTS×4, specs×1 | — | 12 | 1 | — |
| gate | gate_cplx.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | 1 | — | — |
| gate | gate_phase19m.py | library | RESULTS, specs | RESULTS×5, specs×2 | 1 | — | — | — |
| gate | gate_pp.py | results-cited | RESULTS, specs | RESULTS×10, specs×5 | — | 1 | — | — |
| gate | gate_prefix.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | 1 | — | — |
| gate | gate_rarity.py | results-cited | RESULTS, specs | RESULTS×3, specs×3 | — | 8 | 2 | — |
| gate | gate_regate.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| gate | gate_transcripts.py | results-cited | RESULTS, specs | RESULTS×3, specs×2 | — | — | — | — |
| gate | gate_v2_bench.sh | UNCITED | — | — | — | — | — | — |
| gate | gate_zx.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | 10 | 1 | — |
| gatepins | gatepins_freeze.py | spec-cited | specs | specs×5 | — | — | 1 | — |
| gauge | gauge_distance_d256.py | UNCITED | — | — | — | — | — | — |
| gauge | gauge_m4x.py | UNCITED | — | — | — | — | — | — |
| gauge | gauge_slack_rat.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| gen | gen_lab_overview_pdf.py | results-cited | RESULTS | RESULTS×1 | — | — | 1 | — |
| gen | gen_lean_corpus.py | results-cited | RESULTS, specs | RESULTS×3, specs×2 | — | — | — | — |
| gen8 | gen8_pipeline.sh | spec-cited | specs | specs×2 | — | — | 1 | — |
| gen9 | gen9_19m_cuda_control.sh | UNCITED | — | — | — | — | — | — |
| gen9 | gen9_45m_fp32_control.sh | UNCITED | — | — | — | — | — | — |
| gen9 | gen9_45m_probes.sh | UNCITED | — | — | — | — | — | — |
| gen9 | gen9_pipeline.sh | spec-cited | specs | specs×2 | — | — | 1 | — |
| genpins | genpins_freeze.py | UNCITED | — | — | — | — | 1 | — |
| gradmap0 | gradmap0_probe.py | library | RESULTS, specs | RESULTS×2, specs×1 | 1 | — | — | — |
| gradmap0 | gradmap0_rd2.py | results-cited | RESULTS, specs | RESULTS×2, specs×2 | — | — | — | — |
| graph | graph_mod_sigma.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| graph | graph_modularity_gen8.py | library | via:RESULTS | — | 1 | — | — | graph_mod_sigma.py |
| grav | grav_posthoc.py | UNCITED | — | — | — | — | — | — |
| grav | grav_probe.py | library | RESULTS | RESULTS×3 | 2 | — | — | — |
| grav1b | grav1b_distance.py | UNCITED | — | — | — | — | — | — |
| grav2 | grav2_spacetime.py | library | via:RESULTS | — | 1 | — | — | p3_grav2.py |
| greedy | greedy_first_gate.py | results-cited | via:RESULTS | — | — | 1 | — | night_28.sh |
| grf | grf_analyze.py | results-cited | RESULTS, specs | RESULTS×5, specs×1 | — | 2 | — | — |
| grf | grf_analyze2.py | results-cited | RESULTS | RESULTS×4 | — | 1 | — | — |
| grf | grf_capture.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | 3 | 1 | — |
| grf | grf_capture2.py | results-cited | RESULTS | RESULTS×7 | — | 2 | — | — |
| grf | grf_corpus.py | results-cited | RESULTS, specs | RESULTS×5, specs×1 | — | 7 | — | — |
| grf | grf_rider.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| grf | grf_rider2.py | results-cited | RESULTS | RESULTS×6 | — | 1 | — | — |
| grf | grf_rider3.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| grow | grow_decomp1.sh | UNCITED | — | — | — | — | — | — |
| grpo | grpo_shaped.py | UNCITED | — | — | — | 1 | — | — |
| gt2 | gt2_code_arm0.py | reproduce-pinned | REPRODUCE, RESULTS, specs, TOOLING | REPRODUCE×1, RESULTS×2, specs×2, TOOLING×4 | — | 2 | — | — |
| gt2 | gt2_jaccard.py | library | REPRODUCE, RESULTS, specs | REPRODUCE×1, RESULTS×7, specs×23 | 9 | 1 | 3 | — |
| gt3 | gt3_probe_arm0.py | reproduce-pinned | REPRODUCE, RESULTS, TOOLING | REPRODUCE×1, RESULTS×4, TOOLING×3 | — | 1 | — | — |
| gt4 | gt4_dialog_prompts.py | reproduce-pinned | REPRODUCE, RESULTS | REPRODUCE×1, RESULTS×4 | — | — | — | — |
| gt4 | gt4_verbal_core.py | reproduce-pinned | REPRODUCE, RESULTS | REPRODUCE×1, RESULTS×6 | — | — | — | — |
| gt5 | gt5_union_keep.py | reproduce-pinned | REPRODUCE, RESULTS, specs | REPRODUCE×1, RESULTS×2, specs×1 | — | — | — | — |
| gt5c | gt5c_randfill_keep.py | reproduce-pinned | REPRODUCE, RESULTS | REPRODUCE×1, RESULTS×2 | — | — | — | — |
| gt6 | gt6_recall_ladder.py | reproduce-pinned | REPRODUCE, RESULTS | REPRODUCE×1, RESULTS×2 | — | — | — | — |
| gt6 | gt6_resume_arms.sh | tool-referenced | TOOLING | TOOLING×4 | — | 2 | — | — |
| gt7 | gt7_coverage_rederive.py | results-cited | RESULTS, specs | RESULTS×6, specs×2 | — | — | 1 | — |
| gt7 | gt7_draw.py | results-cited | RESULTS, specs | RESULTS×2, specs×4 | — | — | 1 | — |
| gt7 | gt7_run.py | results-cited | RESULTS, specs, TOOLING | RESULTS×25, specs×4, TOOLING×16 | — | 10 | — | — |
| head | head_autopsy.py | library | RESULTS, via:specs | RESULTS×1 | 1 | — | — | — |
| head | head_census.py | spec-cited | specs | specs×2 | — | — | 1 | — |
| holdout | holdout_gate.py | spec-cited | specs | specs×1 | — | 1 | — | — |
| holdout | holdout_v2.py | UNCITED | — | — | — | 2 | — | — |
| hot | hot_chain.sh | results-cited | RESULTS | RESULTS×3 | — | — | 1 | — |
| int2 | int2_regate.sh | UNCITED | — | — | — | — | — | — |
| int3 | int3_rider.py | spec-cited | specs | specs×2 | — | — | — | — |
| jointperm | jointperm_distance.py | UNCITED | — | — | — | — | — | — |
| judge | judge_decode.py | spec-cited | specs | specs×3 | — | — | — | — |
| k2h | k2h_gateladder.py | results-cited | RESULTS | RESULTS×8 | — | 3 | — | — |
| k2h | k2h_manifest.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| k2h | k2h_residues.py | results-cited | RESULTS | RESULTS×5 | — | 1 | — | — |
| k2h | k2h_residuesverify.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| k2h | k2h_stagecensus.py | library | RESULTS | RESULTS×11 | 2 | 7 | — | — |
| k2h | k2h_stagecensusverify.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| k2h | k2h_tagmove_check.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| k2h | k2h_transport.py | results-cited | RESULTS | RESULTS×6 | — | 2 | — | — |
| k2h | k2h_transportverify.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| k2h | k2h_wallaudit.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| k3 | k3_expert_demo.py | library | RESULTS, specs | RESULTS×8, specs×4 | — | 1 | 3 | — |
| keff | keff_probe.py | results-cited | RESULTS, TOOLING | RESULTS×2, TOOLING×2 | — | 1 | — | — |
| kv | kv_after_night.sh | UNCITED | — | — | — | — | — | — |
| kv | kv_equiv.py | UNCITED | — | — | — | 1 | 1 | — |
| l9 | l9_probe.py | results-cited | via:RESULTS | — | — | 3 | — | hot_chain.sh |
| lam | lam_merge_review.py | results-cited | via:RESULTS, via:specs | — | — | 1 | — | night31b_cuda.sh |
| lean | lean_check.py | reproduce-pinned | REPRODUCE, RESULTS, specs, TOOLING | REPRODUCE×1, RESULTS×13, specs×10, TOOLING×4 | — | 2 | — | — |
| lean | lean_sample_build.py | results-cited | RESULTS, specs | RESULTS×2, specs×2 | — | — | — | — |
| legacy | legacy_diet_audit.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| len | len_vs_l4.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — | — | — |
| lloydmax | lloydmax_race.py | library | — | — | 1 | — | — | — |
| loss | loss_floor_census.py | results-cited | RESULTS, TOOLING | RESULTS×4, TOOLING×2 | — | 1 | — | — |
| lyap | lyap_compare.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | 2 | — | — |
| lyapunov | lyapunov_birth.sh | UNCITED | — | — | — | — | — | — |
| mac | mac_day_chain.sh | UNCITED | — | — | — | — | — | — |
| make | make_altpairs.py | results-cited | RESULTS | RESULTS×1 | — | — | 1 | — |
| make | make_ruleablate_shards.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | — | — | — |
| make | make_union_diet.py | results-cited | specs, via:RESULTS | specs×1 | — | 1 | — | night_zx.sh |
| margin | margin_by_level.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| margin | margin_by_ply.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| margin | margin_census.py | UNCITED | — | — | — | — | — | — |
| margin | margin_vs_branching.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mass | mass_on_valid.py | spec-cited | specs | specs×4 | — | — | — | — |
| mathworld0 | mathworld0.py | library | RESULTS, specs | RESULTS×27, specs×2 | 2 | 18 | — | — |
| mathworld0 | mathworld0_coldreplay.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | — | — | — |
| mathworld1 | mathworld1_abv2desk.py | results-cited | RESULTS | RESULTS×5 | — | — | — | — |
| mathworld1 | mathworld1_actionbasis_census.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_actionfinal.py | library | RESULTS | RESULTS×15 | 4 | 17 | — | — |
| mathworld1 | mathworld1_actionprog.py | results-cited | RESULTS | RESULTS×3 | — | — | 1 | — |
| mathworld1 | mathworld1_actionprog2.py | results-cited | RESULTS | RESULTS×6 | — | — | — | — |
| mathworld1 | mathworld1_actionsem.py | library | RESULTS | RESULTS×29 | 10 | 27 | — | — |
| mathworld1 | mathworld1_actionsite.py | results-cited | RESULTS | RESULTS×4 | — | — | 1 | — |
| mathworld1 | mathworld1_actiontok.py | library | RESULTS | RESULTS×71 | 49 | 61 | — | — |
| mathworld1 | mathworld1_active.py | results-cited | RESULTS | RESULTS×5 | — | — | — | — |
| mathworld1 | mathworld1_autopsy.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_axfixture.py | library | RESULTS | RESULTS×6 | 2 | 7 | — | — |
| mathworld1 | mathworld1_birth.py | library | RESULTS | RESULTS×41 | 13 | 37 | — | — |
| mathworld1 | mathworld1_cayley.py | library | RESULTS | RESULTS×28 | 5 | 5 | — | — |
| mathworld1 | mathworld1_cayleyverify.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_census.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_cl1cost.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_cl1pop.py | library | RESULTS | RESULTS×2 | 2 | 3 | — | — |
| mathworld1 | mathworld1_cl1run.py | library | RESULTS | RESULTS×2 | 2 | 2 | — | — |
| mathworld1 | mathworld1_cl1splice.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_cycle.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| mathworld1 | mathworld1_decisionatlas.py | results-cited | RESULTS | RESULTS×3 | — | 1 | — | — |
| mathworld1 | mathworld1_decisionatlasverify.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_execbench.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_export.py | results-cited | RESULTS, specs | RESULTS×8, specs×1 | — | — | 1 | — |
| mathworld1 | mathworld1_frontier.py | UNCITED | — | — | — | — | — | — |
| mathworld1 | mathworld1_liveness.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_longctx_census.py | results-cited | RESULTS | RESULTS×5 | — | — | — | — |
| mathworld1 | mathworld1_longctx_census2.py | results-cited | RESULTS | RESULTS×5 | — | — | — | — |
| mathworld1 | mathworld1_matsub.py | library | RESULTS | RESULTS×4 | 1 | 1 | — | — |
| mathworld1 | mathworld1_matsub2.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_morphology.py | results-cited | RESULTS | RESULTS×6 | — | 1 | — | — |
| mathworld1 | mathworld1_morphologyverify.py | results-cited | RESULTS | RESULTS×6 | — | — | — | — |
| mathworld1 | mathworld1_nestedswap.py | results-cited | RESULTS | RESULTS×3 | — | 1 | — | — |
| mathworld1 | mathworld1_nestedswapverify.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_pdcov.py | results-cited | RESULTS | RESULTS×5 | — | — | — | — |
| mathworld1 | mathworld1_prband.py | library | RESULTS | RESULTS×12 | 6 | 4 | — | — |
| mathworld1 | mathworld1_prband2atlas.py | results-cited | RESULTS | RESULTS×6 | — | — | — | — |
| mathworld1 | mathworld1_prband2atlasagg.py | library | RESULTS | RESULTS×2 | 1 | — | 1 | — |
| mathworld1 | mathworld1_prband2atlasfresh.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_prband2atlasfreshagg.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_prband2atlasfreshverify.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_prband2atlasscore.py | library | RESULTS | RESULTS×12 | 1 | 2 | 1 | — |
| mathworld1 | mathworld1_prband2atlasverify.py | results-cited | RESULTS | RESULTS×3 | — | — | 1 | — |
| mathworld1 | mathworld1_prband2branch.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_prband2cf.py | results-cited | RESULTS | RESULTS×11 | — | 2 | — | — |
| mathworld1 | mathworld1_prband2cf_verify.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_prband2desk.py | library | RESULTS | RESULTS×7 | 1 | 1 | — | — |
| mathworld1 | mathworld1_prband2desk_verify.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_prband2fresh.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_prband2freshagg.py | UNCITED | — | — | — | — | — | — |
| mathworld1 | mathworld1_prband2freshfreeze.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_prband2freshprompts.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_prband2freshverify.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_prband2nuis.py | library | RESULTS | RESULTS×8 | 1 | 1 | — | — |
| mathworld1 | mathworld1_prband2prod.py | results-cited | RESULTS | RESULTS×6 | — | — | — | — |
| mathworld1 | mathworld1_prband2prod_verify.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_prband2score.py | library | RESULTS | RESULTS×19 | 7 | 5 | — | — |
| mathworld1 | mathworld1_prband2score_verify.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| mathworld1 | mathworld1_prband_anat.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_prband_hce.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_prband_verify.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_regret.py | results-cited | RESULTS | RESULTS×5 | — | — | 1 | — |
| mathworld1 | mathworld1_regret_walllift.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_respath.py | library | RESULTS | RESULTS×20 | 5 | 6 | — | — |
| mathworld1 | mathworld1_respath20.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_retrolabel.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_rolecensus.py | results-cited | RESULTS | RESULTS×3 | — | 1 | — | — |
| mathworld1 | mathworld1_rolecensusverify.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_scoreqal.py | results-cited | RESULTS | RESULTS×5 | — | — | — | — |
| mathworld1 | mathworld1_srepr_export.py | library | RESULTS | RESULTS×25 | 9 | 19 | — | — |
| mathworld1 | mathworld1_stateobs.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_substrate_desk.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_svpadj.py | library | RESULTS | RESULTS×25 | 15 | 18 | — | — |
| mathworld1 | mathworld1_svpbirth.py | library | RESULTS | RESULTS×75 | 65 | 53 | 2 | — |
| mathworld1 | mathworld1_svpcalscore.py | results-cited | RESULTS | RESULTS×5 | — | — | — | — |
| mathworld1 | mathworld1_svpcalscore16.py | results-cited | RESULTS | RESULTS×5 | — | 2 | — | — |
| mathworld1 | mathworld1_svpcalscore17.py | results-cited | RESULTS | RESULTS×7 | — | 2 | — | — |
| mathworld1 | mathworld1_svpcalscore18.py | results-cited | RESULTS | RESULTS×8 | — | 3 | — | — |
| mathworld1 | mathworld1_svpchal.py | library | RESULTS | RESULTS×32 | 17 | 16 | — | — |
| mathworld1 | mathworld1_svpchal2.py | library | RESULTS | RESULTS×10 | 7 | 8 | 1 | — |
| mathworld1 | mathworld1_svpchalscore.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_svpcode.py | library | RESULTS | RESULTS×54 | 45 | 46 | — | — |
| mathworld1 | mathworld1_svpcovdesk.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_svpdcl.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_svpdesign.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_svpdiet.py | library | RESULTS | RESULTS×13 | 7 | 6 | — | — |
| mathworld1 | mathworld1_svpdiet2.py | library | RESULTS | RESULTS×12 | 5 | 6 | — | — |
| mathworld1 | mathworld1_svpdiet3.py | library | RESULTS | RESULTS×17 | 4 | 9 | — | — |
| mathworld1 | mathworld1_svpeval.py | library | RESULTS | RESULTS×26 | 7 | 16 | — | — |
| mathworld1 | mathworld1_svpeval2.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_svpeval3.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_svpfbcredit.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_svpfhadj.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_svpfhbirth.py | results-cited | RESULTS | RESULTS×4 | — | 1 | — | — |
| mathworld1 | mathworld1_svpfocal.py | results-cited | RESULTS | RESULTS×9 | — | 5 | — | — |
| mathworld1 | mathworld1_svpfoclrepl.py | results-cited | RESULTS | RESULTS×4 | — | 1 | — | — |
| mathworld1 | mathworld1_svpfofresh.py | results-cited | RESULTS | RESULTS×9 | — | 2 | — | — |
| mathworld1 | mathworld1_svpfoheld.py | library | RESULTS | RESULTS×10 | 1 | 5 | — | — |
| mathworld1 | mathworld1_svpfohrepl.py | library | RESULTS | RESULTS×3 | 1 | 2 | — | — |
| mathworld1 | mathworld1_svpforder.py | library | RESULTS | RESULTS×25 | 19 | 17 | 1 | — |
| mathworld1 | mathworld1_svpforepl.py | library | RESULTS | RESULTS×25 | 1 | 4 | — | — |
| mathworld1 | mathworld1_svpfreeact.py | results-cited | RESULTS | RESULTS×3 | — | 1 | — | — |
| mathworld1 | mathworld1_svpgbirth.py | results-cited | RESULTS | RESULTS×5 | — | 1 | — | — |
| mathworld1 | mathworld1_svpgbirth16.py | results-cited | RESULTS | RESULTS×11 | — | 3 | — | — |
| mathworld1 | mathworld1_svpgbirth17.py | results-cited | RESULTS | RESULTS×7 | — | 2 | — | — |
| mathworld1 | mathworld1_svpgbirth18.py | results-cited | RESULTS | RESULTS×7 | — | 2 | — | — |
| mathworld1 | mathworld1_svpgenadj.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_svpgeodesk.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_svpgriddesk.py | library | RESULTS | RESULTS×5 | 3 | 3 | — | — |
| mathworld1 | mathworld1_svpgriddesk2.py | library | RESULTS | RESULTS×4 | 1 | 2 | — | — |
| mathworld1 | mathworld1_svpgriddesk3.py | results-cited | RESULTS | RESULTS×3 | — | 1 | 1 | — |
| mathworld1 | mathworld1_svpgriddesk4.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| mathworld1 | mathworld1_svpheldout16.py | results-cited | RESULTS | RESULTS×4 | — | 1 | — | — |
| mathworld1 | mathworld1_svpheldout17.py | results-cited | RESULTS | RESULTS×6 | — | 1 | — | — |
| mathworld1 | mathworld1_svpheldout18.py | results-cited | RESULTS | RESULTS×5 | — | 1 | — | — |
| mathworld1 | mathworld1_svpldesk.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_svpnuisdesk.py | library | RESULTS | RESULTS×10 | 4 | 4 | — | — |
| mathworld1 | mathworld1_svpp2qual.py | library | RESULTS | RESULTS×7 | 5 | 5 | — | — |
| mathworld1 | mathworld1_svppoutscore16.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_svprep.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_svpsemdesk.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_svpsuppdesk.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_svptokdesk.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mathworld1 | mathworld1_svptokonset.py | results-cited | RESULTS | RESULTS×3 | — | 2 | — | — |
| mathworld1 | mathworld1_terminal.py | UNCITED | — | — | — | — | — | — |
| mathworld1 | mathworld1_transposition.py | results-cited | RESULTS | RESULTS×5 | — | 1 | — | — |
| mathworld1 | mathworld1_transpositionverify.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_unprod_probe.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| mathworld1 | mathworld1_unprodsem.py | library | RESULTS | RESULTS×17 | 2 | 15 | — | — |
| mathworld1 | mathworld1_yield.py | UNCITED | — | — | — | — | — | — |
| mathworld1 | mathworld1_zdpdesk.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| matryoshka | matryoshka_r1.py | results-cited | RESULTS, specs | RESULTS×3, specs×2 | — | — | — | — |
| matryoshka | matryoshka_r2.py | spec-cited | specs | specs×1 | — | — | — | — |
| merge | merge_space1.sh | UNCITED | — | — | — | — | — | — |
| merge | merge_space2.sh | UNCITED | — | — | — | — | — | — |
| merge | merge_space3.sh | UNCITED | — | — | — | — | — | — |
| merge | merge_space4.sh | UNCITED | — | — | — | — | — | — |
| merge | merge_space5.sh | UNCITED | — | — | — | — | — | — |
| metabolic | metabolic_d2.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| metabolic | metabolic_hot.py | results-cited | RESULTS | RESULTS×1 | — | 1 | — | — |
| metabolic | metabolic_v3.py | spec-cited | specs | specs×1 | — | — | — | — |
| metabolic | metabolic_v4.py | UNCITED | — | — | — | — | — | — |
| metabolic | metabolic_v5.py | results-cited | RESULTS, specs | RESULTS×3, specs×2 | — | 2 | — | — |
| metallicity | metallicity_diets.py | results-cited | RESULTS | RESULTS×3 | — | 1 | — | — |
| metallicity1 | metallicity1.sh | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| mezo | mezo_signal_desk.py | library | RESULTS | RESULTS×6 | 1 | 1 | — | — |
| mezo0 | mezo0_launch.sh | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| moe | moe_gt1.py | library | REPRODUCE, RESULTS, specs, TOOLING | REPRODUCE×4, RESULTS×11, specs×7, TOOLING×8 | 4 | 6 | 2 | — |
| moe | moe_gt1_arm2.py | library | REPRODUCE, RESULTS, specs, TOOLING | REPRODUCE×6, RESULTS×47, specs×7, TOOLING×31 | 16 | 26 | 2 | — |
| morning | morning_run.sh | UNCITED | — | — | — | — | — | — |
| mps | mps_sigma_gates.sh | UNCITED | — | — | — | — | — | — |
| muon | muon_3ep_d256.py | results-cited | via:RESULTS | — | — | 1 | — | night_28_mac.sh |
| night | night_28.sh | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| night | night_28_mac.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| night | night_45m_union.sh | results-cited | RESULTS, specs | RESULTS×3, specs×1 | — | 1 | 1 | — |
| night | night_calib.sh | UNCITED | — | — | — | — | — | — |
| night | night_g9.sh | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| night | night_gates.sh | UNCITED | — | — | — | — | — | — |
| night | night_rat.sh | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| night | night_rat_s2.sh | spec-cited | specs | specs×1 | — | — | — | — |
| night | night_sr.sh | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| night | night_zx.sh | results-cited | RESULTS, specs | RESULTS×2, specs×2 | — | — | — | — |
| night | night_zx2.sh | results-cited | RESULTS | RESULTS×2 | — | 1 | — | — |
| night | night_zx3.sh | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| night | night_zx45_x2.sh | results-cited | RESULTS | RESULTS×2 | — | — | 1 | — |
| night2 | night2_mac.sh | results-cited | RESULTS | RESULTS×1 | — | — | 1 | — |
| night2 | night2_mac_shift2.sh | UNCITED | — | — | — | — | — | — |
| night28b | night28b.sh | UNCITED | — | — | — | — | — | — |
| night29 | night29.sh | spec-cited | specs | specs×1 | — | — | — | — |
| night29b | night29b.sh | spec-cited | specs | specs×1 | — | — | — | — |
| night30 | night30.sh | spec-cited | specs | specs×1 | — | — | — | — |
| night30 | night30_mac.py | UNCITED | — | — | — | — | — | — |
| night30b | night30b.sh | UNCITED | — | — | — | — | — | — |
| night31 | night31.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| night31 | night31_cuda.sh | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| night31 | night31_mac.sh | UNCITED | — | — | — | — | — | — |
| night31b | night31b_cuda.sh | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — | — | — |
| nineteen | nineteen_m_displace.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — | — | — |
| ogd0 | ogd0_launch.sh | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| oma1 | oma1_launch.sh | results-cited | RESULTS, specs | RESULTS×5, specs×1 | — | — | — | — |
| onecycle | onecycle_component_audit.py | library | RESULTS, via:specs | RESULTS×32 | 1 | 2 | — | — |
| optimizer | optimizer_geometry_desk.py | library | RESULTS, specs | RESULTS×35, specs×1 | 2 | 4 | — | — |
| optimizer | optimizer_memory_ablation.py | library | RESULTS, specs | RESULTS×46, specs×1 | 3 | 3 | 1 | — |
| oracle | oracle_worker.py | library | REPRODUCE, RESULTS, specs, via:TOOLING | REPRODUCE×1, RESULTS×3, specs×7 | 1 | 2 | 5 | — |
| ozaki | ozaki_2b_bisect.py | spec-cited | specs, TOOLING | specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_2b_check.py | results-cited | RESULTS, specs, TOOLING | RESULTS×1, specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_2b_debug.py | spec-cited | specs, TOOLING | specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_2b_ident.py | spec-cited | specs, TOOLING | specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_cuda.py | spec-cited | specs, TOOLING | specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_cuda2.py | results-cited | RESULTS, specs, TOOLING | RESULTS×2, specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_cuda3.py | results-cited | RESULTS, specs, TOOLING | RESULTS×2, specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_cuda4.py | results-cited | RESULTS, specs, TOOLING | RESULTS×2, specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_cuda5.py | results-cited | RESULTS, specs, TOOLING | RESULTS×2, specs×4, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_cuda6.py | results-cited | RESULTS, specs, TOOLING | RESULTS×2, specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_fused.py | results-cited | RESULTS, specs, TOOLING | RESULTS×3, specs×4, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_rung1.py | results-cited | RESULTS, specs, TOOLING | RESULTS×2, specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_rung1b.py | results-cited | RESULTS, specs, TOOLING | RESULTS×4, specs×3, TOOLING×1 | — | — | — | — |
| ozaki | ozaki_rung2bc.py | results-cited | RESULTS, specs, TOOLING | RESULTS×1, specs×5, TOOLING×1 | — | 4 | — | — |
| p2 | p2_crown_draws.py | UNCITED | — | — | — | — | — | — |
| p3 | p3_autopsy.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| p3 | p3_bits.py | results-cited | RESULTS, specs | RESULTS×6, specs×1 | — | — | — | — |
| p3 | p3_ffnslack.py | results-cited | RESULTS, specs | RESULTS×6, specs×1 | — | — | — | — |
| p3 | p3_grav2.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| p3 | p3_quat.py | results-cited | RESULTS, specs | RESULTS×2, specs×2 | — | — | — | — |
| p3 | p3_stream2x2.py | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — | — | — |
| p3 | p3_umoe_soft.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| p4 | p4_arms_0801.sh | library | RESULTS, specs | RESULTS×5, specs×11 | — | 2 | — | — |
| pack | pack_baselines.py | spec-cited | specs | specs×1 | — | — | — | — |
| pack | pack_c6.py | library | RESULTS, specs | RESULTS×2, specs×1 | 4 | — | — | — |
| pack | pack_c7.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| pack | pack_crystal.py | results-cited | RESULTS, specs | RESULTS×4, specs×3 | — | — | — | — |
| pack | pack_decode.py | library | RESULTS, specs | RESULTS×7, specs×3 | 1 | — | 1 | — |
| pack | pack_determinism.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| pack | pack_gemv.py | spec-cited | specs | specs×1 | — | — | — | — |
| pack | pack_p2a.py | UNCITED | — | — | — | — | — | — |
| pack | pack_rans.py | results-cited | RESULTS, specs | RESULTS×7, specs×3 | — | — | — | — |
| pack | pack_tiered.py | spec-cited | specs | specs×1 | — | — | — | — |
| paper | paper_figs.py | UNCITED | — | — | — | — | — | — |
| perturbation | perturbation_response_gram_desk.py | library | RESULTS, specs | RESULTS×4, specs×1 | 1 | 2 | 1 | — |
| phase | phase_portrait_precompute.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | — | — | — |
| phase4 | phase4_rewrite.py | spec-cited | specs | specs×1 | — | — | — | — |
| phase4 | phase4_sites.py | spec-cited | specs | specs×3 | — | — | 1 | — |
| phase4 | phase4_unboot.py | UNCITED | — | — | — | — | 1 | — |
| phase5 | phase5_deadcode.py | spec-cited | specs | specs×1 | — | — | — | — |
| phys | phys_probe.py | UNCITED | — | — | — | — | — | — |
| pincer | pincer_dist_probe.py | results-cited | RESULTS, specs | RESULTS×2, specs×2 | — | — | — | — |
| pincer | pincer_dist_report.py | results-cited | via:RESULTS, via:specs | — | — | 1 | — | pincer_dist_probe.py |
| pincer | pincer_labels_v2.py | library | RESULTS, specs | RESULTS×2, specs×1 | 1 | — | — | — |
| pincer | pincer_r0.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| pincer | pincer_r0b.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| pincer | pincer_r1_indist.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| pincer | pincer_r1_probe.py | results-cited | RESULTS | RESULTS×2 | — | — | 1 | — |
| pincer | pincer_r1b_labels.py | library | RESULTS, specs | RESULTS×2, specs×3 | — | 1 | 1 | — |
| pincer | pincer_r8.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| place1 | place1_gravity.py | results-cited | RESULTS, specs | RESULTS×5, specs×2 | — | — | — | — |
| polar | polar_snap.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| poly3 | poly3_pipeline.sh | spec-cited | specs | specs×2 | — | — | 1 | — |
| poly4 | poly4_pipeline.sh | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 1 | 1 | — |
| poly4 | poly4_watcher.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| poly5 | poly5_pipeline.sh | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | 1 | 1 | — |
| poly5 | poly5_watcher.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| practice | practice_mine.py | spec-cited | specs | specs×2 | — | — | — | — |
| prband2fresh | prband2fresh_train.sh | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| prefix | prefix_pair.sh | UNCITED | — | — | — | — | — | — |
| prgd0 | prgd0_launch.sh | library | RESULTS, specs | RESULTS×4, specs×2 | — | 1 | — | — |
| probe | probe_int_device_parity.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| prologue | prologue_arms.py | library | — | — | 1 | — | — | — |
| prologue | prologue_gates.sh | UNCITED | — | — | — | — | — | — |
| ptq4 | ptq4_arms.py | UNCITED | — | — | — | — | — | — |
| ptq4 | ptq4_gates.sh | UNCITED | — | — | — | 1 | — | — |
| qcuda | qcuda_tower_qualify.py | UNCITED | — | — | — | — | — | — |
| quat | quat_commutant.py | library | RESULTS, specs | RESULTS×2, specs×3 | 1 | — | — | — |
| quat | quat_convert.py | library | RESULTS, specs | RESULTS×5, specs×3 | 1 | — | — | — |
| quick | quick_exact_3080.sh | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| qwen | qwen_alttok_derive.py | results-cited | RESULTS, specs | RESULTS×3, specs×2 | — | 1 | — | — |
| qwen | qwen_attrib_adjudicate.py | library | specs | specs×2 | 1 | 1 | 1 | — |
| qwen | qwen_ble2_adjudicate.py | library | RESULTS, specs | RESULTS×5, specs×2 | 1 | 1 | — | — |
| qwen | qwen_ble2_autopsy.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_blem_perf.py | results-cited | RESULTS | RESULTS×4 | — | 1 | — | — |
| qwen | qwen_blem_perf2.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_blem_perf3.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| qwen | qwen_capacity27b.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| qwen | qwen_census_night.sh | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_cheap_readout.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| qwen | qwen_cuda_rung0.py | UNCITED | — | — | — | — | — | — |
| qwen | qwen_cuda_rung1.py | UNCITED | — | — | — | — | — | — |
| qwen | qwen_cuda_rung2.py | UNCITED | — | — | — | — | — | — |
| qwen | qwen_cuda_rung3.py | UNCITED | — | — | — | — | — | — |
| qwen | qwen_cuda_rung4.py | library | RESULTS, specs | RESULTS×9, specs×2 | 1 | 4 | 1 | — |
| qwen | qwen_cycle_impulse.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_displace_extract.py | spec-cited | specs | specs×1 | — | — | 1 | — |
| qwen | qwen_effort_probe.py | results-cited | RESULTS, via:specs | RESULTS×7 | — | 9 | — | — |
| qwen | qwen_effort_quant.py | results-cited | RESULTS | RESULTS×7 | — | — | 1 | — |
| qwen | qwen_effort_tower.py | results-cited | RESULTS, via:specs | RESULTS×3 | — | 1 | — | — |
| qwen | qwen_family_probe.py | library | RESULTS, via:specs | RESULTS×2 | 1 | 1 | — | — |
| qwen | qwen_headswap_impulse.py | results-cited | RESULTS, specs | RESULTS×5, specs×1 | — | — | — | — |
| qwen | qwen_homeo_actuator.py | results-cited | RESULTS | RESULTS×8 | — | 3 | — | — |
| qwen | qwen_homeo_adjudicate.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| qwen | qwen_hsimpulse_adjudicate.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| qwen | qwen_hsimpulse_color.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_ioattrib_adjudicate.py | library | RESULTS, specs | RESULTS×2, specs×1 | 1 | 1 | — | — |
| qwen | qwen_lband_adjudicate.py | library | RESULTS | RESULTS×2 | 1 | 1 | — | — |
| qwen | qwen_loop_state.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| qwen | qwen_loop_state_adjudicate.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_loop_state_color2.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_loop_state_headswap.py | results-cited | RESULTS, via:specs | RESULTS×7 | — | 2 | — | — |
| qwen | qwen_margin_census.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_mips_census.py | results-cited | RESULTS | RESULTS×5 | — | — | — | — |
| qwen | qwen_model1_score.py | library | RESULTS, specs | RESULTS×5, specs×2 | — | 3 | 3 | — |
| qwen | qwen_model2_adjudicate.py | library | RESULTS | RESULTS×2 | 1 | 1 | — | — |
| qwen | qwen_qualify.py | spec-cited | specs, TOOLING | specs×1, TOOLING×2 | — | 1 | 1 | — |
| qwen | qwen_recompose.py | library | RESULTS, specs | RESULTS×1, specs×1 | 1 | 1 | 1 | — |
| qwen | qwen_residual_census.py | library | RESULTS | RESULTS×4 | 1 | 1 | — | — |
| qwen | qwen_residual_loo.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_rk_adjudicate.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_rk_census.py | results-cited | RESULTS | RESULTS×2 | — | — | 1 | — |
| qwen | qwen_rk_rider.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_runtime0r.py | library | RESULTS | RESULTS×1 | 1 | 1 | 2 | — |
| qwen | qwen_stream_probe.py | library | RESULTS, via:specs | RESULTS×2 | 2 | 3 | — | — |
| qwen | qwen_teacher_pass.py | library | RESULTS | RESULTS×2 | 2 | 1 | — | — |
| qwen | qwen_teacher_sidecar.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| qwen | qwen_topset_census.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| qwen | qwen_tower_ladder.py | results-cited | RESULTS, via:specs | RESULTS×18 | — | 11 | — | — |
| qwen | qwen_tree_adjudicate.py | library | RESULTS, specs | RESULTS×2, specs×1 | — | 1 | — | — |
| qwen | qwen_whole0t.py | library | RESULTS, specs | RESULTS×3, specs×1 | 1 | 1 | 2 | — |
| random | random_direction_control.py | library | RESULTS, specs | RESULTS×10, specs×3 | 1 | 2 | — | — |
| rank | rank_read.py | spec-cited | specs | specs×2 | — | — | 1 | — |
| rat | rat_deploy.py | results-cited | RESULTS, specs | RESULTS×3, specs×7 | — | 2 | 1 | — |
| rat | rat_repair.py | results-cited | via:RESULTS, via:specs | — | — | 1 | — | quick_exact_3080.sh |
| rational | rational_snap.py | results-cited | RESULTS, specs | RESULTS×6, specs×1 | — | 4 | 2 | — |
| rdc1 | rdc1_launch.sh | results-cited | RESULTS, specs | RESULTS×12, specs×4 | — | — | — | — |
| retention | retention_watcher.sh | UNCITED | — | — | — | — | — | — |
| rev2 | rev2_d768.py | results-cited | RESULTS, specs | RESULTS×3, specs×1 | — | — | — | — |
| rev3 | rev3_crown.py | results-cited | RESULTS, specs | RESULTS×7, specs×3 | — | — | — | — |
| rev4 | rev4_zx45.py | results-cited | RESULTS | RESULTS×2 | — | 1 | — | — |
| rot | rot_commutant.py | library | RESULTS, specs | RESULTS×1, specs×1 | 3 | — | — | — |
| rot | rot_convert.py | spec-cited | specs | specs×1 | — | — | — | — |
| rot | rot_snap_anatomy.py | UNCITED | — | — | — | — | — | — |
| rotinstr | rotinstr_control.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| routedb | routedb_basis.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| routedb | routedb_replay.py | library | RESULTS | RESULTS×8 | 3 | 4 | — | — |
| routedb | routedb_replay2.py | results-cited | RESULTS | RESULTS×9 | — | 1 | — | — |
| routedb | routedb_time.py | library | RESULTS | RESULTS×5 | 1 | 1 | — | — |
| routedb | routedb_time0r.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| ruleablate1 | ruleablate1_driver.sh | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — | — | — |
| rulepolicy0 | rulepolicy0_census.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| run | run_snap_gates.sh | spec-cited | specs | specs×1 | — | — | — | — |
| run | run_snap_knee.sh | UNCITED | — | — | — | — | — | — |
| saturation | saturation_s2.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| saturation | saturation_s2b.py | spec-cited | specs | specs×1 | — | — | — | — |
| scaffold | scaffold_review.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| scorer | scorer_s1_battery.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| scorer | scorer_s2_data.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| scorer | scorer_s2_train.py | spec-cited | specs | specs×1 | — | — | — | — |
| seed | seed_audit.py | spec-cited | specs | specs×2 | — | — | — | — |
| seeds | seeds_ladder_0804.sh | UNCITED | — | — | — | — | — | — |
| series | series_probe.py | results-cited | specs, via:RESULTS | specs×6 | — | 13 | 2 | poly4_pipeline.sh |
| sg | sg_credit.py | library | RESULTS, specs | RESULTS×12, specs×3 | 9 | 4 | — | — |
| sg | sg_crosspos_desk.py | library | RESULTS, via:specs | RESULTS×12 | 1 | 2 | 1 | — |
| sg | sg_failure_desk.py | library | RESULTS, via:specs | RESULTS×7 | 2 | 2 | — | — |
| sg | sg_failure_labels.py | library | RESULTS, specs | RESULTS×2, specs×1 | 1 | 1 | — | — |
| sg | sg_integrity_smoke.py | results-cited | RESULTS, specs | RESULTS×7, specs×3 | — | 1 | 1 | — |
| sg | sg_qualgate.py | library | RESULTS, specs | RESULTS×7, specs×1 | 1 | 3 | — | — |
| sg | sg_target_audit.py | library | RESULTS | RESULTS×5 | 1 | 1 | — | — |
| sg7 | sg7_credit.py | library | RESULTS, specs | RESULTS×5, specs×1 | 3 | 2 | — | — |
| sg7 | sg7_integrity_smoke.py | results-cited | RESULTS, specs | RESULTS×3, specs×1 | — | 1 | — | — |
| sg7 | sg7_qualgate.py | library | RESULTS, specs | RESULTS×4, specs×1 | 1 | 3 | — | — |
| sg7 | sg7_scale_census.py | results-cited | RESULTS, specs | RESULTS×3, specs×1 | — | — | — | — |
| sgbb7 | sgbb7_keepset_prune.py | UNCITED | — | — | — | — | — | — |
| sgbb7 | sgbb7_launch.sh | UNCITED | — | — | — | — | — | — |
| sgbb7 | sgbb7_qual_driver.sh | library | RESULTS, specs | RESULTS×5, specs×2 | — | 2 | — | — |
| sgbb7 | sgbb7_smoke_prune.py | UNCITED | — | — | — | — | — | — |
| sgwriter1 | sgwriter1_keepset_prune.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| sgwriter1 | sgwriter1_qual_driver.sh | results-cited | RESULTS, specs | RESULTS×11, specs×2 | — | — | — | — |
| sgxpos0 | sgxpos0_launch.sh | spec-cited | specs | specs×1 | — | — | — | — |
| smoke | smoke_prune.py | UNCITED | — | — | — | — | 1 | — |
| snap | snap_alloc.py | spec-cited | specs | specs×2 | — | — | 1 | — |
| snap | snap_anatomy.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| softnext1 | softnext1_driver.sh | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| softprompt | softprompt_sampler_probe.py | spec-cited | specs, TOOLING | specs×1, TOOLING×2 | — | 1 | — | — |
| softprompt1 | softprompt1.py | library | RESULTS, specs, via:TOOLING | RESULTS×6, specs×2 | 1 | 2 | — | — |
| softprompt1 | softprompt1.sh | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| softspeed1 | softspeed1_driver.sh | results-cited | RESULTS, specs | RESULTS×4, specs×1 | — | — | — | — |
| soup | soup_gate.py | spec-cited | specs | specs×1 | — | 1 | — | — |
| ssm | ssm_star.py | library | RESULTS, via:TOOLING | RESULTS×2 | 1 | 1 | — | — |
| ssm | ssm_star1.sh | UNCITED | — | — | — | — | — | — |
| stability | stability_atlas.sh | UNCITED | — | — | — | — | — | — |
| star | star_profile.py | results-cited | RESULTS | RESULTS×7 | — | — | — | — |
| stream | stream_wdistill0.py | results-cited | RESULTS | RESULTS×4 | — | — | 1 | — |
| stream | stream_wdistill0s.py | library | RESULTS, via:specs | RESULTS×1 | 2 | 2 | — | — |
| stream | stream_wdistill1.py | library | RESULTS, via:specs | RESULTS×8 | 2 | — | 1 | — |
| streaming | streaming_birth_d256.py | library | RESULTS, via:specs | RESULTS×5 | 1 | — | — | — |
| streamwd | streamwd_complete.py | UNCITED | — | — | — | — | — | — |
| streamwd | streamwd_v2.py | results-cited | RESULTS, specs | RESULTS×2, specs×3 | — | 1 | — | — |
| successors | successors_acceptance.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| sym | sym_birth.py | library | RESULTS, specs | RESULTS×18, specs×7 | 1 | 8 | — | — |
| sym | sym_convert.py | results-cited | RESULTS, specs | RESULTS×2, specs×4 | — | — | — | — |
| sym | sym_spectrum.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — | — | — |
| sym45 | sym45.py | spec-cited | specs | specs×1 | — | 1 | — | — |
| sym45 | sym45_run.sh | UNCITED | — | — | — | — | — | — |
| synonym | synonym_test.py | tool-referenced | TOOLING | TOOLING×1 | — | 1 | — | — |
| tenet | tenet_d1_revgate.py | results-cited | RESULTS, specs | RESULTS×2, specs×2 | — | — | 1 | — |
| tenet | tenet_d2_revdiet.py | library | RESULTS, specs, via:TOOLING | RESULTS×35, specs×1 | 35 | 1 | — | — |
| tenet | tenet_d3_budget.py | library | specs | specs×2 | 2 | — | — | — |
| tenet | tenet_mult_b32.py | results-cited | RESULTS, TOOLING | RESULTS×2, TOOLING×3 | — | 1 | — | — |
| tenet | tenet_mult_census.py | library | RESULTS, specs, TOOLING | RESULTS×2, specs×1, TOOLING×3 | 1 | 1 | — | — |
| tenet | tenet_r1b_micro.py | UNCITED | — | — | — | — | — | — |
| tenet | tenet_w0.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| tenet | tenet_w1_bridge.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | 1 | — |
| tenet | tenet_w1_population.py | UNCITED | — | — | — | — | — | — |
| tenet | tenet_w1_relational.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| tenet | tenet_w1_surfaces.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| ternary | ternary_control.py | UNCITED | — | — | — | — | — | — |
| ternary | ternary_gate.py | UNCITED | — | — | — | — | — | — |
| ternary | ternary_session2.py | UNCITED | — | — | — | 1 | — | — |
| tier | tier_escalate.py | UNCITED | — | — | — | — | — | — |
| tier | tier_retry.py | spec-cited | specs | specs×1 | — | — | — | — |
| train | train_fp64.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| traj | traj_accept.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| tuesday | tuesday_night.sh | UNCITED | — | — | — | — | — | — |
| ugc0 | ugc0_launch.sh | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| umoe | umoe_conserve.py | library | RESULTS, via:specs | RESULTS×4 | 9 | 2 | 1 | — |
| update | update_geometry_census.py | library | RESULTS, specs | RESULTS×37, specs×2 | 4 | 4 | — | — |
| v4flash | v4flash_anatomy.py | spec-cited | specs | specs×1 | — | — | — | — |
| v4flash | v4flash_census.py | results-cited | RESULTS, specs | RESULTS×2, specs×2 | — | — | 1 | — |
| v4flash | v4flash_f1b.py | library | RESULTS, specs, via:TOOLING | RESULTS×2, specs×1 | 2 | — | — | — |
| v4flash | v4flash_f1c.py | library | RESULTS, specs, via:TOOLING | RESULTS×4, specs×1 | 1 | — | 1 | — |
| v4flash | v4flash_f1d.py | results-cited | RESULTS, specs, TOOLING | RESULTS×8, specs×2, TOOLING×4 | — | 2 | — | — |
| v4flash | v4flash_header.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| v4flash | v4flash_router.py | library | RESULTS, specs | RESULTS×2, specs×2 | 3 | — | — | — |
| v4flash | v4flash_rung0.py | results-cited | RESULTS, specs | RESULTS×2, specs×2 | — | — | — | — |
| v4flash | v4flash_rung2b.py | library | RESULTS, specs | RESULTS×2, specs×3 | 1 | — | — | — |
| v4flash | v4flash_rung2b_router.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| v4flash | v4flash_rungA.py | library | RESULTS, specs, via:TOOLING | RESULTS×4, specs×4 | 8 | — | 1 | — |
| v4flash | v4flash_rungd.py | library | RESULTS, specs | RESULTS×5, specs×1 | 1 | — | — | — |
| v4flash | v4flash_rungd2.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| v4flash | v4flash_s0.py | results-cited | RESULTS, specs | RESULTS×5, specs×1 | — | — | — | — |
| v4flash | v4flash_twin.py | library | RESULTS, specs, via:TOOLING | RESULTS×8, specs×1 | 3 | — | — | — |
| verify | verify_intbirth_prims.py | results-cited | RESULTS, specs | RESULTS×4, specs×2 | — | — | — | — |
| vmasm | vmasm.py | library | — | — | 1 | — | — | — |
| vmasm | vmasm_probe.py | UNCITED | — | — | — | — | — | — |
| vrm | vrm_ab.py | UNCITED | — | — | — | 1 | — | — |
| weight | weight_fft_euler.py | UNCITED | — | — | — | — | — | — |
| wfloor | wfloor_ladder.sh | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — | — | — |
| writercaf1 | writercaf1_qual_driver.sh | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| writercaf1 | writercaf1_smoke.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| writerdfa1 | writerdfa1_disc_driver.sh | UNCITED | — | — | — | — | — | — |
| writerdfa1 | writerdfa1_post_driver.sh | UNCITED | — | — | — | — | — | — |
| writerdfa1 | writerdfa1_qual_driver.sh | UNCITED | — | — | — | — | — | — |
| writertraj | writertraj_census.py | library | RESULTS, via:specs | RESULTS×8 | 2 | 2 | — | — |
| writertraj | writertraj_depend.py | library | RESULTS, specs | RESULTS×19, specs×1 | 1 | 4 | — | — |
| writertraj | writertraj_verify.py | results-cited | RESULTS | RESULTS×10 | — | 1 | — | — |
| writertraj0 | writertraj0_driver.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| wsl | wsl.sh | library | RESULTS, specs, TOOLING | RESULTS×15, specs×30, TOOLING×18 | — | 8 | 1 | — |
| xterm | xterm_probe.py | library | RESULTS, specs, via:TOOLING | RESULTS×4, specs×1 | 1 | 1 | — | — |
| xtermdiet1 | xtermdiet1_driver.sh | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| z1 | z1_gate.sh | UNCITED | — | — | — | — | — | — |
| z1s | z1s_hot_watcher.sh | UNCITED | — | — | — | — | — | — |
| zx | zx_chain.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| zx | zx_chain_cuda.sh | UNCITED | — | — | — | — | — | — |
| zx | zx_gate_watcher.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |

## scripts/

| family | file | class | cited by | doc citations | imports | invoked by | mentions | via |
|---|---|---|---|---|---|---|---|---|
|  | __init__.py | spec-cited | specs | specs×2 | — | — | 1 | — |
| adjudicate | adjudicate.py | library | RESULTS, specs, TOOLING | RESULTS×16, specs×3, TOOLING×2 | — | 2 | 4 | — |
| anchor | anchor_guard.py | spec-cited | specs | specs×2 | — | — | 1 | — |
| anim | anim_precompute.py | spec-cited | specs | specs×4 | — | — | 2 | — |
| arena | arena.py | UNCITED | — | — | — | — | — | — |
| arena | arena_qwen.py | spec-cited | specs | specs×2 | — | — | — | — |
| autopsy | autopsy_int.py | UNCITED | — | — | — | — | — | — |
| backfill | backfill_code_commit.py | spec-cited | specs | specs×3 | — | — | — | — |
| bench | bench_adaptive.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | — | — |
| bench | bench_adaptive_draft.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_anneal.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_ansatz_search.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_ansatz_search_2b.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_bandit.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_bestfirst.py | results-cited | RESULTS | RESULTS×1 | — | — | 1 | — |
| bench | bench_bestfirst_llm.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_bestfirst_nnue.py | results-cited | RESULTS | RESULTS×1 | — | — | 1 | — |
| bench | bench_budget_alloc.py | results-cited | RESULTS, TOOLING | RESULTS×2, TOOLING×1 | — | — | — | — |
| bench | bench_commute.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_compile.py | results-cited | RESULTS | RESULTS×1 | — | — | 4 | — |
| bench | bench_control.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_decoding.py | library | RESULTS, via:TOOLING | RESULTS×1 | 7 | — | 1 | — |
| bench | bench_derivation.py | results-cited | RESULTS, specs | RESULTS×1, specs×15 | — | — | — | — |
| bench | bench_dispatch_race_v4.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_distilled_draft.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_engine_regret.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| bench | bench_entropy_beam.py | results-cited | RESULTS | RESULTS×1 | — | — | 1 | — |
| bench | bench_fib_restarts.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_flash_prefill.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_frontier.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | — | — |
| bench | bench_fused.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_fused_ce.py | results-cited | RESULTS | RESULTS×3 | — | — | 1 | — |
| bench | bench_gated.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_gweight.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_hints_ab.py | results-cited | RESULTS, TOOLING | RESULTS×2, TOOLING×1 | — | — | — | — |
| bench | bench_hybrid.py | library | RESULTS | RESULTS×1 | 1 | — | — | — |
| bench | bench_int4_config_sweep.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_int4_gemv.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| bench | bench_interference.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_ksweep.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — | — | — |
| bench | bench_kv_quant_decode.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | 1 | — |
| bench | bench_ladder.py | results-cited | RESULTS, specs | RESULTS×1, specs×3 | — | — | — | — |
| bench | bench_lazy.py | library | RESULTS | RESULTS×3 | 1 | — | — | — |
| bench | bench_llm_gating.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_lookup_static.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_luby.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_magic.py | library | RESULTS, specs | RESULTS×3, specs×1 | 1 | — | — | — |
| bench | bench_markov.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_markov_adaptive.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_metal_kernels.py | results-cited | RESULTS, specs | RESULTS×1, specs×8 | — | — | 2 | — |
| bench | bench_mlx_integration.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_nnue.py | results-cited | RESULTS, specs | RESULTS×1, specs×8 | — | — | 1 | — |
| bench | bench_ode_engine.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_opcap.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | — | — |
| bench | bench_population.py | results-cited | RESULTS | RESULTS×1 | — | — | 1 | — |
| bench | bench_pred_syndromes.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| bench | bench_prefix_reuse.py | library | RESULTS, via:TOOLING | RESULTS×1 | 1 | — | — | — |
| bench | bench_proposer.py | results-cited | RESULTS, specs | RESULTS×1, specs×10 | — | — | 1 | — |
| bench | bench_quant_schemes.py | results-cited | RESULTS | RESULTS×3 | — | — | 1 | — |
| bench | bench_record.py | results-cited | RESULTS, specs | RESULTS×3, specs×1 | — | — | — | — |
| bench | bench_regret_resample.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_rotate_quantize.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — | 1 | — |
| bench | bench_rule_basis.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| bench | bench_stack_winners.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_stacked.py | results-cited | RESULTS, TOOLING | RESULTS×1, TOOLING×1 | — | — | — | — |
| bench | bench_static.py | library | RESULTS, via:TOOLING | RESULTS×1 | 3 | — | — | — |
| bench | bench_step_diversity.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| bench | bench_step_tokens.py | library | RESULTS, specs, via:TOOLING | RESULTS×34, specs×19 | 67 | 1 | 2 | — |
| bench | bench_stitch_poc.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| bench | bench_syndrome_head.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | — | — |
| bench | bench_syndrome_policy.py | library | RESULTS | RESULTS×1 | 1 | — | — | — |
| bench | bench_temp_race.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_tree_verify.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| bench | bench_triton_kernels.py | results-cited | RESULTS | RESULTS×1 | — | — | 1 | — |
| bench | bench_verify_fast.py | library | RESULTS, specs, via:TOOLING | RESULTS×3, specs×14 | 46 | — | 2 | — |
| bench | bench_vge.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | — | — |
| bench | bench_weight_anatomy.py | results-cited | RESULTS, TOOLING | RESULTS×3, TOOLING×1 | — | — | — | — |
| bench | bench_zx.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| bench | bench_zx_r3.py | library | RESULTS | RESULTS×1 | 3 | — | — | — |
| bench | bench_zx_r5.py | library | RESULTS | RESULTS×3 | 2 | — | 1 | — |
| bench | bench_zx_r6.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| bench | bench_zx_r7.py | results-cited | RESULTS | RESULTS×3 | — | — | — | — |
| book | book.py | library | RESULTS, specs, TOOLING | RESULTS×6, specs×6, TOOLING×3 | 1 | 2 | — | — |
| build | build_gen7_diet.py | UNCITED | — | — | — | — | — | — |
| calibrate | calibrate_hce.py | spec-cited | specs | specs×9 | — | — | — | — |
| cite | cite_lookup.py | tool-referenced | TOOLING | TOOLING×3 | — | 1 | 1 | — |
| ckpt | ckpt_manifest.py | library | RESULTS, specs | RESULTS×4, specs×3 | 1 | — | 1 | — |
| claim | claim_lint.py | results-cited | RESULTS, specs, TOOLING | RESULTS×1, specs×6, TOOLING×5 | — | 3 | 1 | — |
| consolidate | consolidate_mathnative.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| control | control_round.py | UNCITED | — | — | — | — | — | — |
| convert | convert_diet_prefix.py | spec-cited | specs | specs×1 | — | — | — | — |
| eval | eval_mathnative.py | UNCITED | — | — | — | — | — | — |
| eval | eval_pruned_moe.py | spec-cited | specs | specs×2 | — | — | — | — |
| eval | eval_ruler.py | spec-cited | specs | specs×1 | — | — | — | — |
| expert | expert_iter_steps.py | library | specs, via:RESULTS | specs×10 | 4 | — | — | farm_v22.py |
| expert | expert_loop.py | library | specs, via:RESULTS | specs×16 | 3 | — | — | step_grpo.py |
| farm | farm_algebra.py | results-cited | RESULTS | RESULTS×4 | — | — | 1 | — |
| farm | farm_l4_calc.py | UNCITED | — | — | — | — | — | — |
| farm | farm_v22.py | results-cited | RESULTS, specs | RESULTS×1, specs×3 | — | — | 1 | — |
| figlib | figlib.py | spec-cited | specs | specs×1 | — | — | — | — |
| fold | fold_book.py | library | specs, TOOLING | specs×1, TOOLING×2 | — | 2 | — | — |
| gen | gen_catalog.py | library | RESULTS, specs | RESULTS×3, specs×1 | 1 | — | — | — |
| gen | gen_codemap.py | library | RESULTS, specs, TOOLING | RESULTS×2, specs×28, TOOLING×13 | — | 7 | 1 | — |
| gen | gen_dispatch_labels.py | results-cited | RESULTS | RESULTS×1 | — | — | 2 | — |
| gen | gen_dispatch_labels_v2.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| gen | gen_figures_web.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | — | — |
| gen | gen_frontier.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| gen | gen_index.py | library | RESULTS, specs, TOOLING | RESULTS×1, specs×11, TOOLING×13 | — | 8 | — | — |
| gen | gen_lake.py | library | RESULTS, specs | RESULTS×3, specs×1 | — | 1 | — | — |
| gen | gen_magic_labels.py | results-cited | RESULTS, specs, TOOLING | RESULTS×3, specs×1, TOOLING×2 | — | — | — | — |
| gen | gen_policy_labels.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| gen | gen_proposer_data.py | results-cited | RESULTS, specs | RESULTS×1, specs×9 | — | — | 1 | — |
| gen | gen_readme.py | library | RESULTS, specs, TOOLING | RESULTS×1, specs×22, TOOLING×4 | 1 | 4 | — | — |
| gen | gen_receipt_lock.py | library | RESULTS, specs, TOOLING | RESULTS×1, specs×3, TOOLING×9 | 1 | 9 | — | — |
| gen | gen_regret_labels.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| gen | gen_results_index.py | library | RESULTS, specs, TOOLING | RESULTS×8, specs×20, TOOLING×9 | 1 | 8 | 2 | — |
| gen | gen_scoreboard.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | — | — |
| gen | gen_syndrome_labels.py | results-cited | RESULTS | RESULTS×2 | — | — | 1 | — |
| grow | grow_mathnative.py | library | RESULTS, specs | RESULTS×1, specs×2 | 1 | — | 1 | — |
| harvest | harvest_champion.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| harvest | harvest_frontier.py | results-cited | RESULTS, specs | RESULTS×1, specs×2 | — | — | — | — |
| list | list_uncurated.py | spec-cited | specs | specs×6 | — | — | — | — |
| liverun | liverun.py | library | RESULTS, specs, TOOLING | RESULTS×18, specs×2, TOOLING×3 | — | 15 | 1 | — |
| log | log_hygiene.py | library | RESULTS, specs | RESULTS×3, specs×5 | 1 | 1 | — | — |
| markov | markov_eval.py | UNCITED | — | — | — | — | — | — |
| markov | markov_prior.py | spec-cited | specs | specs×3 | — | — | — | — |
| mine | mine_highways.py | UNCITED | — | — | — | — | 1 | — |
| mine | mine_prior_update.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| moe | moe_router_stats.py | results-cited | RESULTS, specs | RESULTS×2, specs×3 | — | — | 4 | — |
| obs | obs_from_receipt_0s.py | library | RESULTS, specs, TOOLING | RESULTS×2, specs×1, TOOLING×2 | 1 | 2 | 1 | — |
| obs | obs_from_receipt_0t.py | results-cited | RESULTS, specs | RESULTS×2, specs×1 | — | — | — | — |
| obs | obs_from_receipt_ex6b43knife.py | results-cited | RESULTS | RESULTS×4 | — | — | — | — |
| obs | obs_from_receipt_ex6depth.py | results-cited | RESULTS | RESULTS×6 | — | — | — | — |
| obs | obs_from_receipt_ex6depth1.py | results-cited | RESULTS | RESULTS×6 | — | — | — | — |
| obs | obs_from_receipt_ex6temporal.py | results-cited | RESULTS | RESULTS×10 | — | — | — | — |
| plot | plot_gt1_crest.py | spec-cited | specs | specs×3 | — | — | — | — |
| plot | plot_identity_crest.py | spec-cited | specs | specs×3 | — | — | — | — |
| plot | plot_neurons.py | spec-cited | specs | specs×6 | — | — | 1 | — |
| probe | probe_depth.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| render | render_anim.py | library | RESULTS, specs, TOOLING | RESULTS×1, specs×1, TOOLING×4 | 1 | 1 | — | — |
| render | render_gallery.py | spec-cited | specs | specs×7 | — | — | — | — |
| render | render_hero_neurons.py | spec-cited | specs | specs×4 | — | — | 1 | — |
| results | results_query.py | library | RESULTS, specs, TOOLING | RESULTS×2, specs×23, TOOLING×9 | — | 4 | 1 | — |
| rjob | rjob.py | results-cited | RESULTS, specs, TOOLING | RESULTS×3, specs×3, TOOLING×6 | — | 3 | — | — |
| sol | sol_enrich_results.py | library | specs | specs×3 | 1 | — | — | — |
| sol | sol_generate_tables.py | spec-cited | specs | specs×3 | — | — | — | — |
| step | step_grpo.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | — | — |
| step | step_grpo_micro.py | library | RESULTS, specs, TOOLING | RESULTS×40, specs×26, TOOLING×2 | 95 | 4 | 3 | — |
| sweep | sweep_lookup.py | UNCITED | — | — | — | — | 1 | — |
| sweep | sweep_lookup_mlx.py | UNCITED | — | — | — | — | — | — |
| tabula | tabula_rasa_r0.py | UNCITED | — | — | — | — | — | — |
| tabula | tabula_rasa_r1.py | UNCITED | — | — | — | — | — | — |
| tabula | tabula_rasa_r2.py | UNCITED | — | — | — | — | — | — |
| task | task_arithmetic.py | spec-cited | specs | specs×1 | — | — | — | — |
| task | task_composition.py | spec-cited | specs | specs×1 | — | — | — | — |
| tournament | tournament_birth.py | library | RESULTS, specs | RESULTS×3, specs×1 | 3 | 6 | — | — |
| train | train_calculus.py | library | RESULTS, specs | RESULTS×1, specs×9 | 1 | — | 4 | — |
| train | train_dispatcher.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| train | train_magic_estimator.py | library | RESULTS, via:specs, via:TOOLING | RESULTS×2 | 7 | — | 1 | — |
| train | train_magic_llm.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| train | train_mathnative.py | library | RESULTS, specs, TOOLING | RESULTS×49, specs×17, TOOLING×2 | 113 | 39 | 1 | — |
| train | train_nnue.py | library | RESULTS, specs, via:TOOLING | RESULTS×1, specs×11 | 1 | — | 12 | — |
| train | train_proposer.py | results-cited | RESULTS, specs | RESULTS×1, specs×5 | — | — | — | — |
| train | train_syndrome_decoder.py | results-cited | RESULTS | RESULTS×2 | — | — | 1 | — |
| train | train_syndrome_policy.py | results-cited | RESULTS | RESULTS×2 | — | — | — | — |
| train | train_ternary.py | results-cited | RESULTS, specs | RESULTS×3, specs×5 | — | — | 2 | — |
| train | train_tf32x3.py | results-cited | RESULTS, specs | RESULTS×1, specs×1 | — | — | — | — |
| train | train_value_head.py | results-cited | RESULTS, TOOLING | RESULTS×2, TOOLING×1 | — | — | — | — |
| train | train_weight_reader.py | results-cited | RESULTS, specs | RESULTS×1, specs×4 | — | — | — | — |
| validity | validity_autopsy.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
