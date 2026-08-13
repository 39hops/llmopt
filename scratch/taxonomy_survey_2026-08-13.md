# Taxonomy execution survey — 2026-08-13 (PROPOSALS, unverified)

Source spec: `docs/superpowers/specs/2026-07-24-repo-taxonomy.md` (APPROVED).
Scope note the mover must read first: the spec's target tree reorganizes
`checkpoints/` and `data/` ONLY. No `scripts/` or `scratch/` code file has a
taxonomy destination. Therefore every UNCITED code file below is
`keep-in-place` unless it is a one-night shell launcher (archive-candidate).
The real work in scripts/scratch is section 3 (path retargeting), not moving.

Counts: archive-candidate=55, keep-in-place=146 (total 201)

## 1. UNCITED file dispositions (201)

| path | disposition | reason (5 words) |
|---|---|---|
| scratch/absorb_1e5.py | keep-in-place | live driver; retarget data paths |
| scratch/b768_after_v5.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/basin_probe.py | keep-in-place | live driver; retarget data paths |
| scratch/birth19m_snaps.py | keep-in-place | live driver; retarget data paths |
| scratch/brute_arms_0801.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/brute_b_arms_0801.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/brute_c_arm_0801.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/build_dist_diets.py | keep-in-place | live driver; retarget data paths |
| scratch/build_merged_diet.py | keep-in-place | live driver; retarget data paths |
| scratch/cal_dilute.py | keep-in-place | live driver; retarget data paths |
| scratch/cal_dk_probe.py | keep-in-place | live driver; retarget data paths |
| scratch/callspan_arms.py | keep-in-place | self-contained probe, no taxonomy dir |
| scratch/ce400.py | keep-in-place | sibling-import coupled, breaks if moved |
| scratch/ceiling_probe_cuda.py | keep-in-place | sibling-import coupled, breaks if moved |
| scratch/chain_carry.py | keep-in-place | live driver; retarget data paths |
| scratch/champ_cuda_probe.py | keep-in-place | live driver; retarget data paths |
| scratch/ckpt_delete_pass.py | keep-in-place | housekeeping tool the move needs |
| scratch/ckpt_triage_table.py | keep-in-place | housekeeping tool the move needs |
| scratch/closers_chain.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/complex_birth.py | keep-in-place | live driver; retarget data paths |
| scratch/complex_nnue.py | keep-in-place | live driver; retarget data paths |
| scratch/confluence.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/corner_snap.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/cplx_chain.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/d2_verify.py | keep-in-place | live driver; retarget data paths |
| scratch/day_chain.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/dual_probe.py | keep-in-place | live driver; retarget data paths |
| scratch/duo_mine.py | keep-in-place | live driver; retarget data paths |
| scratch/emission_wall_pair.py | keep-in-place | sibling-import coupled, breaks if moved |
| scratch/export_mb_ref.py | keep-in-place | sibling-import coupled, breaks if moved |
| scratch/fixed_q_snap.py | keep-in-place | self-contained probe, no taxonomy dir |
| scratch/floor_hk1.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/floor_hk1_d256.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/fmt_chain.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/fmt_chain2.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/fmt_pp_watcher.sh | archive-candidate | watcher shell, run already finished |
| scratch/format_delta_prep.py | keep-in-place | live driver; retarget data paths |
| scratch/format_ladder.py | keep-in-place | live driver; retarget data paths |
| scratch/fourier2_modbirth.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/fourier_probe.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/g19_fp32_cell.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/g19_probes_fix.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/g19_sigma_cuda.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/g5_polar.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/gate_batched.py | keep-in-place | sibling-import coupled, breaks if moved |
| scratch/gate_ckpt_cuda.py | keep-in-place | sibling-import coupled, breaks if moved |
| scratch/gate_pp.py | keep-in-place | live driver; retarget data paths |
| scratch/gate_v2_bench.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/gate_zx.py | keep-in-place | live driver; retarget data paths |
| scratch/gauge_distance_d256.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/gauge_m4x.py | keep-in-place | live driver; retarget data paths |
| scratch/gen9_19m_cuda_control.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/gen9_45m_fp32_control.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/gen9_45m_probes.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/gen_lab_overview_pdf.py | keep-in-place | figure generator, writes docs assets |
| scratch/genpins_freeze.py | keep-in-place | self-contained probe, no taxonomy dir |
| scratch/grav1b_distance.py | keep-in-place | live driver; retarget data paths |
| scratch/grav_posthoc.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/greedy_first_gate.py | keep-in-place | sibling-import coupled, breaks if moved |
| scratch/grow_decomp1.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/grpo_shaped.py | keep-in-place | live driver; retarget data paths |
| scratch/gt6_resume_arms.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/holdout_v2.py | keep-in-place | later version of same probe |
| scratch/int2_regate.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/int3_rider.py | keep-in-place | self-contained probe, no taxonomy dir |
| scratch/jointperm_distance.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/kv_after_night.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/kv_equiv.py | keep-in-place | live driver; retarget data paths |
| scratch/l9_probe.py | keep-in-place | sibling-import coupled, breaks if moved |
| scratch/lam_merge_review.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/lyapunov_birth.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/mac_day_chain.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/make_union_diet.py | keep-in-place | live driver; retarget data paths |
| scratch/margin_census.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/merge_space1.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/merge_space2.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/merge_space3.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/merge_space4.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/merge_space5.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/metabolic_v4.py | keep-in-place | later version of same probe |
| scratch/metabolic_v5.py | keep-in-place | later version of same probe |
| scratch/morning_run.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/mps_sigma_gates.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/muon_3ep_d256.py | keep-in-place | live driver; retarget data paths |
| scratch/night28b.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/night2_mac_shift2.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/night30_mac.py | keep-in-place | live driver; retarget data paths |
| scratch/night30b.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/night31_mac.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/night_calib.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/night_gates.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/ozaki_2b_bisect.py | keep-in-place | self-contained probe, no taxonomy dir |
| scratch/ozaki_2b_debug.py | keep-in-place | self-contained probe, no taxonomy dir |
| scratch/ozaki_2b_ident.py | keep-in-place | self-contained probe, no taxonomy dir |
| scratch/ozaki_cuda.py | keep-in-place | self-contained probe, no taxonomy dir |
| scratch/p2_crown_draws.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/pack_p2a.py | keep-in-place | sibling-import coupled, breaks if moved |
| scratch/paper_figs.py | keep-in-place | figure generator, writes docs assets |
| scratch/phase4_unboot.py | keep-in-place | self-contained probe, no taxonomy dir |
| scratch/phys_probe.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/pincer_dist_report.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/poly4_watcher.sh | archive-candidate | watcher shell, run already finished |
| scratch/poly5_watcher.sh | archive-candidate | watcher shell, run already finished |
| scratch/prefix_pair.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/prologue_gates.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/ptq4_arms.py | keep-in-place | sibling-import coupled, breaks if moved |
| scratch/ptq4_gates.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/qwen_displace_extract.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/rat_repair.py | keep-in-place | self-contained probe, no taxonomy dir |
| scratch/retention_watcher.sh | archive-candidate | watcher shell, run already finished |
| scratch/rot_snap_anatomy.py | keep-in-place | live driver; retarget data paths |
| scratch/run_snap_knee.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/seeds_ladder_0804.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/ssm_star1.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/stability_atlas.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/sym45_run.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/synonym_test.py | keep-in-place | live driver; retarget data paths |
| scratch/tenet_r1b_micro.py | keep-in-place | live driver; retarget data paths |
| scratch/tenet_w1_population.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/ternary_control.py | keep-in-place | live driver; retarget data paths |
| scratch/ternary_gate.py | keep-in-place | live driver; retarget data paths |
| scratch/ternary_session2.py | keep-in-place | live driver; retarget data paths |
| scratch/tier_escalate.py | keep-in-place | live driver; retarget data paths |
| scratch/train_fp64.py | keep-in-place | live driver; retarget data paths |
| scratch/tuesday_night.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/v4flash_anatomy.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/vmasm_probe.py | keep-in-place | live driver; retarget data paths |
| scratch/vrm_ab.py | keep-in-place | live driver; retarget data paths |
| scratch/weight_fft_euler.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scratch/z1_gate.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/z1s_hot_watcher.sh | archive-candidate | watcher shell, run already finished |
| scratch/zx_chain.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/zx_chain_cuda.sh | archive-candidate | one-night launcher, schedule expired |
| scratch/zx_gate_watcher.sh | archive-candidate | watcher shell, run already finished |
| scripts/arena.py | keep-in-place | live driver; retarget data paths |
| scripts/autopsy_int.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_adaptive_draft.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_anneal.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_ansatz_search.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_ansatz_search_2b.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_bandit.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_bestfirst.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_bestfirst_llm.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_bestfirst_nnue.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_commute.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_compile.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_control.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_decoding.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_dispatch_race_v4.py | keep-in-place | later version of same probe |
| scripts/bench_distilled_draft.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_entropy_beam.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_fib_restarts.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_flash_prefill.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_fused.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_gated.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_gweight.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_int4_config_sweep.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_interference.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_ksweep.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_kv_quant_decode.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_llm_gating.py | keep-in-place | live driver; retarget data paths |
| scripts/bench_lookup_static.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_luby.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_markov.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_markov_adaptive.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_mlx_integration.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_ode_engine.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_opcap.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_population.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_prefix_reuse.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_regret_resample.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/bench_stack_winners.py | keep-in-place | live driver; retarget data paths |
| scripts/bench_stacked.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_static.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_syndrome_head.py | keep-in-place | live driver; retarget data paths |
| scripts/bench_temp_race.py | keep-in-place | live driver; retarget data paths |
| scripts/bench_tree_verify.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/bench_triton_kernels.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/build_gen7_diet.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/control_round.py | keep-in-place | live driver; retarget data paths |
| scripts/eval_mathnative.py | keep-in-place | live driver; retarget data paths |
| scripts/farm_l4_calc.py | keep-in-place | live driver; retarget data paths |
| scripts/farm_v22.py | keep-in-place | live driver; retarget data paths |
| scripts/gen_dispatch_labels.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/gen_dispatch_labels_v2.py | keep-in-place | later version of same probe |
| scripts/gen_frontier.py | keep-in-place | live driver; retarget data paths |
| scripts/gen_policy_labels.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/gen_regret_labels.py | keep-in-place | live driver; retarget data paths |
| scripts/gen_scoreboard.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/harvest_champion.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/markov_eval.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/markov_prior.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/mine_highways.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/sweep_lookup.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/sweep_lookup_mlx.py | keep-in-place | self-contained probe, no taxonomy dir |
| scripts/tabula_rasa_r0.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/tabula_rasa_r1.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/tabula_rasa_r2.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/task_composition.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/train_dispatcher.py | keep-in-place | hardcodes data/checkpoints; retarget only |
| scripts/train_tf32x3.py | keep-in-place | self-contained probe, no taxonomy dir |

## 2. Cross-import map (importer -> imported module [resolves to])

304 sibling-script import couplings. Every one breaks if either end moves.

- `scripts/arena.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/bench_budget_alloc.py` -> `train_magic_estimator` [scripts/train_magic_estimator.py]
- `scripts/bench_hints_ab.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/bench_llm_gating.py` -> `bench_hybrid` [scripts/bench_hybrid.py]
- `scripts/bench_llm_gating.py` -> `train_magic_estimator` [scripts/train_magic_estimator.py]
- `scripts/bench_stack_winners.py` -> `bench_lazy` [scripts/bench_lazy.py]
- `scripts/bench_stack_winners.py` -> `bench_magic` [scripts/bench_magic.py]
- `scripts/bench_step_diversity.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/bench_step_tokens.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scripts/bench_syndrome_head.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/bench_temp_race.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/bench_verify_fast.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/bench_zx_r5.py` -> `bench_zx_r3` [scripts/bench_zx_r3.py]
- `scripts/bench_zx_r6.py` -> `bench_zx_r3` [scripts/bench_zx_r3.py]
- `scripts/bench_zx_r6.py` -> `bench_zx_r5` [scripts/bench_zx_r5.py]
- `scripts/bench_zx_r7.py` -> `bench_zx_r3` [scripts/bench_zx_r3.py]
- `scripts/bench_zx_r7.py` -> `bench_zx_r5` [scripts/bench_zx_r5.py]
- `scripts/control_round.py` -> `expert_iter_steps` [scripts/expert_iter_steps.py]
- `scripts/control_round.py` -> `expert_loop` [scripts/expert_loop.py]
- `scripts/control_round.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/convert_diet_prefix.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scripts/eval_mathnative.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/eval_mathnative.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scripts/expert_iter_steps.py` -> `train_magic_estimator` [scripts/train_magic_estimator.py]
- `scripts/expert_iter_steps.py` -> `train_calculus` [scripts/train_calculus.py]
- `scripts/expert_loop.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/expert_loop.py` -> `expert_iter_steps` [scripts/expert_iter_steps.py]
- `scripts/farm_l4_calc.py` -> `expert_iter_steps` [scripts/expert_iter_steps.py]
- `scripts/farm_v22.py` -> `expert_iter_steps` [scripts/expert_iter_steps.py]
- `scripts/gen_frontier.py` -> `train_magic_estimator` [scripts/train_magic_estimator.py]
- `scripts/gen_magic_labels.py` -> `train_magic_estimator` [scripts/train_magic_estimator.py]
- `scripts/gen_regret_labels.py` -> `bench_syndrome_policy` [scripts/bench_syndrome_policy.py]
- `scripts/probe_depth.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scripts/probe_depth.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/step_grpo.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/step_grpo.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scripts/step_grpo.py` -> `expert_loop` [scripts/expert_loop.py]
- `scripts/step_grpo_micro.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/step_grpo_micro.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scripts/tournament_birth.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scripts/train_ternary.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scripts/train_value_head.py` -> `train_nnue` [scripts/train_nnue.py]
- `scripts/validity_autopsy.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scripts/validity_autopsy.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scripts/validity_autopsy.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/absorb_1e5.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/absorb_1e5.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/absorb_1e5.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/attractor_census.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/attractor_census2.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/basin_probe.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/birth19m_snaps.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/blackhole_b0.py` -> `capacity_meter` [scratch/capacity_meter.py]
- `scratch/build_dist_diets.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/build_merged_diet.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/cal_dilute.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/cal_dk_probe.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/calib_probe.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/ce400.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/ce_gate_study.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/ce_gate_study.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/ce_gate_study.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/ceiling_probe_cuda.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/ceiling_probe_cuda.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/ceiling_probe_cuda.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/chain_carry.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/champ_cuda_probe.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/champ_cuda_probe.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/champ_cuda_probe.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/clade_stream_d256.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/clade_stream_d256.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/clade_stream_d256.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/complex_birth.py` -> `complex_model` [scratch/complex_model.py]
- `scratch/complex_birth.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/complex_nnue.py` -> `train_magic_estimator` [scripts/train_magic_estimator.py]
- `scratch/complexify_control.py` -> `rot_commutant` [scratch/rot_commutant.py]
- `scratch/d2_verify.py` -> `calib_probe` [scratch/calib_probe.py]
- `scratch/detbwd_diet.py` -> `detbwd_r1` [scratch/detbwd_r1.py]
- `scratch/detbwd_diet.py` -> `detbwd_r2b` [scratch/detbwd_r2b.py]
- `scratch/detbwd_diet.py` -> `detbwd_r3_qw` [scratch/detbwd_r3_qw.py]
- `scratch/detbwd_gravmoe.py` -> `detbwd_diet` [scratch/detbwd_diet.py]
- `scratch/detbwd_gravmoe.py` -> `detbwd_r1` [scratch/detbwd_r1.py]
- `scratch/detbwd_gravmoe.py` -> `detbwd_r2b` [scratch/detbwd_r2b.py]
- `scratch/detbwd_gravmoe.py` -> `detbwd_r3_qw` [scratch/detbwd_r3_qw.py]
- `scratch/detbwd_mb.py` -> `detbwd_r1` [scratch/detbwd_r1.py]
- `scratch/detbwd_mb.py` -> `detbwd_r2b` [scratch/detbwd_r2b.py]
- `scratch/detbwd_mb.py` -> `detbwd_r3_qw` [scratch/detbwd_r3_qw.py]
- `scratch/detbwd_r1b.py` -> `detbwd_r1` [scratch/detbwd_r1.py]
- `scratch/detbwd_r2_adamw.py` -> `detbwd_r1` [scratch/detbwd_r1.py]
- `scratch/detbwd_r2b.py` -> `detbwd_r1` [scratch/detbwd_r1.py]
- `scratch/detbwd_r2b.py` -> `detbwd_r1b` [scratch/detbwd_r1b.py]
- `scratch/detbwd_r2b.py` -> `detbwd_r3_qw` [scratch/detbwd_r3_qw.py]
- `scratch/detbwd_r3_qw.py` -> `detbwd_r1` [scratch/detbwd_r1.py]
- `scratch/detbwd_r3_qw.py` -> `detbwd_r2_adamw` [scratch/detbwd_r2_adamw.py]
- `scratch/determinability_census.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/dual_probe.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/duo_mine.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/duo_mine.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/duo_mine.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/duo_wave.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/duo_wave.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/duo_wave.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/e3_battery.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/emission_wall_pair.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/emission_wall_pair.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/ex1_swap.py` -> `gt2_jaccard` [scratch/gt2_jaccard.py]
- `scratch/ex2_build.py` -> `gt2_jaccard` [scratch/gt2_jaccard.py]
- `scratch/ex3_build.py` -> `gt2_jaccard` [scratch/gt2_jaccard.py]
- `scratch/exchange_test.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/exchange_test.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/export_mb_ref.py` -> `detbwd_r1` [scratch/detbwd_r1.py]
- `scratch/export_mb_ref.py` -> `detbwd_r2b` [scratch/detbwd_r2b.py]
- `scratch/export_mb_ref.py` -> `detbwd_r3_qw` [scratch/detbwd_r3_qw.py]
- `scratch/farm_dist_rows.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/farmer_probe.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/farmer_probe.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/farmer_probe.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/format_delta_prep.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/format_ladder.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/fourier4a_dynamics.py` -> `fourier2b_widemod` [scratch/fourier2b_widemod.py]
- `scratch/fourier_g9.py` -> `fourier2b_widemod` [scratch/fourier2b_widemod.py]
- `scratch/fp64_paired.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/fp64_paired.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/fp64_paired.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/fx3_house.py` -> `pack_decode` [scratch/pack_decode.py]
- `scratch/fx3_house.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/fx3_house.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/gate_batched.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/gate_batched.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/gate_batched.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/gate_ckpt.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/gate_ckpt_cuda.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/gate_pp.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/gate_pp.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/gate_pp.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/gate_prefix.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/gate_prefix.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/gate_prefix.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/gate_rarity.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/gate_rarity.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/gate_rarity.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/gate_transcripts.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/gate_transcripts.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/gate_transcripts.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/gate_zx.py` -> `adjudicate_zx` [scratch/adjudicate_zx.py]
- `scratch/gate_zx.py` -> `complex_model` [scratch/complex_model.py]
- `scratch/gauge_m4x.py` -> `prologue_arms` [scratch/prologue_arms.py]
- `scratch/graph_mod_sigma.py` -> `graph_modularity_gen8` [scratch/graph_modularity_gen8.py]
- `scratch/grav1b_distance.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/grav2_spacetime.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/grav_probe.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/greedy_first_gate.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/greedy_first_gate.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/grpo_shaped.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/grpo_shaped.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/grpo_shaped.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/gt2_code_arm0.py` -> `moe_gt1_arm2` [scratch/moe_gt1_arm2.py]
- `scratch/gt4_verbal_core.py` -> `gt2_jaccard` [scratch/gt2_jaccard.py]
- `scratch/gt5_union_keep.py` -> `gt2_jaccard` [scratch/gt2_jaccard.py]
- `scratch/gt6_recall_ladder.py` -> `gt2_jaccard` [scratch/gt2_jaccard.py]
- `scratch/gt7_coverage_rederive.py` -> `gt2_jaccard` [scratch/gt2_jaccard.py]
- `scratch/gt7_draw.py` -> `gt2_jaccard` [scratch/gt2_jaccard.py]
- `scratch/holdout_gate.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/holdout_gate.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/holdout_v2.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/holdout_v2.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/holdout_v2.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/judge_decode.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/judge_decode.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/keff_probe.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/keff_probe.py` -> `ssm_star` [scratch/ssm_star.py]
- `scratch/kv_equiv.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/kv_equiv.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/l9_probe.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/l9_probe.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/l9_probe.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/legacy_diet_audit.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/lyap_compare.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/make_altpairs.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/make_union_diet.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/margin_by_level.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/margin_by_ply.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/margin_by_ply.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/mass_on_valid.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/matryoshka_r1.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/matryoshka_r2.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/metabolic_d2.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/metabolic_d2.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/metabolic_d2.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/metabolic_hot.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/metabolic_hot.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/metabolic_hot.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/metabolic_v3.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/metabolic_v3.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/metabolic_v3.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/metabolic_v4.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/metabolic_v4.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/metabolic_v4.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/metabolic_v5.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/metabolic_v5.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/metabolic_v5.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/muon_3ep_d256.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/night30_mac.py` -> `capacity_meter` [scratch/capacity_meter.py]
- `scratch/night30_mac.py` -> `pack_c6` [scratch/pack_c6.py]
- `scratch/p3_bits.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/p3_bits.py` -> `tournament_birth` [scripts/tournament_birth.py]
- `scratch/p3_ffnslack.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/p3_grav2.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/p3_quat.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/p3_stream2x2.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/p3_umoe_soft.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/pack_baselines.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/pack_c7.py` -> `capacity_meter` [scratch/capacity_meter.py]
- `scratch/pack_c7.py` -> `pack_c6` [scratch/pack_c6.py]
- `scratch/pack_decode.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/pack_decode.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/pack_determinism.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/pack_p2a.py` -> `pack_c6` [scratch/pack_c6.py]
- `scratch/pincer_labels_v2.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/pincer_r0.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/pincer_r0.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/pincer_r0.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/pincer_r0b.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/pincer_r0b.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/pincer_r0b.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/pincer_r1_indist.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/pincer_r1_indist.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/pincer_r1_probe.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/pincer_r1_probe.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/pincer_r1b_labels.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/pincer_r8.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/pincer_r8.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/pincer_r8.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/place1_gravity.py` -> `pack_c6` [scratch/pack_c6.py]
- `scratch/practice_mine.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/practice_mine.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/practice_mine.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/probe_int_device_parity.py` -> `detbwd_r1` [scratch/detbwd_r1.py]
- `scratch/probe_int_device_parity.py` -> `detbwd_r2b` [scratch/detbwd_r2b.py]
- `scratch/ptq4_arms.py` -> `lloydmax_race` [scratch/lloydmax_race.py]
- `scratch/quat_convert.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/quat_convert.py` -> `quat_commutant` [scratch/quat_commutant.py]
- `scratch/rev2_d768.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/rev2_d768.py` -> `tournament_birth` [scripts/tournament_birth.py]
- `scratch/rev3_crown.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/rev3_crown.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/rev3_crown.py` -> `tournament_birth` [scripts/tournament_birth.py]
- `scratch/rev3_crown.py` -> `grow_mathnative` [scripts/grow_mathnative.py]
- `scratch/rev4_zx45.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/rot_convert.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/rot_convert.py` -> `rot_commutant` [scratch/rot_commutant.py]
- `scratch/rot_snap_anatomy.py` -> `rot_commutant` [scratch/rot_commutant.py]
- `scratch/saturation_s2.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/saturation_s2b.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/scorer_s1_battery.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/scorer_s1_battery.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/scorer_s2_data.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/series_probe.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/softprompt1.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/ssm_star.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/ssm_star.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/successors_acceptance.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/sym_birth.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/sym_convert.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/synonym_test.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/tenet_d1_revgate.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/tenet_d1_revgate.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/tenet_d1_revgate.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/tenet_d1_revgate.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/tenet_d2_revdiet.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/tenet_d2_revdiet.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/tenet_d2_revdiet.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/tenet_mult_b32.py` -> `tenet_mult_census` [scratch/tenet_mult_census.py]
- `scratch/tenet_mult_census.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/tenet_mult_census.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/tenet_mult_census.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/tenet_mult_census.py` -> `tenet_d2_revdiet` [scratch/tenet_d2_revdiet.py]
- `scratch/tenet_r1b_micro.py` -> `tenet_d3_budget` [scratch/tenet_d3_budget.py]
- `scratch/tenet_r1b_micro.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/tenet_r1b_micro.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/ternary_control.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/ternary_control.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/ternary_control.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/ternary_gate.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/ternary_gate.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/ternary_gate.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/ternary_session2.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]
- `scratch/ternary_session2.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/ternary_session2.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/tier_escalate.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/tier_escalate.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/tier_retry.py` -> `bench_step_tokens` [scripts/bench_step_tokens.py]
- `scratch/tier_retry.py` -> `bench_verify_fast` [scripts/bench_verify_fast.py]
- `scratch/train_fp64.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/umoe_conserve.py` -> `train_mathnative` [scripts/train_mathnative.py]
- `scratch/v4flash_f1c.py` -> `v4flash_f1b` [scratch/v4flash_f1b.py]
- `scratch/v4flash_f1d.py` -> `v4flash_f1b` [scratch/v4flash_f1b.py]
- `scratch/v4flash_f1d.py` -> `v4flash_f1c` [scratch/v4flash_f1c.py]
- `scratch/v4flash_rungd.py` -> `v4flash_router` [scratch/v4flash_router.py]
- `scratch/v4flash_rungd2.py` -> `v4flash_router` [scratch/v4flash_router.py]
- `scratch/v4flash_rungd2.py` -> `v4flash_rungd` [scratch/v4flash_rungd.py]
- `scratch/v4flash_twin.py` -> `v4flash_rungA` [scratch/v4flash_rungA.py]
- `scratch/vmasm_probe.py` -> `vmasm` [scratch/vmasm.py]
- `scratch/vrm_ab.py` -> `step_grpo_micro` [scripts/step_grpo_micro.py]

### 2b. sys.path insertions (452 lines, scripts+scratch)

Two shapes. `Path(__file__).parent` inserts survive a move; the literal
`sys.path.insert(0, "scripts")` / `"scratch"` / `"."` shape is CWD-relative and
breaks if the launcher's working directory changes.

- `scratch/absorb_1e5.py:5:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/anatomy.py:13:sys.path.insert(0, ".")`
- `scratch/anatomy.py:14:sys.path.insert(0, "scripts")`
- `scratch/anatomy.py:15:sys.path.insert(0, "scratch")`
- `scratch/attractor_census.py:15:sys.path.insert(0, ".")`
- `scratch/attractor_census.py:16:sys.path.insert(0, "scripts")`
- `scratch/attractor_census2.py:18:sys.path.insert(0, ".")`
- `scratch/attractor_census2.py:19:sys.path.insert(0, "scripts")`
- `scratch/basin_probe.py:7:sys.path.insert(0, ".")`
- `scratch/basin_probe.py:8:sys.path.insert(0, "scripts")`
- `scratch/basin_probe.py:9:sys.path.insert(0, "scratch")`
- `scratch/birth19m_snaps.py:24:sys.path.insert(0, ".")`
- `scratch/birth19m_snaps.py:25:sys.path.insert(0, "scripts")`
- `scratch/birth19m_snaps.py:26:sys.path.insert(0, "scratch")`
- `scratch/blackhole_b0.py:18:sys.path.insert(0, ".")`
- `scratch/blackhole_b0.py:19:sys.path.insert(0, "scratch")`
- `scratch/build_dist_diets.py:17:sys.path.insert(0, ".")`
- `scratch/build_dist_diets.py:18:sys.path.insert(0, "scripts")`
- `scratch/build_merged_diet.py:6:sys.path.insert(0, "scripts")`
- `scratch/cal_dilute.py:12:sys.path.insert(0, ".")`
- `scratch/cal_dilute.py:13:sys.path.insert(0, "scripts")`
- `scratch/cal_dk_probe.py:10:sys.path.insert(0, "scripts")`
- `scratch/cal_dk_probe.py:9:sys.path.insert(0, ".")`
- `scratch/calib_probe.py:12:sys.path.insert(0, ".")`
- `scratch/calib_probe.py:13:sys.path.insert(0, "scripts")`
- `scratch/capacity_meter.py:19:sys.path.insert(0, ".")`
- `scratch/ce_gate_study.py:12:sys.path.insert(0, ".")`
- `scratch/ce_gate_study.py:13:sys.path.insert(0, "scripts")`
- `scratch/ce400.py:5:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/ceiling_probe_cuda.py:5:sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')`
- `scratch/chain_carry.py:10:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/champ_cuda_probe.py:2:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/clade_stream_d256.py:15:sys.path.insert(0, ".")`
- `scratch/clade_stream_d256.py:16:sys.path.insert(0, "scripts")`
- `scratch/complex_birth.py:13:sys.path.insert(0, ".")`
- `scratch/complex_birth.py:14:sys.path.insert(0, "scripts")`
- `scratch/complex_birth.py:15:sys.path.insert(0, "scratch")`
- `scratch/complex_nnue.py:16:sys.path.insert(0, "scripts")`
- `scratch/complexify_control.py:10:sys.path.insert(0, "scripts")`
- `scratch/complexify_control.py:11:sys.path.insert(0, "scratch")`
- `scratch/complexify_control.py:9:sys.path.insert(0, ".")`
- `scratch/corner_snap.py:11:sys.path.insert(0, ".")`
- `scratch/corner_snap.py:12:sys.path.insert(0, "scripts")`
- `scratch/corner_snap.py:13:sys.path.insert(0, "scratch")`
- `scratch/crystal_recreate_test.py:21:sys.path.insert(0, ".")`
- `scratch/desert_v2.py:14:sys.path.insert(0, ".")`
- `scratch/detbwd_diet.py:24:sys.path.insert(0, ".")`
- `scratch/detbwd_diet.py:25:sys.path.insert(0, "scratch")`
- `scratch/detbwd_gravmoe.py:30:sys.path.insert(0, ".")`
- `scratch/detbwd_gravmoe.py:31:sys.path.insert(0, "scratch")`
- `scratch/detbwd_mb.py:25:sys.path.insert(0, ".")`
- `scratch/detbwd_mb.py:26:sys.path.insert(0, "scratch")`
- `scratch/detbwd_plateau.py:22:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scratch/detbwd_plateau.py:23:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scratch/detbwd_r1.py:24:sys.path.insert(0, ".")`
- `scratch/detbwd_r1b.py:17:sys.path.insert(0, ".")`
- `scratch/detbwd_r1b.py:18:sys.path.insert(0, "scratch")`
- `scratch/detbwd_r2_adamw.py:23:sys.path.insert(0, ".")`
- `scratch/detbwd_r2_adamw.py:24:sys.path.insert(0, "scratch")`
- `scratch/detbwd_r2b.py:15:sys.path.insert(0, ".")`
- `scratch/detbwd_r2b.py:16:sys.path.insert(0, "scratch")`
- `scratch/detbwd_r3_qw.py:10:sys.path.insert(0, ".")`
- `scratch/detbwd_r3_qw.py:11:sys.path.insert(0, "scratch")`
- `scratch/determinability_census.py:19:sys.path.insert(0, ".")`
- `scratch/determinability_census.py:20:sys.path.insert(0, "scripts")`
- `scratch/distortion_collapse.py:14:sys.path.insert(0, ".")`
- `scratch/dual_probe.py:7:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/duo_mine.py:8:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/duo_wave.py:7:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/e2_logit_check.py:10:sys.path.insert(0, ".")`
- `scratch/e3_battery.py:13:sys.path.insert(0, ".")`
- `scratch/e3_battery.py:14:sys.path.insert(0, "scripts")`
- `scratch/emission_wall_pair.py:19:sys.path.insert(0, "scripts")`
- `scratch/emission_wall_pair.py:20:sys.path.insert(0, ".")`
- `scratch/engine_scale_export.py:40:    sys.path.insert(0, ".")`
- `scratch/engine_scale_export.py:41:    sys.path.insert(0, "scratch")`
- `scratch/ex1_swap.py:27:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scratch/ex2_build.py:43:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scratch/ex3_build.py:36:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scratch/exact_twin_d56.py:10:sys.path.insert(0, "scratch")`
- `scratch/exact_twin_d56.py:8:sys.path.insert(0, ".")`
- `scratch/exact_twin_d56.py:9:sys.path.insert(0, "scripts")`
- `scratch/exact1_small_cells.py:48:sys.path.insert(0, args.build_dir)`
- `scratch/exchange_test.py:14:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/export_axnn.py:21:sys.path.insert(0, ".")`
- `scratch/export_r2b_ref.py:13:sys.path.insert(0, ".")`
- `scratch/export_r2b_ref.py:14:sys.path.insert(0, "scratch")`
- `scratch/farm_dist_rows.py:14:sys.path.insert(0, ".")`
- `scratch/farm_dist_rows.py:15:sys.path.insert(0, "scripts")`
- `scratch/farmer_probe.py:16:sys.path.insert(0, ".")`
- `scratch/farmer_probe.py:17:sys.path.insert(0, "scripts")`
- `scratch/farmer_probe.py:18:sys.path.insert(0, "scratch")`
- `scratch/fig_magic_scatter.py:26:sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
- `scratch/format_delta_prep.py:10:sys.path.insert(0, "scripts")`
- `scratch/format_delta_prep.py:9:sys.path.insert(0, ".")`
- `scratch/format_ladder.py:19:sys.path.insert(0, ".")`
- `scratch/format_ladder.py:20:sys.path.insert(0, "scripts")`
- `scratch/fourier_g9.py:20:sys.path.insert(0, ".")`
- `scratch/fourier_g9.py:21:sys.path.insert(0, "scripts")`
- `scratch/fourier_g9.py:22:sys.path.insert(0, "scratch")`
- `scratch/fourier2b_widemod.py:15:sys.path.insert(0, ".")`
- `scratch/fourier2b_widemod.py:16:sys.path.insert(0, "scripts")`
- `scratch/fourier3_algdiet.py:26:sys.path.insert(0, ".")`
- `scratch/fourier3_algdiet.py:27:sys.path.insert(0, "scripts")`
- `scratch/fourier4a_dynamics.py:16:sys.path.insert(0, ".")`
- `scratch/fourier4a_dynamics.py:17:sys.path.insert(0, "scripts")`
- `scratch/fp64_paired.py:8:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/frozen_products/g19bn_probe.py:16:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/frozen_products/poly_probe_representative.py:16:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/fx3_house.py:11:sys.path.insert(0, ".")`
- `scratch/fx3_house.py:12:sys.path.insert(0, "scripts")`
- `scratch/fx3_house.py:13:sys.path.insert(0, "scratch")`
- `scratch/g5_polar.py:14:sys.path.insert(0, ".")`
- `scratch/g5_polar.py:15:sys.path.insert(0, "scripts")`
- `scratch/g5_polar.py:16:sys.path.insert(0, "scratch")`
- `scratch/gate_batched.py:10:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/gate_ckpt_cuda.py:2:sys.path.insert(0, 'scripts')`
- `scratch/gate_ckpt_cuda.py:3:sys.path.insert(0, '.')`
- `scratch/gate_ckpt.py:2:sys.path.insert(0, 'scripts')`
- `scratch/gate_ckpt.py:3:sys.path.insert(0, '.')`
- `scratch/gate_cplx.py:10:sys.path.insert(0, ".")`
- `scratch/gate_cplx.py:11:sys.path.insert(0, "scratch")`
- `scratch/gate_cplx.py:9:sys.path.insert(0, "scripts")`
- `scratch/gate_pp.py:13:sys.path.insert(0, ".")`
- `scratch/gate_pp.py:14:sys.path.insert(0, "scripts")`
- `scratch/gate_prefix.py:15:sys.path.insert(0, "scripts")`
- `scratch/gate_prefix.py:16:sys.path.insert(0, ".")`
- `scratch/gate_rarity.py:10:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/gate_regate.py:12:sys.path.insert(0, ".")`
- `scratch/gate_regate.py:13:sys.path.insert(0, "scripts")`
- `scratch/gate_regate.py:14:sys.path.insert(0, "scratch")`
- `scratch/gate_transcripts.py:17:sys.path.insert(0, ".")`
- `scratch/gate_transcripts.py:18:sys.path.insert(0, "scripts")`
- `scratch/gate_transcripts.py:19:sys.path.insert(0, "scratch")`
- `scratch/graph_mod_sigma.py:10:sys.path.insert(0, ".")`
- `scratch/graph_mod_sigma.py:11:sys.path.insert(0, "scratch")`
- `scratch/grav_posthoc.py:18:sys.path.insert(0, ".")`
- `scratch/grav_posthoc.py:19:sys.path.insert(0, "scripts")`
- `scratch/grav_posthoc.py:20:sys.path.insert(0, "scratch")`
- `scratch/grav_probe.py:13:sys.path.insert(0, ".")`
- `scratch/grav_probe.py:14:sys.path.insert(0, "scripts")`
- `scratch/grav_probe.py:15:sys.path.insert(0, "scratch")`
- `scratch/grav1b_distance.py:11:sys.path.insert(0, ".")`
- `scratch/grav1b_distance.py:12:sys.path.insert(0, "scripts")`
- `scratch/grav1b_distance.py:13:sys.path.insert(0, "scratch")`
- `scratch/grav2_spacetime.py:16:sys.path.insert(0, ".")`
- `scratch/grav2_spacetime.py:17:sys.path.insert(0, "scripts")`
- `scratch/grav2_spacetime.py:18:sys.path.insert(0, "scratch")`
- `scratch/greedy_first_gate.py:11:sys.path.insert(0, ".")`
- `scratch/greedy_first_gate.py:12:sys.path.insert(0, "scripts")`
- `scratch/grpo_shaped.py:11:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/gt2_code_arm0.py:27:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scratch/gt2_code_arm0.py:28:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scratch/gt3_probe_arm0.py:21:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scratch/gt3_probe_arm0.py:22:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scratch/gt4_verbal_core.py:22:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scratch/gt5_union_keep.py:19:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scratch/gt6_recall_ladder.py:30:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scratch/gt7_coverage_rederive.py:47:sys.path.insert(0, ".")`
- `scratch/gt7_coverage_rederive.py:48:sys.path.insert(0, "scratch")`
- `scratch/gt7_draw.py:44:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scratch/gt7_run.py:28:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scratch/head_autopsy.py:10:sys.path.insert(0, ".")`
- `scratch/head_autopsy.py:11:sys.path.insert(0, "scripts")`
- `scratch/head_census.py:10:sys.path.insert(0, "scratch")`
- `scratch/head_census.py:8:sys.path.insert(0, ".")`
- `scratch/head_census.py:9:sys.path.insert(0, "scripts")`
- `scratch/holdout_gate.py:7:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/holdout_v2.py:7:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/judge_decode.py:10:sys.path.insert(0, ".")`
- `scratch/judge_decode.py:11:sys.path.insert(0, "scripts")`
- `scratch/keff_probe.py:28:sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))`
- `scratch/keff_probe.py:29:sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))`
- `scratch/kv_equiv.py:4:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/l9_probe.py:6:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/lam_merge_review.py:12:sys.path.insert(0, ".")`
- `scratch/lam_merge_review.py:13:sys.path.insert(0, "scripts")`
- `scratch/lam_merge_review.py:14:sys.path.insert(0, "scratch")`
- `scratch/legacy_diet_audit.py:18:sys.path.insert(0, ".")`
- `scratch/legacy_diet_audit.py:19:sys.path.insert(0, "scripts")`
- `scratch/legacy_diet_audit.py:20:sys.path.insert(0, "scratch")`
- `scratch/loss_floor_census.py:22:sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`
- `scratch/lyap_compare.py:10:sys.path.insert(0, ".")`
- `scratch/lyap_compare.py:11:sys.path.insert(0, "scripts")`
- `scratch/make_altpairs.py:19:sys.path.insert(0, ".")`
- `scratch/make_altpairs.py:20:sys.path.insert(0, "scripts")`
- `scratch/make_union_diet.py:9:sys.path.insert(0, "scripts")`
- `scratch/margin_by_level.py:19:sys.path.insert(0, ".")`
- `scratch/margin_by_level.py:20:sys.path.insert(0, "scripts")`
- `scratch/margin_by_ply.py:18:sys.path.insert(0, ".")`
- `scratch/margin_by_ply.py:19:sys.path.insert(0, "scripts")`
- `scratch/margin_vs_branching.py:15:sys.path.insert(0, ".")`
- `scratch/margin_vs_branching.py:16:sys.path.insert(0, "scripts")`
- `scratch/mass_on_valid.py:11:sys.path.insert(0, ".")`
- `scratch/mass_on_valid.py:12:sys.path.insert(0, "scripts")`
- `scratch/matryoshka_r1.py:12:sys.path.insert(0, ".")`
- `scratch/matryoshka_r1.py:13:sys.path.insert(0, "scripts")`
- `scratch/matryoshka_r1.py:14:sys.path.insert(0, "scratch")`
- `scratch/matryoshka_r2.py:10:sys.path.insert(0, ".")`
- `scratch/matryoshka_r2.py:11:sys.path.insert(0, "scripts")`
- `scratch/matryoshka_r2.py:12:sys.path.insert(0, "scratch")`
- `scratch/metabolic_d2.py:10:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/metabolic_hot.py:9:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/metabolic_v3.py:24:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/metabolic_v4.py:11:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/metabolic_v5.py:14:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/metallicity_diets.py:29:sys.path.insert(0, os.path.dirname(os.path.dirname(`
- `scratch/moe_gt1_arm2.py:37:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scratch/moe_gt1.py:36:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scratch/muon_3ep_d256.py:15:sys.path.insert(0, ".")`
- `scratch/muon_3ep_d256.py:16:sys.path.insert(0, "scripts")`
- `scratch/nineteen_m_displace.py:14:sys.path.insert(0, ".")`
- `scratch/oracle_worker.py:23:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scratch/p2_crown_draws.py:11:sys.path.insert(0, ".")`
- `scratch/p2_crown_draws.py:12:sys.path.insert(0, "scripts")`
- `scratch/p3_autopsy.py:18:sys.path.insert(0, ".")`
- `scratch/p3_autopsy.py:19:sys.path.insert(0, "scripts")`
- `scratch/p3_autopsy.py:20:sys.path.insert(0, "scratch")`
- `scratch/p3_bits.py:20:sys.path.insert(0, ".")`
- `scratch/p3_bits.py:21:sys.path.insert(0, "scripts")`
- `scratch/p3_bits.py:22:sys.path.insert(0, "scratch")`
- `scratch/p3_ffnslack.py:20:sys.path.insert(0, ".")`
- `scratch/p3_ffnslack.py:21:sys.path.insert(0, "scripts")`
- `scratch/p3_ffnslack.py:22:sys.path.insert(0, "scratch")`
- `scratch/p3_grav2.py:15:sys.path.insert(0, ".")`
- `scratch/p3_grav2.py:16:sys.path.insert(0, "scripts")`
- `scratch/p3_grav2.py:17:sys.path.insert(0, "scratch")`
- `scratch/p3_quat.py:24:sys.path.insert(0, ".")`
- `scratch/p3_quat.py:25:sys.path.insert(0, "scripts")`
- `scratch/p3_quat.py:26:sys.path.insert(0, "scratch")`
- `scratch/p3_stream2x2.py:23:sys.path.insert(0, ".")`
- `scratch/p3_stream2x2.py:36:sys.path.insert(0, "scratch")`
- `scratch/p3_umoe_soft.py:11:sys.path.insert(0, ".")`
- `scratch/p3_umoe_soft.py:12:sys.path.insert(0, "scripts")`
- `scratch/p3_umoe_soft.py:13:sys.path.insert(0, "scratch")`
- `scratch/pack_baselines.py:12:sys.path.insert(0, ".")`
- `scratch/pack_baselines.py:13:sys.path.insert(0, "scripts")`
- `scratch/pack_c6.py:12:sys.path.insert(0, ".")`
- `scratch/pack_c7.py:12:sys.path.insert(0, ".")`
- `scratch/pack_c7.py:13:sys.path.insert(0, "scratch")`
- `scratch/pack_crystal.py:16:sys.path.insert(0, ".")`
- `scratch/pack_crystal.py:17:sys.path.insert(0, "scripts")`
- `scratch/pack_decode.py:18:sys.path.insert(0, ".")`
- `scratch/pack_decode.py:19:sys.path.insert(0, "scripts")`
- `scratch/pack_determinism.py:14:sys.path.insert(0, ".")`
- `scratch/pack_determinism.py:15:sys.path.insert(0, "scripts")`
- `scratch/pack_gemv.py:14:sys.path.insert(0, ".")`
- `scratch/pack_rans.py:14:sys.path.insert(0, ".")`
- `scratch/pack_tiered.py:13:sys.path.insert(0, ".")`
- `scratch/pack_tiered.py:14:sys.path.insert(0, "scripts")`
- `scratch/phase4_sites.py:3:rows are migrated. --bootstrap lists sys.path.insert sites instead.`
- `scratch/phase4_unboot.py:1:"""One-shot Phase 4.3: delete sys.path.insert bootstrap lines in the`
- `scratch/phase4_unboot.py:16:        if "sys.path.insert" in lines[i]:`
- `scratch/pincer_dist_probe.py:33:sys.path.insert(0, ".")`
- `scratch/pincer_dist_probe.py:34:sys.path.insert(0, "scripts")`
- `scratch/pincer_labels_v2.py:28:sys.path.insert(0, ".")`
- `scratch/pincer_labels_v2.py:29:sys.path.insert(0, "scripts")`
- `scratch/pincer_r0.py:16:sys.path.insert(0, ".")`
- `scratch/pincer_r0.py:17:sys.path.insert(0, "scripts")`
- `scratch/pincer_r0b.py:14:sys.path.insert(0, ".")`
- `scratch/pincer_r0b.py:15:sys.path.insert(0, "scripts")`
- `scratch/pincer_r1_indist.py:32:sys.path.insert(0, ".")`
- `scratch/pincer_r1_indist.py:33:sys.path.insert(0, "scripts")`
- `scratch/pincer_r1_probe.py:18:sys.path.insert(0, ".")`
- `scratch/pincer_r1_probe.py:19:sys.path.insert(0, "scripts")`
- `scratch/pincer_r1b_labels.py:18:sys.path.insert(0, ".")`
- `scratch/pincer_r1b_labels.py:19:sys.path.insert(0, "scripts")`
- `scratch/pincer_r8.py:16:sys.path.insert(0, ".")`
- `scratch/pincer_r8.py:17:sys.path.insert(0, "scripts")`
- `scratch/place1_gravity.py:8:sys.path.insert(0, ".")`
- `scratch/place1_gravity.py:9:sys.path.insert(0, "scratch")`
- `scratch/polar_snap.py:12:sys.path.insert(0, ".")`
- `scratch/polar_snap.py:13:sys.path.insert(0, "scripts")`
- `scratch/polar_snap.py:14:sys.path.insert(0, "scratch")`
- `scratch/practice_mine.py:15:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/probe_int_device_parity.py:22:sys.path.insert(0, ".")`
- `scratch/probe_int_device_parity.py:23:sys.path.insert(0, "scratch")`
- `scratch/quat_commutant.py:11:sys.path.insert(0, ".")`
- `scratch/quat_convert.py:12:sys.path.insert(0, ".")`
- `scratch/quat_convert.py:13:sys.path.insert(0, "scripts")`
- `scratch/quat_convert.py:14:sys.path.insert(0, "scratch")`
- `scratch/rank_read.py:7:sys.path.insert(0, ".")`
- `scratch/rank_read.py:8:sys.path.insert(0, "scripts")`
- `scratch/rank_read.py:9:sys.path.insert(0, "scratch")`
- `scratch/rev2_d768.py:22:sys.path.insert(0, ".")`
- `scratch/rev2_d768.py:23:sys.path.insert(0, "scripts")`
- `scratch/rev2_d768.py:24:sys.path.insert(0, "scratch")`
- `scratch/rev3_crown.py:34:sys.path.insert(0, ".")`
- `scratch/rev3_crown.py:35:sys.path.insert(0, "scripts")`
- `scratch/rev3_crown.py:36:sys.path.insert(0, "scratch")`
- `scratch/rev4_zx45.py:24:sys.path.insert(0, ".")`
- `scratch/rev4_zx45.py:25:sys.path.insert(0, "scripts")`
- `scratch/rev4_zx45.py:26:sys.path.insert(0, "scratch")`
- `scratch/rot_commutant.py:10:sys.path.insert(0, ".")`
- `scratch/rot_convert.py:10:sys.path.insert(0, ".")`
- `scratch/rot_convert.py:11:sys.path.insert(0, "scripts")`
- `scratch/rot_convert.py:12:sys.path.insert(0, "scratch")`
- `scratch/rot_snap_anatomy.py:10:sys.path.insert(0, ".")`
- `scratch/rot_snap_anatomy.py:11:sys.path.insert(0, "scripts")`
- `scratch/rot_snap_anatomy.py:12:sys.path.insert(0, "scratch")`
- `scratch/rotinstr_control.py:15:sys.path.insert(0, ".")`
- `scratch/saturation_s2.py:8:sys.path.insert(0, ".")`
- `scratch/saturation_s2.py:9:sys.path.insert(0, "scripts")`
- `scratch/saturation_s2b.py:8:sys.path.insert(0, ".")`
- `scratch/saturation_s2b.py:9:sys.path.insert(0, "scripts")`
- `scratch/scaffold_review.py:10:sys.path.insert(0, "scratch")`
- `scratch/scaffold_review.py:8:sys.path.insert(0, ".")`
- `scratch/scaffold_review.py:9:sys.path.insert(0, "scripts")`
- `scratch/scorer_s1_battery.py:22:sys.path.insert(0, ".")`
- `scratch/scorer_s1_battery.py:23:sys.path.insert(0, "scripts")`
- `scratch/scorer_s2_data.py:20:sys.path.insert(0, ".")`
- `scratch/scorer_s2_data.py:21:sys.path.insert(0, "scripts")`
- `scratch/scorer_s2_train.py:27:sys.path.insert(0, ".")`
- `scratch/scorer_s2_train.py:28:sys.path.insert(0, "scripts")`
- `scratch/series_probe.py:10:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/snap_alloc.py:10:sys.path.insert(0, "scripts")`
- `scratch/snap_alloc.py:11:sys.path.insert(0, "scratch")`
- `scratch/snap_alloc.py:9:sys.path.insert(0, ".")`
- `scratch/snap_anatomy.py:15:sys.path.insert(0, ".")`
- `scratch/snap_anatomy.py:16:sys.path.insert(0, "scripts")`
- `scratch/softprompt_sampler_probe.py:29:sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`
- `scratch/softprompt_sampler_probe.py:30:sys.path.insert(0, "scripts")`
- `scratch/softprompt1.py:24:sys.path.insert(0, os.path.dirname(os.path.dirname(`
- `scratch/softprompt1.py:26:sys.path.insert(0, "scripts")`
- `scratch/soup_gate.py:10:sys.path.insert(0, "scripts")`
- `scratch/soup_gate.py:11:sys.path.insert(0, "scratch")`
- `scratch/soup_gate.py:9:sys.path.insert(0, ".")`
- `scratch/ssm_star.py:33:sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))`
- `scratch/ssm_star.py:34:sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))`
- `scratch/star_profile.py:16:sys.path.insert(0, ".")`
- `scratch/star_profile.py:17:sys.path.insert(0, "scripts")`
- `scratch/streaming_birth_d256.py:17:sys.path.insert(0, ".")`
- `scratch/successors_acceptance.py:16:sys.path.insert(0, ".")`
- `scratch/successors_acceptance.py:17:sys.path.insert(0, "scripts")`
- `scratch/sym_birth.py:11:sys.path.insert(0, ".")`
- `scratch/sym_birth.py:12:sys.path.insert(0, "scripts")`
- `scratch/sym_birth.py:13:sys.path.insert(0, "scratch")`
- `scratch/sym_convert.py:13:sys.path.insert(0, ".")`
- `scratch/sym_convert.py:14:sys.path.insert(0, "scripts")`
- `scratch/sym_convert.py:15:sys.path.insert(0, "scratch")`
- `scratch/sym_spectrum.py:11:sys.path.insert(0, ".")`
- `scratch/sym_spectrum.py:12:sys.path.insert(0, "scripts")`
- `scratch/sym_spectrum.py:13:sys.path.insert(0, "scratch")`
- `scratch/sym45.py:12:sys.path.insert(0, ".")`
- `scratch/sym45.py:13:sys.path.insert(0, "scripts")`
- `scratch/synonym_test.py:8:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/tenet_d1_revgate.py:37:sys.path.insert(0, ".")`
- `scratch/tenet_d1_revgate.py:38:sys.path.insert(0, "scripts")`
- `scratch/tenet_d2_revdiet.py:44:sys.path.insert(0, ".")`
- `scratch/tenet_d2_revdiet.py:45:sys.path.insert(0, "scripts")`
- `scratch/tenet_mult_b32.py:14:sys.path.insert(0, ".")`
- `scratch/tenet_mult_b32.py:15:sys.path.insert(0, "scripts")`
- `scratch/tenet_mult_b32.py:16:sys.path.insert(0, "scratch")`
- `scratch/tenet_mult_census.py:33:sys.path.insert(0, ".")`
- `scratch/tenet_mult_census.py:34:sys.path.insert(0, "scripts")`
- `scratch/tenet_mult_census.py:35:sys.path.insert(0, "scratch")`
- `scratch/tenet_r1b_micro.py:28:sys.path.insert(0, ".")`
- `scratch/tenet_r1b_micro.py:29:sys.path.insert(0, "scripts")`
- `scratch/tenet_r1b_micro.py:30:sys.path.insert(0, "scratch")`
- `scratch/tenet_w0.py:31:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scratch/tenet_w1_bridge.py:31:sys.path.insert(0, ".")`
- `scratch/tenet_w1_relational.py:38:sys.path.insert(0, ".")`
- `scratch/tenet_w1_surfaces.py:35:sys.path.insert(0, ".")`
- `scratch/ternary_control.py:5:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/ternary_gate.py:5:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/ternary_session2.py:8:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scratch/tier_escalate.py:10:sys.path.insert(0, "scripts")`
- `scratch/tier_escalate.py:11:sys.path.insert(0, "scratch")`
- `scratch/tier_escalate.py:9:sys.path.insert(0, ".")`
- `scratch/tier_retry.py:10:sys.path.insert(0, ".")`
- `scratch/tier_retry.py:11:sys.path.insert(0, "scripts")`
- `scratch/tier_retry.py:12:sys.path.insert(0, "scratch")`
- `scratch/train_fp64.py:10:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scratch/train_fp64.py:9:sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))`
- `scratch/traj_accept.py:21:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scratch/umoe_conserve.py:26:sys.path.insert(0, ".")`
- `scratch/umoe_conserve.py:27:sys.path.insert(0, "scripts")`
- `scratch/umoe_conserve.py:28:sys.path.insert(0, "scratch")`
- `scratch/v4flash_census.py:23:sys.path.insert(0, ".")`
- `scratch/v4flash_census.py:24:sys.path.insert(0, "scratch")`
- `scratch/v4flash_f1b.py:32:sys.path.insert(0, "scratch")`
- `scratch/v4flash_f1c.py:31:sys.path.insert(0, "scratch")`
- `scratch/v4flash_f1d.py:40:sys.path.insert(0, "scratch")`
- `scratch/v4flash_router.py:26:sys.path.insert(0, ".")`
- `scratch/v4flash_router.py:27:sys.path.insert(0, "scratch")`
- `scratch/v4flash_rung0.py:30:sys.path.insert(0, ".")`
- `scratch/v4flash_rung2b_router.py:23:sys.path.insert(0, ".")`
- `scratch/v4flash_rung2b_router.py:24:sys.path.insert(0, "scratch")`
- `scratch/v4flash_rung2b.py:29:sys.path.insert(0, ".")`
- `scratch/v4flash_rung2b.py:30:sys.path.insert(0, "scratch")`
- `scratch/v4flash_rungA.py:31:sys.path.insert(0, ".")`
- `scratch/v4flash_rungd.py:50:sys.path.insert(0, ".")`
- `scratch/v4flash_rungd.py:51:sys.path.insert(0, "scratch")`
- `scratch/v4flash_rungd2.py:55:sys.path.insert(0, ".")`
- `scratch/v4flash_rungd2.py:56:sys.path.insert(0, "scratch")`
- `scratch/v4flash_s0.py:47:sys.path.insert(0, ".")`
- `scratch/v4flash_s0.py:48:sys.path.insert(0, "scratch")`
- `scratch/v4flash_twin.py:373:        sys.path.insert(0, "scratch")`
- `scratch/verify_intbirth_prims.py:21:sys.path.insert(0, BUILD)`
- `scratch/vrm_ab.py:9:sys.path.insert(0, "."); sys.path.insert(0, "scripts")`
- `scripts/bench_budget_alloc.py:43:    sys.path.insert(0, "scripts")`
- `scripts/bench_fused_ce.py:17:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/bench_hints_ab.py:30:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/bench_hints_ab.py:31:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/bench_metal_kernels.py:12:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/bench_pred_syndromes.py:59:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/bench_rotate_quantize.py:12:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/bench_step_diversity.py:32:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/bench_step_diversity.py:33:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/bench_stitch_poc.py:26:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/bench_syndrome_policy.py:38:    sys.path.insert(0, "scripts")`
- `scripts/bench_verify_fast.py:27:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/bench_verify_fast.py:28:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/bench_weight_anatomy.py:33:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/book.py:42:sys.path.insert(0, str(REPO_ROOT))`
- `scripts/consolidate_mathnative.py:18:sys.path.insert(0, str(Path(__file__).parent))`
- `scripts/consolidate_mathnative.py:19:sys.path.insert(0, str(Path(__file__).parent.parent))`
- `scripts/convert_diet_prefix.py:19:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/convert_diet_prefix.py:20:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/eval_pruned_moe.py:17:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/eval_ruler.py:15:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/expert_iter_steps.py:252:        sys_path = str(Path(__file__).resolve().parent)`
- `scripts/expert_iter_steps.py:254:        if sys_path not in _s.path:`
- `scripts/expert_iter_steps.py:255:            _s.path.insert(0, sys_path)`
- `scripts/expert_iter_steps.py:509:    sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/expert_loop.py:10:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/expert_loop.py:11:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/farm_algebra.py:37:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/gen_catalog.py:28:sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))`
- `scripts/gen_lake.py:18:sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
- `scripts/gen_magic_labels.py:143:    sys.path.insert(0, "scripts")`
- `scripts/grow_mathnative.py:20:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/moe_router_stats.py:21:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/probe_depth.py:15:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/probe_depth.py:16:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/step_grpo_micro.py:23:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/step_grpo_micro.py:24:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/step_grpo.py:30:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/step_grpo.py:31:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/task_arithmetic.py:18:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/tournament_birth.py:14:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/tournament_birth.py:15:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/train_calculus.py:13:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/train_mathnative.py:20:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/train_proposer.py:14:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/train_ternary.py:16:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/train_ternary.py:17:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/train_value_head.py:150:    sys.path.insert(0, "scripts")`
- `scripts/train_value_head.py:48:    sys.path.insert(0, "scripts")`
- `scripts/train_weight_reader.py:22:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- `scripts/validity_autopsy.py:27:sys.path.insert(0, str(Path(__file__).resolve().parent))`
- `scripts/validity_autopsy.py:28:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`

## 3. Hardcoded path globs

### 3a. scripts/ + llmopt/ (203 hits) — the spec's gate step 1

- `llmopt/lab/config.py:47:"logs/gate.jsonl`
- `llmopt/lab/config.py:50:"logs/gate.jsonl`
- `llmopt/lab/shards.py:39:"checkpoints/v4flash_f1`
- `llmopt/runs/lake.py:173:"data/catalog/models.jsonl`
- `llmopt/runs/lake.py:176:"data/catalog/models.jsonl`
- `llmopt/runs/lake.py:40:"data/lake`
- `llmopt/runs/receipts.py:78:"logs/myrun/steps.jsonl`
- `llmopt/vendor/axiom/nn_exact_ref.py:33:"data/qual`
- `scripts/arena.py:96:"checkpoints/step_lora.pt`
- `scripts/autopsy_int.py:68:"checkpoints/nnue_eval.pt`
- `scripts/bench_adaptive.py:44:"checkpoints/proposer_lora.pt`
- `scripts/bench_anneal.py:114:"checkpoints/nnue_eval.pt`
- `scripts/bench_bandit.py:113:"checkpoints/nnue_eval.pt`
- `scripts/bench_bandit.py:71:"checkpoints/markov_prior.json`
- `scripts/bench_bestfirst_llm.py:49:"checkpoints/proposer_lora.pt`
- `scripts/bench_bestfirst_llm.py:85:"checkpoints/nnue_eval.pt`
- `scripts/bench_bestfirst_nnue.py:130:"checkpoints/nnue_eval.pt`
- `scripts/bench_bestfirst_nnue.py:72:"checkpoints/markov_prior.json`
- `scripts/bench_bestfirst.py:31:"checkpoints/markov_prior.json`
- `scripts/bench_budget_alloc.py:31:"checkpoints/magic_estimator_v5.pt`
- `scripts/bench_commute.py:127:"checkpoints/nnue_eval.pt`
- `scripts/bench_engine_regret.py:228:"data/engine_regret_labels.jsonl`
- `scripts/bench_engine_regret.py:30:"checkpoints/engine_regret_probe.pt`
- `scripts/bench_flash_prefill.py:13:"data/flash_tile_labels.jsonl`
- `scripts/bench_frontier.py:110:"checkpoints/proposer_lora.pt`
- `scripts/bench_frontier.py:111:"checkpoints/proposer_lora_r2.pt`
- `scripts/bench_frontier.py:71:"data/frontier_r1_roots.json`
- `scripts/bench_fused.py:148:"checkpoints/nnue_eval.pt`
- `scripts/bench_fused.py:80:"checkpoints/proposer_lora.pt`
- `scripts/bench_fused.py:83:"checkpoints/value_head_v2.pt`
- `scripts/bench_fused.py:84:"checkpoints/value_head.pt`
- `scripts/bench_gated.py:40:"checkpoints/rule_gate.pt`
- `scripts/bench_gweight.py:25:"checkpoints/markov_prior.json`
- `scripts/bench_hints_ab.py:137:"checkpoints/step_lora.pt`
- `scripts/bench_hints_ab.py:82:"checkpoints/pred_syndromes_l15.pt`
- `scripts/bench_hybrid.py:46:"checkpoints/proposer_lora.pt`
- `scripts/bench_int4_config_sweep.py:25:"data/int4_config_labels.jsonl`
- `scripts/bench_interference.py:122:"checkpoints/nnue_eval.pt`
- `scripts/bench_lazy.py:126:"checkpoints/nnue_eval.pt`
- `scripts/bench_llm_gating.py:41:"checkpoints/magic_estimator.pt`
- `scripts/bench_magic.py:139:"checkpoints/nnue_eval.pt`
- `scripts/bench_markov_adaptive.py:32:"data/proposer_train.jsonl`
- `scripts/bench_markov.py:29:"data/proposer_train.jsonl`
- `scripts/bench_nnue.py:105:"checkpoints/nnue_eval.pt`
- `scripts/bench_opcap.py:66:"checkpoints/nnue_eval.pt`
- `scripts/bench_pred_syndromes.py:259:"data/pred_syndrome_orbitals.jsonl`
- `scripts/bench_pred_syndromes.py:418:"checkpoints/pred_syndromes_emb.pt`
- `scripts/bench_pred_syndromes.py:419:"checkpoints/pred_syndromes_emb_orb.pt`
- `scripts/bench_pred_syndromes.py:531:"checkpoints/pred_syndromes_lora.pt`
- `scripts/bench_pred_syndromes.py:63:"data/pred_syndrome_labels.jsonl`
- `scripts/bench_pred_syndromes.py:64:"checkpoints/pred_syndromes.pt`
- `scripts/bench_pred_syndromes.py:94:"data/step_chains.jsonl`
- `scripts/bench_proposer.py:48:"checkpoints/nnue_eval.pt`
- `scripts/bench_proposer.py:63:"checkpoints/proposer_lora.pt`
- `scripts/bench_record.py:135:"checkpoints/nnue_eval.pt`
- `scripts/bench_record.py:78:"checkpoints/proposer_lora.pt`
- `scripts/bench_regret_resample.py:422:"data/regret_trace_labels.jsonl`
- `scripts/bench_regret_resample.py:427:"data/regret_pool.jsonl`
- `scripts/bench_regret_resample.py:91:"checkpoints/calculus_lora.pt`
- `scripts/bench_regret_resample.py:92:"checkpoints/regret_probe.pt`
- `scripts/bench_rule_basis.py:101:"checkpoints/nnue_eval.pt`
- `scripts/bench_stack_winners.py:57:"checkpoints/nnue_eval.pt`
- `scripts/bench_step_diversity.py:59:"checkpoints/step_lora.pt`
- `scripts/bench_stitch_poc.py:86:"data/pred_syndrome_labels.jsonl`
- `scripts/bench_stitch_poc.py:88:"data/pred_syndrome_orbitals.jsonl`
- `scripts/bench_syndrome_head.py:197:"checkpoints/step_lora_syn0.pt`
- `scripts/bench_syndrome_head.py:198:"checkpoints/step_lora_syn3.pt`
- `scripts/bench_syndrome_head.py:202:"checkpoints/step_lora_syn0.pt`
- `scripts/bench_syndrome_head.py:204:"checkpoints/step_lora_syn3.pt`
- `scripts/bench_syndrome_head.py:38:"data/step_chains.jsonl`
- `scripts/bench_syndrome_head.py:40:"data/pred_syndrome_labels.jsonl`
- `scripts/bench_syndrome_policy.py:36:"checkpoints/syndrome_policy.pt`
- `scripts/bench_temp_race.py:30:"checkpoints/step_lora.pt`
- `scripts/bench_verify_fast.py:42:"data/step_chains.jsonl`
- `scripts/bench_weight_anatomy.py:36:"checkpoints/step_lora_pre_grpo_backup.pt`
- `scripts/bench_weight_anatomy.py:37:"checkpoints/step_lora_control.pt`
- `scripts/bench_weight_anatomy.py:38:"checkpoints/step_lora_dietB.pt`
- `scripts/bench_weight_anatomy.py:39:"checkpoints/step_lora_syn0.pt`
- `scripts/bench_weight_anatomy.py:40:"checkpoints/step_lora_syn3.pt`
- `scripts/bench_weight_anatomy.py:42:"checkpoints/step_lora_grpo.pt`
- `scripts/bench_weight_anatomy.py:43:"checkpoints/step_lora_pre_grpo_backup.pt`
- `scripts/bench_zx_r3.py:28:"checkpoints/zx_markov_prior.json`
- `scripts/bench_zx_r7.py:34:"checkpoints/zx_markov_prior_v2.json`
- `scripts/build_gen7_diet.py:21:"data/micromodel_chains_shard*.jsonl`
- `scripts/build_gen7_diet.py:23:"data/step_chains.jsonl`
- `scripts/build_gen7_diet.py:24:"data/micromodel_v22_shard*.jsonl`
- `scripts/build_gen7_diet.py:26:"data/micromodel_l8_shard*.jsonl`
- `scripts/build_gen7_diet.py:29:"data/micromodel_gen4_sidecar.jsonl`
- `scripts/build_gen7_diet.py:30:"data/micromodel_l9a_shard*.jsonl`
- `scripts/build_gen7_diet.py:41:"data/gen7_diet.jsonl`
- `scripts/consolidate_mathnative.py:23:"data/micromodel_grpo_mined.jsonl`
- `scripts/control_round.py:24:"data/step_chains_r23.jsonl`
- `scripts/control_round.py:25:"checkpoints/step_lora_control.pt`
- `scripts/control_round.py:40:"checkpoints/step_lora.pt`
- `scripts/convert_diet_prefix.py:38:"data/gen4_diet_infix.jsonl`
- `scripts/convert_diet_prefix.py:39:"data/gen4_diet_prefix.jsonl`
- `scripts/eval_mathnative.py:131:"checkpoints/mathnative_19m.pt`
- `scripts/eval_mathnative.py:56:"data/micromodel_chains_shard*.jsonl`
- `scripts/eval_mathnative.py:57:"data/micromodel_algebra_shard*.jsonl`
- `scripts/eval_mathnative.py:58:"data/micromodel_calc_l4_shard*.jsonl`
- `scripts/eval_mathnative.py:59:"data/step_chains.jsonl`
- `scripts/eval_pruned_moe.py:25:"checkpoints/router_stats.json`
- `scripts/expert_iter_steps.py:257:"checkpoints/magic_estimator_v7.pt`
- `scripts/expert_iter_steps.py:26:"data/step_chains.jsonl`
- `scripts/expert_iter_steps.py:27:"checkpoints/step_lora.pt`
- `scripts/expert_iter_steps.py:423:"data/ode_chains.jsonl`
- `scripts/expert_loop.py:129:"checkpoints/step_lora.pt`
- `scripts/expert_loop.py:147:"checkpoints/step_lora_r{round_no}.pt`
- `scripts/expert_loop.py:75:"data/step_chains.jsonl`
- `scripts/farm_algebra.py:42:"data/micromodel_algebra_shard0.jsonl`
- `scripts/farm_l4_calc.py:25:"data/micromodel_calc_l4_shard0.jsonl`
- `scripts/farm_v22.py:143:"data/micromodel_v22_shard{a.part}.jsonl`
- `scripts/gen_catalog.py:86:"checkpoints/`
- `scripts/gen_dispatch_labels_v2.py:147:"data/dispatch_labels_v2.jsonl`
- `scripts/gen_dispatch_labels.py:124:"data/dispatch_labels.jsonl`
- `scripts/gen_frontier.py:148:"data/frontier_gaps.jsonl`
- `scripts/gen_frontier.py:88:"checkpoints/magic_estimator.pt`
- `scripts/gen_magic_labels.py:145:"checkpoints/magic_estimator.pt`
- `scripts/gen_magic_labels.py:232:"data/magic_labels.jsonl`
- `scripts/gen_policy_labels.py:119:"data/magic_labels_all.jsonl`
- `scripts/gen_policy_labels.py:121:"data/policy_labels.jsonl`
- `scripts/gen_proposer_data.py:75:"data/proposer_{split}.jsonl`
- `scripts/gen_proposer_data.py:92:"data/proposer_{split}_roots.json`
- `scripts/gen_regret_labels.py:139:"data/regret_labels.jsonl`
- `scripts/gen_syndrome_labels.py:81:"data/magic_labels_all_rf.jsonl`
- `scripts/gen_syndrome_labels.py:83:"data/syndrome_labels.jsonl`
- `scripts/harvest_champion.py:70:"data/champion_harvest.jsonl`
- `scripts/harvest_frontier.py:111:"data/frontier_r1.jsonl`
- `scripts/harvest_frontier.py:147:"data/frontier_r1_roots.json`
- `scripts/harvest_frontier.py:49:"checkpoints/proposer_lora.pt`
- `scripts/log_hygiene.py:63:"logs/archive/logs/archive`
- `scripts/log_hygiene.py:85:"logs/`
- `scripts/markov_eval.py:150:"checkpoints/nnue_eval.pt`
- `scripts/markov_eval.py:35:"data/proposer_train.jsonl`
- `scripts/markov_prior.py:25:"data/proposer_train.jsonl`
- `scripts/markov_prior.py:26:"data/proposer_eval.jsonl`
- `scripts/mine_highways.py:8:"data/proposer_train.jsonl`
- `scripts/mine_prior_update.py:32:"checkpoints/markov_prior.json`
- `scripts/moe_router_stats.py:29:"checkpoints/router_stats.json`
- `scripts/plot_gt1_crest.py:138:"docs/assets/gallery/gt1-crest-small-multiples{suffix}.png`
- `scripts/plot_identity_crest.py:83:"docs/assets/gallery/identity-crest-fresh-seeds{suffix}.png`
- `scripts/render_gallery.py:25:"docs/assets/gallery`
- `scripts/step_grpo_micro.py:246:"checkpoints/mathnative_19m.pt`
- `scripts/step_grpo_micro.py:40:"checkpoints/mathnative_grpo.pt`
- `scripts/step_grpo_micro.py:41:"data/micromodel_grpo_mined.jsonl`
- `scripts/step_grpo.py:198:"checkpoints/step_lora.pt`
- `scripts/step_grpo.py:266:"checkpoints/step_lora_grpo_tmp.pt`
- `scripts/step_grpo.py:310:"checkpoints/step_lora.pt`
- `scripts/step_grpo.py:56:"checkpoints/step_lora_grpo.pt`
- `scripts/step_grpo.py:57:"data/step_chains.jsonl`
- `scripts/tabula_rasa_r0.py:102:"data/tr_round0.jsonl`
- `scripts/tabula_rasa_r0.py:122:"data/tr_round0_roots.json`
- `scripts/tabula_rasa_r1.py:122:"data/tr_round0_roots.json`
- `scripts/tabula_rasa_r1.py:123:"data/tr_round1.jsonl`
- `scripts/tabula_rasa_r1.py:152:"data/tr_round1_roots.json`
- `scripts/tabula_rasa_r1.py:55:"checkpoints/proposer_tr_r1.pt`
- `scripts/tabula_rasa_r2.py:107:"checkpoints/proposer_tr_r1.pt`
- `scripts/tabula_rasa_r2.py:108:"checkpoints/proposer_tr_r2.pt`
- `scripts/tabula_rasa_r2.py:109:"data/tr_round0_roots.json`
- `scripts/tabula_rasa_r2.py:110:"data/tr_round1_roots.json`
- `scripts/tabula_rasa_r2.py:111:"data/tr_round2.jsonl`
- `scripts/tabula_rasa_r2.py:139:"data/tr_round2_roots.json`
- `scripts/task_arithmetic.py:29:"checkpoints/calculus_lora.pt`
- `scripts/task_composition.py:36:"checkpoints/diff_only_lora.pt`
- `scripts/task_composition.py:37:"checkpoints/int_only_lora.pt`
- `scripts/tournament_birth.py:88:"checkpoints/tourn_{a.alpha}{a.tag}.pt`
- `scripts/train_calculus.py:31:"checkpoints/int_only_lora.pt`
- `scripts/train_dispatcher.py:78:"data/dispatch_labels_mac.jsonl`
- `scripts/train_dispatcher.py:79:"data/dispatch_labels_3080.jsonl`
- `scripts/train_dispatcher.py:82:"checkpoints/dispatcher.pt`
- `scripts/train_magic_estimator.py:119:"data/magic_labels.jsonl`
- `scripts/train_magic_estimator.py:122:"checkpoints/magic_estimator.pt`
- `scripts/train_magic_llm.py:132:"checkpoints/magic_llm.pt`
- `scripts/train_magic_llm.py:139:"data/magic_labels.jsonl`
- `scripts/train_magic_llm.py:49:"checkpoints/proposer_lora.pt`
- `scripts/train_mathnative.py:24:"checkpoints/mathnative_19m.pt`
- `scripts/train_mathnative.py:25:"checkpoints/mathnative_19m_v2.pt`
- `scripts/train_mathnative.py:38:"data/micromodel_chains_shard*.jsonl`
- `scripts/train_mathnative.py:40:"data/step_chains.jsonl`
- `scripts/train_mathnative.py:42:"data/micromodel_algebra_shard*.jsonl`
- `scripts/train_mathnative.py:45:"data/micromodel_calc_l4_shard*.jsonl`
- `scripts/train_mathnative.py:49:"data/micromodel_v22_shard*.jsonl`
- `scripts/train_mathnative.py:54:"data/micromodel_l8_shard*.jsonl`
- `scripts/train_mathnative.py:57:"data/gen7_diet.jsonl`
- `scripts/train_mathnative.py:61:"data/micromodel_gen4_sidecar.jsonl`
- `scripts/train_nnue.py:176:"checkpoints/nnue_eval.pt`
- `scripts/train_proposer.py:152:"data/proposer_train.jsonl`
- `scripts/train_proposer.py:154:"data/proposer_eval.jsonl`
- `scripts/train_proposer.py:28:"checkpoints/proposer_lora.pt`
- `scripts/train_proposer.py:84:"data/proposer_train.jsonl`
- `scripts/train_proposer.py:85:"data/proposer_eval.jsonl`
- `scripts/train_syndrome_decoder.py:79:"checkpoints/syndrome_decoder.pt`
- `scripts/train_syndrome_decoder.py:86:"data/syndrome_labels.jsonl`
- `scripts/train_syndrome_policy.py:86:"checkpoints/rule_gate.pt`
- `scripts/train_syndrome_policy.py:87:"checkpoints/syndrome_policy.pt`
- `scripts/train_syndrome_policy.py:96:"data/policy_labels.jsonl`
- `scripts/train_ternary.py:44:"checkpoints/mathnative_45m_ternary.pt`
- `scripts/train_value_head.py:159:"checkpoints/value_head_v2.pt`
- `scripts/train_value_head.py:160:"checkpoints/value_head.pt`
- `scripts/train_value_head.py:35:"data/value_labels.jsonl`
- `scripts/train_value_head.py:81:"checkpoints/proposer_lora.pt`
- `scripts/train_weight_reader.py:29:"checkpoints/weight_reader_results.json`
- `scripts/validity_autopsy.py:154:"data/autopsy_structural.jsonl`

### 3b. scratch/ (470 hits, same grep, listed for completeness)

- `scratch/absorb_1e5.py:32:"checkpoints/mathnative_gen6_ternary_latent.pt`
- `scratch/attractor_census.py:23:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/attractor_census.py:30:"logs/data_ceil/attractor_census_d256.jsonl`
- `scratch/attractor_census.py:62:"logs/data_ceil`
- `scratch/attractor_census2.py:26:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/attractor_census2.py:34:"logs/data_ceil/attractor_census2_d256.jsonl`
- `scratch/attractor_census2.py:92:"logs/data_ceil`
- `scratch/basin_probe.py:26:"checkpoints/umoe_lb_s{SEED}.pt`
- `scratch/birth19m_snaps.py:30:"checkpoints/gallery19m_s{SEED}.pt`
- `scratch/blackhole_b0.py:24:"checkpoints/blackhole_q3_parts`
- `scratch/blackhole_b0.py:25:"logs/blackhole_atlas.jsonl`
- `scratch/build_dist_diets.py:24:"data/dist_rows_d256.jsonl`
- `scratch/build_dist_diets.py:31:"data/diet_dosectl_d256.jsonl`
- `scratch/build_dist_diets.py:32:"data/diet_dist_d256.jsonl`
- `scratch/build_merged_diet.py:12:"data/micromodel_l9a_shard*.jsonl`
- `scratch/build_merged_diet.py:13:"data/pull_l9a/*.jsonl`
- `scratch/build_merged_diet.py:32:"data/merged_diet.jsonl`
- `scratch/cal_dilute.py:25:"checkpoints/cal_dilute_{int`
- `scratch/cal_dk_probe.py:19:"checkpoints/sym_birth_dense_mps_h8_ema.pt`
- `scratch/capacity_meter.py:78:"checkpoints/sym_birth_dense_mps_h8_ema.pt`
- `scratch/capacity_meter.py:80:"checkpoints/sym_birth_dense_mps_L4_ema.pt`
- `scratch/capacity_meter.py:82:"checkpoints/cplx_none.pt`
- `scratch/ce_gate_study.py:25:"checkpoints/mathnative_wfloor_d256_muon.pt`
- `scratch/ce_gate_study.py:26:"checkpoints/mathnative_wfloor_d256_stream3.pt`
- `scratch/ce_gate_study.py:27:"checkpoints/mathnative_wfloor_d256_stream4.pt`
- `scratch/ce_gate_study.py:28:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/chain_carry.py:101:"checkpoints/cc_{ARM}.pt`
- `scratch/chain_carry.py:21:"data/micromodel_chains_shard*.jsonl`
- `scratch/champ_cuda_probe.py:14:"checkpoints/mathnative_gen6_grown.pt`
- `scratch/churn_judge_eval.py:21:"logs/opus/moe_gt1_perprob.jsonl`
- `scratch/ckpt_delete_pass.py:45:"checkpoints/`
- `scratch/ckpt_delete_pass.py:47:"logs/triage/uncited_but_consumed.json`
- `scratch/ckpt_delete_pass.py:48:"logs/triage/triage_table.jsonl`
- `scratch/ckpt_delete_pass.py:71:"logs/triage/delete_pass_{host}.json`
- `scratch/ckpt_inventory.py:16:"logs/triage/inventory.jsonl`
- `scratch/ckpt_triage_table.py:22:"logs/triage/mac_inventory.jsonl`
- `scratch/ckpt_triage_table.py:23:"logs/triage/wsl_inventory.jsonl`
- `scratch/ckpt_triage_table.py:62:"logs/triage/triage_table.jsonl`
- `scratch/ckpt_triage_table.py:67:"logs/triage/triage_table.md`
- `scratch/clade_stream_d256.py:41:"checkpoints/mathnative_wfloor_d256_clade2.pt`
- `scratch/clade_stream_d256.py:43:"checkpoints/mathnative_wfloor_d256_clade.pt`
- `scratch/complex_birth.py:36:"checkpoints/cplx_{alpha}{a.tag}.pt`
- `scratch/complex_nnue.py:92:"data/magic_labels_v7.jsonl`
- `scratch/complexify_control.py:29:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/confluence.py:6:"checkpoints/mathnative_gen6_grown.pt`
- `scratch/confluence.py:7:"checkpoints/metabolic_live.pt`
- `scratch/corner_snap.py:20:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/corner_snap.py:21:"checkpoints/sym_circ8_b.pt`
- `scratch/crystal_recreate_test.py:32:"checkpoints/qwen05b_base_l14gate.pt`
- `scratch/crystal_recreate_test.py:42:"checkpoints/step_lora.pt`
- `scratch/crystal_recreate_test.py:43:"checkpoints/step_lora_grpo.pt`
- `scratch/crystal_recreate_test.py:44:"checkpoints/step_lora_pre_grpo_backup.pt`
- `scratch/crystal_recreate_test.py:50:"checkpoints/mathnative_19m_gen8.pt`
- `scratch/crystal_recreate_test.py:95:"figs/2026-08-08/crystal-recreate-test.png`
- `scratch/d2_verify.py:12:"checkpoints/metab_d2_fp64.pt`
- `scratch/d2_verify.py:13:"checkpoints/metab_d2_dd.pt`
- `scratch/detbwd_diet.py:41:"data/micromodel_gen4_sidecar.jsonl`
- `scratch/detbwd_gravmoe.py:283:"data/micromodel_gen4_sidecar.jsonl`
- `scratch/determinability_census.py:48:"logs/data_ceil/determinability_gen4.jsonl`
- `scratch/determinability_census.py:63:"logs/data_ceil`
- `scratch/distortion_collapse.py:101:"checkpoints/snap19m_q{q}.pt`
- `scratch/distortion_collapse.py:64:"checkpoints/cplx_none.pt`
- `scratch/distortion_collapse.py:80:"checkpoints/sym_birth_dense_w56_ema.pt`
- `scratch/distortion_collapse.py:98:"checkpoints/mathnative_19m.pt`
- `scratch/dual_probe.py:21:"data/phys_probe.jsonl`
- `scratch/duo_mine.py:24:"data/*.jsonl`
- `scratch/duo_mine.py:58:"checkpoints/mathnative_gen6_grown.pt`
- `scratch/duo_mine.py:63:"checkpoints/mathnative_gen6_ternary.pt`
- `scratch/duo_mine.py:69:"data/duo_mined_shard1.jsonl`
- `scratch/duo_wave.py:22:"data/*.jsonl`
- `scratch/duo_wave.py:56:"checkpoints/mathnative_gen6_grown.pt`
- `scratch/duo_wave.py:61:"checkpoints/mathnative_gen6_ternary.pt`
- `scratch/e2_logit_check.py:17:"data/scorer_s2_battery20.txt`
- `scratch/e2_logit_check.py:19:"data/scorer_s2_battery20_meta.jsonl`
- `scratch/e2_logit_check.py:21:"data/scorer_s2_expected_logits.txt`
- `scratch/e2_logit_check.py:30:"checkpoints/scorer_s2_dist.pt`
- `scratch/e3_battery.py:24:"checkpoints/scorer_s2_dist.pt`
- `scratch/e3_battery.py:72:"data/e3_battery50.txt`
- `scratch/e3_battery.py:75:"data/e3_battery50_meta.jsonl`
- `scratch/e3_battery.py:78:"data/e3_expected_greedy.txt`
- `scratch/e3_battery.py:82:"data/e3_battery50_meta.jsonl`
- `scratch/e3_battery.py:82:"data/e3_battery50.txt`
- `scratch/e3_battery.py:83:"data/e3_expected_greedy.txt`
- `scratch/engine_scale_export.py:55:"data/micromodel_gen4_sidecar.jsonl`
- `scratch/ex1_swap.py:102:"checkpoints/{name}.json`
- `scratch/ex1_swap.py:118:"checkpoints/{name}.json`
- `scratch/ex1_swap.py:41:"checkpoints/moe_gt1_arm0.json`
- `scratch/ex1_swap.py:44:"checkpoints/gt3_core_keep.json`
- `scratch/ex1_swap.py:76:"logs/opus/gt3_prose_traj.jsonl`
- `scratch/ex1_swap.py:77:"logs/opus/gt4_dialog_traj.jsonl`
- `scratch/ex1_swap.py:82:"checkpoints/{hi_f}.json`
- `scratch/ex1_swap.py:84:"checkpoints/{lo_f}.json`
- `scratch/ex2_build.py:101:"checkpoints/{lo_f}.json`
- `scratch/ex2_build.py:139:"checkpoints/{name}.json`
- `scratch/ex2_build.py:53:"checkpoints/moe_gt1_arm0.json`
- `scratch/ex2_build.py:56:"checkpoints/gt3_core_keep.json`
- `scratch/ex2_build.py:93:"logs/opus/gt3_prose_traj.jsonl`
- `scratch/ex2_build.py:94:"logs/opus/gt4_dialog_traj.jsonl`
- `scratch/ex2_build.py:99:"checkpoints/{hi_f}.json`
- `scratch/ex3_build.py:100:"logs/opus/gt3_prose_traj.jsonl`
- `scratch/ex3_build.py:101:"logs/opus/gt4_dialog_traj.jsonl`
- `scratch/ex3_build.py:103:"logs/opus/moe_gt1_traj_v2.jsonl`
- `scratch/ex3_build.py:42:"checkpoints/moe_gt1_arm0.json`
- `scratch/ex3_build.py:45:"checkpoints/gt3_core_keep.json`
- `scratch/ex3_build.py:62:"checkpoints/{hf}.json`
- `scratch/ex3_build.py:64:"checkpoints/{lf}.json`
- `scratch/ex3_build.py:82:"checkpoints/{name}.json`
- `scratch/exact_twin_d56.py:18:"checkpoints/sym_birth_dense_w56_ema.pt`
- `scratch/exact_twin_d56.py:73:"checkpoints/exact_twin_d56_q16.pt`
- `scratch/exact1_small_cells.py:52:"logs/exact1small/{args.cell}`
- `scratch/exchange_test.py:138:"checkpoints/exchange_p1.pt`
- `scratch/farm_dist_rows.py:39:"data/dist_rows_d256.jsonl`
- `scratch/farmer_probe.py:29:"checkpoints/sym_birth_dense_revfarm_ema.pt`
- `scratch/fig_magic_scatter.py:132:"figs/2026-08-09/magic_scatter.png`
- `scratch/fig_magic_scatter.py:62:"data/magic_labels_all_rf.jsonl`
- `scratch/fig_magic_scatter.py:64:"checkpoints/magic_estimator_rf.pt`
- `scratch/format_delta_prep.py:24:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/format_delta_prep.py:65:"checkpoints/fmt_row_emb.pt`
- `scratch/format_ladder.py:108:"data/altpairs_rows.jsonl`
- `scratch/format_ladder.py:36:"checkpoints/fmt_{FORMAT}_{SCHED}.pt`
- `scratch/fourier_g9.py:35:"checkpoints/fourier_g9_{ALPHA}.pt`
- `scratch/fourier_probe.py:12:"checkpoints/sym_birth_dense_mps_h8_ema.pt`
- `scratch/fourier2_modbirth.py:19:"data/nt_callspan_pilot500.jsonl`
- `scratch/fourier2_modbirth.py:20:"checkpoints/fourier2_modbirth.pt`
- `scratch/fourier2b_widemod.py:30:"checkpoints/fourier2b_widemod.pt`
- `scratch/fourier3_algdiet.py:42:"checkpoints/fourier3_algdiet.pt`
- `scratch/fourier4a_dynamics.py:31:"checkpoints/fourier4a_dynamics.pt`
- `scratch/fp64_paired.py:39:"checkpoints/mathnative_gen6_ternary_latent.pt`
- `scratch/frozen_products/g19bn_probe.py:48:"data/series_probe_1e.jsonl`
- `scratch/frozen_products/poly_probe_representative.py:48:"data/poly3_probe.jsonl`
- `scratch/g5_polar.py:61:"checkpoints/cplx_G5_dep.pt`
- `scratch/g5_polar.py:62:"checkpoints/cplx_none.pt`
- `scratch/gate_pp.py:33:"logs/pp_{label}.jsonl`
- `scratch/gate_rarity.py:26:"data/*.jsonl`
- `scratch/gate_regate.py:20:"checkpoints/umoe_lb_s1.pt`
- `scratch/gate_transcripts.py:13:"checkpoints/umoe_gravmoe_s1.pt`
- `scratch/gate_transcripts.py:28:"checkpoints/umoe_gravmoe_s1.pt`
- `scratch/gate_zx.py:43:"data/zx_farm1_held.jsonl`
- `scratch/gauge_distance_d256.py:14:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/gauge_distance_d256.py:15:"checkpoints/mathnative_wfloor_d256_s2.pt`
- `scratch/gauge_distance_d256.py:16:"checkpoints/mathnative_wfloor_d256_s3.pt`
- `scratch/gauge_distance_d256.py:17:"checkpoints/mathnative_wfloor_d256_pack.pt`
- `scratch/gauge_distance_d256.py:18:"checkpoints/mathnative_wfloor_d256_stream4.pt`
- `scratch/gauge_distance_d256.py:19:"checkpoints/mathnative_wfloor_d256_clade2.pt`
- `scratch/gauge_m4x.py:12:"checkpoints/mathnative_19m_infixtwin.pt`
- `scratch/gauge_slack_rat.py:17:"checkpoints/mathnative_19m_mac_fp32.pt`
- `scratch/gauge_slack_rat.py:18:"checkpoints/mathnative_19m_mac_fp32_s2.pt`
- `scratch/gauge_slack_rat.py:19:"checkpoints/mathnative_19m_mac_ratq6_dep.pt`
- `scratch/gauge_slack_rat.py:20:"checkpoints/mathnative_19m_mac_ratq6_s2_dep.pt`
- `scratch/gen_lab_overview_pdf.py:107:"docs/assets/identity-crest-fresh-seeds.png`
- `scratch/gen_lab_overview_pdf.py:135:"docs/assets/neurons-qwen-vs-19m.png`
- `scratch/gen_lab_overview_pdf.py:16:"figs/2026-08-08/lab-overview.pdf`
- `scratch/gen_lab_overview_pdf.py:83:"docs/assets/neurons-19m-zoom.png`
- `scratch/gen_lab_overview_pdf.py:98:"docs/assets/gt1-crest-small-multiples.png`
- `scratch/graph_mod_sigma.py:15:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/graph_mod_sigma.py:16:"checkpoints/mathnative_wfloor_d256_s2.pt`
- `scratch/graph_mod_sigma.py:17:"checkpoints/mathnative_wfloor_d256_s3.pt`
- `scratch/graph_modularity_gen8.py:59:"checkpoints/mathnative_19m_gen8.pt`
- `scratch/graph_modularity_gen8.py:60:"checkpoints/mathnative_19m.pt`
- `scratch/grav_posthoc.py:26:"checkpoints/umoe_lb_s1.pt`
- `scratch/grav_probe.py:24:"checkpoints/umoe_lb_s{SEED}.pt`
- `scratch/grav1b_distance.py:22:"checkpoints/umoe_lb_s{SEED}.pt`
- `scratch/grav2_spacetime.py:31:"checkpoints/grav2_{ARM}_s{SEED}.pt`
- `scratch/grpo_shaped.py:95:"checkpoints/mathnative_gen6_grown.pt`
- `scratch/grpo_shaped.py:96:"checkpoints/grpo_shaped.pt`
- `scratch/grpo_shaped.py:97:"checkpoints/grpo_shaped.pt`
- `scratch/gt2_code_arm0.py:108:"logs/opus/gt2_code_traj.jsonl`
- `scratch/gt2_code_arm0.py:35:"checkpoints/gt2_code_arm0.json`
- `scratch/gt2_code_arm0.py:37:"logs/opus/moe_gt1.jsonl`
- `scratch/gt2_code_arm0.py:38:"logs/opus/moe_gt1_smoke.jsonl`
- `scratch/gt2_jaccard.py:31:"logs/opus/moe_gt1_traj_v2.jsonl`
- `scratch/gt2_jaccard.py:32:"logs/opus/gt2_phys_traj.jsonl`
- `scratch/gt2_jaccard.py:33:"logs/opus/gt2_code_traj.jsonl`
- `scratch/gt2_jaccard.py:57:"checkpoints/gt2_{d}_arm0_decode.json`
- `scratch/gt4_dialog_prompts.py:57:"checkpoints/gt3_prose_prompts.json`
- `scratch/gt4_dialog_prompts.py:60:"checkpoints/gt4_dialog_prompts.json`
- `scratch/gt4_verbal_core.py:27:"logs/opus/moe_gt1_traj_v2.jsonl`
- `scratch/gt4_verbal_core.py:28:"logs/opus/gt2_phys_traj.jsonl`
- `scratch/gt4_verbal_core.py:29:"logs/opus/gt2_code_traj.jsonl`
- `scratch/gt4_verbal_core.py:30:"logs/opus/gt3_proofs_traj.jsonl`
- `scratch/gt4_verbal_core.py:31:"logs/opus/gt3_prose_traj.jsonl`
- `scratch/gt4_verbal_core.py:32:"logs/opus/gt4_dialog_traj.jsonl`
- `scratch/gt4_verbal_core.py:63:"checkpoints/gt3_core_keep.json`
- `scratch/gt5_union_keep.py:24:"checkpoints/gt3_core_keep.json`
- `scratch/gt5_union_keep.py:25:"logs/opus/gt3_prose_traj.jsonl`
- `scratch/gt5_union_keep.py:26:"logs/opus/gt4_dialog_traj.jsonl`
- `scratch/gt5_union_keep.py:29:"checkpoints/gt5_union_keep.json`
- `scratch/gt5c_randfill_keep.py:21:"checkpoints/gt3_core_keep.json`
- `scratch/gt5c_randfill_keep.py:23:"checkpoints/gt5_union_keep.json`
- `scratch/gt5c_randfill_keep.py:33:"checkpoints/gt5c_keep_r{draw}.json`
- `scratch/gt6_recall_ladder.py:32:"checkpoints/moe_gt1_arm0.json`
- `scratch/gt6_recall_ladder.py:35:"checkpoints/gt3_core_keep.json`
- `scratch/gt6_recall_ladder.py:72:"checkpoints/{name}.json`
- `scratch/gt6_recall_ladder.py:87:"logs/opus/gt3_prose_traj.jsonl`
- `scratch/gt6_recall_ladder.py:88:"logs/opus/gt4_dialog_traj.jsonl`
- `scratch/gt7_coverage_rederive.py:125:"checkpoints/{a}.json`
- `scratch/gt7_coverage_rederive.py:92:"checkpoints/moe_gt1_arm0.json`
- `scratch/gt7_coverage_rederive.py:95:"checkpoints/gt3_core_keep.json`
- `scratch/gt7_coverage_rederive.py:96:"logs/opus/gt3_prose_traj.jsonl`
- `scratch/gt7_coverage_rederive.py:97:"logs/opus/gt4_dialog_traj.jsonl`
- `scratch/gt7_draw.py:110:"checkpoints/{name}.json`
- `scratch/gt7_draw.py:126:"logs/opus/gt3_prose_traj.jsonl`
- `scratch/gt7_draw.py:127:"logs/opus/gt4_dialog_traj.jsonl`
- `scratch/gt7_draw.py:46:"checkpoints/moe_gt1_arm0.json`
- `scratch/gt7_draw.py:49:"checkpoints/gt3_core_keep.json`
- `scratch/gt7_run.py:42:"logs/gt7/gt7_answers.jsonl`
- `scratch/gt7_run.py:79:"checkpoints/{arm}.json`
- `scratch/head_autopsy.py:17:"checkpoints/sym_birth_dense_mps_h8_ema.pt`
- `scratch/head_census.py:19:"checkpoints/sym_birth_dense_w56_ema.pt`
- `scratch/holdout_gate.py:27:"data/*.jsonl`
- `scratch/holdout_v2.py:18:"data/*.jsonl`
- `scratch/jointperm_distance.py:16:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/jointperm_distance.py:17:"checkpoints/mathnative_wfloor_d256_s2.pt`
- `scratch/jointperm_distance.py:18:"checkpoints/mathnative_wfloor_d256_s3.pt`
- `scratch/jointperm_distance.py:19:"checkpoints/mathnative_wfloor_d256_stream4.pt`
- `scratch/jointperm_distance.py:20:"checkpoints/mathnative_wfloor_d256_clade2.pt`
- `scratch/judge_decode.py:104:"logs/pp_judge_decode.jsonl`
- `scratch/judge_decode.py:20:"checkpoints/calib_d256_ctl.pt`
- `scratch/k3_expert_demo.py:118:"checkpoints/k3_silu_tab.pt`
- `scratch/k3_expert_demo.py:31:"checkpoints/k3_expert_l45_e7`
- `scratch/kv_equiv.py:53:"checkpoints/mathnative_19m_v21.pt`
- `scratch/lam_merge_review.py:24:"checkpoints/umoe_gravmoe_g{lam}_cuda_s1.pt`
- `scratch/make_altpairs.py:44:"data/altpairs_shard{idx}.jsonl`
- `scratch/make_altpairs.py:75:"data/altpairs_rows.jsonl`
- `scratch/make_altpairs.py:77:"data/altpairs_shard{i}.jsonl`
- `scratch/make_union_diet.py:14:"data/zx_farm1_train.jsonl`
- `scratch/make_union_diet.py:15:"data/union_math_zx.jsonl`
- `scratch/margin_by_level.py:27:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/margin_by_level.py:33:"logs/data_ceil/margins_d256_L1-7.jsonl`
- `scratch/margin_by_level.py:62:"logs/data_ceil`
- `scratch/margin_by_ply.py:27:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/margin_by_ply.py:34:"logs/data_ceil/margins_by_ply_d256.jsonl`
- `scratch/margin_by_ply.py:63:"logs/data_ceil`
- `scratch/margin_census.py:13:"checkpoints/merged_grown_latent.pt`
- `scratch/margin_vs_branching.py:23:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/margin_vs_branching.py:27:"logs/data_ceil/determinability_gen4.jsonl`
- `scratch/margin_vs_branching.py:29:"logs/data_ceil/margin_vs_branching_d256.jsonl`
- `scratch/margin_vs_branching.py:73:"logs/data_ceil`
- `scratch/mass_on_valid.py:22:"checkpoints/mathnative_wfloor_d256_muon.pt`
- `scratch/mass_on_valid.py:23:"checkpoints/mathnative_wfloor_d256_stream3.pt`
- `scratch/mass_on_valid.py:24:"checkpoints/mathnative_wfloor_d256_stream4.pt`
- `scratch/mass_on_valid.py:25:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/matryoshka_r1.py:29:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/matryoshka_r1.py:30:"checkpoints/matryoshka_d256.pt`
- `scratch/matryoshka_r2.py:22:"checkpoints/sym_birth_dense_w56_ema.pt`
- `scratch/matryoshka_r2.py:23:"checkpoints/matryoshka_d56_3tier.pt`
- `scratch/metabolic_d2.py:205:"checkpoints/metab_d2_{ARM}.pt`
- `scratch/metabolic_hot.py:116:"checkpoints/metabolic_{TAG}.pt`
- `scratch/metabolic_hot.py:24:"checkpoints/metabolic_{TAG}.pt`
- `scratch/metabolic_hot.py:33:"checkpoints/mathnative_gen6_grown.pt`
- `scratch/metabolic_hot.py:43:"data/metabolic_{TAG}_sidecar.jsonl`
- `scratch/metabolic_hot.py:44:"checkpoints/metabolic_{TAG}_snap.pt`
- `scratch/metabolic_v3.py:183:"checkpoints/metab_v3_{LABEL}.pt`
- `scratch/metabolic_v4.py:118:"data/practice_rows_v4.jsonl`
- `scratch/metabolic_v4.py:237:"checkpoints/metab_v4.pt`
- `scratch/metabolic_v5.py:131:"data/practice_rows_v5.jsonl`
- `scratch/metabolic_v5.py:132:"data/stuck_states_v5.jsonl`
- `scratch/metabolic_v5.py:255:"checkpoints/metab_v5_s1.pt`
- `scratch/metallicity_diets.py:32:"data/metallicity`
- `scratch/metallicity_diets.py:38:"data/micromodel_chains_shard*.jsonl`
- `scratch/metallicity_diets.py:40:"data/step_chains.jsonl`
- `scratch/metallicity_diets.py:41:"data/micromodel_v22_shard*.jsonl`
- `scratch/metallicity_diets.py:44:"data/micromodel_gen4_sidecar.jsonl`
- `scratch/moe_gt1_arm2.py:48:"checkpoints/moe_gt1_arm0.json`
- `scratch/moe_gt1_arm2.py:60:"logs/opus/moe_gt1.jsonl`
- `scratch/moe_gt1_arm2.py:61:"logs/opus/moe_gt1_smoke.jsonl`
- `scratch/moe_gt1_arm2.py:63:"logs/opus/moe_gt1_perprob.jsonl`
- `scratch/moe_gt1_arm2.py:64:"logs/opus/moe_gt1_perprob_smoke.jsonl`
- `scratch/moe_gt1.py:178:"logs/opus/moe_gt1_traj.jsonl`
- `scratch/moe_gt1.py:58:"checkpoints/moe_gt1_arm0.json`
- `scratch/moe_gt1.py:60:"logs/opus/moe_gt1.jsonl`
- `scratch/moe_gt1.py:61:"logs/opus/moe_gt1_smoke.jsonl`
- `scratch/muon_3ep_d256.py:26:"checkpoints/muon3ep_d256.pt`
- `scratch/night30_mac.py:130:"checkpoints/blackhole_q3_parts/part-*.npz`
- `scratch/nineteen_m_displace.py:24:"checkpoints/snap19m_q32.pt`
- `scratch/nineteen_m_displace.py:26:"checkpoints/snap19m_q4.pt`
- `scratch/nineteen_m_displace.py:49:"figs/2026-08-08/nineteen-m-quant-displace.png`
- `scratch/nineteen_m_displace.py:57:"checkpoints/snap19m_q4.pt`
- `scratch/p2_crown_draws.py:21:"checkpoints/mathnative_gen6_grown.pt`
- `scratch/p2_crown_draws.py:23:"checkpoints/merged_grown.pt`
- `scratch/p3_autopsy.py:23:"checkpoints/sym_birth_dense_mps_h8_s{SEED}_ema.pt`
- `scratch/p3_bits.py:30:"checkpoints/tourn_T_p3r9_s{SEED}.pt`
- `scratch/p3_bits.py:31:"checkpoints/tourn_T_p3r9_s{SEED}_latent.pt`
- `scratch/p3_bits.py:33:"checkpoints/p3r9_fp32_s{SEED}.pt`
- `scratch/p3_ffnslack.py:30:"checkpoints/sym_birth_{ARM}{TAG}.pt`
- `scratch/p3_quat.py:30:"checkpoints/p3r7_quat_{ARM}_s{SEED}.pt`
- `scratch/p3_stream2x2.py:32:"checkpoints/p3r4_stream_{ARM}_s{SEED}.pt`
- `scratch/pack_baselines.py:22:"checkpoints/sym_birth_dense_mps_h8_ema.pt`
- `scratch/pack_crystal.py:114:"checkpoints/packed_{tag}.npz`
- `scratch/pack_crystal.py:25:"checkpoints/sym_birth_dense_mps_L4_ema.pt`
- `scratch/pack_crystal.py:27:"checkpoints/sym_birth_dense_mps_h8_ema.pt`
- `scratch/pack_decode.py:22:"checkpoints/sym_birth_dense_mps_h8_ema.pt`
- `scratch/pack_decode.py:23:"checkpoints/p3_tables.pt`
- `scratch/pack_determinism.py:23:"checkpoints/sym_birth_dense_mps_h8_ema.pt`
- `scratch/pack_rans.py:42:"checkpoints/packed_*.npz`
- `scratch/pack_rans.py:75:"checkpoints/blackhole_q3_parts/part-*.npz`
- `scratch/pack_tiered.py:20:"checkpoints/matryoshka_d56_3tier.pt`
- `scratch/phys_probe.py:38:"data/phys_probe.jsonl`
- `scratch/pincer_dist_probe.py:100:"checkpoints/mathnative_wfloor_d256_stream4.pt`
- `scratch/pincer_dist_probe.py:101:"checkpoints/fmt_oneshot_1p.pt`
- `scratch/pincer_dist_probe.py:102:"checkpoints/fmt_backpairs_1p.pt`
- `scratch/pincer_dist_probe.py:147:"logs/pp_dist_probe.jsonl`
- `scratch/pincer_dist_probe.py:47:"logs/pp_*.jsonl`
- `scratch/pincer_dist_probe.py:99:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/pincer_dist_report.py:16:"logs/pp_dist_probe.jsonl`
- `scratch/pincer_labels_v2.py:60:"logs/pincer/labels_v2.jsonl`
- `scratch/pincer_r0.py:27:"checkpoints/fmt_oneshot_1p.pt`
- `scratch/pincer_r0.py:36:"logs/pp_r0_conjecture.jsonl`
- `scratch/pincer_r0b.py:31:"checkpoints/fmt_oneshot_1p.pt`
- `scratch/pincer_r0b.py:53:"logs/pp_r0b_readout.jsonl`
- `scratch/pincer_r1_indist.py:64:"logs/pincer/indist_{label}.jsonl`
- `scratch/pincer_r1_probe.py:31:"logs/pp_*.jsonl`
- `scratch/pincer_r1_probe.py:49:"logs/pp_{label}.jsonl`
- `scratch/pincer_r8.py:44:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/pincer_r8.py:45:"checkpoints/fmt_oneshot_1p.pt`
- `scratch/pincer_r8.py:46:"checkpoints/fmt_backpairs_1p.pt`
- `scratch/pincer_r8.py:49:"logs/pp_pairs_3e.jsonl`
- `scratch/pincer_r8.py:92:"logs/pp_r8_meet.jsonl`
- `scratch/place1_gravity.py:47:"logs/place1_traces.npy`
- `scratch/place1_gravity.py:55:"logs/place1_traces.npy`
- `scratch/place1_gravity.py:56:"logs/place1_traces.npy`
- `scratch/polar_snap.py:21:"checkpoints/cplx_none.pt`
- `scratch/practice_mine.py:30:"data/*.jsonl`
- `scratch/practice_mine.py:51:"checkpoints/mathnative_gen6_grown.pt`
- `scratch/practice_mine.py:56:"checkpoints/mathnative_gen6_ternary.pt`
- `scratch/practice_mine.py:59:"data/practice_rows_{TAG}.jsonl`
- `scratch/practice_mine.py:60:"data/stuck_states_{TAG}.jsonl`
- `scratch/quat_commutant.py:15:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/quat_commutant.py:16:"checkpoints/mathnative_wfloor_d256_s2.pt`
- `scratch/quat_commutant.py:17:"checkpoints/mathnative_19m.pt`
- `scratch/quat_convert.py:26:"checkpoints/quat_convert_{ARM}.pt`
- `scratch/quat_convert.py:31:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/qwen_displace_extract.py:31:"checkpoints/qwen05b_{tag}_l14gate.pt`
- `scratch/rank_read.py:17:"checkpoints/sym_birth_dense_w56_ema.pt`
- `scratch/rev2_d768.py:33:"checkpoints/tourn_T_rev2d768_s{SEED}.pt`
- `scratch/rev2_d768.py:34:"checkpoints/tourn_T_rev2d768_s{SEED}_latent.pt`
- `scratch/rev2_d768.py:36:"checkpoints/rev2_fp32_768_s{SEED}.pt`
- `scratch/rev3_crown.py:45:"checkpoints/tourn_T_crown_s{SEED}.pt`
- `scratch/rev3_crown.py:46:"checkpoints/tourn_T_crown_s{SEED}_latent.pt`
- `scratch/rev3_crown.py:48:"checkpoints/crown_c_birth_s{SEED}.pt`
- `scratch/rev3_crown.py:49:"checkpoints/crown_c_grown_s{SEED}.pt`
- `scratch/rev3_crown.py:94:"data/merged_diet.jsonl`
- `scratch/rev4_zx45.py:30:"checkpoints/union_45m_s{SEED}.pt`
- `scratch/rev4_zx45.py:56:"data/union_math_zx.jsonl`
- `scratch/rot_commutant.py:14:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/rot_commutant.py:15:"checkpoints/mathnative_wfloor_d256_s2.pt`
- `scratch/rot_commutant.py:16:"checkpoints/mathnative_wfloor_d256_muon.pt`
- `scratch/rot_commutant.py:17:"checkpoints/mathnative_19m.pt`
- `scratch/rot_commutant.py:20:"checkpoints/cplx_none.pt`
- `scratch/rot_commutant.py:21:"checkpoints/cplx_G5.pt`
- `scratch/rot_convert.py:24:"checkpoints/rot_convert_{ARM}.pt`
- `scratch/rot_convert.py:29:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/rot_snap_anatomy.py:19:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/rotinstr_control.py:18:"checkpoints/fourier2b_widemod.pt`
- `scratch/saturation_s2.py:26:"checkpoints/sat_s2.pt`
- `scratch/saturation_s2.py:35:"checkpoints/sat_s2.pt`
- `scratch/saturation_s2b.py:16:"data/gen8_diet.jsonl`
- `scratch/saturation_s2b.py:22:"data/sat_b_widened.jsonl`
- `scratch/saturation_s2b.py:39:"checkpoints/sat_s2b.pt`
- `scratch/saturation_s2b.py:40:"data/sat_b_widened.jsonl`
- `scratch/saturation_s2b.py:48:"checkpoints/sat_s2b.pt`
- `scratch/scaffold_review.py:29:"checkpoints/umoe_gravmoe_s{S}.pt`
- `scratch/scaffold_review.py:55:"checkpoints/umoe_channel_s{S}.pt`
- `scratch/scaffold_review.py:61:"checkpoints/umoe_channel_s{S}.pt`
- `scratch/scorer_s1_battery.py:128:"data/vcache_shard{i}.jsonl`
- `scratch/scorer_s1_battery.py:140:"data/scorer_battery_v1.jsonl`
- `scratch/scorer_s1_battery.py:190:"data/scorer_battery_v1.jsonl`
- `scratch/scorer_s1_battery.py:32:"data/value_cache.jsonl`
- `scratch/scorer_s1_battery.py:53:"logs/pp_*.jsonl`
- `scratch/scorer_s1_battery.py:89:"data/vcache_shard{idx}.jsonl`
- `scratch/scorer_s2_data.py:115:"data/vcache_s2_shard{idx}.jsonl`
- `scratch/scorer_s2_data.py:154:"data/vcache_s2_shard{i}.jsonl`
- `scratch/scorer_s2_data.py:165:"data/scorer_train_v1.jsonl`
- `scratch/scorer_s2_data.py:31:"data/value_cache.jsonl`
- `scratch/scorer_s2_data.py:57:"data/enum_s2_shard{idx}.jsonl`
- `scratch/scorer_s2_data.py:91:"data/enum_s2_shard{i}.jsonl`
- `scratch/scorer_s2_train.py:106:"logs/pp_s2_train.jsonl`
- `scratch/scorer_s2_train.py:137:"checkpoints/scorer_s2_{arm}.pt`
- `scratch/scorer_s2_train.py:203:"logs/pp_s2_train.jsonl`
- `scratch/scorer_s2_train.py:34:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/scorer_s2_train.py:43:"data/scorer_battery_v1.jsonl`
- `scratch/scorer_s2_train.py:46:"data/scorer_train_v1.jsonl`
- `scratch/series_probe.py:42:"data/series_probe.jsonl`
- `scratch/snap_alloc.py:22:"checkpoints/sym_birth_dense_w56_ema.pt`
- `scratch/snap_anatomy.py:22:"checkpoints/mathnative_19m.pt`
- `scratch/snap_anatomy.py:27:"data/micromodel_gen4_sidecar.jsonl`
- `scratch/softprompt1.py:35:"logs/softprompt1`
- `scratch/softprompt1.py:80:"data/metallicity/z3.jsonl`
- `scratch/star_profile.py:23:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/star_profile.py:29:"logs/star_profile/star_profile_d256.jsonl`
- `scratch/star_profile.py:59:"logs/star_profile`
- `scratch/streaming_birth_d256.py:126:"checkpoints/mathnative_wfloor_d256_muon_mx.pt`
- `scratch/streaming_birth_d256.py:131:"checkpoints/mathnative_wfloor_d256_muon.pt`
- `scratch/streaming_birth_d256.py:136:"checkpoints/mathnative_wfloor_d256_stream4.pt`
- `scratch/streaming_birth_d256.py:145:"checkpoints/mathnative_wfloor_d256_stream3.pt`
- `scratch/streaming_birth_d256.py:157:"checkpoints/mathnative_wfloor_d256_stream2.pt`
- `scratch/streaming_birth_d256.py:30:"checkpoints/mathnative_wfloor_d256_stream.pt`
- `scratch/streaming_birth_d256.py:80:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/streaming_birth_d256.py:87:"checkpoints/mathnative_wfloor_d256_tref.pt`
- `scratch/sym_birth.py:33:"checkpoints/sym_birth_{ARM}{TAG}.pt`
- `scratch/sym_convert.py:27:"checkpoints/sym_{GROUP}_{ARM}.pt`
- `scratch/sym_convert.py:62:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/sym_spectrum.py:25:"checkpoints/mathnative_wfloor_d256.pt`
- `scratch/sym45.py:20:"checkpoints/union_45m.pt`
- `scratch/sym45.py:21:"checkpoints/union_45m_c8.pt`
- `scratch/sym45.py:22:"data/union_math_zx.jsonl`
- `scratch/sym45.py:60:"checkpoints/union_45m_c8_projinit.pt`
- `scratch/synonym_test.py:71:"checkpoints/mathnative_19m_v21.pt`
- `scratch/tenet_d2_revdiet.py:47:"data/gen4_replay_status.jsonl`
- `scratch/tenet_d2_revdiet.py:48:"data/gen4_reverse_certified.jsonl`
- `scratch/tenet_d2_revdiet.py:49:"data/gen4_forward_certified.jsonl`
- `scratch/tenet_mult_b32.py:39:"logs/mult0/census_b32.jsonl`
- `scratch/tenet_mult_census.py:155:"logs/mult0/census.jsonl`
- `scratch/tenet_r1b_micro.py:39:"checkpoints/sym_birth_dense_fwdcert.pt`
- `scratch/tenet_r1b_micro.py:40:"checkpoints/sym_birth_dense_revcert.pt`
- `scratch/tenet_r1b_micro.py:41:"logs/tenet_r1b_{ARM}.jsonl`
- `scratch/tenet_w1_bridge.py:82:"data/w1_population_manifest.jsonl`
- `scratch/tenet_w1_population.py:35:"data/w1_population_manifest.jsonl`
- `scratch/tenet_w1_population.py:40:"data/gen4_forward_certified.jsonl`
- `scratch/tenet_w1_population.py:41:"data/gen4_reverse_certified.jsonl`
- `scratch/tenet_w1_population.py:46:"checkpoints/sym_birth_dense{tag}.pt`
- `scratch/tenet_w1_relational.py:102:"data/w1_population_manifest.jsonl`
- `scratch/tenet_w1_surfaces.py:117:"data/w1_population_manifest.jsonl`
- `scratch/ternary_control.py:14:"checkpoints/mathnative_gen6_ternary_latent.pt`
- `scratch/ternary_control.py:22:"checkpoints/ternary_control_deployed.pt`
- `scratch/ternary_gate.py:14:"checkpoints/ternary_nnue_latent.pt`
- `scratch/ternary_gate.py:22:"checkpoints/ternary_nnue_deployed.pt`
- `scratch/ternary_session2.py:122:"checkpoints/ternary_s2_latent.pt`
- `scratch/ternary_session2.py:37:"checkpoints/ternary_nnue_latent.pt`
- `scratch/ternary_session2.py:50:"data/ternary_s2_sidecar.jsonl`
- `scratch/ternary_session2.py:51:"checkpoints/ternary_s2_snap.pt`
- `scratch/tier_escalate.py:55:"checkpoints/matryoshka_d56_3tier.pt`
- `scratch/tier_retry.py:55:"checkpoints/matryoshka_d56.pt`
- `scratch/train_fp64.py:16:"checkpoints/fp64_birth.pt`
- `scratch/umoe_conserve.py:58:"checkpoints/umoe_{ARM}{TAG}{FTAG}{GTAG}{OTAG}_s{SEED}.pt`
- `scratch/v4flash_anatomy.py:28:"logs/v4flash_anatomy`
- `scratch/v4flash_anatomy.py:51:"checkpoints/v4flash_f1`
- `scratch/v4flash_census.py:28:"logs/opus/v4_census.jsonl`
- `scratch/v4flash_census.py:40:"logs/opus`
- `scratch/v4flash_f1b.py:35:"checkpoints/v4flash_vendor`
- `scratch/v4flash_f1c.py:37:"checkpoints/v4flash_f1`
- `scratch/v4flash_f1d.py:334:"logs/opus`
- `scratch/v4flash_f1d.py:47:"checkpoints/v4flash_f1`
- `scratch/v4flash_f1d.py:62:"logs/opus/v4_f1d.jsonl`
- `scratch/v4flash_router.py:32:"logs/opus/v4_router.jsonl`
- `scratch/v4flash_router.py:73:"logs/opus`
- `scratch/v4flash_rung0.py:112:"logs/opus`
- `scratch/v4flash_rung0.py:38:"checkpoints/v4flash_sample`
- `scratch/v4flash_rung0.py:39:"logs/opus/v4_rung0.jsonl`
- `scratch/v4flash_rung2b_router.py:34:"logs/opus/v4_rung2b_router.jsonl`
- `scratch/v4flash_rung2b_router.py:77:"logs/opus`
- `scratch/v4flash_rung2b.py:120:"logs/opus`
- `scratch/v4flash_rung2b.py:41:"logs/opus/v4_rung2b.jsonl`
- `scratch/v4flash_rungA.py:36:"checkpoints/v4flash_sample`
- `scratch/v4flash_rungA.py:48:"checkpoints/k3_silu_tab.pt`
- `scratch/v4flash_rungd.py:165:"logs/opus`
- `scratch/v4flash_rungd.py:61:"logs/opus/v4_rungd.jsonl`
- `scratch/v4flash_rungd2.py:111:"logs/opus`
- `scratch/v4flash_rungd2.py:67:"logs/opus/v4_rungd2.jsonl`
- `scratch/v4flash_s0.py:154:"logs/opus`
- `scratch/v4flash_s0.py:52:"checkpoints/v4flash_sample`
- `scratch/v4flash_s0.py:55:"logs/opus/v4_s0.jsonl`
- `scratch/v4flash_twin.py:369:"checkpoints/v4flash_sample/layers.22.ffn.experts.0.w1.weight.bin`
- `scratch/vmasm_probe.py:18:"data/vmasm_probe.jsonl`
- `scratch/vmasm.py:105:"data/{out}_diet.jsonl`
- `scratch/vmasm.py:109:"data/{out}_probe.jsonl`
- `scratch/vrm_ab.py:119:"checkpoints/vrm_{name}.pt`
- `scratch/vrm_ab.py:16:"checkpoints/mathnative_19m_v21.pt`
- `scratch/vrm_ab.py:71:"data/micromodel_chains_shard*.jsonl`
- `scratch/weight_fft_euler.py:15:"checkpoints/merged_grown_latent.pt`

## 4. Entry-point inventory (`__main__`) — smoke-launch gate (283 files)

- `scratch/adjudicate_zx.py`
- `scratch/assets_classify.py`
- `scratch/attractor_census.py`
- `scratch/attractor_census2.py`
- `scratch/basin_probe.py`
- `scratch/blackhole_b0.py`
- `scratch/boundary_or_bulk.py`
- `scratch/cal_dilute.py`
- `scratch/cal_dk_probe.py`
- `scratch/calib_probe.py`
- `scratch/capacity_meter.py`
- `scratch/churn_judge_eval.py`
- `scratch/detbwd_diet.py`
- `scratch/detbwd_gravmoe.py`
- `scratch/detbwd_mb.py`
- `scratch/detbwd_plateau.py`
- `scratch/detbwd_r1.py`
- `scratch/detbwd_r1b.py`
- `scratch/detbwd_r2_adamw.py`
- `scratch/detbwd_r2b.py`
- `scratch/detbwd_r3_qw.py`
- `scratch/determinability_census.py`
- `scratch/distortion_collapse.py`
- `scratch/emission_wall_pair.py`
- `scratch/engine_scale_export.py`
- `scratch/ex1_swap.py`
- `scratch/ex2_build.py`
- `scratch/ex3_build.py`
- `scratch/export_mb_ref.py`
- `scratch/export_r2b_ref.py`
- `scratch/fig_magic_scatter.py`
- `scratch/fourier_g9.py`
- `scratch/fourier_probe.py`
- `scratch/fourier2_modbirth.py`
- `scratch/fourier2b_widemod.py`
- `scratch/fourier3_algdiet.py`
- `scratch/fourier4a_dynamics.py`
- `scratch/fx3_house.py`
- `scratch/g5_polar.py`
- `scratch/gate_batched.py`
- `scratch/gate_regate.py`
- `scratch/gate_transcripts.py`
- `scratch/graph_mod_sigma.py`
- `scratch/graph_modularity_gen8.py`
- `scratch/grav_posthoc.py`
- `scratch/grav_probe.py`
- `scratch/grav1b_distance.py`
- `scratch/grav2_spacetime.py`
- `scratch/grpo_shaped.py`
- `scratch/gt2_code_arm0.py`
- `scratch/gt2_jaccard.py`
- `scratch/gt3_probe_arm0.py`
- `scratch/gt4_dialog_prompts.py`
- `scratch/gt4_verbal_core.py`
- `scratch/gt6_recall_ladder.py`
- `scratch/gt7_coverage_rederive.py`
- `scratch/gt7_draw.py`
- `scratch/gt7_run.py`
- `scratch/head_autopsy.py`
- `scratch/k3_expert_demo.py`
- `scratch/keff_probe.py`
- `scratch/kv_equiv.py`
- `scratch/lam_merge_review.py`
- `scratch/lean_check.py`
- `scratch/lloydmax_race.py`
- `scratch/loss_floor_census.py`
- `scratch/margin_by_level.py`
- `scratch/margin_by_ply.py`
- `scratch/margin_vs_branching.py`
- `scratch/metallicity_diets.py`
- `scratch/moe_gt1_arm2.py`
- `scratch/moe_gt1.py`
- `scratch/night30_mac.py`
- `scratch/nineteen_m_displace.py`
- `scratch/oracle_worker.py`
- `scratch/p3_autopsy.py`
- `scratch/p3_grav2.py`
- `scratch/p3_umoe_soft.py`
- `scratch/pack_baselines.py`
- `scratch/pack_c6.py`
- `scratch/pack_c7.py`
- `scratch/pack_crystal.py`
- `scratch/pack_decode.py`
- `scratch/pack_determinism.py`
- `scratch/pack_gemv.py`
- `scratch/pack_p2a.py`
- `scratch/pack_rans.py`
- `scratch/pack_tiered.py`
- `scratch/paper_figs.py`
- `scratch/phase4_rewrite.py`
- `scratch/pincer_labels_v2.py`
- `scratch/pincer_r1_indist.py`
- `scratch/place1_gravity.py`
- `scratch/probe_int_device_parity.py`
- `scratch/prologue_arms.py`
- `scratch/ptq4_arms.py`
- `scratch/quat_commutant.py`
- `scratch/rot_commutant.py`
- `scratch/rotinstr_control.py`
- `scratch/scaffold_review.py`
- `scratch/scorer_s2_train.py`
- `scratch/seed_audit.py`
- `scratch/softprompt_sampler_probe.py`
- `scratch/softprompt1.py`
- `scratch/ssm_star.py`
- `scratch/star_profile.py`
- `scratch/synonym_test.py`
- `scratch/tenet_d1_revgate.py`
- `scratch/tenet_d2_revdiet.py`
- `scratch/tenet_d3_budget.py`
- `scratch/tenet_mult_census.py`
- `scratch/tenet_r1b_micro.py`
- `scratch/tenet_w0.py`
- `scratch/tenet_w1_bridge.py`
- `scratch/tenet_w1_population.py`
- `scratch/tenet_w1_relational.py`
- `scratch/tenet_w1_surfaces.py`
- `scratch/umoe_conserve.py`
- `scratch/v4flash_anatomy.py`
- `scratch/v4flash_census.py`
- `scratch/v4flash_f1b.py`
- `scratch/v4flash_f1c.py`
- `scratch/v4flash_f1d.py`
- `scratch/v4flash_header.py`
- `scratch/v4flash_router.py`
- `scratch/v4flash_rung0.py`
- `scratch/v4flash_rung2b_router.py`
- `scratch/v4flash_rung2b.py`
- `scratch/v4flash_rungA.py`
- `scratch/v4flash_rungd.py`
- `scratch/v4flash_rungd2.py`
- `scratch/v4flash_s0.py`
- `scratch/v4flash_twin.py`
- `scratch/vmasm.py`
- `scratch/vrm_ab.py`
- `scripts/arena.py`
- `scripts/autopsy_int.py`
- `scripts/backfill_code_commit.py`
- `scripts/bench_adaptive_draft.py`
- `scripts/bench_adaptive.py`
- `scripts/bench_anneal.py`
- `scripts/bench_ansatz_search_2b.py`
- `scripts/bench_ansatz_search.py`
- `scripts/bench_bandit.py`
- `scripts/bench_bestfirst_llm.py`
- `scripts/bench_bestfirst_nnue.py`
- `scripts/bench_bestfirst.py`
- `scripts/bench_budget_alloc.py`
- `scripts/bench_commute.py`
- `scripts/bench_compile.py`
- `scripts/bench_control.py`
- `scripts/bench_decoding.py`
- `scripts/bench_derivation.py`
- `scripts/bench_dispatch_race_v4.py`
- `scripts/bench_distilled_draft.py`
- `scripts/bench_engine_regret.py`
- `scripts/bench_entropy_beam.py`
- `scripts/bench_fib_restarts.py`
- `scripts/bench_flash_prefill.py`
- `scripts/bench_frontier.py`
- `scripts/bench_fused_ce.py`
- `scripts/bench_fused.py`
- `scripts/bench_gated.py`
- `scripts/bench_gweight.py`
- `scripts/bench_hints_ab.py`
- `scripts/bench_hybrid.py`
- `scripts/bench_int4_config_sweep.py`
- `scripts/bench_int4_gemv.py`
- `scripts/bench_interference.py`
- `scripts/bench_ksweep.py`
- `scripts/bench_kv_quant_decode.py`
- `scripts/bench_ladder.py`
- `scripts/bench_lazy.py`
- `scripts/bench_llm_gating.py`
- `scripts/bench_lookup_static.py`
- `scripts/bench_luby.py`
- `scripts/bench_magic.py`
- `scripts/bench_markov_adaptive.py`
- `scripts/bench_markov.py`
- `scripts/bench_metal_kernels.py`
- `scripts/bench_mlx_integration.py`
- `scripts/bench_nnue.py`
- `scripts/bench_ode_engine.py`
- `scripts/bench_opcap.py`
- `scripts/bench_population.py`
- `scripts/bench_pred_syndromes.py`
- `scripts/bench_prefix_reuse.py`
- `scripts/bench_proposer.py`
- `scripts/bench_quant_schemes.py`
- `scripts/bench_record.py`
- `scripts/bench_regret_resample.py`
- `scripts/bench_rotate_quantize.py`
- `scripts/bench_rule_basis.py`
- `scripts/bench_stack_winners.py`
- `scripts/bench_stacked.py`
- `scripts/bench_static.py`
- `scripts/bench_step_diversity.py`
- `scripts/bench_step_tokens.py`
- `scripts/bench_stitch_poc.py`
- `scripts/bench_syndrome_head.py`
- `scripts/bench_syndrome_policy.py`
- `scripts/bench_temp_race.py`
- `scripts/bench_tree_verify.py`
- `scripts/bench_triton_kernels.py`
- `scripts/bench_verify_fast.py`
- `scripts/bench_vge.py`
- `scripts/bench_weight_anatomy.py`
- `scripts/bench_zx_r3.py`
- `scripts/bench_zx_r5.py`
- `scripts/bench_zx_r6.py`
- `scripts/bench_zx_r7.py`
- `scripts/bench_zx.py`
- `scripts/book.py`
- `scripts/build_gen7_diet.py`
- `scripts/calibrate_hce.py`
- `scripts/ckpt_manifest.py`
- `scripts/consolidate_mathnative.py`
- `scripts/control_round.py`
- `scripts/convert_diet_prefix.py`
- `scripts/eval_mathnative.py`
- `scripts/eval_pruned_moe.py`
- `scripts/eval_ruler.py`
- `scripts/expert_iter_steps.py`
- `scripts/expert_loop.py`
- `scripts/farm_algebra.py`
- `scripts/farm_l4_calc.py`
- `scripts/farm_v22.py`
- `scripts/gen_catalog.py`
- `scripts/gen_codemap.py`
- `scripts/gen_dispatch_labels_v2.py`
- `scripts/gen_dispatch_labels.py`
- `scripts/gen_figures_web.py`
- `scripts/gen_frontier.py`
- `scripts/gen_index.py`
- `scripts/gen_lake.py`
- `scripts/gen_magic_labels.py`
- `scripts/gen_policy_labels.py`
- `scripts/gen_proposer_data.py`
- `scripts/gen_readme.py`
- `scripts/gen_regret_labels.py`
- `scripts/gen_syndrome_labels.py`
- `scripts/grow_mathnative.py`
- `scripts/harvest_champion.py`
- `scripts/harvest_frontier.py`
- `scripts/list_uncurated.py`
- `scripts/log_hygiene.py`
- `scripts/markov_eval.py`
- `scripts/markov_prior.py`
- `scripts/mine_prior_update.py`
- `scripts/moe_router_stats.py`
- `scripts/plot_gt1_crest.py`
- `scripts/plot_identity_crest.py`
- `scripts/plot_neurons.py`
- `scripts/probe_depth.py`
- `scripts/render_gallery.py`
- `scripts/render_hero_neurons.py`
- `scripts/rjob.py`
- `scripts/sol_enrich_results.py`
- `scripts/sol_generate_tables.py`
- `scripts/step_grpo_micro.py`
- `scripts/step_grpo.py`
- `scripts/sweep_lookup_mlx.py`
- `scripts/sweep_lookup.py`
- `scripts/tabula_rasa_r0.py`
- `scripts/tabula_rasa_r1.py`
- `scripts/tabula_rasa_r2.py`
- `scripts/task_arithmetic.py`
- `scripts/task_composition.py`
- `scripts/tournament_birth.py`
- `scripts/train_calculus.py`
- `scripts/train_dispatcher.py`
- `scripts/train_magic_estimator.py`
- `scripts/train_magic_llm.py`
- `scripts/train_mathnative.py`
- `scripts/train_nnue.py`
- `scripts/train_proposer.py`
- `scripts/train_syndrome_decoder.py`
- `scripts/train_syndrome_policy.py`
- `scripts/train_ternary.py`
- `scripts/train_tf32x3.py`
- `scripts/train_value_head.py`
- `scripts/train_weight_reader.py`
- `scripts/validity_autopsy.py`

## 5. Frozen files (never move; edit only under the dual-copy/adoption path)

CODEMAP marks 311 files in scratch/ + scripts/ as results-cited,
spec-cited, reproduce-pinned, or cited-library. None of them has a taxonomy
destination, so none should move. The hazard is different and it is the
headline finding of this survey:

**181 frozen files contain hardcoded `data/` or `checkpoints/` strings.**
Gate step 1 (rewrite every path reference) requires editing evidence-record
files. That collision needs an explicit ruling before the move runs.

### 5a. Frozen AND path-touching (181) — the collision set

- `scratch/attractor_census.py` — results-cited, cited by RESULTS
- `scratch/attractor_census2.py` — results-cited, cited by RESULTS, specs
- `scratch/blackhole_b0.py` — results-cited, cited by RESULTS, specs
- `scratch/capacity_meter.py` — library, cited by RESULTS, specs
- `scratch/ce_gate_study.py` — spec-cited, cited by specs
- `scratch/churn_judge_eval.py` — reproduce-pinned, cited by REPRODUCE
- `scratch/ckpt_inventory.py` — spec-cited, cited by specs
- `scratch/clade_stream_d256.py` — results-cited, cited by RESULTS
- `scratch/complexify_control.py` — results-cited, cited by RESULTS, specs
- `scratch/crystal_recreate_test.py` — spec-cited, cited by specs
- `scratch/detbwd_diet.py` — library, cited by RESULTS, specs
- `scratch/detbwd_gravmoe.py` — library, cited by RESULTS, specs
- `scratch/determinability_census.py` — results-cited, cited by RESULTS
- `scratch/distortion_collapse.py` — results-cited, cited by RESULTS
- `scratch/duo_wave.py` — spec-cited, cited by specs
- `scratch/e2_logit_check.py` — results-cited, cited by RESULTS, specs
- `scratch/e3_battery.py` — spec-cited, cited by specs
- `scratch/engine_scale_export.py` — spec-cited, cited by specs
- `scratch/ex1_swap.py` — results-cited, cited by RESULTS, specs
- `scratch/ex2_build.py` — results-cited, cited by RESULTS, specs
- `scratch/ex3_build.py` — results-cited, cited by RESULTS, specs
- `scratch/exact1_small_cells.py` — results-cited, cited by RESULTS, specs
- `scratch/exact_twin_d56.py` — spec-cited, cited by specs
- `scratch/exchange_test.py` — results-cited, cited by RESULTS
- `scratch/farm_dist_rows.py` — results-cited, cited by RESULTS, specs
- `scratch/farmer_probe.py` — results-cited, cited by RESULTS, specs
- `scratch/fig_magic_scatter.py` — spec-cited, cited by specs
- `scratch/fourier2b_widemod.py` — library, cited by RESULTS
- `scratch/fourier3_algdiet.py` — results-cited, cited by RESULTS
- `scratch/fourier4a_dynamics.py` — results-cited, cited by RESULTS
- `scratch/fourier_g9.py` — results-cited, cited by RESULTS
- `scratch/fp64_paired.py` — spec-cited, cited by specs
- `scratch/gate_rarity.py` — results-cited, cited by RESULTS
- `scratch/gate_regate.py` — results-cited, cited by RESULTS, specs
- `scratch/gate_transcripts.py` — results-cited, cited by RESULTS, specs
- `scratch/gauge_slack_rat.py` — results-cited, cited by RESULTS
- `scratch/graph_mod_sigma.py` — results-cited, cited by RESULTS
- `scratch/grav_probe.py` — library, cited by RESULTS
- `scratch/gt2_code_arm0.py` — reproduce-pinned, cited by REPRODUCE, RESULTS, specs
- `scratch/gt2_jaccard.py` — library, cited by REPRODUCE, RESULTS, specs
- `scratch/gt4_dialog_prompts.py` — reproduce-pinned, cited by REPRODUCE, RESULTS
- `scratch/gt4_verbal_core.py` — reproduce-pinned, cited by REPRODUCE, RESULTS
- `scratch/gt5_union_keep.py` — reproduce-pinned, cited by REPRODUCE, RESULTS, specs
- `scratch/gt5c_randfill_keep.py` — reproduce-pinned, cited by REPRODUCE, RESULTS
- `scratch/gt6_recall_ladder.py` — reproduce-pinned, cited by REPRODUCE, RESULTS
- `scratch/gt7_coverage_rederive.py` — results-cited, cited by RESULTS, specs
- `scratch/gt7_draw.py` — spec-cited, cited by specs
- `scratch/gt7_run.py` — results-cited, cited by RESULTS, specs
- `scratch/head_census.py` — spec-cited, cited by specs
- `scratch/holdout_gate.py` — spec-cited, cited by specs
- `scratch/judge_decode.py` — spec-cited, cited by specs
- `scratch/k3_expert_demo.py` — results-cited, cited by RESULTS, specs
- `scratch/make_altpairs.py` — results-cited, cited by RESULTS
- `scratch/margin_by_level.py` — results-cited, cited by RESULTS
- `scratch/margin_by_ply.py` — results-cited, cited by RESULTS
- `scratch/margin_vs_branching.py` — results-cited, cited by RESULTS
- `scratch/mass_on_valid.py` — spec-cited, cited by specs
- `scratch/matryoshka_r1.py` — results-cited, cited by RESULTS, specs
- `scratch/matryoshka_r2.py` — spec-cited, cited by specs
- `scratch/metabolic_d2.py` — results-cited, cited by RESULTS, specs
- `scratch/metabolic_hot.py` — results-cited, cited by RESULTS
- `scratch/metabolic_v3.py` — spec-cited, cited by specs
- `scratch/metallicity_diets.py` — results-cited, cited by RESULTS
- `scratch/moe_gt1.py` — library, cited by REPRODUCE, RESULTS, specs
- `scratch/moe_gt1_arm2.py` — library, cited by REPRODUCE, RESULTS, specs
- `scratch/nineteen_m_displace.py` — results-cited, cited by RESULTS, specs
- `scratch/p3_autopsy.py` — results-cited, cited by RESULTS
- `scratch/p3_bits.py` — results-cited, cited by RESULTS
- `scratch/p3_ffnslack.py` — results-cited, cited by RESULTS
- `scratch/p3_quat.py` — results-cited, cited by RESULTS
- `scratch/p3_stream2x2.py` — results-cited, cited by RESULTS
- `scratch/pack_baselines.py` — spec-cited, cited by specs
- `scratch/pack_crystal.py` — results-cited, cited by RESULTS, specs
- `scratch/pack_decode.py` — library, cited by RESULTS
- `scratch/pack_determinism.py` — results-cited, cited by RESULTS, specs
- `scratch/pack_rans.py` — results-cited, cited by RESULTS, specs
- `scratch/pack_tiered.py` — spec-cited, cited by specs
- `scratch/pincer_dist_probe.py` — results-cited, cited by RESULTS
- `scratch/pincer_labels_v2.py` — library, cited by RESULTS
- `scratch/pincer_r0.py` — results-cited, cited by RESULTS
- `scratch/pincer_r0b.py` — results-cited, cited by RESULTS
- `scratch/pincer_r1_indist.py` — results-cited, cited by RESULTS
- `scratch/pincer_r1_probe.py` — results-cited, cited by RESULTS
- `scratch/pincer_r8.py` — results-cited, cited by RESULTS
- `scratch/place1_gravity.py` — results-cited, cited by RESULTS, specs
- `scratch/polar_snap.py` — results-cited, cited by RESULTS
- `scratch/practice_mine.py` — spec-cited, cited by specs
- `scratch/quat_commutant.py` — library, cited by RESULTS, specs
- `scratch/quat_convert.py` — library, cited by RESULTS, specs
- `scratch/rank_read.py` — spec-cited, cited by specs
- `scratch/rev2_d768.py` — results-cited, cited by RESULTS, specs
- `scratch/rev3_crown.py` — results-cited, cited by RESULTS, specs
- `scratch/rev4_zx45.py` — results-cited, cited by RESULTS
- `scratch/rot_commutant.py` — library, cited by RESULTS, specs
- `scratch/rot_convert.py` — spec-cited, cited by specs
- `scratch/rotinstr_control.py` — results-cited, cited by RESULTS
- `scratch/saturation_s2.py` — results-cited, cited by RESULTS
- `scratch/saturation_s2b.py` — spec-cited, cited by specs
- `scratch/scaffold_review.py` — results-cited, cited by RESULTS
- `scratch/scorer_s1_battery.py` — results-cited, cited by RESULTS
- `scratch/scorer_s2_data.py` — results-cited, cited by RESULTS, specs
- `scratch/scorer_s2_train.py` — spec-cited, cited by specs
- `scratch/series_probe.py` — spec-cited, cited by specs
- `scratch/snap_alloc.py` — spec-cited, cited by specs
- `scratch/snap_anatomy.py` — results-cited, cited by RESULTS
- `scratch/softprompt1.py` — results-cited, cited by RESULTS, specs
- `scratch/star_profile.py` — results-cited, cited by RESULTS
- `scratch/streaming_birth_d256.py` — library, cited by RESULTS
- `scratch/sym45.py` — spec-cited, cited by specs
- `scratch/sym_birth.py` — library, cited by RESULTS, specs
- `scratch/sym_convert.py` — results-cited, cited by RESULTS, specs
- `scratch/sym_spectrum.py` — spec-cited, cited by specs
- `scratch/tenet_d2_revdiet.py` — library, cited by RESULTS, specs
- `scratch/tenet_mult_b32.py` — results-cited, cited by RESULTS
- `scratch/tenet_mult_census.py` — library, cited by RESULTS, specs
- `scratch/tenet_w1_bridge.py` — results-cited, cited by RESULTS, specs
- `scratch/tenet_w1_relational.py` — results-cited, cited by RESULTS, specs
- `scratch/tenet_w1_surfaces.py` — results-cited, cited by RESULTS, specs
- `scratch/tier_retry.py` — spec-cited, cited by specs
- `scratch/umoe_conserve.py` — library, cited by RESULTS
- `scratch/v4flash_census.py` — results-cited, cited by RESULTS, specs
- `scratch/v4flash_f1b.py` — library, cited by RESULTS
- `scratch/v4flash_f1c.py` — library, cited by RESULTS
- `scratch/v4flash_f1d.py` — results-cited, cited by RESULTS, specs
- `scratch/v4flash_router.py` — library, cited by RESULTS, specs
- `scratch/v4flash_rung0.py` — results-cited, cited by RESULTS, specs
- `scratch/v4flash_rung2b.py` — library, cited by RESULTS, specs
- `scratch/v4flash_rung2b_router.py` — results-cited, cited by RESULTS
- `scratch/v4flash_rungA.py` — library, cited by RESULTS, specs
- `scratch/v4flash_rungd.py` — library, cited by RESULTS
- `scratch/v4flash_rungd2.py` — results-cited, cited by RESULTS
- `scratch/v4flash_s0.py` — results-cited, cited by RESULTS
- `scratch/v4flash_twin.py` — library, cited by RESULTS
- `scripts/bench_adaptive.py` — spec-cited, cited by specs
- `scripts/bench_budget_alloc.py` — results-cited, cited by RESULTS
- `scripts/bench_engine_regret.py` — results-cited, cited by RESULTS
- `scripts/bench_frontier.py` — spec-cited, cited by specs
- `scripts/bench_hints_ab.py` — results-cited, cited by RESULTS
- `scripts/bench_lazy.py` — library, cited by RESULTS
- `scripts/bench_magic.py` — library, cited by RESULTS
- `scripts/bench_nnue.py` — spec-cited, cited by specs
- `scripts/bench_pred_syndromes.py` — results-cited, cited by RESULTS
- `scripts/bench_proposer.py` — spec-cited, cited by specs
- `scripts/bench_record.py` — results-cited, cited by RESULTS
- `scripts/bench_rule_basis.py` — results-cited, cited by RESULTS
- `scripts/bench_step_diversity.py` — results-cited, cited by RESULTS
- `scripts/bench_stitch_poc.py` — results-cited, cited by RESULTS
- `scripts/bench_verify_fast.py` — library, cited by specs
- `scripts/bench_weight_anatomy.py` — results-cited, cited by RESULTS
- `scripts/bench_zx_r7.py` — results-cited, cited by RESULTS
- `scripts/consolidate_mathnative.py` — results-cited, cited by RESULTS
- `scripts/convert_diet_prefix.py` — spec-cited, cited by specs
- `scripts/eval_pruned_moe.py` — spec-cited, cited by specs
- `scripts/expert_iter_steps.py` — library, cited by specs
- `scripts/expert_loop.py` — library, cited by specs
- `scripts/farm_algebra.py` — results-cited, cited by RESULTS
- `scripts/gen_catalog.py` — results-cited, cited by RESULTS, specs
- `scripts/gen_magic_labels.py` — results-cited, cited by RESULTS
- `scripts/gen_proposer_data.py` — spec-cited, cited by specs
- `scripts/gen_syndrome_labels.py` — results-cited, cited by RESULTS
- `scripts/harvest_frontier.py` — spec-cited, cited by specs
- `scripts/log_hygiene.py` — results-cited, cited by RESULTS, specs
- `scripts/mine_prior_update.py` — results-cited, cited by RESULTS
- `scripts/moe_router_stats.py` — results-cited, cited by RESULTS, specs
- `scripts/plot_gt1_crest.py` — spec-cited, cited by specs
- `scripts/plot_identity_crest.py` — spec-cited, cited by specs
- `scripts/step_grpo.py` — spec-cited, cited by specs
- `scripts/step_grpo_micro.py` — library, cited by RESULTS, specs
- `scripts/task_arithmetic.py` — spec-cited, cited by specs
- `scripts/train_calculus.py` — library, cited by specs
- `scripts/train_magic_estimator.py` — library, cited by RESULTS
- `scripts/train_magic_llm.py` — results-cited, cited by RESULTS
- `scripts/train_mathnative.py` — library, cited by RESULTS, specs
- `scripts/train_nnue.py` — library, cited by specs
- `scripts/train_proposer.py` — spec-cited, cited by specs
- `scripts/train_syndrome_decoder.py` — results-cited, cited by RESULTS
- `scripts/train_syndrome_policy.py` — results-cited, cited by RESULTS
- `scripts/train_ternary.py` — results-cited, cited by RESULTS, specs
- `scripts/train_value_head.py` — results-cited, cited by RESULTS
- `scripts/train_weight_reader.py` — spec-cited, cited by specs
- `scripts/validity_autopsy.py` — results-cited, cited by RESULTS

### 5b. All frozen files (311)

- `scratch/anatomy.py` — results-cited (RESULTS, specs)
- `scratch/assets_classify.py` — spec-cited (specs)
- `scratch/attractor_census.py` — results-cited (RESULTS)
- `scratch/attractor_census2.py` — results-cited (RESULTS, specs)
- `scratch/blackhole_b0.py` — results-cited (RESULTS, specs)
- `scratch/boundary_or_bulk.py` — results-cited (RESULTS)
- `scratch/calib_dist_birth.sh` — spec-cited (specs)
- `scratch/calib_probe.py` — library (RESULTS, specs)
- `scratch/calib_snap_gates.sh` — spec-cited (specs)
- `scratch/capacity_meter.py` — library (RESULTS, specs)
- `scratch/ce_gate_study.py` — spec-cited (specs)
- `scratch/churn_judge_eval.py` — reproduce-pinned (REPRODUCE)
- `scratch/ckpt_inventory.py` — spec-cited (specs)
- `scratch/clade_stream_d256.py` — results-cited (RESULTS)
- `scratch/complex_model.py` — library (specs)
- `scratch/complexify_control.py` — results-cited (RESULTS, specs)
- `scratch/crystal_recreate_test.py` — spec-cited (specs)
- `scratch/desert_v2.py` — spec-cited (specs)
- `scratch/detbwd_diet.py` — library (RESULTS, specs)
- `scratch/detbwd_gravmoe.py` — library (RESULTS, specs)
- `scratch/detbwd_mb.py` — library (RESULTS, specs)
- `scratch/detbwd_plateau.py` — results-cited (RESULTS, specs)
- `scratch/detbwd_r1.py` — library (RESULTS)
- `scratch/detbwd_r1b.py` — library (RESULTS)
- `scratch/detbwd_r2_adamw.py` — library (RESULTS)
- `scratch/detbwd_r2b.py` — library (RESULTS, specs)
- `scratch/detbwd_r3_qw.py` — library (RESULTS)
- `scratch/determinability_census.py` — results-cited (RESULTS)
- `scratch/distortion_collapse.py` — results-cited (RESULTS)
- `scratch/duo_wave.py` — spec-cited (specs)
- `scratch/e2_logit_check.py` — results-cited (RESULTS, specs)
- `scratch/e3_battery.py` — spec-cited (specs)
- `scratch/engine_scale_export.py` — spec-cited (specs)
- `scratch/ex1_swap.py` — results-cited (RESULTS, specs)
- `scratch/ex2_build.py` — results-cited (RESULTS, specs)
- `scratch/ex3_build.py` — results-cited (RESULTS, specs)
- `scratch/exact1_small_cells.py` — results-cited (RESULTS, specs)
- `scratch/exact_twin_d56.py` — spec-cited (specs)
- `scratch/exchange_test.py` — results-cited (RESULTS)
- `scratch/export_axnn.py` — results-cited (RESULTS)
- `scratch/export_r2b_ref.py` — results-cited (RESULTS)
- `scratch/farm_dist_rows.py` — results-cited (RESULTS, specs)
- `scratch/farmer_probe.py` — results-cited (RESULTS, specs)
- `scratch/fig_magic_scatter.py` — spec-cited (specs)
- `scratch/fourier2b_widemod.py` — library (RESULTS)
- `scratch/fourier3_algdiet.py` — results-cited (RESULTS)
- `scratch/fourier4a_dynamics.py` — results-cited (RESULTS)
- `scratch/fourier_g9.py` — results-cited (RESULTS)
- `scratch/fp64_paired.py` — spec-cited (specs)
- `scratch/fx3_house.py` — results-cited (RESULTS, specs)
- `scratch/g19_bf16_isolation.sh` — spec-cited (specs)
- `scratch/gate_ckpt.py` — results-cited (RESULTS, specs)
- `scratch/gate_cplx.py` — spec-cited (specs)
- `scratch/gate_prefix.py` — results-cited (RESULTS)
- `scratch/gate_rarity.py` — results-cited (RESULTS)
- `scratch/gate_regate.py` — results-cited (RESULTS, specs)
- `scratch/gate_transcripts.py` — results-cited (RESULTS, specs)
- `scratch/gatepins_freeze.py` — spec-cited (specs)
- `scratch/gauge_slack_rat.py` — results-cited (RESULTS)
- `scratch/gen8_pipeline.sh` — spec-cited (specs)
- `scratch/gen9_pipeline.sh` — spec-cited (specs)
- `scratch/gen_lean_corpus.py` — results-cited (RESULTS, specs)
- `scratch/graph_mod_sigma.py` — results-cited (RESULTS)
- `scratch/grav_probe.py` — library (RESULTS)
- `scratch/gt2_code_arm0.py` — reproduce-pinned (REPRODUCE, RESULTS, specs)
- `scratch/gt2_jaccard.py` — library (REPRODUCE, RESULTS, specs)
- `scratch/gt3_probe_arm0.py` — reproduce-pinned (REPRODUCE, RESULTS)
- `scratch/gt4_dialog_prompts.py` — reproduce-pinned (REPRODUCE, RESULTS)
- `scratch/gt4_verbal_core.py` — reproduce-pinned (REPRODUCE, RESULTS)
- `scratch/gt5_union_keep.py` — reproduce-pinned (REPRODUCE, RESULTS, specs)
- `scratch/gt5c_randfill_keep.py` — reproduce-pinned (REPRODUCE, RESULTS)
- `scratch/gt6_recall_ladder.py` — reproduce-pinned (REPRODUCE, RESULTS)
- `scratch/gt7_coverage_rederive.py` — results-cited (RESULTS, specs)
- `scratch/gt7_draw.py` — spec-cited (specs)
- `scratch/gt7_run.py` — results-cited (RESULTS, specs)
- `scratch/head_census.py` — spec-cited (specs)
- `scratch/holdout_gate.py` — spec-cited (specs)
- `scratch/hot_chain.sh` — results-cited (RESULTS)
- `scratch/judge_decode.py` — spec-cited (specs)
- `scratch/k3_expert_demo.py` — results-cited (RESULTS, specs)
- `scratch/keff_probe.py` — results-cited (RESULTS)
- `scratch/lean_check.py` — reproduce-pinned (REPRODUCE, RESULTS, specs)
- `scratch/lean_sample_build.py` — results-cited (RESULTS, specs)
- `scratch/legacy_diet_audit.py` — results-cited (RESULTS)
- `scratch/loss_floor_census.py` — results-cited (RESULTS)
- `scratch/lyap_compare.py` — results-cited (RESULTS, specs)
- `scratch/make_altpairs.py` — results-cited (RESULTS)
- `scratch/margin_by_level.py` — results-cited (RESULTS)
- `scratch/margin_by_ply.py` — results-cited (RESULTS)
- `scratch/margin_vs_branching.py` — results-cited (RESULTS)
- `scratch/mass_on_valid.py` — spec-cited (specs)
- `scratch/matryoshka_r1.py` — results-cited (RESULTS, specs)
- `scratch/matryoshka_r2.py` — spec-cited (specs)
- `scratch/metabolic_d2.py` — results-cited (RESULTS, specs)
- `scratch/metabolic_hot.py` — results-cited (RESULTS)
- `scratch/metabolic_v3.py` — spec-cited (specs)
- `scratch/metallicity1.sh` — results-cited (RESULTS)
- `scratch/metallicity_diets.py` — results-cited (RESULTS)
- `scratch/moe_gt1.py` — library (REPRODUCE, RESULTS, specs)
- `scratch/moe_gt1_arm2.py` — library (REPRODUCE, RESULTS, specs)
- `scratch/night29.sh` — spec-cited (specs)
- `scratch/night29b.sh` — spec-cited (specs)
- `scratch/night2_mac.sh` — results-cited (RESULTS)
- `scratch/night30.sh` — spec-cited (specs)
- `scratch/night31.sh` — results-cited (RESULTS)
- `scratch/night31_cuda.sh` — results-cited (RESULTS)
- `scratch/night31b_cuda.sh` — results-cited (RESULTS, specs)
- `scratch/night_28.sh` — results-cited (RESULTS)
- `scratch/night_28_mac.sh` — results-cited (RESULTS)
- `scratch/night_45m_union.sh` — results-cited (RESULTS, specs)
- `scratch/night_g9.sh` — results-cited (RESULTS, specs)
- `scratch/night_rat.sh` — results-cited (RESULTS, specs)
- `scratch/night_rat_s2.sh` — spec-cited (specs)
- `scratch/night_sr.sh` — results-cited (RESULTS)
- `scratch/night_zx.sh` — results-cited (RESULTS, specs)
- `scratch/night_zx2.sh` — results-cited (RESULTS)
- `scratch/night_zx3.sh` — results-cited (RESULTS)
- `scratch/night_zx45_x2.sh` — results-cited (RESULTS)
- `scratch/nineteen_m_displace.py` — results-cited (RESULTS, specs)
- `scratch/oracle_worker.py` — reproduce-pinned (REPRODUCE, RESULTS, specs)
- `scratch/ozaki_2b_check.py` — results-cited (RESULTS)
- `scratch/ozaki_cuda2.py` — results-cited (RESULTS)
- `scratch/ozaki_cuda3.py` — results-cited (RESULTS)
- `scratch/ozaki_cuda4.py` — results-cited (RESULTS)
- `scratch/ozaki_cuda5.py` — results-cited (RESULTS, specs)
- `scratch/ozaki_cuda6.py` — results-cited (RESULTS)
- `scratch/ozaki_fused.py` — results-cited (RESULTS)
- `scratch/ozaki_rung1.py` — results-cited (RESULTS)
- `scratch/ozaki_rung1b.py` — results-cited (RESULTS)
- `scratch/ozaki_rung2bc.py` — results-cited (RESULTS)
- `scratch/p3_autopsy.py` — results-cited (RESULTS)
- `scratch/p3_bits.py` — results-cited (RESULTS)
- `scratch/p3_ffnslack.py` — results-cited (RESULTS)
- `scratch/p3_grav2.py` — results-cited (RESULTS)
- `scratch/p3_quat.py` — results-cited (RESULTS)
- `scratch/p3_stream2x2.py` — results-cited (RESULTS)
- `scratch/p3_umoe_soft.py` — results-cited (RESULTS)
- `scratch/p4_arms_0801.sh` — results-cited (RESULTS, specs)
- `scratch/pack_baselines.py` — spec-cited (specs)
- `scratch/pack_c6.py` — library (RESULTS, specs)
- `scratch/pack_c7.py` — results-cited (RESULTS)
- `scratch/pack_crystal.py` — results-cited (RESULTS, specs)
- `scratch/pack_decode.py` — library (RESULTS)
- `scratch/pack_determinism.py` — results-cited (RESULTS, specs)
- `scratch/pack_gemv.py` — spec-cited (specs)
- `scratch/pack_rans.py` — results-cited (RESULTS, specs)
- `scratch/pack_tiered.py` — spec-cited (specs)
- `scratch/phase4_rewrite.py` — spec-cited (specs)
- `scratch/phase4_sites.py` — spec-cited (specs)
- `scratch/phase5_deadcode.py` — spec-cited (specs)
- `scratch/pincer_dist_probe.py` — results-cited (RESULTS)
- `scratch/pincer_labels_v2.py` — library (RESULTS)
- `scratch/pincer_r0.py` — results-cited (RESULTS)
- `scratch/pincer_r0b.py` — results-cited (RESULTS)
- `scratch/pincer_r1_indist.py` — results-cited (RESULTS)
- `scratch/pincer_r1_probe.py` — results-cited (RESULTS)
- `scratch/pincer_r1b_labels.py` — results-cited (RESULTS, specs)
- `scratch/pincer_r8.py` — results-cited (RESULTS)
- `scratch/place1_gravity.py` — results-cited (RESULTS, specs)
- `scratch/polar_snap.py` — results-cited (RESULTS)
- `scratch/poly3_pipeline.sh` — spec-cited (specs)
- `scratch/poly4_pipeline.sh` — spec-cited (specs)
- `scratch/poly5_pipeline.sh` — spec-cited (specs)
- `scratch/practice_mine.py` — spec-cited (specs)
- `scratch/probe_int_device_parity.py` — results-cited (RESULTS)
- `scratch/quat_commutant.py` — library (RESULTS, specs)
- `scratch/quat_convert.py` — library (RESULTS, specs)
- `scratch/quick_exact_3080.sh` — results-cited (RESULTS, specs)
- `scratch/rank_read.py` — spec-cited (specs)
- `scratch/rat_deploy.py` — results-cited (RESULTS, specs)
- `scratch/rational_snap.py` — results-cited (RESULTS, specs)
- `scratch/rev2_d768.py` — results-cited (RESULTS, specs)
- `scratch/rev3_crown.py` — results-cited (RESULTS, specs)
- `scratch/rev4_zx45.py` — results-cited (RESULTS)
- `scratch/rot_commutant.py` — library (RESULTS, specs)
- `scratch/rot_convert.py` — spec-cited (specs)
- `scratch/rotinstr_control.py` — results-cited (RESULTS)
- `scratch/run_snap_gates.sh` — spec-cited (specs)
- `scratch/saturation_s2.py` — results-cited (RESULTS)
- `scratch/saturation_s2b.py` — spec-cited (specs)
- `scratch/scaffold_review.py` — results-cited (RESULTS)
- `scratch/scorer_s1_battery.py` — results-cited (RESULTS)
- `scratch/scorer_s2_data.py` — results-cited (RESULTS, specs)
- `scratch/scorer_s2_train.py` — spec-cited (specs)
- `scratch/seed_audit.py` — spec-cited (specs)
- `scratch/series_probe.py` — spec-cited (specs)
- `scratch/snap_alloc.py` — spec-cited (specs)
- `scratch/snap_anatomy.py` — results-cited (RESULTS)
- `scratch/softprompt1.py` — results-cited (RESULTS, specs)
- `scratch/softprompt1.sh` — results-cited (RESULTS)
- `scratch/softprompt_sampler_probe.py` — spec-cited (specs)
- `scratch/soup_gate.py` — spec-cited (specs)
- `scratch/ssm_star.py` — library (RESULTS)
- `scratch/star_profile.py` — results-cited (RESULTS)
- `scratch/streaming_birth_d256.py` — library (RESULTS)
- `scratch/successors_acceptance.py` — results-cited (RESULTS)
- `scratch/sym45.py` — spec-cited (specs)
- `scratch/sym_birth.py` — library (RESULTS, specs)
- `scratch/sym_convert.py` — results-cited (RESULTS, specs)
- `scratch/sym_spectrum.py` — spec-cited (specs)
- `scratch/tenet_d1_revgate.py` — results-cited (RESULTS, specs)
- `scratch/tenet_d2_revdiet.py` — library (RESULTS, specs)
- `scratch/tenet_d3_budget.py` — library (specs)
- `scratch/tenet_mult_b32.py` — results-cited (RESULTS)
- `scratch/tenet_mult_census.py` — library (RESULTS, specs)
- `scratch/tenet_w0.py` — results-cited (RESULTS)
- `scratch/tenet_w1_bridge.py` — results-cited (RESULTS, specs)
- `scratch/tenet_w1_relational.py` — results-cited (RESULTS, specs)
- `scratch/tenet_w1_surfaces.py` — results-cited (RESULTS, specs)
- `scratch/tier_retry.py` — spec-cited (specs)
- `scratch/traj_accept.py` — results-cited (RESULTS, specs)
- `scratch/umoe_conserve.py` — library (RESULTS)
- `scratch/v4flash_census.py` — results-cited (RESULTS, specs)
- `scratch/v4flash_f1b.py` — library (RESULTS)
- `scratch/v4flash_f1c.py` — library (RESULTS)
- `scratch/v4flash_f1d.py` — results-cited (RESULTS, specs)
- `scratch/v4flash_header.py` — results-cited (RESULTS)
- `scratch/v4flash_router.py` — library (RESULTS, specs)
- `scratch/v4flash_rung0.py` — results-cited (RESULTS, specs)
- `scratch/v4flash_rung2b.py` — library (RESULTS, specs)
- `scratch/v4flash_rung2b_router.py` — results-cited (RESULTS)
- `scratch/v4flash_rungA.py` — library (RESULTS, specs)
- `scratch/v4flash_rungd.py` — library (RESULTS)
- `scratch/v4flash_rungd2.py` — results-cited (RESULTS)
- `scratch/v4flash_s0.py` — results-cited (RESULTS)
- `scratch/v4flash_twin.py` — library (RESULTS)
- `scratch/verify_intbirth_prims.py` — results-cited (RESULTS, specs)
- `scratch/wfloor_ladder.sh` — results-cited (RESULTS)
- `scratch/wsl.sh` — results-cited (RESULTS, specs)
- `scripts/__init__.py` — results-cited (RESULTS, specs)
- `scripts/backfill_code_commit.py` — spec-cited (specs)
- `scripts/bench_adaptive.py` — spec-cited (specs)
- `scripts/bench_budget_alloc.py` — results-cited (RESULTS)
- `scripts/bench_derivation.py` — spec-cited (specs)
- `scripts/bench_engine_regret.py` — results-cited (RESULTS)
- `scripts/bench_frontier.py` — spec-cited (specs)
- `scripts/bench_fused_ce.py` — results-cited (RESULTS)
- `scripts/bench_hints_ab.py` — results-cited (RESULTS)
- `scripts/bench_int4_gemv.py` — results-cited (RESULTS)
- `scripts/bench_ladder.py` — spec-cited (specs)
- `scripts/bench_lazy.py` — library (RESULTS)
- `scripts/bench_magic.py` — library (RESULTS)
- `scripts/bench_metal_kernels.py` — spec-cited (specs)
- `scripts/bench_nnue.py` — spec-cited (specs)
- `scripts/bench_pred_syndromes.py` — results-cited (RESULTS)
- `scripts/bench_proposer.py` — spec-cited (specs)
- `scripts/bench_quant_schemes.py` — results-cited (RESULTS)
- `scripts/bench_record.py` — results-cited (RESULTS)
- `scripts/bench_rotate_quantize.py` — spec-cited (specs)
- `scripts/bench_rule_basis.py` — results-cited (RESULTS)
- `scripts/bench_step_diversity.py` — results-cited (RESULTS)
- `scripts/bench_step_tokens.py` — library (RESULTS, specs)
- `scripts/bench_stitch_poc.py` — results-cited (RESULTS)
- `scripts/bench_verify_fast.py` — library (specs)
- `scripts/bench_vge.py` — spec-cited (specs)
- `scripts/bench_weight_anatomy.py` — results-cited (RESULTS)
- `scripts/bench_zx.py` — results-cited (RESULTS)
- `scripts/bench_zx_r5.py` — library (RESULTS)
- `scripts/bench_zx_r6.py` — results-cited (RESULTS)
- `scripts/bench_zx_r7.py` — results-cited (RESULTS)
- `scripts/book.py` — library (RESULTS, specs)
- `scripts/calibrate_hce.py` — spec-cited (specs)
- `scripts/ckpt_manifest.py` — results-cited (RESULTS, specs)
- `scripts/consolidate_mathnative.py` — results-cited (RESULTS)
- `scripts/convert_diet_prefix.py` — spec-cited (specs)
- `scripts/eval_pruned_moe.py` — spec-cited (specs)
- `scripts/eval_ruler.py` — spec-cited (specs)
- `scripts/expert_iter_steps.py` — library (specs)
- `scripts/expert_loop.py` — library (specs)
- `scripts/farm_algebra.py` — results-cited (RESULTS)
- `scripts/figlib.py` — spec-cited (specs)
- `scripts/gen_catalog.py` — results-cited (RESULTS, specs)
- `scripts/gen_codemap.py` — results-cited (RESULTS, specs)
- `scripts/gen_figures_web.py` — spec-cited (specs)
- `scripts/gen_index.py` — spec-cited (specs)
- `scripts/gen_lake.py` — results-cited (RESULTS, specs)
- `scripts/gen_magic_labels.py` — results-cited (RESULTS)
- `scripts/gen_proposer_data.py` — spec-cited (specs)
- `scripts/gen_readme.py` — spec-cited (specs)
- `scripts/gen_results_index.py` — results-cited (RESULTS, specs)
- `scripts/gen_syndrome_labels.py` — results-cited (RESULTS)
- `scripts/grow_mathnative.py` — library (specs)
- `scripts/harvest_frontier.py` — spec-cited (specs)
- `scripts/list_uncurated.py` — spec-cited (specs)
- `scripts/log_hygiene.py` — results-cited (RESULTS, specs)
- `scripts/mine_prior_update.py` — results-cited (RESULTS)
- `scripts/moe_router_stats.py` — results-cited (RESULTS, specs)
- `scripts/plot_gt1_crest.py` — spec-cited (specs)
- `scripts/plot_identity_crest.py` — spec-cited (specs)
- `scripts/plot_neurons.py` — spec-cited (specs)
- `scripts/probe_depth.py` — results-cited (RESULTS)
- `scripts/render_hero_neurons.py` — spec-cited (specs)
- `scripts/results_query.py` — spec-cited (specs)
- `scripts/rjob.py` — results-cited (RESULTS, specs)
- `scripts/sol_enrich_results.py` — library (specs)
- `scripts/sol_generate_tables.py` — spec-cited (specs)
- `scripts/step_grpo.py` — spec-cited (specs)
- `scripts/step_grpo_micro.py` — library (RESULTS, specs)
- `scripts/task_arithmetic.py` — spec-cited (specs)
- `scripts/train_calculus.py` — library (specs)
- `scripts/train_magic_estimator.py` — library (RESULTS)
- `scripts/train_magic_llm.py` — results-cited (RESULTS)
- `scripts/train_mathnative.py` — library (RESULTS, specs)
- `scripts/train_nnue.py` — library (specs)
- `scripts/train_proposer.py` — spec-cited (specs)
- `scripts/train_syndrome_decoder.py` — results-cited (RESULTS)
- `scripts/train_syndrome_policy.py` — results-cited (RESULTS)
- `scripts/train_ternary.py` — results-cited (RESULTS, specs)
- `scripts/train_value_head.py` — results-cited (RESULTS)
- `scripts/train_weight_reader.py` — spec-cited (specs)
- `scripts/validity_autopsy.py` — results-cited (RESULTS)

