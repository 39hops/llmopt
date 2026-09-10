"""SYNTHETIC-GRADIENT-WRITER-1 integrity smoke (AMENDMENT -ARENA fold 5
checks (a) to (g), AMENDMENT -SEAL fold A check (h) and the two endpoint
checks (i), (j)); mechanical; the only optimizer steps it takes are on a
COPY of a smoke checkpoint. Cases: the frozenbb1_smoke FROZEN final (seed
11; arena shape: emb + blocks 0..3 frozen, SG blocks 4..7, arena constants)
and the FULL final in full-stack mode (SG blocks 0..7, full-stack
constants), each with LINEAR and MLP-256, one 32-row probe chunk.

Asserted per (mode, family):
 (a) teacher logits == writer logits at the same pre-update state
     (max |diff| <= TOL_LOGITS, fp32 CPU);
 (b) the true teacher loss cannot accumulate any parameter gradient
     (autograd.grad over model params all None; .grad all None after
     loss_T.backward());
 (c) the predictor regression loss reaches predictor parameters only;
 (d) block parameters receive model-step gradients solely from the
     detached synthetic surrogate: bit-identical to a run with the same
     hat_delta loaded as constants (no predictor, no teacher);
 (e) head / norm gradients bit-identical to the plain CE with x_8
     detached; frozen tensors (arena) carry no grad;
 (f) the applied hat_delta predates the predictor step (the predictor
     recomputed after its step differs, the applied value is the earlier);
 (g) skipping the predictor update leaves the block gradients bit-identical;
 (h) FOLD A, same-pre-update teacher order: the cached target of the
     registered step equals a teacher target computed at W_t before a
     deliberately nonzero model step, bit-exactly, and differs from the
     target recomputed at W_{t+1} after that step (the fixture changes:
     max |T_post - T_pre| > 0);
 (i) forced-delta^BP endpoint: with hat_delta_l := delta^BP_l (teacher
     targets substituted for s_l * G), the SG-block gradients equal the
     frozen-top backprop gradients (dfa_credit.hybrid_objective(k=8) with
     freeze_lower, the FROZEN-BACKBONE-1 arm) within TOL_GRAD relative
     to the gradient scale, and one AdamW step from the same state lands
     within TOL_PARAM (arena only: the endpoint IS the FROZEN arm);
 (j) zero-output predictor endpoint: first-step SG-block gradients are
     exactly zero, head / norm gradients nonzero, and every block motion
     equals AdamW's decoupled weight decay p *= 1 - lr * wd; frozen
     tensors bit-identical after the step.
Writes one row per case to logs/sgwriter1/integrity_smoke_seal.jsonl (the
pre-seal receipt logs/sgwriter1/integrity_smoke.jsonl stays frozen).
Usage: .venv/bin/python scratch/sg_integrity_smoke.py
"""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402
from dfa_credit import block_params, ce_loss, freeze_lower, hybrid_objective  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402
from sg_credit import build_predictors, run_sg_step, sg_step_terms, true_hidden_errors, writer_forward  # noqa: E402

OUT = Path("logs/sgwriter1/integrity_smoke_seal.jsonl")
AUDIT = Path("logs/sgaudit0/audit.json")
SMOKE_BIRTHS = Path("logs/frozenbb1/smoke.jsonl")
TOL_LOGITS = 1e-5
TOL_GRAD = 1e-5      # relative to max |g| of the BP reference, per tensor
TOL_PARAM = 1e-6     # absolute, after one AdamW step
ARENA = [4, 5, 6, 7]
FULL = list(range(8))


def load_model(sd, tok):
    m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    m.load_state_dict(sd)
    return m.train()


def grads_of(params):
    return [None if p.grad is None else p.grad.detach().clone() for p in params]


def same(a, b):
    return all((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y)) for x, y in zip(a, b))


def fresh(sd, tok, arena, family, sg_blocks):
    torch.manual_seed(0)
    model = load_model(sd, tok)
    frozen = freeze_lower(model, 4)[0] if arena else []
    preds = build_predictors(family, sg_blocks, seed=101)
    return model, frozen, preds


def run_case(sd, tok, ids, mask, labels, sg_blocks, consts, family, arena):
    rec = {"mode": "arena" if arena else "full", "family": family, "sg_blocks": sg_blocks, "checks": {}}
    C = rec["checks"]
    model, frozen, preds = fresh(sd, tok, arena, family, sg_blocks)
    rec["frozen_tensors"] = len(frozen)
    model_params = [p for p in model.parameters() if p.requires_grad]
    phi = list(preds.parameters())
    blocks_p = [p for l in sg_blocks for p in block_params(model, l)]
    hn = [model.norm.g, model.head.weight]

    # zero-init: hat_delta == 0
    T = sg_step_terms(model, preds, ids, mask, labels, sg_blocks, consts)
    C["zero_init_hat_delta"] = all(float(T["hat_delta"][l].abs().max()) == 0.0 for l in sg_blocks)
    # (a)
    C["a_logit_maxdiff"] = float((T["logits_T"] - T["logits"].detach()).abs().max())
    C["a_ok"] = C["a_logit_maxdiff"] <= TOL_LOGITS
    # (b)
    from sg_credit import teacher_targets
    logits_T2, _ = teacher_targets(model, ids, mask)
    loss_T2 = ce_loss(logits_T2, labels)
    g = torch.autograd.grad(loss_T2, [p for p in model.parameters() if p.requires_grad], allow_unused=True, retain_graph=True)
    C["b_teacher_grad_all_none"] = all(x is None for x in g)
    model.zero_grad(set_to_none=True)
    loss_T2.backward()
    C["b_teacher_backward_leaves_grad_none"] = all(p.grad is None for p in model.parameters())
    C["b_ok"] = C["b_teacher_grad_all_none"] and C["b_teacher_backward_leaves_grad_none"]
    # (c) with a non-zero predictor state
    with torch.no_grad():
        for p in phi:
            p.add_(torch.randn_like(p) * 1e-3)
    T = sg_step_terms(model, preds, ids, mask, labels, sg_blocks, consts)
    gm = torch.autograd.grad(T["pred_loss"], model_params, allow_unused=True, retain_graph=True)
    gp = torch.autograd.grad(T["pred_loss"], phi, allow_unused=True, retain_graph=True)
    C["c_pred_loss_no_model_grad"] = all(x is None for x in gm)
    C["c_pred_loss_reaches_all_phi"] = all(x is not None for x in gp)
    C["c_ok"] = C["c_pred_loss_no_model_grad"] and C["c_pred_loss_reaches_all_phi"]
    C["pred_loss_value"] = float(T["pred_loss"].detach())
    C["pred_loss_per_block"] = {str(k): v for k, v in T["pred_loss_per_block"].items()}
    C["align"] = {str(k): v for k, v in T["align"].items()}
    # (d)/(e)
    model.zero_grad(set_to_none=True)
    hat_applied = {l: T["hat_delta"][l].clone() for l in sg_blocks}
    T["total"].backward()
    g_total, g_hn = grads_of(blocks_p), grads_of(hn)
    C["e_frozen_grads_none"] = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    model.zero_grad(set_to_none=True)
    outs, logits = writer_forward(model, ids, mask, sg_blocks)
    (ce_loss(logits, labels) + sum((hat_applied[l] * outs[l]).sum() for l in sg_blocks)).backward()
    g_const, g_hn_const = grads_of(blocks_p), grads_of(hn)
    C["d_block_grads_bitexact_v_constants"] = same(g_total, g_const)
    model.zero_grad(set_to_none=True)
    outs, logits = writer_forward(model, ids, mask, sg_blocks)
    ce_loss(logits, labels).backward()
    g_hn_ce = grads_of(hn)
    C["e_headnorm_grads_bitexact_v_true_ce"] = same(g_hn, g_hn_ce) and same(g_hn_const, g_hn_ce)
    # (g) + (f): the registered step with and without the predictor update
    model_opt = torch.optim.AdamW(model_params, lr=3e-4, weight_decay=0.01)
    pred_opt = torch.optim.AdamW(phi, lr=3e-4, weight_decay=0.0)
    model.zero_grad(set_to_none=True)
    T2 = sg_step_terms(model, preds, ids, mask, labels, sg_blocks, consts)
    hat_before = {l: T2["hat_delta"][l].clone() for l in sg_blocks}
    T2["total"].backward(retain_graph=True)
    g_before = grads_of(blocks_p)
    pred_opt.zero_grad(set_to_none=True)
    T2["pred_loss"].backward()
    pred_opt.step()
    g_after = grads_of(blocks_p)
    C["g_block_grads_unchanged_by_pred_step"] = same(g_before, g_after)
    with torch.no_grad():
        outs_now, _ = writer_forward(model, ids, mask, sg_blocks)
        hat_after = {l: float(consts[str(l)]) * preds[str(l)](outs_now[l], T2["e"]) for l in sg_blocks}
    C["f_applied_equals_pre_step_hat"] = all(torch.equal(hat_before[l], T2["hat_delta"][l]) for l in sg_blocks)
    C["f_post_step_predictor_differs"] = any(not torch.equal(hat_after[l], hat_before[l]) for l in sg_blocks)
    C["f_ok"] = C["f_applied_equals_pre_step_hat"] and C["f_post_step_predictor_differs"]

    # (h) FOLD A: cached target == pre-step teacher target, != post-step target
    model, frozen, preds = fresh(sd, tok, arena, family, sg_blocks)
    with torch.no_grad():
        for p in preds.parameters():
            p.add_(torch.randn_like(p) * 1e-2)      # nonzero credit so the model step is nonzero on the blocks
    model_params = [p for p in model.parameters() if p.requires_grad]
    phi = list(preds.parameters())
    model_opt = torch.optim.AdamW(model_params, lr=3e-4, weight_decay=0.01)
    pred_opt = torch.optim.AdamW(phi, lr=3e-4, weight_decay=0.0)
    _, _, T_pre = true_hidden_errors(model, ids, mask, labels, sg_blocks)       # at W_t, before anything
    T3 = sg_step_terms(model, preds, ids, mask, labels, sg_blocks, consts)     # the registered step caches its target
    run_sg_step(T3, model_params, phi, model_opt, pred_opt)                     # W_t -> W_{t+1}
    _, _, T_post = true_hidden_errors(model, ids, mask, labels, sg_blocks)     # at W_{t+1}
    C["h_cached_equals_pre_step_target"] = all(torch.equal(T3["targets"][l], T_pre[l]) for l in sg_blocks)
    C["h_post_step_target_differs"] = max(float((T_post[l] - T_pre[l]).abs().max()) for l in sg_blocks) > 0.0
    C["h_cached_not_post_step"] = any(not torch.equal(T3["targets"][l], T_post[l]) for l in sg_blocks)
    C["h_post_minus_pre_maxabs"] = max(float((T_post[l] - T_pre[l]).abs().max()) for l in sg_blocks)
    C["h_ok"] = C["h_cached_equals_pre_step_target"] and C["h_post_step_target_differs"] and C["h_cached_not_post_step"]

    # (i) forced-delta^BP endpoint (arena only: the endpoint is the FROZEN-BACKBONE-1 arm)
    if arena:
        model, frozen, preds = fresh(sd, tok, arena, family, sg_blocks)
        model_params = [p for p in model.parameters() if p.requires_grad]
        blocks_p = [p for l in sg_blocks for p in block_params(model, l)]
        _, _, tg = true_hidden_errors(model, ids, mask, labels, sg_blocks)
        model.zero_grad(set_to_none=True)
        outs, logits = writer_forward(model, ids, mask, sg_blocks)
        (ce_loss(logits, labels) + sum((tg[l] * outs[l]).sum() for l in sg_blocks)).backward()
        g_forced = grads_of(blocks_p + [model.norm.g, model.head.weight])
        torch.nn.utils.clip_grad_norm_(model_params, 1.0)
        opt_f = torch.optim.AdamW(model_params, lr=3e-4, weight_decay=0.01)
        opt_f.step()
        p_forced = [p.detach().clone() for p in model_params]
        # reference: the FROZEN-BACKBONE-1 arm (hybrid_objective k=8 over freeze_lower)
        model_r, _, _ = fresh(sd, tok, arena, family, sg_blocks)
        params_r = [p for p in model_r.parameters() if p.requires_grad]
        blocks_r = [p for l in sg_blocks for p in block_params(model_r, l)]
        model_r.zero_grad(set_to_none=True)
        hybrid_objective(model_r, [], ids, mask, labels, 8)["loss"].backward()
        g_ref = grads_of(blocks_r + [model_r.norm.g, model_r.head.weight])
        rel = max(float((a - b).abs().max()) / max(float(b.abs().max()), 1e-30) for a, b in zip(g_forced, g_ref))
        torch.nn.utils.clip_grad_norm_(params_r, 1.0)
        opt_r = torch.optim.AdamW(params_r, lr=3e-4, weight_decay=0.01)
        opt_r.step()
        pdiff = max(float((a - b.detach()).abs().max()) for a, b in zip(p_forced, params_r))
        C["i_forced_grad_rel_maxdiff"] = rel
        C["i_forced_param_maxdiff_after_step"] = pdiff
        C["i_ok"] = rel <= TOL_GRAD and pdiff <= TOL_PARAM
    else:
        C["i_ok"] = True
        C["i_note"] = "full-stack: no frozen-top reference; endpoint identity is arena-only"

    # (j) zero-output endpoint
    model0, frozen0, preds0 = fresh(sd, tok, arena, family, sg_blocks)
    params0 = [p for p in model0.parameters() if p.requires_grad]
    opt0 = torch.optim.AdamW(params0, lr=3e-4, weight_decay=0.01)
    before = {n: p.detach().clone() for n, p in model0.named_parameters()}
    T0 = sg_step_terms(model0, preds0, ids, mask, labels, sg_blocks, consts)
    opt0.zero_grad(set_to_none=True)
    T0["total"].backward()
    blk0 = [p for l in sg_blocks for p in block_params(model0, l)]
    C["j_block_grads_exactly_zero"] = all(p.grad is not None and float(p.grad.abs().max()) == 0.0 for p in blk0)
    C["j_headnorm_grads_nonzero"] = all(float(p.grad.abs().max()) > 0 for p in (model0.norm.g, model0.head.weight))
    torch.nn.utils.clip_grad_norm_(params0, 1.0)
    opt0.step()
    lr = opt0.param_groups[0]["lr"]
    C["j_block_move_is_pure_decay"] = all(torch.allclose(p.detach(), before[n] * (1 - lr * 0.01), rtol=0, atol=1e-7)
                                          for n, p in model0.named_parameters() if p.requires_grad and n not in ("norm.g", "head.weight"))
    C["j_frozen_bitexact_after_step"] = all(torch.equal(before[n], p.detach()) for n, p in model0.named_parameters() if not p.requires_grad)
    C["j_ok"] = C["j_block_grads_exactly_zero"] and C["j_headnorm_grads_nonzero"] and C["j_block_move_is_pure_decay"] and C["j_frozen_bitexact_after_step"]

    keys = ("zero_init_hat_delta", "a_ok", "b_ok", "c_ok", "d_block_grads_bitexact_v_constants", "e_headnorm_grads_bitexact_v_true_ce",
            "e_frozen_grads_none", "f_ok", "g_block_grads_unchanged_by_pred_step", "h_ok", "i_ok", "j_ok")
    rec["failed"] = [k for k in keys if not C[k]]
    rec["ok"] = not rec["failed"]
    return rec


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    tok = TM.MathTokenizer()
    _, rows, probe_digest, _ = probe_rows(tok)
    ids, mask, labels = probe_tensors(rows[:32], tok)
    audit = json.loads(AUDIT.read_text())
    consts_arena = audit["normalization_constants"]["arena_frozen_4_7"]
    consts_full = audit["normalization_constants"]["full_stack_0_7"]
    births = [json.loads(l) for l in SMOKE_BIRTHS.open() if '"kind": "birth"' in l]
    by_mode = {b["mode"]: b for b in births if b.get("smoke") and b.get("final")}
    rows_out = []
    for mode, fam in (("zero", "linear"), ("zero", "mlp256"), ("bp", "linear"), ("bp", "mlp256")):
        b = by_mode[mode]
        sd = torch.load(Path(b["outdir"]) / "final.pt", map_location="cpu")
        assert state_digest(sd) == b["final"]["state_digest"]
        arena = mode == "zero"
        rec = run_case(sd, tok, ids, mask, labels, ARENA if arena else FULL, consts_arena if arena else consts_full, fam, arena)
        rec.update({"kind": "sg_integrity_smoke", "commit": commit, "probe_token_digest": probe_digest, "state_digest": state_digest(sd),
                    "source_cell": b["cell"], "tol": {"logits": TOL_LOGITS, "grad_rel": TOL_GRAD, "param": TOL_PARAM},
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
        rows_out.append(rec)
        c = rec["checks"]
        print(f"[sgsmoke] {rec['mode']} {fam}: ok={rec['ok']} a={c['a_logit_maxdiff']:.1e} h_post-pre={c['h_post_minus_pre_maxabs']:.2e} "
              f"i_rel={c.get('i_forced_grad_rel_maxdiff', float('nan')):.1e} i_p={c.get('i_forced_param_maxdiff_after_step', float('nan')):.1e} "
              f"pred_loss={c['pred_loss_value']:.3f} failed={rec['failed']}", flush=True)
    with OUT.open("a") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"[sgsmoke] {sum(r['ok'] for r in rows_out)}/{len(rows_out)} cases ok -> {OUT}")
    return 0 if all(r["ok"] for r in rows_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
