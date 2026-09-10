"""SYNTHETIC-GRADIENT-WRITER-1 integrity smoke (AMENDMENT -ARENA fold 5,
checks (a) to (g)), mechanical, zero main-model training beyond the two
in-smoke steps it takes on a COPY of the smoke checkpoint. Runs on the
frozenbb1_smoke FROZEN final (seed 11; the arena shape: blocks 0..3 + emb
frozen at their checkpoint values, SG blocks 4..7) with the arena constants
of OBSERVATION SG-PREDICTOR-AUDIT-0, and on the FULL final in full-stack
mode (SG blocks 0..7, full-stack constants), one 32-row probe chunk each.

Asserted per (mode, family):
 (a) teacher logits == writer logits at the same pre-update state: max
     |diff| <= TOL_LOGITS (fp32 CPU; the two forwards differ only by the
     functional_call dispatch);
 (b) the true teacher loss cannot accumulate block-parameter gradients:
     autograd.grad(loss_T, block params, allow_unused=True) is all None
     and every block .grad stays None after loss_T.backward();
 (c) the predictor regression loss reaches predictor parameters only:
     autograd.grad(pred_loss, model params, allow_unused=True) all None,
     autograd.grad(pred_loss, phi) all not None;
 (d) block parameters receive model-step gradients solely from the
     detached synthetic surrogate: grads from total.backward() are
     bit-identical to grads from a second run in which the same hat_delta
     values are loaded as constants (no predictor, no teacher);
 (e) head / norm receive their true CE gradient only: their grads from
     total.backward() are bit-identical to grads of the plain CE with x_8
     detached; frozen tensors (arena) have no grad;
 (f) the hat_delta used for the model update was produced before the
     predictor step: hat_delta recorded before pred_opt.step() equals the
     one applied (identity by construction; asserted by recomputing
     G_phi after the predictor step and showing it DIFFERS while the
     applied credit is the earlier value);
 (g) skipping the predictor update after hat_delta production leaves the
     current block gradient bit-identical (run with and without step 4).
 Plus: zero-init predictors give hat_delta == 0, the first model step's
 block gradients are EXACTLY zero (head / norm gradients nonzero) and the
 blocks move only by AdamW's decoupled weight decay (p *= 1 - lr * wd,
 the stock optimizer's own motion, not credit); the arena's frozen
 tensors are bit-identical after a model step.

Writes one row per (mode, family) to logs/sgwriter1/integrity_smoke.jsonl.
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
from dfa_credit import block_params, ce_loss, freeze_lower  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402
from sg_credit import build_predictors, sg_step_terms, writer_forward  # noqa: E402

OUT = Path("logs/sgwriter1/integrity_smoke.jsonl")
AUDIT = Path("logs/sgaudit0/audit.json")
SMOKE_BIRTHS = Path("logs/frozenbb1/smoke.jsonl")
TOL_LOGITS = 1e-5
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


def run_case(sd, tok, ids, mask, labels, sg_blocks, consts, family, arena):
    rec = {"mode": "arena" if arena else "full", "family": family, "sg_blocks": sg_blocks, "checks": {}}
    torch.manual_seed(0)
    model = load_model(sd, tok)
    if arena:
        frozen, trainable = freeze_lower(model, 4)
        rec["frozen_tensors"] = len(frozen)
    model_params = [p for p in model.parameters() if p.requires_grad]
    preds = build_predictors(family, sg_blocks, seed=101)
    phi = list(preds.parameters())
    blocks_p = [p for l in sg_blocks for p in block_params(model, l)]
    hn = [model.norm.g, model.head.weight]

    # ---- zero-init: hat_delta == 0 and G == 0 at step 0
    T = sg_step_terms(model, preds, ids, mask, labels, sg_blocks, consts)
    rec["checks"]["zero_init_hat_delta"] = all(float(T["hat_delta"][l].abs().max()) == 0.0 for l in sg_blocks)

    # ---- (a) teacher logits == writer logits
    loss_T, logits_T, targets = T["teacher"]()
    rec["checks"]["a_logit_maxdiff"] = float((logits_T - T["logits"].detach()).abs().max())
    rec["checks"]["a_ok"] = rec["checks"]["a_logit_maxdiff"] <= TOL_LOGITS
    rec["checks"]["a_loss_T"] = float(loss_T)
    rec["checks"]["a_loss_W"] = float(T["loss"].detach())

    # ---- (b) the teacher loss cannot reach block parameters
    logits_T2, outs_T2 = __import__("sg_credit").teacher_targets(model, ids, mask, sg_blocks)
    loss_T2 = ce_loss(logits_T2, labels)
    g = torch.autograd.grad(loss_T2, blocks_p + hn, allow_unused=True, retain_graph=True)
    rec["checks"]["b_teacher_grad_all_none"] = all(x is None for x in g)
    model.zero_grad(set_to_none=True)
    loss_T2.backward()
    rec["checks"]["b_teacher_backward_leaves_grad_none"] = all(p.grad is None for p in model.parameters())
    rec["checks"]["b_ok"] = rec["checks"]["b_teacher_grad_all_none"] and rec["checks"]["b_teacher_backward_leaves_grad_none"]

    # ---- (c) predictor loss reaches phi only (use a non-zero predictor state so the loss is not trivially flat)
    with torch.no_grad():
        for p in phi:
            p.add_(torch.randn_like(p) * 1e-3)
    T = sg_step_terms(model, preds, ids, mask, labels, sg_blocks, consts)
    _, _, targets = T["teacher"]()
    pl = T["pred_loss"](targets)
    gm = torch.autograd.grad(pl, model_params, allow_unused=True, retain_graph=True)
    gp = torch.autograd.grad(pl, phi, allow_unused=True, retain_graph=True)
    rec["checks"]["c_pred_loss_no_model_grad"] = all(x is None for x in gm)
    rec["checks"]["c_pred_loss_reaches_all_phi"] = all(x is not None for x in gp)
    rec["checks"]["c_ok"] = rec["checks"]["c_pred_loss_no_model_grad"] and rec["checks"]["c_pred_loss_reaches_all_phi"]
    rec["checks"]["pred_loss_value"] = float(pl)

    # ---- (d)/(e)/(g): model grads from total.backward() v constants-only run v plain CE; with/without predictor step
    model.zero_grad(set_to_none=True)
    hat_applied = {l: T["hat_delta"][l].clone() for l in sg_blocks}
    T["total"].backward()
    g_total = grads_of(blocks_p)
    g_hn = grads_of(hn)
    frozen_grads_none = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    # constants-only surrogate (no predictor, no teacher)
    model.zero_grad(set_to_none=True)
    outs, logits = writer_forward(model, ids, mask, sg_blocks)
    loss = ce_loss(logits, labels)
    S = sum((hat_applied[l] * outs[l]).sum() for l in sg_blocks)
    (loss + S).backward()
    g_const = grads_of(blocks_p)
    g_hn_const = grads_of(hn)
    rec["checks"]["d_block_grads_bitexact_v_constants"] = same(g_total, g_const)
    # plain CE with x_8 detached: head / norm grads
    model.zero_grad(set_to_none=True)
    outs, logits = writer_forward(model, ids, mask, sg_blocks)
    ce_loss(logits, labels).backward()
    g_hn_ce = grads_of(hn)
    rec["checks"]["e_headnorm_grads_bitexact_v_true_ce"] = same(g_hn, g_hn_ce) and same(g_hn_const, g_hn_ce)
    rec["checks"]["e_frozen_grads_none"] = frozen_grads_none if arena else True
    # (g) skip v take the predictor step: block grads identical (they were computed before)
    pred_opt = torch.optim.AdamW(phi, lr=3e-4, weight_decay=0.0)
    model.zero_grad(set_to_none=True)
    T2 = sg_step_terms(model, preds, ids, mask, labels, sg_blocks, consts)
    hat_before = {l: T2["hat_delta"][l].clone() for l in sg_blocks}
    T2["total"].backward()
    g_before_pred_step = grads_of(blocks_p)
    _, _, targets2 = T2["teacher"]()
    pred_opt.zero_grad(set_to_none=True)
    T2["pred_loss"](targets2).backward()
    pred_opt.step()
    g_after_pred_step = grads_of(blocks_p)
    rec["checks"]["g_block_grads_unchanged_by_pred_step"] = same(g_before_pred_step, g_after_pred_step)
    # (f) the applied credit predates the predictor step: recompute G after the step, it differs; applied == hat_before
    with torch.no_grad():
        outs_now, _ = writer_forward(model, ids, mask, sg_blocks)
        hat_after = {l: float(consts[str(l)]) * preds[str(l)](outs_now[l], T2["e"]) for l in sg_blocks}
    rec["checks"]["f_applied_equals_pre_step_hat"] = all(torch.equal(hat_before[l], T2["hat_delta"][l]) for l in sg_blocks)
    rec["checks"]["f_post_step_predictor_differs"] = any(not torch.equal(hat_after[l], hat_before[l]) for l in sg_blocks)
    rec["checks"]["f_ok"] = rec["checks"]["f_applied_equals_pre_step_hat"] and rec["checks"]["f_post_step_predictor_differs"]
    # ---- first model step from zero-init predictors moves head / norm only
    torch.manual_seed(0)
    model0 = load_model(sd, tok)
    if arena:
        freeze_lower(model0, 4)
    preds0 = build_predictors(family, sg_blocks, seed=101)
    opt = torch.optim.AdamW([p for p in model0.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
    before = {n: p.detach().clone() for n, p in model0.named_parameters()}
    T0 = sg_step_terms(model0, preds0, ids, mask, labels, sg_blocks, consts)
    opt.zero_grad(set_to_none=True)
    T0["total"].backward()
    torch.nn.utils.clip_grad_norm_([p for p in model0.parameters() if p.requires_grad], 1.0)
    opt.step()
    moved = [n for n, p in model0.named_parameters() if not torch.equal(before[n], p.detach())]
    rec["checks"]["first_step_moves_n"] = len(moved)
    # zero-init predictors: block gradients are EXACTLY zero at the first step (no credit); the blocks still move
    # by the decoupled weight decay of the stock AdamW (p *= 1 - lr * wd), which is the optimizer's, not credit
    blk0 = [p for l in sg_blocks for p in block_params(model0, l)]
    rec["checks"]["first_step_block_grads_exactly_zero"] = all(p.grad is not None and float(p.grad.abs().max()) == 0.0 for p in blk0)
    rec["checks"]["first_step_headnorm_grads_nonzero"] = all(float(p.grad.abs().max()) > 0 for p in (model0.norm.g, model0.head.weight))
    rec["checks"]["first_step_block_move_is_pure_decay"] = all(
        torch.allclose(p.detach(), before[n] * (1 - opt.param_groups[0]["lr"] * 0.01), rtol=0, atol=1e-7)
        for n, p in model0.named_parameters() if p.requires_grad and n not in ("norm.g", "head.weight"))
    rec["checks"]["frozen_bitexact_after_step"] = (all(torch.equal(before[n], p.detach()) for n, p in model0.named_parameters() if not p.requires_grad) if arena else True)
    rec["ok"] = all(rec["checks"][k] for k in ("zero_init_hat_delta", "a_ok", "b_ok", "c_ok", "d_block_grads_bitexact_v_constants",
                                                "e_headnorm_grads_bitexact_v_true_ce", "e_frozen_grads_none", "g_block_grads_unchanged_by_pred_step",
                                                "f_ok", "first_step_block_grads_exactly_zero", "first_step_headnorm_grads_nonzero",
                                                "first_step_block_move_is_pure_decay", "frozen_bitexact_after_step"))
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
                    "source_cell": b["cell"], "tol_logits": TOL_LOGITS, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
        rows_out.append(rec)
        print(f"[sgsmoke] {rec['mode']} {fam}: ok={rec['ok']} a_maxdiff={rec['checks']['a_logit_maxdiff']:.2e} moved_n={rec['checks']['first_step_moves_n']}", flush=True)
        bad = [k for k, v in rec["checks"].items() if v is False]
        if bad:
            print(f"[sgsmoke]   FAILED checks: {bad}")
    with OUT.open("a") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"[sgsmoke] {sum(r['ok'] for r in rows_out)}/{len(rows_out)} cases ok -> {OUT}")
    return 0 if all(r["ok"] for r in rows_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
