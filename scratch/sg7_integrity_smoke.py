"""SG-BOUNDARY-BLOCK7-1 integrity smoke: mechanical checks of the SG7 credit
law (scratch/sg7_credit.py) on three arena states, one 32-row probe chunk,
fp32 CPU; the only optimizer steps are on in-memory copies. States: a
fresh seed-11 W_0 (no checkpoint) and the retained seed-27 frozen-BP
control at steps 463 and 15,420 (checkpoints/sgwriter1, digests asserted
against logs/sgwriter1/qual.jsonl; read-only).

Asserted per state:
 (a) SG7 logits bit-identical to the frozen-BP control's forward;
 (b) NO CE LEAK: the true CE reaches no block-7 parameter (autograd.grad
     all None; .grad None after loss.backward); frozen tensors carry no
     grad;
 (c) EXACT BP: blocks 4..6 / norm / head gradients of the SG7 objective
     bit-identical to the control's;
 (d) block-7 gradient bit-identical to the surrogate alone (J^T hat_delta
     with the applied hat_delta loaded as a constant);
 (e) PREDICTOR ISOLATION: L_phi reaches phi only; total reaches no phi;
 (f) FORCED-DELTA ENDPOINT: hat_delta_7 := delta^BP_7 -> blocks 4..6 /
     norm / head bit-identical to the control, block 7 within TOL_GRAD
     (relative to the control's max |g|), and one clipped AdamW step
     (lr 3e-4, wd 0.01, clip 1.0) within TOL_PARAM of the control's;
 (g) ELIGIBILITY: delta^BP_7 and the credit are exactly zero at every
     ineligible position, the credit nonzero at eligible ones (perturbed
     predictor);
 (h) FOLD A: the cached target equals the pre-step teacher target and
     differs from the post-step target after a nonzero step;
 (i) ZERO-INIT: first-step block-7 gradient exactly zero, head / norm and
     blocks 4..6 nonzero; block-7 motion after the step is pure AdamW
     decay; frozen tensors bit-identical;
 (j) the applied hat_delta predates the predictor step;
 (k) INPUT STATS derived at the state from the probe FIT chunks (shapes,
     finiteness, sd floor) and the predictor parameter count 500,736.
Writes logs/sgbb7/integrity_smoke_<HEAD>[-dirty]<INTEG_TAG>.jsonl, or the
path in INTEG_OUT when set (the driver passes the registered
logs/sgbb7/integrity_smoke_launch.jsonl); refuses to overwrite.
Usage: .venv/bin/python scratch/sg7_integrity_smoke.py
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
from dfa_credit import block_params, freeze_lower, hybrid_objective  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402
from sg_credit import run_sg_step, true_hidden_errors  # noqa: E402
from sg7_credit import SG7_BLOCK, build_local_predictor, forced_total, input_stats, n_params, sg7_forward, sg7_step_terms  # noqa: E402

AUDIT = Path("logs/sgaudit0/audit.json")
QUAL = Path("logs/sgwriter1/qual.jsonl")
CONTROL_CELL = "sgq_control_s27_lr0.0003"
CONTROL_STEPS = [463, 15420]
FIT_CHUNKS = [0, 2, 4, 6]
CHUNK = 32
TOL_GRAD = 1e-5
TOL_PARAM = 1e-6
SMOKE_SEED = 11


def grads(params):
    return [None if p.grad is None else p.grad.detach().clone() for p in params]


def same(a, b):
    return all((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y)) for x, y in zip(a, b))


def fresh(sd, tok, seed, perturb=0.0, mu=None, sdv=None):
    torch.manual_seed(0)
    model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    model.load_state_dict(sd)
    model.train()
    frozen, _ = freeze_lower(model, 4)
    pred = build_local_predictor(mu, sdv, seed)
    if perturb:
        with torch.no_grad():
            for p in pred.parameters():
                p.add_(torch.randn_like(p) * perturb)
    return model, frozen, pred


def control_grads(model, ids, mask, labels):
    model.zero_grad(set_to_none=True)
    ob = hybrid_objective(model, [], ids, mask, labels, 8)
    ob["loss"].backward()
    return ob["logits"].detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def run_state(sd, tok, rows, const7, label):
    ids, mask, labels = probe_tensors(rows[:CHUNK], tok)
    fit = [probe_tensors(rows[c * CHUNK:(c + 1) * CHUNK], tok) for c in FIT_CHUNKS]
    torch.manual_seed(0)
    m0 = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    m0.load_state_dict(sd)
    freeze_lower(m0, 4)
    mu, sdv = input_stats(m0, [f[0] for f in fit], [f[1] for f in fit], [f[2] for f in fit])
    C = {}
    rec = {"state": label, "checks": C}
    # (k)
    C["k_stats_shape_ok"] = tuple(mu.shape) == (1, 424) and tuple(sdv.shape) == (1, 424)
    C["k_stats_finite"] = bool(torch.isfinite(mu).all() and torch.isfinite(sdv).all())
    C["k_sd_min"] = float(sdv.min())
    C["k_x8_sd_median"] = float(sdv[0, :384].median())
    C["k_e_sd_median"] = float(sdv[0, 384:].median())
    C["k_n_clamped_features"] = int((sdv <= 1.0000001e-6).sum())      # features whose sd hit the 1e-6 floor (never-emitted logit classes)
    C["k_n_params"] = n_params(build_local_predictor(mu, sdv, 1))
    C["k_ok"] = C["k_stats_shape_ok"] and C["k_stats_finite"] and C["k_sd_min"] >= 9e-7 and C["k_n_params"] == 500736
    # (a), (b), (c), (d), (e) with a perturbed predictor (nonzero credit)
    model, frozen, pred = fresh(sd, tok, 101, perturb=1e-2, mu=mu, sdv=sdv)
    logits_c, gc = control_grads(model, ids, mask, labels)
    b7 = block_params(model, SG7_BLOCK)
    others = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("blocks.7.")]
    phi = list(pred.parameters())
    mp = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    T = sg7_step_terms(model, pred, ids, mask, labels, const7)
    C["a_logits_bitexact_v_control"] = bool(torch.equal(T["logits"].detach(), logits_c))
    C["a_teacher_logit_maxdiff"] = float((T["logits_T"] - T["logits"].detach()).abs().max())
    C["a_ok"] = C["a_logits_bitexact_v_control"] and C["a_teacher_logit_maxdiff"] <= 1e-5
    g7 = torch.autograd.grad(T["loss"], b7, allow_unused=True, retain_graph=True)
    C["b_ce_grad_block7_all_none"] = all(x is None for x in g7)
    model.zero_grad(set_to_none=True)
    T["loss"].backward(retain_graph=True)
    C["b_ce_backward_block7_grad_none"] = all(p.grad is None for p in b7)
    C["b_frozen_grads_none"] = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    C["b_ok"] = C["b_ce_grad_block7_all_none"] and C["b_ce_backward_block7_grad_none"] and C["b_frozen_grads_none"]
    model.zero_grad(set_to_none=True)
    hat = T["hat_delta"][SG7_BLOCK].clone()
    T["total"].backward()
    C["c_others_bitexact_v_control"] = all(torch.equal(p.grad, gc[n]) for n, p in others)
    C["c_ok"] = C["c_others_bitexact_v_control"]
    g7_total = grads(b7)
    model.zero_grad(set_to_none=True)
    _, _, _, x8_L, _ = sg7_forward(model, ids, mask)
    (hat * x8_L).sum().backward()
    C["d_block7_bitexact_v_surrogate"] = same(g7_total, grads(b7))
    C["d_block7_grad_nonzero"] = all(float(g.abs().max()) > 0 for g in g7_total)
    C["d_ok"] = C["d_block7_bitexact_v_surrogate"] and C["d_block7_grad_nonzero"]
    T = sg7_step_terms(model, pred, ids, mask, labels, const7)
    C["e_pred_loss_no_model_grad"] = all(x is None for x in torch.autograd.grad(T["pred_loss"], mp, allow_unused=True, retain_graph=True))
    C["e_pred_loss_reaches_all_phi"] = all(x is not None for x in torch.autograd.grad(T["pred_loss"], phi, allow_unused=True, retain_graph=True))
    C["e_total_no_phi_grad"] = all(x is None for x in torch.autograd.grad(T["total"], phi, allow_unused=True, retain_graph=True))
    C["e_ok"] = C["e_pred_loss_no_model_grad"] and C["e_pred_loss_reaches_all_phi"] and C["e_total_no_phi_grad"]
    C["pred_loss_value"] = float(T["pred_loss"].detach())
    C["baseline_mse"] = T["baseline_mse"]
    C["align_7"] = T["align"][SG7_BLOCK]
    # (f) forced endpoint
    model, frozen, pred = fresh(sd, tok, 101, mu=mu, sdv=sdv)
    import copy
    ctrl = copy.deepcopy(model)
    _, gc = control_grads(ctrl, ids, mask, labels)
    model.zero_grad(set_to_none=True)
    forced_total(model, ids, mask, labels).backward()
    rel7 = max(float((p.grad - gc[n]).abs().max()) / max(float(gc[n].abs().max()), 1e-30) for n, p in model.named_parameters() if n.startswith("blocks.7."))
    C["f_others_bitexact_v_control"] = all(torch.equal(p.grad, gc[n]) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("blocks.7."))
    C["f_block7_rel_maxdiff"] = rel7
    for m in (model, ctrl):
        ps = [p for p in m.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        torch.optim.AdamW(ps, lr=3e-4, weight_decay=0.01).step()
    C["f_param_maxdiff_after_step"] = max(float((p.detach() - q.detach()).abs().max()) for p, q in zip(model.parameters(), ctrl.parameters()))
    C["f_ok"] = C["f_others_bitexact_v_control"] and rel7 <= TOL_GRAD and C["f_param_maxdiff_after_step"] <= TOL_PARAM
    # (g) eligibility
    model, frozen, pred = fresh(sd, tok, 101, perturb=1e-2, mu=mu, sdv=sdv)
    T = sg7_step_terms(model, pred, ids, mask, labels, const7)
    inel = ~T["elig"]
    C["g_n_ineligible"] = int(inel.sum())
    C["g_target_zero_at_ineligible"] = float(T["targets"][SG7_BLOCK][inel].abs().max()) == 0.0
    C["g_credit_zero_at_ineligible"] = float(T["hat_delta"][SG7_BLOCK][inel].abs().max()) == 0.0
    C["g_credit_nonzero_at_eligible"] = float(T["hat_delta"][SG7_BLOCK][T["elig"]].abs().max()) > 0.0
    C["g_ok"] = C["g_n_ineligible"] > 0 and C["g_target_zero_at_ineligible"] and C["g_credit_zero_at_ineligible"] and C["g_credit_nonzero_at_eligible"]
    # (h) FOLD A + (j)
    model, frozen, pred = fresh(sd, tok, 101, perturb=1e-2, mu=mu, sdv=sdv)
    mp = [p for p in model.parameters() if p.requires_grad]
    phi = list(pred.parameters())
    _, _, pre = true_hidden_errors(model, ids, mask, labels, [SG7_BLOCK])
    T = sg7_step_terms(model, pred, ids, mask, labels, const7)
    hat_before = T["hat_delta"][SG7_BLOCK].clone()
    run_sg_step(T, mp, phi, torch.optim.AdamW(mp, lr=3e-4, weight_decay=0.01), torch.optim.AdamW(phi, lr=1e-3, weight_decay=0.0))
    _, _, post = true_hidden_errors(model, ids, mask, labels, [SG7_BLOCK])
    C["h_cached_equals_pre"] = bool(torch.equal(T["targets"][SG7_BLOCK], pre[SG7_BLOCK]))
    C["h_post_minus_pre_maxabs"] = float((post[SG7_BLOCK] - pre[SG7_BLOCK]).abs().max())
    C["h_ok"] = C["h_cached_equals_pre"] and C["h_post_minus_pre_maxabs"] > 0.0
    with torch.no_grad():
        _, _, x8_T, _, _ = sg7_forward(model, ids, mask)
        hat_after = float(const7) * pred(x8_T, T["e"])
    C["j_applied_equals_pre_step_hat"] = bool(torch.equal(hat_before, T["hat_delta"][SG7_BLOCK]))
    C["j_post_step_predictor_differs"] = not torch.equal(hat_after * T["elig"].unsqueeze(-1).float(), hat_before)
    C["j_ok"] = C["j_applied_equals_pre_step_hat"] and C["j_post_step_predictor_differs"]
    # (i) zero-init endpoint
    model, frozen, pred = fresh(sd, tok, 101, mu=mu, sdv=sdv)
    mp = [p for p in model.parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(mp, lr=3e-4, weight_decay=0.01)
    T = sg7_step_terms(model, pred, ids, mask, labels, const7)
    opt.zero_grad(set_to_none=True)
    T["total"].backward()
    C["i_block7_grads_exactly_zero"] = all(p.grad is not None and float(p.grad.abs().max()) == 0.0 for p in block_params(model, SG7_BLOCK))
    C["i_others_nonzero"] = all(float(p.grad.abs().max()) > 0 for n, p in model.named_parameters() if p.requires_grad and not n.startswith("blocks.7."))
    torch.nn.utils.clip_grad_norm_(mp, 1.0)
    opt.step()
    lr = opt.param_groups[0]["lr"]
    C["i_block7_move_is_pure_decay"] = all(torch.allclose(p.detach(), before[n] * (1 - lr * 0.01), rtol=0, atol=1e-7) for n, p in model.named_parameters() if n.startswith("blocks.7."))
    C["i_frozen_bitexact_after_step"] = all(torch.equal(before[n], p.detach()) for n, p in model.named_parameters() if not p.requires_grad)
    C["i_ok"] = C["i_block7_grads_exactly_zero"] and C["i_others_nonzero"] and C["i_block7_move_is_pure_decay"] and C["i_frozen_bitexact_after_step"]
    keys = [k for k in C if k.endswith("_ok")]
    rec["failed"] = [k for k in keys if not C[k]]
    rec["ok"] = not rec["failed"]
    rec["n_frozen"] = len(frozen)
    return rec


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip():
        commit += "-dirty"
    out = Path(os.environ.get("INTEG_OUT") or f"logs/sgbb7/integrity_smoke_{commit}{os.environ.get('INTEG_TAG', '')}.jsonl")   # the driver passes the registered literal path
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists")
    tok = TM.MathTokenizer()
    _, rows, probe_digest, _ = probe_rows(tok)
    audit = json.loads(AUDIT.read_text())
    const7 = audit["normalization_constants"]["arena_frozen_4_7"]["7"]
    births = [json.loads(l) for l in QUAL.open() if '"kind": "birth"' in l]
    ctrl = next(b for b in births if b["cell"] == CONTROL_CELL)
    states = []
    torch.manual_seed(SMOKE_SEED)
    m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    states.append(("w0_seed11", {k: v.detach().clone() for k, v in m.state_dict().items()}, None))
    for step in CONTROL_STEPS:
        p = Path(ctrl["outdir"]) / f"step_{step:05d}.pt"
        sd = torch.load(p, map_location="cpu")
        assert state_digest(sd) == ctrl["snapshots"][str(step)]["state_digest"], f"{p}: digest v receipt"
        states.append((f"control_s27_step{step}", sd, str(p)))
    rows_out = []
    for label, sd, path in states:
        rec = run_state(sd, tok, rows, const7, label)
        rec.update({"kind": "sg7_integrity_smoke", "prereg": "SG-BOUNDARY-BLOCK7-1", "commit": commit, "path": path, "state_digest": state_digest(sd),
                    "probe_token_digest": probe_digest, "const7": const7, "tol": {"grad_rel": TOL_GRAD, "param": TOL_PARAM},
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
        rows_out.append(rec)
        c = rec["checks"]
        print(f"[sg7smoke] {label}: ok={rec['ok']} f_rel7={c['f_block7_rel_maxdiff']:.1e} f_p={c['f_param_maxdiff_after_step']:.1e} h_post-pre={c['h_post_minus_pre_maxabs']:.2e} "
              f"pred_loss={c['pred_loss_value']:.3f} baseline={c['baseline_mse']:.3f} g_inel={c['g_n_ineligible']} x8sd={c['k_x8_sd_median']:.2f} failed={rec['failed']}", flush=True)
    with out.open("a") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"[sg7smoke] {sum(r['ok'] for r in rows_out)}/{len(rows_out)} states ok -> {out}")
    return 0 if all(r["ok"] for r in rows_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
