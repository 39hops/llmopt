"""SYNTHETIC-GRADIENT-WRITER-1 credit law (PRE-REG SYNTHETIC-GRADIENT-
WRITER-1 L69765, AMENDMENT -ARENA L70126, OBSERVATION SG-PREDICTOR-AUDIT-0
constants, AMENDMENT -SEAL folds A and B). The per-block synthetic hidden
error

    hat_delta_l = s_l * G_phi_l(stopgrad(x_{l+1}), stopgrad(e_t))

replaces the true hidden error delta^BP_l = dL/dx_{l+1} for every SG block
l in sg_blocks at the eligible label positions (labels != -100; delta^BP is
exactly zero at the pad positions, so no credit is applied there); block parameters receive J_{f_l}^T hat_delta_l only (the
surrogate sum_l <hat_delta_l.detach(), x_{l+1}> backpropagated through a
block whose input is detached); head.weight and norm.g take the true CE
gradient with x_8 detached (the sealed DFA / frontier treatment); blocks
below the first SG block are frozen at W_0 (the arena) or, in full-stack
mode, are SG blocks themselves with emb receiving hat_delta_0 +
J_{f_0}^T hat_delta_0 (the identity path).

FOLD A, frozen step semantics (every gradient derives from the pre-update
pair (W_t, phi_t)):
  1. writer forward at W_t;
  2. hat_delta_t from phi_t;
  3. parameter-detached teacher pass at W_t, delta^BP_t CACHED;
  4. model synthetic-credit objective backward;
  5. predictor regression against the cached delta^BP_t, backward;
  6. clip model and predictor gradients separately;
  7. model optimizer step;  8. predictor optimizer step;  9. schedulers.
sg_step_terms performs 1 to 3 and returns the objectives; run_sg_step
performs 4 to 9. No teacher target is ever computed after W has moved
(smoked: scratch/sg_integrity_smoke.py check h).

Teacher pass: a forward whose MODEL PARAMETER VIEWS ARE DETACHED CONSTANTS
(torch.func.functional_call on {name: p.detach()}, x_0 a leaf activation),
so the true loss graph contains activations only and cannot reach any
parameter by construction; it reproduces the writer logits at W_t.

FOLD B, predictor regression (the s_l are element-RMS scales, so the loss
is an elementwise MSE, not a hidden-dimension SSE):
  L_l   = mean over eligible label positions (labels != -100) AND hidden
          dimensions of (G_phi_l - delta^BP_l / s_l)^2
  L_phi = mean_l L_l
gradients reach phi only (both predictor inputs are detached). The applied
credit stays hat_delta_l = s_l * G_phi_l with no runtime scaling or gain.
The zero-predictor baseline of the same reduction (G = 0: the mean of
(delta^BP / s)^2) is returned beside it at every step for the registered
prior.

Predictor families: LINEAR G(h, e) = h A + e B + C (Czarnecki's SG(h, y);
Jaderberg's best cDNI family) and MLP-256 (one hidden layer, ReLU); the
OUTPUT layer is zero-initialised in both (Jaderberg), so the writer starts
with exactly zero credit to the SG blocks (their first-step gradients are
zero; any motion is AdamW's decoupled weight decay).
"""
import torch
import torch.nn as nn
from torch.func import functional_call

from dfa_credit import causal_mask, ce_loss

D_MODEL, N_OUT, N_BLOCKS = 384, 40, 8
ARENA_BLOCKS = [4, 5, 6, 7]


class LinearSG(nn.Module):
    def __init__(self, d=D_MODEL, n_out=N_OUT):
        super().__init__()
        self.A = nn.Linear(d, d, bias=False)
        self.B = nn.Linear(n_out, d, bias=True)
        nn.init.zeros_(self.A.weight)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, h, e):
        return self.A(h) + self.B(e)


class MLPSG(nn.Module):
    def __init__(self, d=D_MODEL, n_out=N_OUT, width=256):
        super().__init__()
        self.inp = nn.Linear(d + n_out, width, bias=True)
        self.out = nn.Linear(width, d, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, e):
        return self.out(torch.relu(self.inp(torch.cat([h, e], -1))))


FAMILIES = {"linear": LinearSG, "mlp256": MLPSG}


def build_predictors(family, sg_blocks, seed, d=D_MODEL, n_out=N_OUT):
    """One predictor per SG block, deterministic init from `seed` (the
    zero output layer makes the initial synthetic credit exactly zero
    regardless of the seed; the seed fixes the MLP input layer)."""
    g = torch.Generator().manual_seed(seed)
    preds = nn.ModuleDict()
    for l in sg_blocks:
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
        preds[str(l)] = FAMILIES[family](d, n_out)
    return preds


def writer_forward(model, ids, attn_mask, sg_blocks):
    """The writer-state forward. Every SG block sees a DETACHED input (block
    0 in full-stack mode takes emb's output attached: the identity path);
    blocks below the first SG block are the frozen backbone (no trainable
    tensor upstream); x_8 is detached before norm / head.
    Returns (outs, logits): outs[l] = x_{l+1}."""
    m = causal_mask(ids, attn_mask)
    x = model.emb(ids)
    outs = []
    first_sg = min(sg_blocks)
    for l, b in enumerate(model.blocks):
        if l == 0 and l in sg_blocks:
            xin = x
        elif l >= first_sg:
            xin = x.detach()
        else:
            xin = x
        x, _ = b(xin, m, None)
        outs.append(x)
    logits = model.head(model.norm(x.detach()))
    return outs, logits


def teacher_targets(model, ids, attn_mask):
    """Forward over DETACHED PARAMETER CONSTANTS (functional_call); the
    graph holds activations only. Returns (logits_T, outs_T)."""
    params = {n: p.detach() for n, p in model.named_parameters()}
    m = causal_mask(ids, attn_mask)
    x = functional_call(model.emb, {k[len("emb."):]: v for k, v in params.items() if k.startswith("emb.")}, (ids,))
    x = x.requires_grad_(True)   # x_0 is a leaf ACTIVATION
    outs = []
    for l, b in enumerate(model.blocks):
        pb = {k[len(f"blocks.{l}."):]: v for k, v in params.items() if k.startswith(f"blocks.{l}.")}
        x, _ = functional_call(b, pb, (x, m, None))
        outs.append(x)
    xn = functional_call(model.norm, {k[len("norm."):]: v for k, v in params.items() if k.startswith("norm.")}, (x,))
    logits = functional_call(model.head, {k[len("head."):]: v for k, v in params.items() if k.startswith("head.")}, (xn,))
    return logits, outs


def true_hidden_errors(model, ids, attn_mask, labels, sg_blocks):
    """delta^BP_l = dL/dx_{l+1} for l in sg_blocks at the CURRENT W via the
    parameter-detached teacher pass. Returns (loss_T, logits_T, {l: delta})."""
    logits_T, outs_T = teacher_targets(model, ids, attn_mask)
    loss_T = ce_loss(logits_T, labels)
    grads = torch.autograd.grad(loss_T, [outs_T[l] for l in sg_blocks])
    return loss_T.detach(), logits_T.detach(), {l: g.detach() for l, g in zip(sg_blocks, grads)}


def normalized_mse(g_out, targets, consts, elig, sg_blocks):
    """FOLD B: per block the mean over eligible positions and hidden
    dimensions of (G - delta^BP / s)^2; L_phi = mean over blocks.
    Returns (L_phi, {l: L_l detached float})."""
    n_elig = elig.sum()
    per = {}
    tot = 0.0
    for l in sg_blocks:
        diff = g_out[l] - targets[l] / float(consts[str(l)])
        L_l = (diff.pow(2).mean(-1) * elig).sum() / n_elig
        per[l] = float(L_l.detach())
        tot = tot + L_l
    return tot / len(sg_blocks), per


def alignment(hat_delta, targets, elig, sg_blocks):
    """Descriptive: cosine between the applied credit and delta^BP over the
    eligible positions, per block (never a bar)."""
    out = {}
    for l in sg_blocks:
        a = hat_delta[l][elig].reshape(-1)
        b = targets[l][elig].reshape(-1)
        na, nb = float(a.norm()), float(b.norm())
        out[l] = float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else None
    return out


def sg_step_terms(model, preds, ids, attn_mask, labels, sg_blocks, consts):
    """Steps 1 to 3 of the frozen order at the pre-update pair (W_t, phi_t):
    writer forward, hat_delta from phi_t, teacher targets CACHED. Returns
      loss, e, outs, logits, g_out {l: G_phi_l attached to phi},
      hat_delta {l: s_l * G.detach()}, total (model backward object),
      targets {l: delta^BP_l at W_t, detached, cached}, loss_T, logits_T,
      elig (labels != -100), pred_loss (L_phi attached to phi), pred_loss_per_block,
      align {l: cos(hat_delta_l, delta^BP_l)} (descriptive).
    No optimizer is stepped here."""
    outs, logits = writer_forward(model, ids, attn_mask, sg_blocks)               # 1
    loss = ce_loss(logits, labels)
    e = torch.autograd.grad(loss, logits, retain_graph=True)[0].detach()
    elig = labels != -100
    credit_mask = elig.unsqueeze(-1).to(outs[sg_blocks[0]].dtype)
    g_out, hat_delta, S = {}, {}, 0.0
    for l in sg_blocks:                                                            # 2
        g = preds[str(l)](outs[l].detach(), e)
        g_out[l] = g
        # credit only at eligible label positions: delta^BP is exactly zero elsewhere (pad positions
        # never reach the loss), so the applied credit has a BP counterpart everywhere it is nonzero
        hat_delta[l] = float(consts[str(l)]) * g.detach() * credit_mask
        S = S + (hat_delta[l] * outs[l]).sum()
    total = loss + S
    loss_T, logits_T, targets = true_hidden_errors(model, ids, attn_mask, labels, sg_blocks)   # 3, cached
    pred_loss, per_block = normalized_mse(g_out, targets, consts, elig, sg_blocks)
    zero_g = {l: torch.zeros_like(g_out[l]) for l in sg_blocks}
    baseline, baseline_per_block = normalized_mse(zero_g, targets, consts, elig, sg_blocks)
    return {"loss": loss, "e": e, "outs": outs, "logits": logits, "g_out": g_out, "hat_delta": hat_delta, "total": total,
            "targets": targets, "loss_T": loss_T, "logits_T": logits_T, "elig": elig,
            "pred_loss": pred_loss, "pred_loss_per_block": per_block,
            "baseline_mse": float(baseline.detach()), "baseline_mse_per_block": baseline_per_block,
            "align": alignment(hat_delta, targets, elig, sg_blocks)}


def run_sg_step(T, model_params, phi, model_opt, pred_opt, model_sched=None, pred_sched=None, steps_total=None, skip_pred_update=False):
    """Steps 4 to 9 of the frozen order on the terms T of sg_step_terms.
    Model gradients come from T['total'] alone (the CE reaches head / norm,
    the surrogate reaches the SG blocks); predictor gradients from
    T['pred_loss'] alone; the two are clipped separately (1.0 each) and
    stepped in the order model, predictor; schedulers last."""
    model_opt.zero_grad(set_to_none=True)
    T["total"].backward()                                                          # 4
    if not skip_pred_update:
        pred_opt.zero_grad(set_to_none=True)
        T["pred_loss"].backward()                                                  # 5
    gn_model = torch.nn.utils.clip_grad_norm_(model_params, 1.0)                   # 6
    gn_pred = torch.nn.utils.clip_grad_norm_(phi, 1.0) if not skip_pred_update else None
    model_opt.step()                                                               # 7
    if not skip_pred_update:
        pred_opt.step()                                                            # 8
    if model_sched is not None and (steps_total is None or model_sched.last_epoch < steps_total - 1):
        model_sched.step()                                                         # 9
    if pred_sched is not None and not skip_pred_update and (steps_total is None or pred_sched.last_epoch < steps_total - 1):
        pred_sched.step()
    return {"grad_norm_model": float(gn_model), "grad_norm_pred": (float(gn_pred) if gn_pred is not None else None)}
