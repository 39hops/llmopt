"""SYNTHETIC-GRADIENT-WRITER-1 credit law (PRE-REG SYNTHETIC-GRADIENT-
WRITER-1 L69765, AMENDMENT -ARENA L70126, OBSERVATION SG-PREDICTOR-AUDIT-0
constants). The per-block synthetic hidden error

    hat_delta_l = s_l * G_phi_l(stopgrad(x_{l+1}), stopgrad(e_t))

replaces the true hidden error delta^BP_l = dL/dx_{l+1} for every SG block
l in SG_BLOCKS; block parameters receive J_{f_l}^T hat_delta_l only (the
surrogate sum_l <hat_delta_l.detach(), x_{l+1}> backpropagated through a
block whose input is detached); head.weight and norm.g take the true CE
gradient with x_8 detached (the sealed DFA / frontier treatment); blocks
below the first SG block are frozen at W_0 (the arena) or, in full-stack
mode, are SG blocks themselves with emb receiving hat_delta_0 +
J_{f_0}^T hat_delta_0 (the identity path).

Teacher pass (integrity fold 5, preferred implementation): the true targets
delta^BP_l come from a forward whose MODEL PARAMETER VIEWS ARE DETACHED
CONSTANTS (torch.func.functional_call on {name: p.detach()}) while
activation differentiation stays enabled; the true loss graph therefore
cannot reach any block parameter by construction. The teacher forward is
computed at the same pre-update state as the writer forward and reproduces
its logits (asserted in the integrity smoke).

Predictor regression: L_phi = sum_l mean_{tokens with mask=1} ||G_phi_l(.)
- delta^BP_l / s_l||^2, gradients reach phi only (both predictor inputs are
detached). Ordering per step: (1) writer forward, (2) hat_delta produced
from the CURRENT predictor state, (3) model surrogate backward + model
step, (4) teacher targets, predictor loss backward, predictor step. Skipping
(4) leaves the model gradients bit-identical (smoked).

Predictor families: LINEAR G(h, e) = h A + e B + C (Czarnecki's SG(h, y);
Jaderberg's best cDNI family) and MLP-256 (one hidden layer, ReLU); the
OUTPUT layer is zero-initialised in both (Jaderberg), so the writer starts
as 'no credit' to the SG blocks.
"""
import torch
import torch.nn as nn
from torch.func import functional_call

from dfa_credit import causal_mask, ce_loss

D_MODEL, N_OUT, N_BLOCKS = 384, 40, 8


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
    """The writer-state forward. Blocks below min(sg_blocks) run attached
    only if they are SG blocks (full-stack mode); in the arena they are
    frozen (requires_grad False) and their outputs carry no graph to any
    trainable tensor. Every SG block sees a DETACHED input; x_8 is detached
    before norm/head. Returns (outs, logits): outs[l] = x_{l+1}."""
    m = causal_mask(ids, attn_mask)
    x = model.emb(ids)
    outs = []
    first_sg = min(sg_blocks)
    for l, b in enumerate(model.blocks):
        if l == 0 and l in sg_blocks:
            xin = x                      # emb takes the identity path of hat_delta_0
        elif l >= first_sg:
            xin = x.detach()             # SG block input (or the frozen -> SG boundary)
        else:
            xin = x                      # frozen lower stack (no trainable tensor upstream)
        x, _ = b(xin, m, None)
        outs.append(x)
    logits = model.head(model.norm(x.detach()))
    return outs, logits


def teacher_targets(model, ids, attn_mask, sg_blocks=None):
    """True hidden errors delta^BP_l = dL/dx_{l+1} for l in sg_blocks from a
    forward over DETACHED PARAMETER CONSTANTS (functional_call); the loss
    graph contains activations only. Returns (loss_T, logits_T, {l: delta})."""
    params = {n: p.detach() for n, p in model.named_parameters()}
    m = causal_mask(ids, attn_mask)
    x = functional_call(model.emb, {k[len("emb."):]: v for k, v in params.items() if k.startswith("emb.")}, (ids,))
    x = x.requires_grad_(True)   # x_0 is a leaf ACTIVATION: the graph below holds activations only
    outs = []
    for l, b in enumerate(model.blocks):
        pb = {k[len(f"blocks.{l}."):]: v for k, v in params.items() if k.startswith(f"blocks.{l}.")}
        x, _ = functional_call(b, pb, (x, m, None))
        outs.append(x)
    xn = functional_call(model.norm, {k[len("norm."):]: v for k, v in params.items() if k.startswith("norm.")}, (x,))
    logits = functional_call(model.head, {k[len("head."):]: v for k, v in params.items() if k.startswith("head.")}, (xn,))
    return logits, outs


def sg_step_terms(model, preds, ids, attn_mask, labels, sg_blocks, consts):
    """Everything one step needs, in the registered order, WITHOUT stepping
    any optimizer. Returns a dict:
      loss        : the writer CE (true gradient reaches head / norm only)
      e           : dL/dlogits, detached
      g_out       : {l: G_phi_l(h, e)} attached to phi (for the predictor loss)
      hat_delta   : {l: s_l * g_out[l].detach()} (the credit applied)
      total       : loss + sum_l <hat_delta_l, x_{l+1}>  (model backward object)
      teacher()   : closure computing (loss_T, logits_T, {l: delta^BP_l}) at
                    the same pre-update state via the parameter-detached pass
      pred_loss(targets): closure giving the masked normalized regression loss
    """
    outs, logits = writer_forward(model, ids, attn_mask, sg_blocks)
    loss = ce_loss(logits, labels)
    e = torch.autograd.grad(loss, logits, retain_graph=True)[0].detach()
    g_out, hat_delta, S = {}, {}, 0.0
    for l in sg_blocks:
        g = preds[str(l)](outs[l].detach(), e)
        g_out[l] = g
        hat_delta[l] = float(consts[str(l)]) * g.detach()
        S = S + (hat_delta[l] * outs[l]).sum()
    total = loss + S
    valid = attn_mask.bool()

    def teacher():
        logits_T, outs_T = teacher_targets(model, ids, attn_mask, sg_blocks)
        loss_T = ce_loss(logits_T, labels)
        grads = torch.autograd.grad(loss_T, [outs_T[l] for l in sg_blocks])
        return loss_T.detach(), logits_T.detach(), {l: g.detach() for l, g in zip(sg_blocks, grads)}

    def pred_loss(targets):
        tot = 0.0
        for l in sg_blocks:
            diff = g_out[l] - targets[l] / float(consts[str(l)])
            tot = tot + (diff.pow(2).sum(-1) * valid).sum() / valid.sum()
        return tot

    return {"loss": loss, "e": e, "outs": outs, "logits": logits, "g_out": g_out, "hat_delta": hat_delta, "total": total,
            "teacher": teacher, "pred_loss": pred_loss}
