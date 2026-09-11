"""SG-BOUNDARY-BLOCK7-1 credit law (PRE-REG SG-BOUNDARY-BLOCK7-1): a LOCAL
boundary writer for block 7 only, over the frozen-random-backbone arena.

Arena: emb + blocks 0..3 frozen bit-exact at W_0 (dfa_credit.freeze_lower
(model, 4)); blocks 4..6 by EXACT backprop; block 7 by synthetic credit
only; norm + head by the exact CE gradient. Motivated by the block-7
exception of VERDICT SG-CROSSPOS-REPRESENTABILITY-0 (block 7's true hidden
error is a function of its local input (x_8, e) on the trained control:
HELDOUT ratio 0.10 to 0.12 in both arms); the top-four SG family stays
CLOSED. Nothing here reverses that verdict.

The SG7 arm's one true-CE graph (sg7_forward):
    x_4 = blocks 0..3 (frozen, no graph) of emb(ids)
    x_7 = blocks 4..6 (real parameters, ATTACHED input)      exact BP
    x_8^T = block 7 with DETACHED PARAMETER VIEWS on the ATTACHED x_7
            (torch.func.functional_call over {name: p.detach()}): the true
            block-7 Jacobian carries the exact CE credit into blocks 4..6,
            and the true CE cannot reach any block-7 parameter
    logits = head(norm(x_8^T))                                 exact CE
and a separate LOCAL block-7 forward with REAL parameters on the DETACHED
input, x_8^L = block_7(x_7.detach()), which supplies the synthetic-credit
surrogate S = <hat_delta_7, x_8^L>: S.backward() gives block 7 exactly
J_{f_7}^T hat_delta_7 and nothing else. total = L + S is the one model
backward object; L reaches blocks 4..6 / norm / head only, S reaches
block 7 only.

Synthetic credit (the SG-1 law restricted to block 7):
    hat_delta_7 = s_7 * G_phi(stopgrad(x_8), stopgrad(e)) * 1[labels != -100]
with s_7 the sealed arena constant of block 7 (logs/sgaudit0/audit.json,
arena_frozen_4_7["7"]), e = dL/dlogits, and G_phi the LOCAL predictor of
the cross-position desk (scratch/sg_crosspos_desk.SeqSG under the j == i
mask: a per-token network, in_proj 424 -> 128, two pre-LN blocks whose
self-attention reads the token itself only, FFN 128 -> 512 -> 128, final
LN, zero-initialised out_proj 128 -> 384; 500,736 parameters); the
classes are verbatim copies guarded by a source-identity test. Input
Z = [x_8, e] standardized per feature by FIXED constants (mu, sd) computed
once at W_0 on the frozen probe's FIT chunks (input_stats; recorded in the
receipt) with NO clip: the desk's +-5 clip is dropped because the arena's
x_8 scale grows about 7x over training (seed-27 control on the same FIT
tokens, logs/sgbb7/x8_scale_census.json: per-feature sd median 1.02 ->
7.28, max 1.85 -> 19.5) and a W_0-frozen clip would saturate the late
states.

FOLD A (frozen step order, every gradient from the pre-update pair):
  1 forward (sg7_forward); 2 hat_delta_7 from phi_t; 3 parameter-detached
  teacher pass at W_t, delta^BP_7 CACHED (sg_credit.true_hidden_errors);
  4 model backward of total; 5 predictor backward of L_phi against the
  cached target; 6 separate clips; 7 model step; 8 predictor step; 9
  schedulers.  sg7_step_terms performs 1 to 3; sg_credit.run_sg_step
  performs 4 to 9 unchanged.
FOLD B: L_phi = mean over eligible positions and hidden dims of
  (G - delta^BP_7 / s_7)^2 (sg_credit.normalized_mse on the one block).

Mechanical endpoint (sg7_integrity_smoke check f): with hat_delta_7 :=
delta^BP_7 the trainable gradients and one clipped AdamW update reproduce
the paired frozen-BP control (dfa_credit.hybrid_objective(k=8) over
freeze_lower(model, 4)) under the registered exact / tolerance law.
"""
import math

import torch
import torch.nn as nn
from torch.func import functional_call

from dfa_credit import causal_mask, ce_loss
from sg_credit import alignment, normalized_mse, teacher_targets, true_hidden_errors

SG7_BLOCK = 7
D_IN, D_OUT = 384 + 40, 384


# ---- verbatim copies of scratch/sg_crosspos_desk.py (source-identity test; the ONE delta is
# `device=x.device` on the two torch.arange calls of Attn.forward, so the predictor runs on mps) ----

def alibi_slopes(n_heads):
    return torch.tensor([2.0 ** (-8.0 * (h + 1) / n_heads) for h in range(n_heads)])


def allowed_mask(arm, key_real):
    """[B, T, T] bool: query i may read key j."""
    B, T = key_real.shape
    i = torch.arange(T).view(T, 1)
    j = torch.arange(T).view(1, T)
    if arm == "reverse_causal":
        m = (j >= i).unsqueeze(0) & key_real.unsqueeze(1)
        m = m | torch.eye(T, dtype=torch.bool).unsqueeze(0)     # a pad query keeps itself (no empty softmax row)
    elif arm == "local":
        m = torch.eye(T, dtype=torch.bool).unsqueeze(0).expand(B, T, T)
    else:
        raise ValueError(arm)
    return m


class Attn(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)

    def forward(self, x, allowed):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)
        s = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dh)
        dist = (torch.arange(T, device=x.device).view(1, T) - torch.arange(T, device=x.device).view(T, 1)).abs().to(x.dtype)   # |j - i|
        s = s - self.slopes.view(1, self.h, 1, 1) * dist.view(1, 1, T, T)
        s = s.masked_fill(~allowed.unsqueeze(1), float("-inf"))
        a = torch.softmax(s, -1)
        return self.o(torch.einsum("bhij,bjhd->bihd", a, v).reshape(B, T, d))


class SeqSG(nn.Module):
    """Fixed-capacity sequence predictor: in_proj -> N_LAYERS x (pre-LN attn +
    pre-LN FFN) -> zero-initialised out_proj."""

    def __init__(self, d_in=D_IN, d=128, n_layers=2, n_heads=4, d_ffn=512, d_out=D_OUT):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.att = nn.ModuleList([Attn(d, n_heads) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, d_ffn), nn.GELU(), nn.Linear(d_ffn, d)) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.out = nn.Linear(d, d_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, allowed):
        x = self.inp(z)
        for ln1, att, ln2, ffn in zip(self.ln1, self.att, self.ln2, self.ffn):
            x = x + att(ln1(x), allowed)
            x = x + ffn(ln2(x))
        return self.out(self.lnf(x))

# ---- end verbatim copies ----


class LocalSG(nn.Module):
    """The desk's LOCAL arm as an online predictor: G(h, e) = SeqSG(Z, j == i)
    with Z = ([h, e] - mu) / sd, (mu, sd) fixed buffers (part of the state
    dict, so a snapshot carries them)."""

    def __init__(self, mu, sd):
        super().__init__()
        assert mu.shape == (1, D_IN) and sd.shape == (1, D_IN)
        self.register_buffer("mu", mu.clone().float())
        self.register_buffer("sd", sd.clone().float())
        self.net = SeqSG()

    def forward(self, h, e):
        z = (torch.cat([h, e], -1) - self.mu) / self.sd
        B, T, _ = z.shape
        return self.net(z, allowed_mask("local", torch.ones(B, T, dtype=torch.bool)).to(z.device))


def build_local_predictor(mu, sd, seed):
    torch.manual_seed(seed)
    return LocalSG(mu, sd)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def input_stats(model, ids_list, mask_list, labels_list):
    """(mu, sd) per feature of [x_8, e] over the eligible tokens of the given
    chunks at the CURRENT W (called once at W_0, before any step; the
    parameter-detached teacher pass, so no parameter is touched).
    Deterministic."""
    Z = []
    for ids, mask, labels in zip(ids_list, mask_list, labels_list):
        lg, outs = teacher_targets(model, ids, mask)
        e = torch.autograd.grad(ce_loss(lg, labels), lg)[0]
        el = labels != -100
        Z.append(torch.cat([outs[SG7_BLOCK].detach()[el], e.detach()[el]], -1))
    Z = torch.cat(Z).float()
    return Z.mean(0, keepdim=True), Z.std(0, keepdim=True).clamp_min(1e-6)


def sg7_forward(model, ids, attn_mask):
    """Returns (outs, x7, x8_T, x8_L, logits). outs[l] = x_{l+1} for l < 7
    (blocks 0..3 carry no graph under freeze_lower; 4..6 attached)."""
    m = causal_mask(ids, attn_mask)
    x = model.emb(ids)
    outs = []
    for l in range(SG7_BLOCK):
        x, _ = model.blocks[l](x, m, None)
        outs.append(x)
    x7 = x
    p7 = {n: p.detach() for n, p in model.blocks[SG7_BLOCK].named_parameters()}
    x8_T, _ = functional_call(model.blocks[SG7_BLOCK], p7, (x7, m, None))    # true Jacobian, constant parameters
    x8_L, _ = model.blocks[SG7_BLOCK](x7.detach(), m, None)                    # real parameters, detached input
    logits = model.head(model.norm(x8_T))
    return outs, x7, x8_T, x8_L, logits


def sg7_step_terms(model, pred, ids, attn_mask, labels, const7):
    """Steps 1 to 3 at the pre-update pair (W_t, phi_t). Returns the dict
    run_sg_step consumes ('total', 'pred_loss') plus the SG-1 receipt terms
    keyed by block 7."""
    outs, x7, x8_T, x8_L, logits = sg7_forward(model, ids, attn_mask)            # 1
    loss = ce_loss(logits, labels)
    e = torch.autograd.grad(loss, logits, retain_graph=True)[0].detach()
    elig = labels != -100
    credit_mask = elig.unsqueeze(-1).to(x8_L.dtype)
    g = pred(x8_T.detach(), e)                                                     # 2
    hat = float(const7) * g.detach() * credit_mask
    S = (hat * x8_L).sum()
    total = loss + S
    loss_T, logits_T, targets = true_hidden_errors(model, ids, attn_mask, labels, [SG7_BLOCK])   # 3, cached
    consts = {str(SG7_BLOCK): float(const7)}
    pred_loss, per_block = normalized_mse({SG7_BLOCK: g}, targets, consts, elig, [SG7_BLOCK])
    baseline, baseline_per_block = normalized_mse({SG7_BLOCK: torch.zeros_like(g)}, targets, consts, elig, [SG7_BLOCK])
    return {"loss": loss, "e": e, "outs": outs, "x7": x7, "x8_T": x8_T, "x8_L": x8_L, "logits": logits,
            "g_out": {SG7_BLOCK: g}, "hat_delta": {SG7_BLOCK: hat}, "total": total,
            "targets": targets, "loss_T": loss_T, "logits_T": logits_T, "elig": elig,
            "pred_loss": pred_loss, "pred_loss_per_block": per_block,
            "baseline_mse": float(baseline.detach()), "baseline_mse_per_block": baseline_per_block,
            "align": alignment({SG7_BLOCK: hat}, targets, elig, [SG7_BLOCK])}


def forced_total(model, ids, attn_mask, labels):
    """The forced-delta^BP endpoint objective: total with hat_delta_7 :=
    delta^BP_7 (teacher target). Its backward must reproduce the frozen-BP
    control's trainable gradients."""
    _, _, targets = true_hidden_errors(model, ids, attn_mask, labels, [SG7_BLOCK])
    _, _, _, x8_L, logits = sg7_forward(model, ids, attn_mask)
    return ce_loss(logits, labels) + (targets[SG7_BLOCK] * x8_L).sum()
