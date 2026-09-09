"""WRITER-DFA-1 credit machinery (PRE-REG RESULTS L68321, sealed by
AMENDMENT -SEAL L68543 and AMENDMENT -PRECISION L68644). Shared by the
birth driver (scratch/birth19m_dfa.py), the writer-integrity smoke
(scratch/dfa_leakage_smoke.py) and the offline diagnostics
(scratch/dfa_align.py). Zero side effects at import.

Feedback matrices: B_l in R^{384 x 40}, entries U(-1, 1) from
torch.Generator("cpu").manual_seed(31_000_000 + l), scaled by
s / sqrt(40) (40 = the number of output classes), one per block, fixed.

forward_blocks(model, ids, mask, detach) reproduces MicroLM.forward
(llmopt/train/mathnative.py; no cache, no activation checkpoint) and
returns x_0, the eight block outputs x_1..x_8 and the logits. With
detach=True every block input after x_0 is detached and the head / norm
read x_8.detach(): no gradient crosses any block boundary and nothing
flows from the loss into block 7.

dfa_objective(...) builds e_t = dL/dlogits_t (detached), delta_l = B_l e_t
and the surrogate S = sum_l <delta_l, x_{l+1}>; total = L + S. Calling
total.backward() gives head.weight and norm.g the true gradient of L,
block l's parameters J_{f_l}(x_l)^T delta_l, and emb B_0 e + J_{f_0}^T B_0 e
(the identity path of block 0; x_0 stays attached to block 0 only).
"""
import hashlib
import math

import torch

D_MODEL = 384
N_OUT = 40
N_BLOCKS = 8
FEEDBACK_SEED_BASE = 31_000_000


def build_feedback(s, d=D_MODEL, n_out=N_OUT, n_blocks=N_BLOCKS, seed_base=FEEDBACK_SEED_BASE):
    """Eight fixed feedback matrices (CPU float32), U(-1, 1) * s / sqrt(n_out)."""
    out = []
    for l in range(n_blocks):
        g = torch.Generator("cpu").manual_seed(seed_base + l)
        u = torch.rand(d, n_out, generator=g, dtype=torch.float32) * 2 - 1
        out.append(u * (float(s) / math.sqrt(n_out)))
    return out


def feedback_digest(Bs):
    h = hashlib.sha256()
    for l, B in enumerate(Bs):
        t = B.detach().to("cpu", torch.float32).contiguous()
        h.update(f"B{l}:{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def causal_mask(ids, attn_mask):
    """The MicroLM.forward mask: causal AND key padding, bool (B, 1, T, T)."""
    T = ids.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
    return causal[None, None] & attn_mask[:, None, None, :].bool()


def forward_blocks(model, ids, attn_mask, detach):
    m = causal_mask(ids, attn_mask)
    x = model.emb(ids)
    x0 = x
    outs = []
    for l, b in enumerate(model.blocks):
        xin = x if l == 0 else (x.detach() if detach else x)
        x, _ = b(xin, m, None)
        outs.append(x)
    xf = x.detach() if detach else x
    logits = model.head(model.norm(xf))
    return x0, outs, logits


def ce_loss(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def dfa_objective(model, Bs, ids, attn_mask, labels):
    """Returns dict(loss, e, deltas, outs, logits, x0, total). Only
    total.backward() (or autograd.grad on total) is meant to update
    parameters; the true gradient of L reaches head.weight and norm.g only."""
    x0, outs, logits = forward_blocks(model, ids, attn_mask, detach=True)
    loss = ce_loss(logits, labels)
    e = torch.autograd.grad(loss, logits, retain_graph=True)[0].detach()
    deltas = [e @ B.t() for B in Bs]               # (batch, T, 384) = B_l e_t per token
    S = sum((dl * x).sum() for dl, x in zip(deltas, outs))
    return {"loss": loss, "e": e, "deltas": deltas, "outs": outs, "logits": logits, "x0": x0, "total": loss + S}


def bp_hidden_errors(model, ids, attn_mask, labels):
    """Read-only diagnostic: the true backprop hidden error dL/dx_{l+1} at
    every block output (attached forward). Never used to update any arm."""
    _, outs, logits = forward_blocks(model, ids, attn_mask, detach=False)
    loss = ce_loss(logits, labels)
    grads = torch.autograd.grad(loss, outs)
    return loss.detach(), [g.detach() for g in grads]


def block_params(model, l):
    return [p for n, p in model.named_parameters() if n.startswith(f"blocks.{l}.")]
