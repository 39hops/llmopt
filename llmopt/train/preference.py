"""Preference-optimization losses: DPO, IPO, KTO, ORPO, SimPO, GRPO.

All are pure functions of sequence log-probabilities (summed over
completion tokens), so they slot onto any policy/ref forward pass.
Notation: pc/pr = policy logprob of chosen/rejected, rc/rr = reference.

Cheat sheet — what each changes about DPO:
- DPO: -log sigmoid(beta * ((pc-rc) - (pr-rr))); the implicit-reward
  margin, needs a frozen reference model.
- IPO: (margin - 1/(2 beta))^2 — regression to a fixed margin, doesn't
  push probabilities to extremes when data is noisy.
- KTO: unpaired — desirable and undesirable examples weighted against
  a running KL baseline instead of pairwise margins.
- ORPO: reference-free — CE on chosen plus a log-odds-ratio penalty.
- SimPO: reference-free margin on *length-normalized* logprobs with a
  target margin gamma.
- GRPO: no preference pairs at all — group-relative advantages
  (reward z-scored within a prompt group) weight a clipped PG loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(pc, pr, rc, rr, *, beta: float = 0.1):
    margin = beta * ((pc - rc) - (pr - rr))
    return -F.logsigmoid(margin).mean()


def ipo_loss(pc, pr, rc, rr, *, beta: float = 0.1):
    margin = (pc - rc) - (pr - rr)
    return ((margin - 1 / (2 * beta)) ** 2).mean()


def kto_loss(p, r, desirable, *, beta: float = 0.1, kl_baseline=0.0):
    """p, r: [n] policy/ref logprobs; desirable: [n] bool."""
    reward = beta * (p - r)
    z = kl_baseline
    good = 1 - torch.sigmoid(reward - z)
    bad = 1 - torch.sigmoid(z - reward)
    return torch.where(desirable, good, bad).mean()


def orpo_loss(pc, pr, chosen_ce, *, lam: float = 0.5):
    """chosen_ce: mean CE on chosen tokens (the SFT term). pc/pr are
    mean per-token logprobs."""
    odds = lambda lp: lp - torch.log1p(-torch.exp(lp).clamp(max=1 - 1e-6))
    ratio = -F.logsigmoid(odds(pc) - odds(pr))
    return (chosen_ce + lam * ratio).mean()


def simpo_loss(pc_norm, pr_norm, *, beta: float = 2.0, gamma: float = 1.0):
    """pc_norm/pr_norm: length-normalized (per-token mean) logprobs."""
    return -F.logsigmoid(beta * (pc_norm - pr_norm) - gamma).mean()


def grpo_advantages(rewards, group_ids):
    """Z-score rewards within each prompt group (GRPO's critic-free
    baseline). rewards: [n], group_ids: [n] int."""
    adv = torch.zeros_like(rewards)
    for g in group_ids.unique():
        m = group_ids == g
        r = rewards[m]
        adv[m] = (r - r.mean()) / (r.std() + 1e-8)
    return adv


def grpo_loss(logp_new, logp_old, advantages, *, clip: float = 0.2):
    """PPO-clip objective with group-relative advantages, per sequence."""
    ratio = (logp_new - logp_old).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantages
    return -torch.minimum(unclipped, clipped).mean()
