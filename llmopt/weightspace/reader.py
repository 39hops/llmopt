"""Weight-space reader: a from-scratch transformer that classifies the
function family of a subject MLP from its weights.

Tokenization respects the object's structure: ONE TOKEN PER NEURON —
the neuron's incoming weight row plus bias, zero-padded to a fixed
width — with a learned layer-index embedding added inside the model.
That grouping is deliberate: hidden-neuron permutation symmetry (the
thing the raw/canonical/augmented ablation measures) acts on whole
neurons, so it must act on whole tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llmopt.weightspace.subjects import (
    FAMILIES,
    HIDDEN,
    canonicalize,
    permute_hidden,
)

FEAT = HIDDEN + 1  # widest incoming row (16) + bias
N_TOKENS = 2 * HIDDEN + 1
_LAYER_IDS = torch.tensor([0] * HIDDEN + [1] * HIDDEN + [2])


def tokenize(weights: list[torch.Tensor]) -> torch.Tensor:
    """[33, 17]: rows = neurons of layer1 (16), layer2 (16), output (1);
    features = incoming weights zero-padded to 16, then bias."""
    w1, b1, w2, b2, w3, b3 = weights
    rows = []
    pad1 = torch.zeros(HIDDEN, HIDDEN - w1.shape[1])
    rows.append(torch.cat([w1, pad1, b1[:, None]], dim=1))
    rows.append(torch.cat([w2, b2[:, None]], dim=1))
    rows.append(torch.cat([w3, b3[:, None]], dim=1))
    return torch.cat(rows, dim=0)


class WeightReader(nn.Module):
    def __init__(self, n_families=len(FAMILIES), d_model=128, n_layers=4, n_heads=8):
        super().__init__()
        self.proj = nn.Linear(FEAT, d_model)
        self.layer_emb = nn.Embedding(3, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(d_model, n_families)

    def forward(self, tokens: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        h = self.proj(tokens) + self.layer_emb(layer_ids)
        return self.head(self.encoder(h).mean(dim=1))


def _batch(subjects, canonical: bool, augment: bool, gen=None) -> torch.Tensor:
    toks = []
    for s in subjects:
        w = s.weights
        if canonical:
            w = canonicalize(w)
        elif augment:
            w = permute_hidden(
                w,
                torch.randperm(HIDDEN, generator=gen),
                torch.randperm(HIDDEN, generator=gen),
            )
        toks.append(tokenize(w))
    return torch.stack(toks)


def train_reader(
    subjects, labels, *, epochs=30, augment=False, canonical=False,
    seed=0, batch_size=64, lr=3e-4,
) -> WeightReader:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = WeightReader()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    y_all = torch.tensor(labels)
    lids = _LAYER_IDS[None, :]
    for _ in range(epochs):
        order = torch.randperm(len(subjects), generator=gen)
        for i in range(0, len(subjects), batch_size):
            idx = order[i : i + batch_size]
            x = _batch([subjects[j] for j in idx], canonical, augment, gen)
            logits = model(x, lids.expand(len(idx), -1))
            loss = nn.functional.cross_entropy(logits, y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate_reader(model, subjects, labels, canonical=False) -> float:
    model.eval()
    x = _batch(subjects, canonical, augment=False)
    pred = model(x, _LAYER_IDS[None, :].expand(len(subjects), -1)).argmax(-1)
    return float((pred == torch.tensor(labels)).float().mean())
