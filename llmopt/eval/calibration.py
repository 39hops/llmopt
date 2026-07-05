"""Calibration: does confidence mean what it says?

ECE (expected calibration error): bucket predictions by confidence,
compare each bucket's mean confidence against its accuracy, average the
gaps weighted by bucket mass. 0 = perfectly calibrated. Relevant here
because quantization/eviction/distillation can shift calibration even
when top-1 accuracy survives.
"""

from __future__ import annotations


def ece(confidences, correct, *, n_bins: int = 10) -> float:
    """confidences: [n] max-prob per prediction; correct: [n] bool."""
    import torch

    conf = torch.as_tensor(confidences, dtype=torch.float32)
    ok = torch.as_tensor(correct, dtype=torch.float32)
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            gap = float((conf[m].mean() - ok[m].mean()).abs())
            total += float(m.float().mean()) * gap
    return total


def model_ece(model, token_seqs, *, n_bins: int = 10) -> float:
    """Next-token ECE over sequences: confidence = max softmax prob,
    correct = argmax matches the actual next token."""
    import torch

    confs, oks = [], []
    with torch.inference_mode():
        for seq in token_seqs:
            ids = torch.tensor([list(seq)], device=model.device)
            probs = torch.softmax(model(input_ids=ids).logits[0, :-1], -1)
            conf, pred = probs.max(-1)
            confs.append(conf)
            oks.append(pred == ids[0, 1:])
    return ece(torch.cat(confs), torch.cat(oks), n_bins=n_bins)
