"""Gist tokens: learned prompt compression (Mu et al. 2023).

Instead of dropping tokens (compression.py), teach the model to *summarize
a prefix into k slots*: append k gist tokens after the instruction and
train with a mask where everything after the gists cannot see the
instruction — only the gists. The model is forced to route all
instruction information through k KV entries. At inference the
instruction's KV rows are discarded: a prefix of any length costs k cache
slots.

Pieces (mechanics + a minimal trainer; the paper's scale is a fine-tune
away, the mask is the whole trick):

- add_gist_tokens: grow tokenizer/embeddings with <GIST_i> specials.
- gist_attention_mask: causal mask AND (queries after the gist span
  cannot attend keys before it). Gists themselves still read the
  instruction — they're the bottleneck, not a wall.
- compress_to_gist_kv: run [instruction + gists] once, slice the cache
  down to the gist rows. Continuation attends k slots instead of len(inst).
- gist_training_step: LM loss on the answer with the gist mask applied —
  gradient teaches gist embeddings (and the model) to carry the prefix.
"""

from __future__ import annotations

from typing import Sequence


def add_gist_tokens(model, tokenizer, k: int = 1) -> list[int]:
    """Add <GIST_0>..<GIST_k-1> specials, resize embeddings. Returns ids."""
    names = [f"<GIST_{i}>" for i in range(k)]
    tokenizer.add_special_tokens({"additional_special_tokens": names})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(names)


def gist_attention_mask(
    seq_len: int, gist_start: int, gist_end: int, device=None
):
    """Bool [seq, seq] mask (True = may attend): causal, and queries at
    positions >= gist_end are blinded to keys < gist_start."""
    import torch

    allow = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    allow[gist_end:, :gist_start] = False
    return allow


def to_additive(mask, dtype):
    """Bool mask -> additive float mask [1, 1, seq, seq] for HF attention."""
    import torch

    out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    out.masked_fill_(~mask, torch.finfo(dtype).min)
    return out[None, None]


def compress_to_gist_kv(model, instruction_ids: Sequence[int], gist_ids: Sequence[int]):
    """Run [instruction + gists] once and keep only the gist rows of the KV
    cache. Returns (cache, gist_len): a prefix of len(instruction) tokens
    compressed into len(gist_ids) cache slots.

    Continuation must feed position_ids starting at gist_end (positions
    are preserved, rows are just dropped) and an attention_mask of
    gist_len ones.
    """
    import torch
    from transformers import DynamicCache

    ids = list(instruction_ids) + list(gist_ids)
    g0, g1 = len(instruction_ids), len(ids)
    mask = gist_attention_mask(g1, g0, g1, device=model.device)
    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([ids], device=model.device),
            attention_mask=to_additive(mask, model.dtype),
            use_cache=True,
        )
    cache = DynamicCache()
    for i, layer in enumerate(out.past_key_values.layers):
        cache.update(layer.keys[:, :, g0:g1].clone(), layer.values[:, :, g0:g1].clone(), i)
    return cache, g1 - g0


def gist_training_step(
    model, instruction_ids, gist_ids, answer_ids
) -> "torch.Tensor":
    """One LM step on [instruction, gists, answer] with the gist mask:
    loss only on answer tokens, which can see gists but not instruction.
    Caller owns optimizer/backward. Returns the loss tensor."""
    import torch

    ids = list(instruction_ids) + list(gist_ids) + list(answer_ids)
    g0 = len(instruction_ids)
    g1 = g0 + len(gist_ids)
    mask = gist_attention_mask(len(ids), g0, g1, device=model.device)
    labels = torch.full((1, len(ids)), -100, device=model.device)
    labels[0, g1:] = torch.tensor(list(answer_ids), device=model.device)
    out = model(
        input_ids=torch.tensor([ids], device=model.device),
        attention_mask=to_additive(mask, model.dtype),
        labels=labels,
    )
    return out.loss
