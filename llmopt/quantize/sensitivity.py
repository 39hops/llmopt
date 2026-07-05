"""Per-layer quantization sensitivity: delta-KL vs fp16 reference per layer
per bit-width, via fake-quant (simulate int levels in fp16, swap back after).

Workflow:
  1. precompute_ref_logprobs(...) once on the fp16 model  (ref_logprobs.py)
  2. measure_sensitivity(...) fake-quants one weight matrix at a time,
     re-runs eval set, records mean delta-KL
  3. feed the table to allocator.allocate_bits(...)
"""

from __future__ import annotations

from dataclasses import dataclass

from llmopt.train.ref_logprobs import RefLogprobs, kl_vs_ref


@dataclass(frozen=True)
class LayerSensitivity:
    name: str  # parameter name, e.g. "model.layers.7.mlp.down_proj.weight"
    bits: int
    delta_kl: float  # mean per-token KL(ref || quantized) increase
    n_params: int


def fake_quantize_(weight, bits: int, group_size: int = 64):
    """In-place symmetric per-group round-to-nearest fake quant.

    Returns the original tensor (clone) so the caller can restore.
    """
    import torch

    orig = weight.detach().clone()
    w = weight.detach()
    out_f, in_f = w.shape
    pad = (-in_f) % group_size
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    g = w.view(out_f, -1, group_size)
    qmax = 2 ** (bits - 1) - 1
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (g / scale).round().clamp(-qmax - 1, qmax) * scale
    q = q.view(out_f, -1)[:, :in_f]
    weight.detach().copy_(q)
    return orig


def measure_sensitivity(
    model,
    token_ids,
    refs: list[RefLogprobs],
    *,
    bit_widths: tuple[int, ...] = (2, 4),
    layer_filter=None,
    group_size: int = 64,
    progress: bool = True,
) -> list[LayerSensitivity]:
    """For each 2-D weight (optionally filtered by name), for each bit width:
    fake-quant just that weight, compute mean delta-KL over the eval set
    against precomputed refs, restore the weight. O(layers * bits) evals.

    layer_filter: callable(name) -> bool; default targets transformer
    projection matrices only (skips embeddings and lm_head).
    """
    import torch

    if layer_filter is None:
        def layer_filter(name: str) -> bool:
            return ".layers." in name and name.endswith(".weight")

    targets = [
        (name, p)
        for name, p in model.named_parameters()
        if p.dim() == 2 and layer_filter(name)
    ]
    results: list[LayerSensitivity] = []
    device = next(model.parameters()).device

    with torch.inference_mode():
        for li, (name, param) in enumerate(targets):
            for bits in bit_widths:
                orig = fake_quantize_(param, bits, group_size=group_size)
                kls = []
                for seq, ref in zip(token_ids, refs):
                    ids = torch.tensor([list(seq)], device=device)
                    logits = model(input_ids=ids).logits[0, :-1].float()
                    lp = torch.log_softmax(logits, dim=-1)
                    kls.append(kl_vs_ref(ref, lp))
                param.detach().copy_(orig)
                results.append(
                    LayerSensitivity(
                        name=name,
                        bits=bits,
                        delta_kl=sum(kls) / len(kls),
                        n_params=param.numel(),
                    )
                )
            if progress:
                print(f"[{li + 1}/{len(targets)}] {name} done")
    return results
