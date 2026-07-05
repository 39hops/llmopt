"""Batch precompute and disk-cache reference (teacher) logprobs.

Used as the fp16 baseline for per-layer quant sensitivity (delta-KL needs
full-vocab reference distributions) and for perplexity eval. Works for any
number of sequences -- HumanEval's 164 prompts, WikiText calibration sets, etc.

Storage: one .npz per run under cache_dir, keyed by a content hash of
(model name, token ids, top_k). Full-vocab logprobs for 164 x 1k tokens x
150k vocab is ~100 GB in fp32, so by default we store top-k logprobs +
indices plus the tail mass, which is enough for a tight KL estimate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RefLogprobs:
    """Top-k reference logprobs for one sequence.

    topk_logprobs: [T, k] float32, log p of the k most likely tokens per position
    topk_indices:  [T, k] int32, vocab ids for those entries
    tail_logmass:  [T] float32, log of remaining probability mass (for KL bound)
    target_logprob:[T] float32, log p of the actual next token (for perplexity)
    """

    topk_logprobs: np.ndarray
    topk_indices: np.ndarray
    tail_logmass: np.ndarray
    target_logprob: np.ndarray

    @property
    def perplexity(self) -> float:
        return float(np.exp(-self.target_logprob.mean()))


def _cache_key(model_name: str, token_ids: Sequence[Sequence[int]], top_k: int) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(str(top_k).encode())
    for seq in token_ids:
        h.update(np.asarray(seq, dtype=np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()[:16]


def precompute_ref_logprobs(
    model,
    token_ids: Sequence[Sequence[int]],
    *,
    model_name: str = "",
    top_k: int = 128,
    batch_size: int = 8,
    cache_dir: str | Path | None = None,
    device: str | None = None,
) -> list[RefLogprobs]:
    """Run the reference model over all sequences, return per-sequence RefLogprobs.

    model: HF causal LM (or None if cache hit is guaranteed).
    token_ids: list of token-id lists (ragged fine; batches are length-padded).
    Results are cached to cache_dir keyed by (model_name, inputs, top_k);
    a second call with identical inputs is a pure disk read.
    """
    import torch

    key = _cache_key(model_name, token_ids, top_k)
    cache_path = Path(cache_dir) / f"ref_{key}.npz" if cache_dir else None
    if cache_path and cache_path.exists():
        return _load(cache_path, n_seqs=len(token_ids))

    if model is None:
        raise ValueError(f"cache miss for key {key} and no model provided")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    results: list[RefLogprobs] = []
    with torch.inference_mode():
        for start in range(0, len(token_ids), batch_size):
            chunk = [list(s) for s in token_ids[start : start + batch_size]]
            lens = [len(s) for s in chunk]
            max_len = max(lens)
            input_ids = torch.zeros(len(chunk), max_len, dtype=torch.long)
            attn = torch.zeros(len(chunk), max_len, dtype=torch.long)
            for i, seq in enumerate(chunk):
                input_ids[i, : len(seq)] = torch.tensor(seq)
                attn[i, : len(seq)] = 1
            logits = model(
                input_ids=input_ids.to(device), attention_mask=attn.to(device)
            ).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)

            for i, seq_len in enumerate(lens):
                # position t predicts token t+1; last position predicts nothing
                lp = logprobs[i, : seq_len - 1]  # [T-1, V]
                targets = input_ids[i, 1:seq_len].to(device)
                target_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                tk_lp, tk_idx = lp.topk(top_k, dim=-1)
                tail = torch.log1p(-tk_lp.exp().sum(-1).clamp(max=1 - 1e-7))
                results.append(
                    RefLogprobs(
                        topk_logprobs=tk_lp.cpu().numpy().astype(np.float32),
                        topk_indices=tk_idx.cpu().numpy().astype(np.int32),
                        tail_logmass=tail.cpu().numpy().astype(np.float32),
                        target_logprob=target_lp.cpu().numpy().astype(np.float32),
                    )
                )

    if cache_path:
        _save(cache_path, results, meta={"model": model_name, "top_k": top_k})
    return results


def kl_vs_ref(ref: RefLogprobs, new_logprobs) -> float:
    """Mean per-token KL(ref || new) estimated over ref's top-k support.

    new_logprobs: [T, V] torch tensor or ndarray of log-probs from the
    modified (e.g. quantized) model, positions aligned with ref.
    Tail mass is treated as zero-KL (upper-bounded contribution ignored),
    which is tight when top_k covers ~99.9% of mass.
    """
    import torch

    if not isinstance(new_logprobs, torch.Tensor):
        new_logprobs = torch.from_numpy(np.asarray(new_logprobs))
    ref_lp = torch.from_numpy(ref.topk_logprobs).to(new_logprobs.device)
    idx = torch.from_numpy(ref.topk_indices.astype(np.int64)).to(new_logprobs.device)
    new_at_idx = new_logprobs.gather(-1, idx)
    kl_per_pos = (ref_lp.exp() * (ref_lp - new_at_idx)).sum(-1)
    return float(kl_per_pos.mean())


def _save(path: Path, results: list[RefLogprobs], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for i, r in enumerate(results):
        arrays[f"lp_{i}"] = r.topk_logprobs
        arrays[f"idx_{i}"] = r.topk_indices
        arrays[f"tail_{i}"] = r.tail_logmass
        arrays[f"tgt_{i}"] = r.target_logprob
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def _load(path: Path, n_seqs: int) -> list[RefLogprobs]:
    data = np.load(path)
    return [
        RefLogprobs(
            topk_logprobs=data[f"lp_{i}"],
            topk_indices=data[f"idx_{i}"],
            tail_logmass=data[f"tail_{i}"],
            target_logprob=data[f"tgt_{i}"],
        )
        for i in range(n_seqs)
    ]
