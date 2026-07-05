"""KV-cache helpers for draft/verify decode loops.

Supports both the modern HF `Cache` API (DynamicCache: get_seq_length /
crop) and the legacy tuple-of-(k, v) format with shapes [B, H, T, D].
`crop` drops rejected draft positions after a verify pass; `valid_len`
reports cached sequence length.
"""

from __future__ import annotations


def to_legacy(past):
    """Pass-through normalizer. Modern Cache objects are kept as-is (their
    own crop is used); only exotic containers get converted."""
    return past


def crop(past, length: int):
    """Keep only the first `length` positions of every layer's K/V."""
    if past is None:
        return None
    if hasattr(past, "crop"):  # transformers Cache API (in-place)
        past.crop(length)
        return past
    return tuple((k[:, :, :length], v[:, :, :length]) for k, v in past)


def valid_len(past) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return past[0][0].shape[2]
