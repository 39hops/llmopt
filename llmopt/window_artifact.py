"""Validation and decoding for committed gravmoe training windows."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


def load_contiguous_windows(
    windows_path: Path,
    contract_path: Path,
    sequence_length: int,
) -> list[list[int]]:
    """Load ``tok[T] ++ tgt[T]`` records as contiguous ``T+1`` rows."""
    raw = windows_path.read_bytes()
    contract = json.loads(contract_path.read_text())

    observed_sha = hashlib.sha256(raw).hexdigest()
    expected_sha = contract["windows_sha"]
    if observed_sha != expected_sha:
        raise ValueError(
            "window artifact SHA mismatch: "
            f"expected {expected_sha}, got {observed_sha}"
        )

    record_bytes = 2 * sequence_length * 8
    if not raw or len(raw) % record_bytes:
        raise ValueError(
            "window artifact length must be a non-empty multiple of "
            f"{record_bytes} bytes"
        )

    values = [value for (value,) in struct.iter_unpack("<q", raw)]
    record_values = 2 * sequence_length
    rows = []
    for offset in range(0, len(values), record_values):
        tok = values[offset:offset + sequence_length]
        tgt = values[offset + sequence_length:offset + record_values]
        if tok[1:] != tgt[:-1]:
            raise ValueError(
                "windows must be contiguous next-token slices: "
                "expected tok[1:] == tgt[:-1]"
            )
        rows.append(tok + [tgt[-1]])

    expected_rows_sha = contract.get("windows_rows_sha")
    if expected_rows_sha is not None:
        rows_raw = b"".join(
            struct.pack(f"<{len(row)}q", *row) for row in rows
        )
        observed_rows_sha = hashlib.sha256(rows_raw).hexdigest()
        if observed_rows_sha != expected_rows_sha:
            raise ValueError(
                "reconstructed window rows SHA mismatch: "
                f"expected {expected_rows_sha}, got {observed_rows_sha}"
            )

    return rows
