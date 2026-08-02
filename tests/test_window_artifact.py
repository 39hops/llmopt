import json
import struct

import pytest

from llmopt.window_artifact import load_contiguous_windows


def write_artifact(tmp_path, raw, contract):
    windows_path = tmp_path / "windows.bin"
    contract_path = tmp_path / "contract.json"
    windows_path.write_bytes(raw)
    contract_path.write_text(json.dumps(contract))
    return windows_path, contract_path


def test_reconstructs_contiguous_next_token_rows(tmp_path):
    raw = struct.pack(
        "<12q", 10, 11, 12, 11, 12, 13, 20, 21, 22, 21, 22, 23
    )
    paths = write_artifact(tmp_path, raw, {
        "windows_sha":
            "13884fd32060487a49b0c3ef64dc3a55d82686d8568262df7f8368272959c43e",
        "windows_rows_sha":
            "745e9f70d86dcbf1521c2edbab9758d96d4bd24b75c044b4ef0207b5460d02e0",
    })

    assert load_contiguous_windows(*paths, sequence_length=3) == [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
    ]


def test_refuses_raw_byte_sha_mismatch_before_decoding(tmp_path):
    paths = write_artifact(tmp_path, b"x", {"windows_sha": "0" * 64})

    with pytest.raises(ValueError, match="window artifact SHA mismatch"):
        load_contiguous_windows(*paths, sequence_length=3)


def test_refuses_record_length_not_divisible_by_record_width(tmp_path):
    paths = write_artifact(tmp_path, b"\0" * 8, {
        "windows_sha":
            "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc",
    })

    with pytest.raises(ValueError, match="non-empty multiple of 48 bytes"):
        load_contiguous_windows(*paths, sequence_length=3)


def test_refuses_noncontiguous_token_target_overlap(tmp_path):
    raw = struct.pack("<6q", 10, 11, 12, 11, 99, 13)
    paths = write_artifact(tmp_path, raw, {
        "windows_sha":
            "0e446af04f43711faf079194348ffc7947d2be6a7a2fb6f3bfaad43b9bfac236",
    })

    with pytest.raises(
        ValueError, match="windows must be contiguous next-token slices"
    ):
        load_contiguous_windows(*paths, sequence_length=3)


def test_refuses_reconstructed_row_sha_mismatch(tmp_path):
    raw = struct.pack("<6q", 10, 11, 12, 11, 12, 13)
    paths = write_artifact(tmp_path, raw, {
        "windows_sha":
            "d0697bd3eb0d63ee894783fd839950456a8374d1cddb981b7e4eb7061b68b626",
        "windows_rows_sha": "0" * 64,
    })

    with pytest.raises(ValueError, match="reconstructed window rows SHA mismatch"):
        load_contiguous_windows(*paths, sequence_length=3)
