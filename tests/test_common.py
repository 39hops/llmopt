"""llmopt.common — device/seed/ckpt utilities (spec 2026-08-12 §4.1).

pick_device precedence is torch-like (Artin GO 2026-08-12):
explicit arg > LLMOPT_DEVICE env > cuda > mps > cpu. Placement is
NOT this module's business: .to(dev) stays per-object, and seeded
CPU islands are pins that never route through pick_device.
"""
import random

import pytest

from llmopt.common.device import pick_device
from llmopt.common.seed import srng


def test_pick_device_override_wins(monkeypatch):
    monkeypatch.setenv("LLMOPT_DEVICE", "cpu")
    assert pick_device("mps") == "mps"


def test_pick_device_env_beats_detection(monkeypatch):
    monkeypatch.setenv("LLMOPT_DEVICE", "cpu")
    assert pick_device() == "cpu"


def test_pick_device_detection_order(monkeypatch):
    monkeypatch.delenv("LLMOPT_DEVICE", raising=False)
    torch = pytest.importorskip("torch")
    want = ("cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu")
    assert pick_device() == want


def test_srng_string_seed_law():
    # stable STRING seeds only (house law: tuple hash is
    # per-process randomized)
    a = srng("mathgen", 3, 7000)
    b = srng("mathgen", 3, 7000)
    assert isinstance(a, random.Random)
    assert [a.random() for _ in range(4)] == \
        [b.random() for _ in range(4)]
    assert srng("mathgen", 3, 7001).random() != \
        srng("mathgen", 3, 7000).random()


def test_load_ckpt_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from llmopt.common.ckpt import load_ckpt
    p = tmp_path / "w.pt"
    torch.save({"w": torch.ones(3)}, p)
    ck = load_ckpt(p)
    assert torch.equal(ck["w"], torch.ones(3))
