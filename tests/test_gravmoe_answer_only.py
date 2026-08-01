import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, "scratch")

import detbwd_gravmoe as G  # noqa: E402


def test_answer_region_includes_one_terminator_only():
    # marker [4, 26], answer [31, 32], newline 27, repeated EOS 1
    full = torch.tensor([8, 4, 26, 31, 32, 27, 1, 1],
                        dtype=torch.int64)
    assert G.answer_region(full, [4, 26], {27, 1}) == (3, 5)


def test_answer_region_rejects_missing_marker_or_terminator():
    with pytest.raises(ValueError, match="marker"):
        G.answer_region(torch.tensor([8, 31, 27]), [4, 26], {27, 1})
    with pytest.raises(ValueError, match="terminator"):
        G.answer_region(torch.tensor([8, 4, 26, 31]), [4, 26], {27, 1})


def test_loss_dlogits_masks_scaffold_and_repeated_padding():
    pp = torch.tensor([[100, 200, 212]] * 7, dtype=torch.int64)
    tgt = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.int64)
    eye = torch.eye(3, dtype=torch.int64)
    legacy = (pp - G.Q * eye[tgt]) * 7
    assert torch.equal(G.loss_dlogits(pp, tgt, eye, 7), legacy)

    # split=3, terminator token index=5 -> logit rows 2,3,4 only.
    got = G.loss_dlogits(pp, tgt, eye, 7, (3, 5))
    assert int(got[:2].abs().sum()) == 0
    assert torch.equal(got[2:5], legacy[2:5])
    assert int(got[5:].abs().sum()) == 0


def test_loss_proxy_uses_the_same_rows_as_dlogits():
    pp = torch.tensor([[100, 200, 212]] * 7, dtype=torch.int64)
    tgt = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.int64)
    expected = sum(G.Q - int(pp[i, tgt[i]]) for i in range(2, 5))
    assert G.loss_proxy(pp, tgt, (3, 5)) == expected
