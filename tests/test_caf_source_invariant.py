"""Source-invariant and endpoint-identity guards for CREDIT-ANCHOR-FRONTIER-1
(RESULTS L69122, AMENDMENT -PRECISION L69223 F1): the frontier driver is
the results-cited WRITER-DFA-1 driver plus MODE=hybrid / MODE=zero and
K_BP; its bp and dfa branches and the stock loop lines survive verbatim;
the sealed dfa_credit functions are byte-identical; and on a deterministic
CPU fixture hybrid_objective(k_bp=0) reproduces dfa_objective bit-exactly
while hybrid_objective(k_bp=8) reproduces the stock forward / backward.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DFA = ROOT / "scratch" / "birth19m_dfa.py"
CAF = ROOT / "scratch" / "birth19m_caf.py"
CREDIT = ROOT / "scratch" / "dfa_credit.py"
TRAJ = ROOT / "scratch" / "birth19m_atoms_traj.py"



def _code_lines(path):
    src = path.read_text()
    src = re.sub(r'^""".*?"""', "", src, count=1, flags=re.S)
    return [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]


def _loop_body(path):
    src = path.read_text()
    m = re.search(r"for a, b in stream:\n(.*?)opt\.zero_grad\(\)\n", src, flags=re.S)
    assert m, path
    return [l.strip() for l in m.group(1).splitlines() if l.strip()]


def test_stock_loop_lines_survive():
    caf = set(_code_lines(CAF))
    missing = [l for l in _loop_body(TRAJ) if l not in caf]
    assert not missing, missing


def test_dfa_driver_branches_survive():
    dfa_loop = _loop_body(DFA)
    caf = set(_code_lines(CAF))
    allowed = {'else:', 'grads = {n: p.grad.detach().to("cpu").clone() for n, p in model.named_parameters()}'}
    missing = [l for l in dfa_loop if l not in caf and l not in allowed]
    assert not missing, missing
    src = CAF.read_text()
    for s in ('ob = dfa_objective(model, Bs, ids[:, :-1], mask[:, :-1], labels)',
              'ob = hybrid_objective(model, Bs, ids[:, :-1], mask[:, :-1], labels, K_BP)',
              'ob = hybrid_objective(model, [], ids[:, :-1], mask[:, :-1], labels, 8)',
              'frozen_names, trainable_names = freeze_lower(model, K_BP)',
              'QUAL_SEED, DISCOVERY_SEED, SMOKE_SEED = 23, 2, 11',
              'QUAL_CELLS = [(1.0, 3e-4), (1.0, 1e-4)]', 'QUAL_K = (1, 2, 4)', 'ZERO_LR = 3e-4',
              'assert SEED == QUAL_SEED and MODE in ("hybrid", "zero")', "gate_eval" ):
        assert (s in src) != (s == "gate_eval"), s


def test_sealed_credit_functions_unchanged():
    """The WRITER-DFA-1 part of dfa_credit.py (everything above the frontier
    section) is byte-identical to the version the verdict cites (git blob at
    the booking commit 5568f890)."""
    import subprocess
    old = subprocess.run(["git", "show", "5568f890:scratch/dfa_credit.py"], capture_output=True, text=True, cwd=ROOT).stdout
    if not old:
        return  # shallow checkout
    new = CREDIT.read_text()
    assert new.startswith(old.rstrip("\n")), "sealed dfa_credit.py prefix changed"


def _fixture():
    sys.path.insert(0, str(ROOT / "scratch"))
    sys.path.insert(0, str(ROOT))
    import torch
    from llmopt.train.mathnative import MathTokenizer, build_model
    tok = MathTokenizer()
    torch.manual_seed(11)
    model = build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    g = torch.Generator("cpu").manual_seed(5)
    ids = torch.randint(2, len(tok.vocab), (3, 12), generator=g)
    mask = torch.ones_like(ids)
    mask[1, 9:] = 0
    mask[2, 6:] = 0
    labels = ids[:, 1:].clone()
    labels[mask[:, 1:] == 0] = -100
    return torch, model, ids[:, :-1], mask[:, :-1], labels


def test_k0_identity_with_sealed_dfa_objective():
    torch, model, ids, mask, labels = _fixture()
    from dfa_credit import build_feedback, dfa_objective, hybrid_objective
    Bs = build_feedback(1.0)
    a = dfa_objective(model, Bs, ids, mask, labels)
    b = hybrid_objective(model, Bs, ids, mask, labels, 0)
    assert torch.equal(a["loss"], b["loss"]) and torch.equal(a["e"], b["e"]) and torch.equal(a["total"], b["total"])
    assert all(torch.equal(x, y) for x, y in zip(a["deltas"], b["deltas"])) and len(b["deltas"]) == 8
    ga = torch.autograd.grad(a["total"], list(model.parameters()))
    gb = torch.autograd.grad(b["total"], list(model.parameters()))
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))


def test_k8_identity_with_stock_backprop():
    torch, model, ids, mask, labels = _fixture()
    from dfa_credit import hybrid_objective, ce_loss
    b = hybrid_objective(model, [], ids, mask, labels, 8)
    logits = model(ids, mask)
    loss = ce_loss(logits, labels)
    assert b["e"] is None and b["deltas"] == [] and b["total"] is b["loss"]
    assert float((b["logits"] - logits).abs().max()) <= 1e-7 and abs(float(b["loss"]) - float(loss)) <= 1e-7
    ga = torch.autograd.grad(loss, list(model.parameters()))
    gb = torch.autograd.grad(b["total"], list(model.parameters()))
    assert max(float((x - y).abs().max()) for x, y in zip(ga, gb)) <= 1e-7


def test_freeze_lower_partition():
    torch, model, ids, mask, labels = _fixture()
    from dfa_credit import freeze_lower
    frozen, trainable = freeze_lower(model, 2)
    assert "emb.weight" in frozen and all(f"blocks.{l}." in "|".join(frozen) for l in range(6))
    assert all(n.startswith(("blocks.6.", "blocks.7.", "norm.", "head.")) for n in trainable)
    assert len(frozen) + len(trainable) == 59
