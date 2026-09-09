"""Source-invariant guard for WRITER-DFA-1 (RESULTS L68321 item 10): the
DFA birth driver is the results-cited ATOM-DIET-TRAJECTORY-1 driver's
stock arm with one credit switch. Every recipe line of the trajectory
driver's training loop must survive verbatim in the DFA driver (the BP
branch IS the stock loop), the recipe sentinels must be present, no
gate may run inside the driver, and the DFA credit path must be the
shared scratch/dfa_credit.py (no second implementation).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "scratch" / "birth19m_atoms_traj.py"
DFA = ROOT / "scratch" / "birth19m_dfa.py"
CREDIT = ROOT / "scratch" / "dfa_credit.py"

RECIPE_SENTINELS = [
    "model = TM.build_model(len(tok.vocab), d=384, layers=8,",
    "heads=6, ffn=1536).to(dev)",
    "weight_decay=0.01)",
    "sched = torch.optim.lr_scheduler.OneCycleLR(",
    "opt, max_lr=PEAK_LR, total_steps=steps_total, pct_start=0.03)",
    "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)",
    "if sched.last_epoch < steps_total - 1:",
    "sched.step()",
    "opt.zero_grad()",
    "torch.manual_seed(SEED)",
    "C.assert_noop(enc_stock)    # precondition, fresh, in-process",
    "steps_total = EPOCHS * (len(enc_stock) // BS)",
    "assert steps_total == STEPS_TOTAL_PIN, steps_total",
    "logits = model(ids[:, :-1], mask[:, :-1])",
    "labels[mask[:, 1:] == 0] = -100",
    "loss = torch.nn.functional.cross_entropy(",
    "labels.reshape(-1), ignore_index=-100)",
    "loss.backward()",
    "STEPS_TOTAL_PIN = 15_420",
    "SCHEDULE = [0, 463] + list(range(1_028, STEPS_TOTAL_PIN + 1, 1_028))",
    "EPOCHS, BS = C.EPOCHS, C.BS",
    'os.environ["ARM"] = "off"       # frozen module import side-effects only',
    'os.environ["BIRTH_SEED"] = str(SEED)',
    "QUAL_SEED, DISCOVERY_SEED, SMOKE_SEED = 21, 2, 11",
    "QUAL_CELLS = [(1.0, 3e-4), (1.0, 1e-4), (0.25, 3e-4), (4.0, 1e-4)]",
    'ob = dfa_objective(model, Bs, ids[:, :-1], mask[:, :-1], labels)',
    'ob["total"].backward()',
    "Bs_cpu = build_feedback(S)",
]
# trajectory-driver loop lines that the DFA driver may legitimately lack
ALLOWED_DROPPED_LOOP = set()


def _code_lines(path):
    src = path.read_text()
    src = re.sub(r'^""".*?"""', "", src, count=1, flags=re.S)
    return [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]


def _loop_body(path):
    src = path.read_text()
    m = re.search(r"for a, b in stream:\n(.*?)opt\.zero_grad\(\)\n", src, flags=re.S)
    assert m, path
    return [l.strip() for l in m.group(1).splitlines() if l.strip()]


def test_stock_loop_lines_survive_in_dfa_driver():
    traj_loop = _loop_body(TRAJ)
    dfa_lines = set(_code_lines(DFA))
    missing = [l for l in traj_loop if l not in dfa_lines and l not in ALLOWED_DROPPED_LOOP]
    assert not missing, "stock loop lines missing from the DFA driver:\n  " + "\n  ".join(missing)


def test_recipe_sentinels_present():
    lines = set(_code_lines(DFA))
    missing = [s for s in RECIPE_SENTINELS if s not in lines]
    assert not missing, f"DFA driver lacks recipe lines: {missing}"


def test_no_gate_inside_driver():
    src = DFA.read_text()
    assert "gate_eval" not in src
    assert "llmopt.lab.gate" not in src


def test_single_credit_implementation():
    src = DFA.read_text()
    assert "from dfa_credit import build_feedback, feedback_digest, dfa_objective" in src
    assert "autograd.grad" not in src, "the driver must not carry its own credit path"
    credit = CREDIT.read_text()
    assert "FEEDBACK_SEED_BASE = 31_000_000" in credit
    assert "torch.rand(d, n_out, generator=g, dtype=torch.float32) * 2 - 1" in credit
    assert "float(s) / math.sqrt(n_out)" in credit
    assert "xin = x if l == 0 else (x.detach() if detach else x)" in credit
    assert "xf = x.detach() if detach else x" in credit


def test_seed_law_is_literal():
    src = DFA.read_text()
    assert 'assert SEED == QUAL_SEED and MODE == "dfa"' in src
    assert "assert SEED == DISCOVERY_SEED" in src
    assert "assert SEED == SMOKE_SEED" in src
    assert 'assert QUAL + DISCOVERY + SMOKE == 1' in src


def test_feedback_is_deterministic_and_scaled():
    import math
    import sys
    sys.path.insert(0, str(ROOT / "scratch"))
    import torch
    from dfa_credit import build_feedback, feedback_digest
    a, b = build_feedback(1.0), build_feedback(1.0)
    assert feedback_digest(a) == feedback_digest(b)
    assert len(a) == 8 and all(tuple(B.shape) == (384, 40) for B in a)
    assert all(not torch.equal(a[i], a[j]) for i in range(8) for j in range(i + 1, 8))
    c = build_feedback(0.25)
    assert torch.allclose(c[3], a[3] * 0.25)
    assert float(a[0].abs().max()) <= 1.0 / math.sqrt(40) + 1e-7
