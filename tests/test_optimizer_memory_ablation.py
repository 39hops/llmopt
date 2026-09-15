"""OPTIMIZER-MEMORY-ABLATION-1 guards: the future-stream law matches a
simulation of the trainer's own loop (epoch crossing included) and the
target leg sits at epoch 1 positions 2060..2959; a fresh scheduler stepped
(anchor - 1) times reproduces the audited row for both writer kinds and
its pending step yields the next row; the BAR-1 law equals a real torch
AdamW step contrast (native v exp_avg-zeroed, same batch) in float64;
apply_arm touches exp_avg only; the readout and the ladder are pure and
literal; the like-with-like P0.d law; a small deterministic CPU replay is
bit-exact."""
import copy
import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    prev = os.environ.get("SMOKE")
    os.environ["SMOKE"] = "1"
    for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scratch")):
        if p not in sys.path:
            sys.path.insert(0, p)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        m = importlib.import_module("optimizer_memory_ablation")
    finally:
        os.chdir(cwd)
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


def _trainer_loop_slices(n_enc, bs, epochs):
    """scripts/train_mathnative.py nopack branch, literally: starts, per-epoch random.Random(ep) shuffle, in order."""
    starts = [(i, i + bs) for i in range(0, n_enc - bs, bs)]
    out = []
    for ep in range(epochs):
        idx = list(starts)
        random.Random(ep).shuffle(idx)
        out.extend(tuple(b) for b in idx)
    return starts, out


def test_leg_slices_match_trainer_loop_across_epochs(mod):
    n_enc, bs = 1000, mod.BS
    starts, sim = _trainer_loop_slices(n_enc, bs, 3)
    E = n_enc // bs
    first, n = E - 2, 7                       # crosses the epoch 0 -> 1 boundary
    got = mod.leg_slices(starts, n_enc, first, n)
    assert got == sim[first - 1:first - 1 + n]
    assert mod.leg_slices(starts, n_enc, 1, 1) == [sim[0]]


def test_target_leg_positions(mod):
    n_enc = 164_490
    assert n_enc // mod.BS == 5_140 and 3 * 5_140 == mod.TOTAL
    assert mod.epoch_position(7201, n_enc) == (1, 2060)
    assert mod.epoch_position(8100, n_enc) == (1, 2959)
    assert mod.epoch_position(5141, n_enc) == (1, 0) and mod.epoch_position(5140, n_enc) == (0, 5139)


@pytest.mark.parametrize("kind,anchor", [("stock", 1), ("stock", 900), ("stock", 7200), ("backward", 7200), ("backward", 13500)])
def test_scheduler_resume_matches_audited_table(mod, kind, anchor):
    OA = mod.OA
    rows, _post = OA.table(kind)
    row_a, row_n = rows[anchor - 1], rows[anchor]
    assert row_a[0] == anchor and row_n[0] == anchor + 1
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr=mod.LR, weight_decay=mod.WD)
    opt.param_groups[0]["lr"] = row_a[1]; opt.param_groups[0]["betas"] = (row_a[2], row_a[3])   # as a loaded milestone group would read
    serialized = {"lr": row_a[1], "beta1": row_a[2], "beta2": row_a[3], "wd": row_a[4]}
    _sched, nxt = mod.resume_sched(kind, opt, anchor, serialized)
    assert nxt == {"lr": row_n[1], "beta1": row_n[2], "beta2": row_n[3], "wd": row_n[4]}


def test_scheduler_resume_refuses_mismatch(mod):
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr=mod.LR, weight_decay=mod.WD)
    with pytest.raises(SystemExit):
        mod.resume_sched("stock", opt, 900, {"lr": 1.0, "beta1": 0.5, "beta2": 0.999, "wd": 0.01})


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(5, 4)
        self.b = torch.nn.Linear(4, 3)


def _segs_of(model):
    keys = sorted(k for k, _ in model.named_parameters())
    shapes = dict((k, tuple(v.shape)) for k, v in model.named_parameters())
    segs, off = [], 0
    for k in keys:
        n = int(np.prod(shapes[k])); segs.append((k, off, off + n, "G")); off += n
    return segs, off


def _warm(model, steps, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for _ in range(steps):
        x = torch.randn(8, 5, dtype=torch.float64)
        model.b(model.a(x)).pow(2).mean().backward(); opt.step(); opt.zero_grad()
    return opt


def test_bar1_law_equals_real_step_contrast(mod):
    """The contrast W_Z(1) - W_C(1) of two real torch AdamW steps from the same state on the same clipped gradient equals
    -a_carry_given_batch to 1e-12 of the update scale, and n_pred = ||a_carry|| / ||u_C||; the c = 0 approximation differs."""
    torch.manual_seed(1)
    base = _Tiny().to(torch.float64)
    opt0 = _warm(base, 6, seed=2)
    segs, d = _segs_of(base)
    grp = {"lr": 2.5e-4, "beta1": 0.88, "beta2": 0.999, "wd": 0.01}
    x = torch.randn(8, 5, dtype=torch.float64)
    # gradient (clipped) on the batch
    base.zero_grad(); base.b(base.a(x)).pow(2).mean().backward()
    names = [n for n, _ in base.named_parameters()]; params = dict(base.named_parameters())
    raw = np.concatenate([params[k].grad.reshape(-1).numpy() for k, _, _, _ in segs])
    c = raw * mod.OG.clip_coef(float(np.linalg.norm(raw)))
    base.zero_grad()
    step_next = int(opt0.state[params[names[0]]]["step"].item()) + 1
    law = mod.bar1_law(base, opt0, c, segs, step_next, grp)
    assert law["identity_residual"] < 1e-12
    W0 = mod.flat({k: v.detach() for k, v in base.state_dict().items()}, segs, d)
    outs = {}
    for arm in ("C", "Z"):
        m = _Tiny().to(torch.float64); m.load_state_dict(base.state_dict())
        o = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); o.load_state_dict(copy.deepcopy(opt0.state_dict()))   # load_state_dict shares tensors with its source
        o.param_groups[0].update({"lr": grp["lr"], "betas": (grp["beta1"], grp["beta2"])})
        assert mod.apply_arm(o, arm) == (len(names) if arm == "Z" else 0)
        m.b(m.a(x)).pow(2).mean().backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()
        outs[arm] = mod.flat({k: v.detach() for k, v in m.state_dict().items()}, segs, d) - W0
    contrast = outs["Z"] - outs["C"]
    assert abs(np.linalg.norm(contrast) - law["norms"]["carry"]) <= 1e-12 * max(law["norms"]["carry"], 1.0)
    assert abs(np.linalg.norm(outs["C"]) - law["norms"]["uC"]) <= 1e-12 * law["norms"]["uC"]
    assert abs(np.linalg.norm(contrast) / np.linalg.norm(outs["C"]) - law["n_pred"]) <= 1e-9
    assert law["n_c0_approx"] != law["n_pred"]


def test_apply_arm_touches_exp_avg_only(mod):
    m = _Tiny().to(torch.float64); o = _warm(m, 3, seed=3)
    before = {id(p): (o.state[p]["exp_avg"].clone(), o.state[p]["exp_avg_sq"].clone(), o.state[p]["step"].clone()) for p in o.param_groups[0]["params"]}
    mod.apply_arm(o, "E")
    for p in o.param_groups[0]["params"]:
        ea, sq, st = before[id(p)]
        assert torch.allclose(o.state[p]["exp_avg"], ea * (1 - mod.EPS_TWIN)) and torch.equal(o.state[p]["exp_avg_sq"], sq) and torch.equal(o.state[p]["step"], st)
    mod.apply_arm(o, "Z")
    for p in o.param_groups[0]["params"]:
        ea, sq, st = before[id(p)]
        assert float(o.state[p]["exp_avg"].abs().sum()) == 0.0 and torch.equal(o.state[p]["exp_avg_sq"], sq) and torch.equal(o.state[p]["step"], st)
    assert mod.EPS_TWIN == 0.01
    with pytest.raises(ValueError):
        mod.apply_arm(o, "V")


def test_readout_and_ladder_pure_and_literal(mod):
    segs = [("x", 0, 2, "BLOCK0"), ("y", 2, 4, "OUTSIDE")]
    W0 = np.zeros(4); WC = np.array([1.0, 0.0, 0.0, 0.0]); WX = np.array([1.0, 0.3, 0.0, 0.4])
    r = mod.readout(W0, WC, WX, segs)
    assert abs(r["n"] - 0.5) < 1e-12 and abs(r["group_share"]["BLOCK0"] - 0.36) < 1e-12 and abs(r["group_share"]["OUTSIDE"] - 0.64) < 1e-12
    assert (mod.FORGOTTEN, mod.PERSISTENT, mod.SENSITIVE, mod.CE_FLOOR, mod.CE_MULT, mod.BAR1_TOL, mod.BAR1_BAND) == (0.05, 0.25, 0.10, 0.005, 3.0, 0.02, (0.80, 1.00))
    assert (mod.ENV_MULT, mod.ENV_FLOOR, mod.ENV_CAP, mod.ENV_CE) == (2.0, 0.02, 0.25, 0.01)
    assert mod.HORIZONS == [1, 2, 3] and mod.LEG == 3     # SMOKE constants under the fixture
    w = {"n_Z": {1: 0.90, 2: 0.30, 3: 0.05}, "n_E": {1: 0.009, 2: 0.003, 3: 0.001}, "dCE_Z": {1: 0.1, 2: 0.0, 3: 0.005}, "dCE_E": {1: 0.0, 2: 0.0, 3: 0.0}, "n_pred": 0.91, "h_end": 3}
    a = mod.adjudicate(w)
    assert (a["bar1"], a["path"], a["sensitivity"], a["function"], a["argmin_h_nZ"]) == ("PASS", "FORGOTTEN", "SPECIFIC", "NEUTRAL", 3)
    assert a["r"] == pytest.approx({"1": 1.0, "2": 1.0, "3": 0.5})
    w2 = dict(w, n_Z={1: 0.94, 2: 0.3, 3: 0.25}, n_E={1: 0.009, 2: 0.05, 3: 0.10}, dCE_Z={1: 0, 2: 0, 3: 0.0151}, dCE_E={1: 0, 2: 0, 3: 0.005})
    a2 = mod.adjudicate(w2)
    assert (a2["bar1"], a2["path"], a2["sensitivity"], a2["function"]) == ("INSTRUMENT-FAULT", "PERSISTENT", "SENSITIVE", "HARMED")
    assert mod.program_label({"A": a, "B": a}) == "FORGOTTEN-SPECIFIC+FUNCTION-NEUTRAL"
    assert mod.program_label({"A": a, "B": dict(a, path="PERSISTENT")}) == "MIXED-BY-WRITER"
    assert mod.program_label({"A": a, "B": a2}) == "INSTRUMENT-FAULT"
    assert mod.p0d(0.01, 0.001, 0.0) == "PASS" and mod.p0d(0.05, 0.02, 0.0) == "FAIL" and mod.p0d(0.3, 0.2, 0.0) == "FAIL"
    assert mod.p0d(0.01, 0.3, 0.0) == "NOT-ADJUDICABLE" and mod.p0d(0.01, 0.001, 0.02) == "FAIL" and mod.p0d(0.04, 0.02, 0.01) == "PASS"


def test_cpu_replay_is_bit_exact(mod):
    """Two identical small CPU legs under the instrument's determinism settings give byte-identical weights and optimizer digests."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    outs = []
    for _ in range(2):
        torch.manual_seed(5)
        m = _Tiny(); o = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
        for i in range(4):
            x = torch.arange(40, dtype=torch.float32).reshape(8, 5) / (7.0 + i)
            m.b(m.a(x)).pow(2).mean().backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step(); o.zero_grad()
        outs.append(({k: v.detach().clone() for k, v in m.state_dict().items()}, mod.OG.opt_state_digest(o)))
    assert mod.sd_equal(outs[0][0], outs[1][0]) and outs[0][1] == outs[1][1]
