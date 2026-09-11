"""Source-invariant guard for SG-BOUNDARY-BLOCK7-1: the birth driver is the
results-cited SYNTHETIC-GRADIENT-WRITER-1 driver with the SG-1 branch
replaced by the SG7 branch; the stock loop lines and the zero-control
branch survive verbatim; the SG7 law is literal (seed 28, arena K_BP 4,
block 7 only, the one LOCAL family, PLR7 1e-3, model LR 3e-4, the block-7
constant from the locked audit receipt, FOLD A via sg7_step_terms +
run_sg_step, W_0 input statistics from the probe FIT chunks); the gate
script's law is literal (floor 24, band 7); the credit module's law is
literal (detached parameter views for the true path, detached input for the
surrogate, credit masked to eligible positions)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FB = ROOT / "scratch" / "birth19m_fb.py"
SG = ROOT / "scratch" / "birth19m_sg.py"
SG7 = ROOT / "scratch" / "birth19m_sg7.py"
TRAJ = ROOT / "scratch" / "birth19m_atoms_traj.py"
GATE = ROOT / "scratch" / "sg7_qualgate.py"
CREDIT = ROOT / "scratch" / "sg7_credit.py"
DRIVER = ROOT / "scratch" / "sgbb7_qual_driver.sh"


def _code_lines(path):
    src = path.read_text()
    src = re.sub(r'^""".*?"""', "", src, count=1, flags=re.S)
    return [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]


def _loop_body(path):
    src = path.read_text()
    m = re.search(r"for a, b in stream:\n(.*?)opt\.zero_grad\(\)\n", src, flags=re.S)
    assert m, path
    return [l.strip() for l in m.group(1).splitlines() if l.strip() and not l.strip().startswith("#")]


def test_stock_loop_lines_survive():
    sg7 = set(_code_lines(SG7))
    assert not [l for l in _loop_body(TRAJ) if l not in sg7]


def test_zero_control_branch_verbatim_v_fb_and_sg1():
    fb = _code_lines(FB)
    sg7 = set(_code_lines(SG7))
    i = next(k for k, l in enumerate(fb) if l.startswith("ob = hybrid_objective(model, [], ids[:, :-1]"))
    for l in fb[i:i + 3]:
        assert l in sg7, l
    for l in ("frozen_names, trainable_names = freeze_lower(model, K_BP)", 'raise SystemExit(f"ABORT: frozen tensor {n} moved from W_0")',
              "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)", "if sched.last_epoch < steps_total - 1:",
              "run_sg_step(T, model_params, phi, opt, pred_opt, sched, pred_sched, steps_total)"):
        assert l in sg7, l
    # every non-SG line of the SG-1 driver survives (the diff is the SG -> SG7 branch and its names only)
    sg1 = _code_lines(SG)
    missing = [l for l in sg1 if l not in sg7 and not any(k in l for k in ("SG_", "sg_", "SG ", "SG=", "SG)", "SG:", '"sg"', "FAMILY", "PLR", "SYNTHETIC", "sgwriter1", "sgq", "build_predictors", "SG_BLOCKS", "consts", "preds", "predictors", "s_4..7", "audit receipt"))]
    assert not missing, missing


def test_sg7_law_literal():
    src = SG7.read_text()
    for s in ("SG7_SEED = 28", "SG7_K = 4", "SG7_BLOCK = 7", 'SG7_FAMILY = "local"', "PLR7 = 1e-3", "PROBE_FIT_CHUNKS = [0, 2, 4, 6]",
              'assert PEAK_LR == 3e-4, "the model recipe is the stock LR 3e-4"', 'AUDIT = Path("logs/sgaudit0/audit.json")',
              'consts = {"7": audit["normalization_constants"]["arena_frozen_4_7"]["7"]}', "pred_opt = torch.optim.AdamW(phi, lr=PLR, weight_decay=0.0)",
              'T = sg7_step_terms(model, preds, ids[:, :-1], mask[:, :-1], labels, consts["7"])',
              "run_sg_step(T, model_params, phi, opt, pred_opt, sched, pred_sched, steps_total)",
              'ROOT = Path(f"checkpoints/sgbb7_smoke{SMOKE_TAG}" if SMOKE else "checkpoints/sgbb7")',
              "pred_seed = 1000 + SEED", "mu, sdv = input_stats(model, [f[0] for f in fit], [f[1] for f in fit], [f[2] for f in fit])",
              'assert "FAMILY" not in os.environ and "PLR" not in os.environ, "SG7 has one frozen recipe: no FAMILY / PLR knobs"',
              'if SG7:\n        assert dev == "mps"'):
        assert s in src, s
    assert "gate_eval" not in src
    g = GATE.read_text()
    assert "FLOOR = 24" in g and "BAND = 7" in g and "SEED = 11 if SMOKE else 28" in g
    assert "c - BAND <= g <= c + BAND" in g
    d = DRIVER.read_text()
    assert "SG7=1 MODE=zero K_BP=4 SEED=28 LR=3e-4" in d and "SG7=1 MODE=sg7 SEED=28 LR=3e-4" in d and d.count("birth19m_sg7.py") == 2


def test_credit_law_literal():
    src = CREDIT.read_text()
    for s in ("p7 = {n: p.detach() for n, p in model.blocks[SG7_BLOCK].named_parameters()}",
              "x8_T, _ = functional_call(model.blocks[SG7_BLOCK], p7, (x7, m, None))",
              "x8_L, _ = model.blocks[SG7_BLOCK](x7.detach(), m, None)",
              "logits = model.head(model.norm(x8_T))",
              "hat = float(const7) * g.detach() * credit_mask",
              "S = (hat * x8_L).sum()",
              "g = pred(x8_T.detach(), e)",
              "loss_T, logits_T, targets = true_hidden_errors(model, ids, attn_mask, labels, [SG7_BLOCK])   # 3, cached",
              "z = (torch.cat([h, e], -1) - self.mu) / self.sd"):
        assert s in src, s
    assert "clamp" not in src.split("class LocalSG")[1].split("def build_local_predictor")[0]   # no input clip in the online predictor
