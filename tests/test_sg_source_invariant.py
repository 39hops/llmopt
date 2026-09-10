"""Source-invariant guard for SYNTHETIC-GRADIENT-WRITER-1: the birth driver
is the results-cited FROZEN-BACKBONE-1 driver plus the SG branch; the stock
loop lines and the zero-control branch survive verbatim; the SG law is
literal (seed 27, arena K_BP 4, blocks 4..7, families linear / mlp256,
predictor LRs 3e-4 / 3e-5, model LR 3e-4, constants from the locked audit
receipt, FOLD A via sg_step_terms + run_sg_step); the gate script's ladder
law is literal (floor 24, band 7, frozen order)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FB = ROOT / "scratch" / "birth19m_fb.py"
SG = ROOT / "scratch" / "birth19m_sg.py"
TRAJ = ROOT / "scratch" / "birth19m_atoms_traj.py"
GATE = ROOT / "scratch" / "sg_qualgate.py"
CREDIT = ROOT / "scratch" / "sg_credit.py"


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
    sg = set(_code_lines(SG))
    assert not [l for l in _loop_body(TRAJ) if l not in sg]


def test_zero_control_branch_verbatim():
    fb = _code_lines(FB)
    sg = set(_code_lines(SG))
    i = next(k for k, l in enumerate(fb) if l.startswith("ob = hybrid_objective(model, [], ids[:, :-1]"))
    for l in fb[i:i + 3]:
        assert l in sg, l
    for l in ("frozen_names, trainable_names = freeze_lower(model, K_BP)", 'raise SystemExit(f"ABORT: frozen tensor {n} moved from W_0")',
              "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)", "if sched.last_epoch < steps_total - 1:"):
        assert l in sg, l


def test_sg_law_literal():
    src = SG.read_text()
    for s in ("SG_SEED = 27", "SG_K = 4", "SG_BLOCKS = [4, 5, 6, 7]", 'SG_FAMILIES = ("linear", "mlp256")', "SG_PLRS = (3e-4, 3e-5)",
              'assert PEAK_LR == 3e-4, "the model recipe is the stock LR 3e-4"', 'AUDIT = Path("logs/sgaudit0/audit.json")',
              'consts = audit["normalization_constants"]["arena_frozen_4_7"]', "pred_opt = torch.optim.AdamW(phi, lr=PLR, weight_decay=0.0)",
              "T = sg_step_terms(model, preds, ids[:, :-1], mask[:, :-1], labels, SG_BLOCKS, consts)",
              "run_sg_step(T, model_params, phi, opt, pred_opt, sched, pred_sched, steps_total)",
              'ROOT = Path(f"checkpoints/sgwriter1_smoke{SMOKE_TAG}" if SMOKE else "checkpoints/sgwriter1")',
              "pred_seed = 1000 + SEED"):
        assert s in src, s
    assert "gate_eval" not in src


def test_arena_constants_pinned_to_the_locked_audit_receipt():
    import json
    audit = json.loads((ROOT / "logs" / "sgaudit0" / "audit.json").read_text())["normalization_constants"]["arena_frozen_4_7"]
    mirror = json.loads((ROOT / "docs" / "preregs" / "synthetic-gradient-writer-1.json").read_text())["credit_law"]["constants_arena"]
    assert mirror == audit
    assert {k: round(v, 9) for k, v in audit.items()} == {"4": 5.464e-06, "5": 4.479e-06, "6": 3.564e-06, "7": 2.441e-06}


def test_fold_b_and_fold_a_literal_in_credit():
    src = CREDIT.read_text()
    assert "L_l = (diff.pow(2).mean(-1) * elig).sum() / n_elig" in src
    assert "elig = labels != -100" in src
    assert "hat_delta[l] = float(consts[str(l)]) * g.detach() * credit_mask" in src
    assert "baseline, baseline_per_block = normalized_mse(zero_g, targets, consts, elig, sg_blocks)" in src
    assert "return tot / len(sg_blocks), per" in src
    assert "loss_T, logits_T, targets = true_hidden_errors(model, ids, attn_mask, labels, sg_blocks)   # 3, cached" in src
    assert 'T["total"].backward()' in src and 'T["pred_loss"].backward()' in src
    assert src.index('T["total"].backward()') < src.index("model_opt.step()") < src.index("pred_opt.step()")


def test_ladder_law_literal():
    src = GATE.read_text()
    for s in ("FLOOR = 24", "BAND = 7", 'LADDER = (("linear", 3e-4), ("linear", 3e-5), ("mlp256", 3e-4), ("mlp256", 3e-5))',
              "match = bool(r[\"finite\"] and g is not None and c - BAND <= g <= c + BAND)",
              'st.update({"verdict": "NOT-RESOLVABLE-CONTROL", "stop": True, "next": None, "selected": None})',
              'st.update({"verdict": "ACCESSIBILITY-ONLY", "stop": True, "next": None, "selected": None})',
              'st.update({"verdict": "LADDER-UNSTABLE", "stop": True, "next": None, "selected": None})'):
        assert s in src, s
