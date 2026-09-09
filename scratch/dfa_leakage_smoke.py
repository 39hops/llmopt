"""WRITER-DFA-1 writer-integrity smoke (AMENDMENT -PRECISION L68644 P1):
mechanical gradient-leakage invariants of the DFA credit path and BP-mode
parity of the driver against the stock training step, on a deterministic
CPU float32 fixture (torch.manual_seed(11) model, the first four rows of
the frozen probe batch for the invariants; the first stock batch of
epoch 0 for the one-step parity). Frozen tolerance TOL = 1e-7 (max abs
diff) for the head / norm gradient check and the parity checks;
everything else is bit-exact (torch.equal). Any failure is an
implementation BLOCKER. Writes logs/writerdfa1/leakage.json (refuses to
overwrite) and exits non-zero on any failed invariant.

Usage: .venv/bin/python scratch/dfa_leakage_smoke.py
"""
import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ["ARM"] = "off"
os.environ["BIRTH_SEED"] = "11"

import torch  # noqa: E402

import birth19m_curric as C  # noqa: E402
import train_mathnative as TM  # noqa: E402
from dfa_credit import build_feedback, dfa_objective, bp_hidden_errors, ce_loss, block_params  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402

OUT = Path("logs/writerdfa1/leakage.json")
TOL = 1e-7
FIXTURE_SEED = 11
N_FIX = 4
R = {"prereg": "WRITER-DFA-1", "amendment": "PRECISION P1", "tol": TOL, "fixture_seed": FIXTURE_SEED, "checks": {}}


def ok(name, cond, **info):
    R["checks"][name] = {"pass": bool(cond), **info}
    print(f"[leak] {'PASS' if cond else 'FAIL'} {name} {json.dumps(info) if info else ''}", flush=True)
    return bool(cond)


def fixture_model(tok):
    torch.manual_seed(FIXTURE_SEED)
    return TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)


def maxdiff(a, b):
    return float((a.double() - b.double()).abs().max())


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    torch.use_deterministic_algorithms(True)
    tok = TM.MathTokenizer()
    row_ids, rows, probe_digest, _ = probe_rows(tok)
    R["probe_token_digest"] = probe_digest
    ids, mask, labels = probe_tensors(rows[:N_FIX], tok)
    R["fixture"] = {"rows": row_ids[:N_FIX], "T": int(ids.shape[1]), "n_labels": int((labels != -100).sum())}
    model = fixture_model(tok)
    params = dict(model.named_parameters())
    names = list(params)
    blk = {l: [n for n in names if n.startswith(f"blocks.{l}.")] for l in range(8)}
    Bs = build_feedback(1.0)
    ob = dfa_objective(model, Bs, ids, mask, labels)
    allp = [params[n] for n in names]
    # (i) loss reaches no block parameter and not emb
    g = torch.autograd.grad(ob["loss"], allp, allow_unused=True, retain_graph=True)
    reached = [n for n, gi in zip(names, g) if gi is not None]
    ok("i_loss_reaches_head_norm_only", sorted(reached) == ["head.weight", "norm.g"], reached=reached)
    # (ii) e detached
    ok("ii_e_detached", (not ob["e"].requires_grad) and ob["e"].grad_fn is None, shape=list(ob["e"].shape))
    # (iii) S_l reaches block l only (+ emb for l = 0)
    all_iii = True
    detail = {}
    for l in range(8):
        S_l = (ob["deltas"][l] * ob["outs"][l]).sum()
        gl = torch.autograd.grad(S_l, allp, allow_unused=True, retain_graph=True)
        got = sorted(n for n, gi in zip(names, gl) if gi is not None)
        want = sorted(blk[l] + (["emb.weight"] if l == 0 else []))
        detail[str(l)] = {"ok": got == want, "extra": sorted(set(got) - set(want)), "missing": sorted(set(want) - set(got))}
        all_iii &= got == want
    ok("iii_surrogate_locality", all_iii, per_block=detail)
    # (iv) B_j (j != l) swap leaves block l's gradient bit-identical
    gl_ref = {l: torch.autograd.grad(ob["total"], block_params(model, l), retain_graph=True) for l in range(8)}
    all_iv = True
    for l in range(8):
        Bs2 = [B if j == l else torch.rand(B.shape, generator=torch.Generator("cpu").manual_seed(777 + j)) * 2 - 1 for j, B in enumerate(Bs)]
        ob2 = dfa_objective(model, Bs2, ids, mask, labels)
        assert torch.equal(ob2["e"], ob["e"]), "forward state drifted between passes"
        g2 = torch.autograd.grad(ob2["total"], block_params(model, l))
        all_iv &= all(torch.equal(a, b) for a, b in zip(gl_ref[l], g2))
    ok("iv_other_feedback_independence", all_iv)
    # (v) delivered hidden error == B_l e exactly, and != the BP hidden error
    loss_bp, dbp = bp_hidden_errors(model, ids, mask, labels)
    all_v, diff_bp = True, {}
    for l in range(8):
        delivered = torch.autograd.grad(ob["total"], ob["outs"][l], retain_graph=True)[0]
        all_v &= torch.equal(delivered, ob["deltas"][l])
        diff_bp[str(l)] = maxdiff(delivered, dbp[l])
        all_v &= diff_bp[str(l)] > TOL
    ok("v_delivered_error_is_B_e_not_bp", all_v, max_abs_diff_v_bp=diff_bp, loss_dfa=float(ob["loss"]), loss_bp=float(loss_bp))
    # (vi) head / norm true gradient == stock gradient at the same weights
    gh = torch.autograd.grad(ob["loss"], [params["head.weight"], params["norm.g"]], retain_graph=True)
    logits_s = model(ids, mask)
    loss_s = ce_loss(logits_s, labels)
    gs = torch.autograd.grad(loss_s, [params["head.weight"], params["norm.g"]])
    d_head, d_norm = maxdiff(gh[0], gs[0]), maxdiff(gh[1], gs[1])
    ok("vi_head_norm_true_gradient", d_head <= TOL and d_norm <= TOL and abs(float(ob["loss"]) - float(loss_s)) <= TOL,
       max_abs_diff_head=d_head, max_abs_diff_norm=d_norm, loss_diff=abs(float(ob["loss"]) - float(loss_s)))

    # BP parity: the driver's BP mode v the stock step, one step, first stock batch, CPU, seed 11
    enc, _ = C.encode_with_levels(C.load_excised_rows(), tok)
    a, b = C.stock_epoch_stream(len(enc), 0)[0]
    steps_total = C.EPOCHS * (len(enc) // C.BS)
    with tempfile.TemporaryDirectory() as td:
        dumps = {}
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")
        for mode in ("bp", "dfa"):
            p = Path(td) / f"{mode}.pt"
            env = {**os.environ, "SMOKE": "1", "MODE": mode, "SEED": "11", "DEVICE": "cpu", "SMOKE_STEPS": "1", "EMIT": "0",
                   "TAG": f"_fixture_{stamp}", "GRADDUMP": str(p), "S": "1", "LR": "3e-4"}
            for k in ("QUAL", "DISCOVERY", "DRYRUN"):
                env.pop(k, None)
            r = subprocess.run([sys.executable, "scratch/birth19m_dfa.py"], env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-2000:])
                raise SystemExit(f"driver {mode} fixture run failed rc {r.returncode}")
            dumps[mode] = torch.load(p, map_location="cpu")
            assert dumps[mode]["batch"] == [a, b], (dumps[mode]["batch"], [a, b])
        # stock reference step: birth19m_atoms_traj.py loop lines
        def stock_step(step_fn):
            dev = "cpu"
            torch.manual_seed(FIXTURE_SEED)
            m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps_total, pct_start=0.03)
            batch = enc[a:b]
            L = max(len(s) for s in batch)
            ids_ = torch.tensor([s + [tok.pad_id] * (L - len(s)) for s in batch], device=dev)
            mask_ = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in batch], device=dev)
            labels_ = ids_[:, 1:].clone()
            labels_[mask_[:, 1:] == 0] = -100
            step_fn(m, ids_, mask_, labels_)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
            opt.step()
            sched.step()
            opt.zero_grad()
            return grads, {k: v.detach().clone() for k, v in m.state_dict().items()}

        def bp_fn(m, ids_, mask_, labels_):
            logits = m(ids_[:, :-1], mask_[:, :-1])
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels_.reshape(-1), ignore_index=-100)
            loss.backward()

        def dfa_fn(m, ids_, mask_, labels_):
            dfa_objective(m, build_feedback(1.0), ids_[:, :-1], mask_[:, :-1], labels_)["total"].backward()

        for mode, fn in (("bp", bp_fn), ("dfa", dfa_fn)):
            g_ref, p_ref = stock_step(fn)
            g_drv, p_drv = dumps[mode]["grads"], dumps[mode]["params"]
            assert set(g_ref) == set(g_drv) and set(p_ref) == set(p_drv)
            dg = max(maxdiff(g_ref[k], g_drv[k]) for k in g_ref)
            dp = max(maxdiff(p_ref[k], p_drv[k]) for k in p_ref)
            ok(f"parity_{mode}_driver_v_reference", dg <= TOL and dp <= TOL, max_abs_diff_grads=dg, max_abs_diff_params=dp,
               driver_loss=dumps[mode]["loss"], lr_at_step1=dumps[mode]["lr_at_step1"], batch=[a, b])
        # the two writers differ on the block gradients (sanity that the switch is live)
        dbg = max(maxdiff(dumps["bp"]["grads"][k], dumps["dfa"]["grads"][k]) for k in dumps["bp"]["grads"] if k.startswith("blocks."))
        ok("switch_live_block_grads_differ", dbg > TOL, max_abs_diff_bp_v_dfa_block_grads=dbg)
    R["all_pass"] = all(c["pass"] for c in R["checks"].values())
    R["torch"] = torch.__version__
    R["code_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    R["tree_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    R["run_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(R, indent=1))
    print(f"[leak] {'ALL PASS' if R['all_pass'] else 'FAILURES'} -> {OUT}")
    sys.exit(0 if R["all_pass"] else 1)


if __name__ == "__main__":
    main()
