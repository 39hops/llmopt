"""CREDIT-ANCHOR-FRONTIER-1 writer-integrity smoke (AMENDMENT -PRECISION
L69223 F1 / F2 / F3-freeze), on the WRITER-DFA-1 deterministic CPU float32
fixture (torch.manual_seed(11) W_0; the first four probe rows for the
objective checks; the first stock batch of epoch 0 for the one-step driver
comparisons). Frozen tolerance TOL = 1e-7 (max abs diff); bit-exact
(torch.equal) where the paths execute the same ops. Any failure is an
implementation BLOCKER. Writes logs/writercaf1/leakage.json (refuses to
overwrite), exit 1 on any failure.

F1 endpoint identities: K_BP = 0 v the sealed dfa_objective and the sealed
driver MODE=dfa one-step dump; K_BP = 8 v the stock path and the sealed
driver MODE=bp one-step dump (post-clip gradients and post-step params).
F2 per-k leakage (k = 1, 2, 4): top blocks CE gradient present and
surrogate absent; lower blocks CE gradient absent and delivered error ==
B_l e exactly; boundary output of block 7-k unreachable by the loss; e
detached; emb reached by the block-0 surrogate only; head / norm gradient
== stock (<= TOL); the BP segment's block gradients v stock recorded.
F3 zero-credit freeze law: MODE=zero K_BP one-step dump has no gradient on
frozen tensors, frozen tensors equal W_0 after the step, trained tensors
moved.

Usage: .venv/bin/python scratch/caf_leakage_smoke.py
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

import train_mathnative as TM  # noqa: E402
from dfa_credit import build_feedback, dfa_objective, hybrid_objective, ce_loss, block_params, freeze_lower  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402

OUT = Path("logs/writercaf1/leakage.json")
TOL = 1e-7
FIXTURE_SEED = 11
N_FIX = 4
KS = (1, 2, 4)
R = {"prereg": "CREDIT-ANCHOR-FRONTIER-1", "amendment": "PRECISION F1/F2/F3", "tol": TOL, "fixture_seed": FIXTURE_SEED, "checks": {}}


def ok(name, cond, **info):
    R["checks"][name] = {"pass": bool(cond), **info}
    print(f"[caf-leak] {'PASS' if cond else 'FAIL'} {name} {json.dumps(info) if info else ''}", flush=True)
    return bool(cond)


def fixture_model(tok):
    torch.manual_seed(FIXTURE_SEED)
    return TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)


def maxdiff(a, b):
    return float((a.double() - b.double()).abs().max())


def run_driver(script, mode, k_bp, dump, stamp, s="1", lr="3e-4"):
    env = {**os.environ, "SMOKE": "1", "MODE": mode, "SEED": "11", "DEVICE": "cpu", "SMOKE_STEPS": "1", "EMIT": "0",
           "TAG": f"_fixture_{stamp}", "GRADDUMP": str(dump), "S": s, "LR": lr}
    if k_bp is not None:
        env["K_BP"] = str(k_bp)
    for k in ("QUAL", "DISCOVERY", "DRYRUN"):
        env.pop(k, None)
    r = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"{script} {mode} k={k_bp} fixture run failed rc {r.returncode}")
    return torch.load(dump, map_location="cpu")


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    torch.use_deterministic_algorithms(True)
    tok = TM.MathTokenizer()
    row_ids, rows, probe_digest, _ = probe_rows(tok)
    R["probe_token_digest"] = probe_digest
    ids, mask, labels = probe_tensors(rows[:N_FIX], tok)
    model = fixture_model(tok)
    params = dict(model.named_parameters())
    names = list(params)
    allp = [params[n] for n in names]
    blk = {l: [n for n in names if n.startswith(f"blocks.{l}.")] for l in range(8)}
    Bs = build_feedback(1.0)
    # ---- F1 objective-level identities
    a = dfa_objective(model, Bs, ids, mask, labels)
    b0 = hybrid_objective(model, Bs, ids, mask, labels, 0)
    ga = torch.autograd.grad(a["total"], allp)
    gb = torch.autograd.grad(b0["total"], allp)
    ok("F1_k0_objective_identity", torch.equal(a["loss"], b0["loss"]) and torch.equal(a["e"], b0["e"]) and all(torch.equal(x, y) for x, y in zip(a["deltas"], b0["deltas"]))
       and all(torch.equal(x, y) for x, y in zip(ga, gb)), n_params=len(allp))
    b8 = hybrid_objective(model, [], ids, mask, labels, 8)
    logits_s = model(ids, mask)
    loss_s = ce_loss(logits_s, labels)
    gs = torch.autograd.grad(loss_s, allp)
    g8 = torch.autograd.grad(b8["total"], allp)
    ok("F1_k8_objective_identity", maxdiff(b8["logits"], logits_s) <= TOL and abs(float(b8["loss"]) - float(loss_s)) <= TOL and max(maxdiff(x, y) for x, y in zip(gs, g8)) <= TOL,
       max_abs_diff_logits=maxdiff(b8["logits"], logits_s), max_abs_diff_grads=max(maxdiff(x, y) for x, y in zip(gs, g8)))
    # ---- F2 per-k leakage
    for k in KS:
        first_bp = 8 - k
        ob = hybrid_objective(model, Bs, ids, mask, labels, k)
        top = sum((blk[l] for l in range(first_bp, 8)), [])
        low = sum((blk[l] for l in range(first_bp)), [])
        gl = torch.autograd.grad(ob["loss"], allp, allow_unused=True, retain_graph=True)
        reached = {n for n, gi in zip(names, gl) if gi is not None}
        want = set(top) | {"head.weight", "norm.g"}
        S = sum((dl * x).sum() for dl, x in zip(ob["deltas"], ob["outs"][:first_bp]))   # the surrogate terms themselves (total - loss would carry the loss graph as zero tensors)
        gsur = torch.autograd.grad(S, allp, allow_unused=True, retain_graph=True)
        sur_reached = {n for n, gi in zip(names, gsur) if gi is not None}
        want_sur = set(low) | {"emb.weight"}
        # boundary: loss must not reach the output of block 7-k; delivered error there == B_{7-k} e
        gb_ = torch.autograd.grad(ob["loss"], ob["outs"][first_bp - 1], allow_unused=True, retain_graph=True)[0]
        delivered_ok = all(torch.equal(torch.autograd.grad(ob["total"], ob["outs"][l], retain_graph=True)[0], ob["deltas"][l]) for l in range(first_bp))
        gh = torch.autograd.grad(ob["loss"], [params["head.weight"], params["norm.g"]], retain_graph=True)
        d_head, d_norm = maxdiff(gh[0], gs[names.index("head.weight")]), maxdiff(gh[1], gs[names.index("norm.g")])
        top_v_stock = max(maxdiff(gl[names.index(n)], gs[names.index(n)]) for n in top)
        ok(f"F2_k{k}_top_bp_present_surrogate_absent", reached == want and not (sur_reached & set(top)), reached_extra=sorted(reached - want), reached_missing=sorted(want - reached))
        ok(f"F2_k{k}_lower_ce_absent_surrogate_local", not (reached & set(low)) and sur_reached == want_sur, surrogate_extra=sorted(sur_reached - want_sur), surrogate_missing=sorted(want_sur - sur_reached))
        ok(f"F2_k{k}_boundary_detached_delivered_B_e", gb_ is None and delivered_ok and len(ob["deltas"]) == first_bp)
        ok(f"F2_k{k}_e_detached_head_norm_true", (not ob["e"].requires_grad) and ob["e"].grad_fn is None and d_head <= TOL and d_norm <= TOL,
           max_abs_diff_head=d_head, max_abs_diff_norm=d_norm, bp_segment_block_grads_v_stock_max_abs_diff=top_v_stock)
    # ---- driver-level one-step comparisons (F1 endpoints, F3 freeze)
    enc, _ = __import__("birth19m_curric").encode_with_levels(__import__("birth19m_curric").load_excised_rows(), tok)
    a_, b_ = __import__("birth19m_curric").stock_epoch_stream(len(enc), 0)[0]
    with tempfile.TemporaryDirectory() as td:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")
        d = Path(td)
        sealed_dfa = run_driver("scratch/birth19m_dfa.py", "dfa", None, d / "sdfa.pt", stamp)
        sealed_bp = run_driver("scratch/birth19m_dfa.py", "bp", None, d / "sbp.pt", stamp)
        # K_BP = 0 / 8 are not qualification k values; the driver refuses them in QUAL only, smoke allows any 0..8
        caf_k0 = run_driver("scratch/birth19m_caf.py", "hybrid", 0, d / "k0.pt", stamp)
        caf_k8 = run_driver("scratch/birth19m_caf.py", "hybrid", 8, d / "k8.pt", stamp)
        for name, x, y in (("F1_k0_driver_v_sealed_dfa", caf_k0, sealed_dfa), ("F1_k8_driver_v_sealed_bp", caf_k8, sealed_bp)):
            assert x["batch"] == y["batch"] == [a_, b_]
            dg = max(maxdiff(x["grads"][n], y["grads"][n]) for n in y["grads"])
            dp = max(maxdiff(x["params"][n], y["params"][n]) for n in y["params"])
            ok(name, set(x["grads"]) == set(y["grads"]) and dg <= TOL and dp <= TOL, max_abs_diff_grads=dg, max_abs_diff_params=dp, loss_x=x["loss"], loss_y=y["loss"])
        z = run_driver("scratch/birth19m_caf.py", "zero", 2, d / "z2.pt", stamp)
        w0 = fixture_model(tok).state_dict()
        frozen, trainable = freeze_lower(fixture_model(tok), 2)
        frozen_unmoved = all(torch.equal(z["params"][n], w0[n]) for n in frozen)
        trained_moved = all(not torch.equal(z["params"][n], w0[n]) for n in trainable)
        ok("F3_zero_k2_freeze_law", (not (set(z["grads"]) & set(frozen))) and set(z["grads"]) == set(trainable) and frozen_unmoved and trained_moved,
           n_frozen=len(frozen), n_trainable=len(trainable), loss=z["loss"])
    R["all_pass"] = all(c["pass"] for c in R["checks"].values())
    R["torch"] = torch.__version__
    R["code_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    R["tree_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    R["run_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(R, indent=1))
    print(f"[caf-leak] {'ALL PASS' if R['all_pass'] else 'FAILURES'} -> {OUT}")
    sys.exit(0 if R["all_pass"] else 1)


if __name__ == "__main__":
    main()
