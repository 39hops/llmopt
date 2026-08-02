"""Deterministic gravmoe pair (spec 2026-08-01-deterministic-
gravmoe): the mb bridge model with each Body's FFN split into E=4
experts behind an integer switch_top1 router (multiplicative
top_p gate, fx3 convention), plus an integer gravity relaxation
every K optimizer steps. All arms share seed/init/windows; the
only variable is lambda = LN/LD.

Conventions (part of the contract):
- Router: r = rdiv(int_mm(h2, wr), Q) [T,E]; p_r = softmax_rows
  (r, exp, PQ); top = lowest index among argmax ties; y =
  rdiv(out_top * top_p, PQ). Backward: d(out_e) via top_p,
  d(top_p) scattered into dp_r, softmax_bwd, router + h2 paths.
- Gravity (wide Q_w space, per body, kinds (wg, wu, wd), expert
  index order): mean finalized ONCE via one rdiv, then each
  expert's pull rounded: w_e += rdiv((mean - w_e) * LN, LD).
- Draw order: emb, per body [wq wk wv wo g1 g2, wr, then experts
  e0..e{E-1} each (wg, wu, wd)], g_f, then windows are the diet
  bridge's (identical ids sha).
Readouts: cycle-mean diet loss; expert agreement (bit-identical
expert outputs on the 8 windows' h2 states, pairwise %); merge
test (experts averaged to dense, exact loss delta).
Env: LN/LD (default 0/1 = lb), STEPS (2000), SHIFT (14), K (100).
Usage: .venv/bin/python scratch/detbwd_gravmoe.py
"""
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
os.environ.setdefault("SHIFT", "14")
import torch  # noqa: E402

import detbwd_mb as M  # noqa: E402
from detbwd_diet import draw_windows  # noqa: E402
from llmopt.window_artifact import load_contiguous_windows  # noqa: E402
from detbwd_r1 import Q, int_mm, rdiv  # noqa: E402
from detbwd_r1 import lut  # noqa: E402
from detbwd_r2b import (  # noqa: E402
    GBOOST, PQ, T, build_exp_table, build_silu_tables,
    rope_tables, softmax_bwd, softmax_rows)
from detbwd_r3_qw import IntAdamWQw  # noqa: E402

V = 40
M.V = V
E = int(os.environ.get("E", "4"))
D, F = M.D, M.F   # r2b's dims (DIM/FFN env knobs, defaults 64/128)
LN = int(os.environ.get("LN", "0"))
LD = int(os.environ.get("LD", "1"))
# DIET-COND rung: residual-WRITING matrices (wo, e{j}.wd) drawn
# at +-Q/8 (GPT-2/muP residual scaling, integerized). Same draw
# ORDER, different bound — a contract fork, never a tweak.
COND = os.environ.get("COND") == "1"
QW = (Q // 8) if COND else Q
# QK-COND rung: q/k logit scale. QK=1 draws wq/wk at +-Q/8
# (softer attention at init). TAU=1 adds a LEARNED per-body
# integer temperature tt (Q-scale, init Q): s = rdiv(s0*Q, tt)
# after the fixed SCALE division — dynamic attention sharpness,
# trained by the same optimizer.
QK = os.environ.get("QK") == "1"
QKW = (Q // 8) if QK else Q
TAU = os.environ.get("TAU") == "1"
K = int(os.environ.get("K", "100"))
STEPS = int(os.environ.get("STEPS", "2000"))
# gravmoe contract boost: the top_p gate shrinks backward values
# by up to E x (top_p/PQ in [1/E, 1]) — 4x the R2b boost restores
# the dense chain's quantization budget (boost is linear-lossless
# up to the final unboost, measured in R2b's 64->256 sweep)
GB = int(os.environ.get("GB", str(GBOOST * 4)))
SHIFT = M.SHIFT
EKEYS = tuple(f"e{j}.{k}" for j in range(E)
              for k in ("wg", "wu", "wd"))


class MoBody(M.Body):
    """Body with the FFN behind an E-expert switch_top1 router."""
    KEYS = (("wq", "wk", "wv", "wo", "g1", "g2", "wr") + EKEYS
            + (("tt",) if TAU else ()))

    def __init__(self):
        mk = lambda *sh: torch.randint(-Q, Q + 1, sh,
                                       dtype=torch.int64)
        mkc = lambda *sh: torch.randint(-QW, QW + 1, sh,
                                        dtype=torch.int64)
        mkq = (lambda *sh: torch.randint(-QKW, QKW + 1, sh,
                                         dtype=torch.int64)) \
            if QK else mk
        # shared draws first, then router, then experts (spec);
        # wo and e{j}.wd are the residual writers (mkc under COND).
        # REVIEWER CATCH (2026-08-01 review): the first cut rebound
        # mk itself, so QK=1 softened wq/wk/wv/wr/wg/wu — the booked
        # B-arms are GLOBAL-soft arms (amendment in RESULTS). QK now
        # means wq/wk ONLY, as the pre-reg contract stated.
        self.w = {"wq": mkq(M.DH, D), "wk": mkq(M.DH, D),
                  "wv": mk(M.DH, D), "wo": mkc(D, M.DH),
                  "g1": torch.full((D,), Q, dtype=torch.int64),
                  "g2": torch.full((D,), Q, dtype=torch.int64)}
        self.w["wr"] = mk(E, D)
        for j in range(E):
            self.w[f"e{j}.wg"] = mk(F, D)
            self.w[f"e{j}.wu"] = mk(F, D)
            self.w[f"e{j}.wd"] = mkc(D, F)
        if TAU:      # learned temperature, Q-scale, init 1.0 (=Q)
            self.w["tt"] = torch.full((1,), Q, dtype=torch.int64)

    def _ffn_fwd(self, h2, j, tab, c, sel):
        """Expert j forward on selected rows; caches under j."""
        w = self.w
        gp = rdiv(int_mm(h2, w[f"e{j}.wg"]), Q)
        u = rdiv(int_mm(h2, w[f"e{j}.wu"]), Q)
        sg = lut(tab["silu"], gp, lambda z: z)
        f = rdiv(sg * u, Q)
        out = rdiv(int_mm(f, w[f"e{j}.wd"]), Q)
        c[f"ffn{j}"] = {"gp": gp, "u": u, "sg": sg, "f": f,
                        "sel": sel, "h2": h2}
        return out

    def fwd(self, x, tab):
        # attention half identical to Body — reuse by running the
        # parent up to h2 manually (copied lines kept minimal)
        w, c = self.w, {}
        cos, sin, t_exp = tab["cos"], tab["sin"], tab["exp"]
        c["x"] = x
        h1, c["i1"] = M.rms_fwd(x, w["g1"])
        c["h1"] = h1
        q = rdiv(int_mm(h1, w["wq"]), Q)
        k = rdiv(int_mm(h1, w["wk"]), Q)
        v = rdiv(int_mm(h1, w["wv"]), Q)
        c["v"] = v
        qr, kr = (M.rope_fwd(q, cos, sin),
                  M.rope_fwd(k, cos, sin))
        c["qr"], c["kr"] = qr, kr
        s = rdiv(int_mm(qr, kr), M.SCALE)
        if TAU:
            s = rdiv(s * Q, w["tt"])
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        s = torch.where(causal, s, torch.full_like(s, -(1 << 40)))
        c["s"] = s
        p = softmax_rows(s, t_exp, PQ)
        c["p"] = p
        a = rdiv(int_mm(p, v.transpose(0, 1)), PQ)
        c["a"] = a
        pre1 = x + rdiv(int_mm(a, w["wo"]), Q)
        c["m1"] = (pre1.abs() <= M.ACT_CLAMP).to(torch.int64)
        x1 = torch.clamp(pre1, -M.ACT_CLAMP, M.ACT_CLAMP)
        c["x1"] = x1
        h2, c["i2"] = M.rms_fwd(x1, w["g2"])
        c["h2"] = h2
        # --- switch_top1 router (contract conventions above)
        r = rdiv(int_mm(h2, w["wr"]), Q)
        p_r = softmax_rows(r, t_exp, PQ)
        c["p_r"] = p_r
        mx = p_r.max(-1, keepdim=True).values
        top = (p_r == mx).to(torch.int64).argmax(-1)  # lowest idx
        c["top"] = top
        top_p = p_r[torch.arange(T), top]
        c["top_p"] = top_p
        out = torch.zeros(T, D, dtype=torch.int64)
        for j in range(E):
            sel = (top == j)
            if not bool(sel.any()):
                c[f"ffn{j}"] = None
                continue
            out[sel] = self._ffn_fwd(h2[sel], j, tab, c, sel)
        c["out"] = out
        y = rdiv(out * top_p[:, None], PQ)
        pre2 = x1 + y
        c["m2"] = (pre2.abs() <= M.ACT_CLAMP).to(torch.int64)
        return torch.clamp(pre2, -M.ACT_CLAMP, M.ACT_CLAMP), c

    def bwd(self, dxin, c, tab):
        w = self.w
        cos, sin, td = tab["cos"], tab["sin"], tab["dsilu"]
        G = {k: torch.zeros_like(w[k]) for k in EKEYS}
        dx2 = dxin * c["m2"]
        # gate: y = rdiv(out * top_p, PQ). The gated grad is kept
        # EXACT (dx2*top_p, no rounding) and each consumer folds
        # the /PQ into its own single rdiv — pre-rounding dout
        # shrank the expert leg by up to E x and cost a rounding,
        # measured dh2_expt cosine 0.9235 at b0 (twin decomposition
        # 2026-08-01); folding is the rdiv-grouping-correct form.
        dgate = dx2 * c["top_p"][:, None]        # exact, PQ-scaled
        # dtop_p at BOOST scale like every other dp fed to
        # softmax_bwd (the attention convention divides by Q, not
        # PQ — /PQ underweighted the router path 16x, measured
        # dx-to-b0 cosine 0.825 -> 0.94 on the fix)
        dtop_p = rdiv((c["out"] * dx2).sum(-1), Q)
        dp_r = torch.zeros(T, E, dtype=torch.int64)
        dp_r[torch.arange(T), c["top"]] = dtop_p
        dr = softmax_bwd(c["p_r"], dp_r, PQ)
        G["wr"] = rdiv(int_mm(dr.transpose(0, 1),
                              c["h2"].transpose(0, 1)), Q)
        dh2 = rdiv(int_mm(dr, w["wr"].transpose(0, 1)), Q)
        # experts
        for j in range(E):
            cj = c[f"ffn{j}"]
            if cj is None:
                continue
            sel = cj["sel"]
            df = rdiv(int_mm(dgate[sel],
                             w[f"e{j}.wd"].transpose(0, 1)),
                      PQ * Q)
            G[f"e{j}.wd"] = rdiv(int_mm(
                dgate[sel].transpose(0, 1),
                cj["f"].transpose(0, 1)), PQ * Q)
            du = rdiv(cj["sg"] * df, Q)
            dgp = rdiv(rdiv(cj["u"] * df, Q)
                       * lut(td, cj["gp"],
                             lambda z: torch.full_like(z, Q)), Q)
            dh2_j = rdiv(int_mm(du, w[f"e{j}.wu"].transpose(0, 1))
                         + int_mm(dgp,
                                  w[f"e{j}.wg"].transpose(0, 1)),
                         Q)
            G[f"e{j}.wu"] = rdiv(int_mm(du.transpose(0, 1),
                                        cj["h2"].transpose(0, 1)),
                                 Q)
            G[f"e{j}.wg"] = rdiv(int_mm(dgp.transpose(0, 1),
                                        cj["h2"].transpose(0, 1)),
                                 Q)
            dh2[sel] = dh2[sel] + dh2_j
        dx1, G["g2"] = M.rms_bwd(dh2, c["x1"], w["g2"], c["i2"])
        dx1 = (dx1 + dx2) * c["m1"]
        da = rdiv(int_mm(dx1, w["wo"].transpose(0, 1)), Q)
        G["wo"] = rdiv(int_mm(dx1.transpose(0, 1),
                              c["a"].transpose(0, 1)), Q)
        dp = rdiv(int_mm(da, c["v"]), Q)
        dv = rdiv(int_mm(c["p"].transpose(0, 1),
                         da.transpose(0, 1)), PQ)
        ds = softmax_bwd(c["p"], dp, PQ)
        if TAU:
            # s = s0*Q/tt: dtt at boost scale (masked entries have
            # ds=0, so the -(1<<40) fill never enters the sum)
            G["tt"] = -rdiv((ds * c["s"]).sum().reshape(1),
                            w["tt"])
            ds = rdiv(ds * Q, w["tt"])
        dqr = rdiv(int_mm(ds, c["kr"].transpose(0, 1)), M.SCALE)
        dkr = rdiv(int_mm(ds.transpose(0, 1),
                          c["qr"].transpose(0, 1)), M.SCALE)
        dq = M.rope_bwd(dqr, cos, sin)
        dk = M.rope_bwd(dkr, cos, sin)
        G["wq"] = rdiv(int_mm(dq.transpose(0, 1),
                              c["h1"].transpose(0, 1)), Q)
        G["wk"] = rdiv(int_mm(dk.transpose(0, 1),
                              c["h1"].transpose(0, 1)), Q)
        G["wv"] = rdiv(int_mm(dv.transpose(0, 1),
                              c["h1"].transpose(0, 1)), Q)
        dh1 = rdiv(int_mm(dq, w["wq"].transpose(0, 1))
                   + int_mm(dk, w["wk"].transpose(0, 1))
                   + int_mm(dv, w["wv"].transpose(0, 1)), Q)
        dx0, G["g1"] = M.rms_bwd(dh1, c["x"], w["g1"], c["i1"])
        dx0 = dx0 + dx1
        return G, dx0


GATE = os.environ.get("GATE") == "1"
WINDOWS_BIN = os.environ.get("WINDOWS_BIN")
WINDOWS_CONTRACT = os.environ.get("WINDOWS_CONTRACT")
if bool(WINDOWS_BIN) != bool(WINDOWS_CONTRACT):
    raise ValueError(
        "WINDOWS_BIN and WINDOWS_CONTRACT must be provided together")
ARTIFACT_MODE = bool(WINDOWS_BIN)
TRAJECTORY_ONLY = os.environ.get("TRAJECTORY_ONLY") == "1"
if TRAJECTORY_ONLY and not ARTIFACT_MODE:
    raise ValueError("TRAJECTORY_ONLY requires committed artifact inputs")
if TRAJECTORY_ONLY and not GATE:
    raise ValueError("TRAJECTORY_ONLY requires GATE=1")
if GATE and ARTIFACT_MODE and not TRAJECTORY_ONLY:
    raise ValueError("gate artifact inputs require TRAJECTORY_ONLY=1")
# GRAVMOE-SS rung: parallel scheduled sampling (one-step,
# deterministic). After SSW warmup steps, each training step runs
# a first forward on the truth input and replaces the input
# positions after the row's "Step: " marker with the model's own
# previous-position greedy predictions; targets stay truth.
SS = os.environ.get("SS") == "1"
SSW = int(os.environ.get("SSW", "500"))
ANSWER_ONLY = os.environ.get("ANSWER_ONLY") == "1"
if ANSWER_ONLY and not GATE:
    raise ValueError("ANSWER_ONLY requires GATE=1")
if ANSWER_ONLY and SS:
    raise ValueError("ANSWER_ONLY and SS are separate mechanisms")

GATE_DIET_PATH = "data/micromodel_gen4_sidecar.jsonl"
PINNED_DIET_SHA = \
    "809bce4215a24164ecbf5e951d77507d455bfd1923d08fe39aa02942b11a200b"
PINNED_TRAIN_ROWS_SHA = \
    "32cc244bf28fdadf01b343ae16fe1a55200ffe9fab9bd784e8abd739b12ef2c0"
PINNED_FULL_ROWS_SHA = \
    "78f8aef992debe6ec74e4701fba23167ff5fda1d4294546b9f7621605429798a"


def _require_sha(label, observed, expected):
    if observed != expected:
        raise ValueError(
            f"{label} SHA drift: expected {expected}, got {observed}")


def assert_gate_diet_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    observed = h.hexdigest()
    _require_sha("gate diet", observed, PINNED_DIET_SHA)
    return observed


def assert_gate_row_shas(ids):
    train_sha = hashlib.sha256(ids[:8].numpy().tobytes()).hexdigest()
    full_sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    _require_sha("gate train-row", train_sha, PINNED_TRAIN_ROWS_SHA)
    _require_sha("gate full-row", full_sha, PINNED_FULL_ROWS_SHA)
    return train_sha, full_sha


def find_split(full, mark):
    """Index just past the LAST 'Step: ' marker, or None."""
    lm = len(mark)
    for t0 in range(T - lm, 0, -1):
        if full[t0:t0 + lm].tolist() == mark:
            return t0 + lm
    return None


def answer_region(full, mark, terminator_ids):
    """Return (first answer token, first newline/EOS token)."""
    split = find_split(full, mark)
    if split is None:
        raise ValueError("final Step marker not found")
    terms = set(int(t) for t in terminator_ids)
    for pos in range(split, len(full)):
        if int(full[pos]) in terms:
            return split, pos
    raise ValueError("answer terminator not found")


def token_accuracy_counts(generated, full, region):
    split, terminator = region
    standard_g = generated[split:T]
    standard_t = full[split:T]
    suffix_g = generated[split:terminator + 1]
    suffix_t = full[split:terminator + 1]
    return {
        "standard_hits": int((standard_g == standard_t).sum()),
        "standard_total": int(standard_t.numel()),
        "suffix_hits": int((suffix_g == suffix_t).sum()),
        "suffix_total": int(suffix_t.numel()),
    }


def assert_disjoint_prompts(ids, splits, cut):
    keys = [tuple(ids[i, :splits[i]].tolist())
            for i in range(ids.shape[0])]
    train, heldout = set(keys[:cut]), set(keys[cut:])
    overlap = train & heldout
    if overlap:
        raise ValueError(f"train/heldout prompt overlap: {len(overlap)}")
    return len(train), len(heldout), 0


def loss_dlogits(pp, tgt, eye, boost, region=None):
    dlogits = (pp - Q * eye[tgt]) * boost
    if region is None:
        return dlogits
    split, terminator = region
    keep = torch.zeros(pp.shape[0], dtype=torch.int64)
    keep[split - 1:terminator] = 1
    return dlogits * keep[:, None]


def loss_proxy(pp, tgt, region=None):
    err = Q - pp[torch.arange(tgt.shape[0]), tgt]
    if region is not None:
        split, terminator = region
        err = err[split - 1:terminator]
    return int(err.sum())


def draw_complete(n, diet_path=None):
    """First n diet rows whose FULL text fits T+1 tokens (padded
    with eos) — complete steps, so the oracle can score free-run
    generations. Returns (ids [n, T+1], truth step strings)."""
    if diet_path is None:
        diet_path = GATE_DIET_PATH
    assert_gate_diet_sha(diet_path)
    import json as _json
    from scripts.train_mathnative import MathTokenizer
    tok = MathTokenizer()
    rows, texts = [], []
    with open(diet_path) as f:
        for line in f:
            r = _json.loads(line)
            t = (f"Current: {r['cur']}\nHints: none\n"
                 f"Step: {r['nxt']}\n")
            try:
                ids = tok.encode(t) + [tok.eos_id]
            except ValueError:
                continue
            if len(ids) <= T + 1:
                ids = ids + [tok.eos_id] * (T + 1 - len(ids))
                rows.append(ids)
                texts.append(r["nxt"])
            if len(rows) == n:
                break
    assert len(rows) == n
    return torch.tensor(rows, dtype=torch.int64), texts, tok


def _fork_call(worker, args, timeout):
    import multiprocessing as mp
    import time
    ctx = mp.get_context("fork")
    receiver, sender = ctx.Pipe(duplex=False)
    p = ctx.Process(target=worker, args=(sender, *args))
    deadline = time.monotonic() + timeout
    p.start()
    sender.close()
    result = None
    try:
        if receiver.poll(timeout):
            try:
                result = receiver.recv()
            except (EOFError, OSError):
                result = None
        p.join(max(0.0, deadline - time.monotonic()))
        if p.is_alive():
            p.kill()
            p.join()
            return None
        return result
    finally:
        receiver.close()
        if p.is_alive():
            p.kill()
            p.join()


def _sympy_worker(sender, a, b):
    try:
        import sympy as sp
        ea = sp.sympify(a)
        eb = sp.sympify(b)
        sender.send((True, bool(sp.simplify(ea - eb) == 0)))
    except Exception:
        sender.send((False, False))


def sympy_assess(a, b, timeout=10):
    return _fork_call(_sympy_worker, (a, b), timeout) or (False, False)


def sympy_equiv(a, b, timeout=10):
    return sympy_assess(a, b, timeout)[1]


def gate(m, ids, truths, tok, tab, label):
    """Free-run validity gate: prefix through 'Step: ', greedy
    decode the rest, oracle-score the produced step."""
    mark = tok.encode("Step: ")
    nrows = ids.shape[0]
    metrics = {
        "solves": 0,
        "parseable": 0,
        "terminated": 0,
        "standard_hits": 0,
        "standard_total": 0,
        "suffix_hits": 0,
        "suffix_total": 0,
    }
    terminators = {tok.eos_id, tok.id["\n"]}
    for wi in range(ids.shape[0]):
        w = ids[wi, :T].clone()
        full = ids[wi]
        region = answer_region(full, mark, terminators)
        split = region[0]
        for t in range(split, T):
            lg, _ = m.fwd(w, tab)
            w[t] = int(lg[t - 1].argmax())
        counts = token_accuracy_counts(w, full, region)
        for name, count in counts.items():
            metrics[name] += count
        terminated = any(int(t) in terminators
                         for t in w[region[0]:T])
        metrics["terminated"] += int(terminated)
        # decode generated step text up to eos/newline
        keep = []
        for t in w[split:T].tolist():
            if t in terminators:
                break
            keep.append(t)
        gen_s = tok.decode(keep).strip()
        parseable, equivalent = sympy_assess(gen_s, truths[wi]) \
            if gen_s else (False, False)
        metrics["parseable"] += int(parseable)
        metrics["solves"] += int(equivalent)
    print(f"[gate] {label}: solves {metrics['solves']}/{nrows} "
          f"parseable {metrics['parseable']}/{nrows} "
          f"terminated {metrics['terminated']}/{nrows}", flush=True)
    print(f"[gate] {label}: token-acc standard "
          f"{metrics['standard_hits']}/{metrics['standard_total']} suffix "
          f"{metrics['suffix_hits']}/{metrics['suffix_total']}", flush=True)
    return metrics


class GMB(M.MB):
    """MB with MoBody bodies (param_items/bwd re-keyed)."""

    def __init__(self):
        self.emb = torch.randint(-Q, Q + 1, (V, D),
                                 dtype=torch.int64)
        self.bodies = [MoBody() for _ in range(M.NBLK)]
        self.g_f = torch.full((D,), Q, dtype=torch.int64)

    def param_items(self):
        yield "emb", self.emb
        for i, b in enumerate(self.bodies):
            for k in MoBody.KEYS:
                yield f"b{i}.{k}", b.w[k]
        yield "g_f", self.g_f

    def bwd(self, dlogits, c, tab):
        G = {}
        g_head = rdiv(int_mm(dlogits.transpose(0, 1),
                             c["hf"].transpose(0, 1)), Q)
        dhf = rdiv(int_mm(dlogits, self.emb.transpose(0, 1)), Q)
        dx, G["g_f"] = M.rms_bwd(dhf, c["xf"], self.g_f,
                                 c["i_f"])
        for i in range(M.NBLK - 1, -1, -1):
            Gb, dx = self.bodies[i].bwd(dx, c["bodies"][i], tab)
            for k in MoBody.KEYS:
                G[f"b{i}.{k}"] = Gb[k]
        g_tok = torch.zeros(V, D, dtype=torch.int64)
        g_tok.index_add_(0, c["tok"], dx)
        G["emb"] = g_head + g_tok
        return G


def relax(wide):
    """The gravity event, per spec: per body, kinds order, mean
    finalized once, per-expert pulls rounded."""
    for i in range(M.NBLK):
        for kind in ("wg", "wu", "wd"):
            keys = [f"b{i}.e{j}.{kind}" for j in range(E)]
            mean = rdiv(sum(wide[k] for k in keys), E)
            for k in keys:
                wide[k] += rdiv((mean - wide[k]) * LN, LD)


def agreement(m, wins, tab):
    """Pairwise % of probe tokens where expert outputs agree:
    strict (bit-identical) AND coarse (within +-1 LSB, the spec's
    graded metric — reviewer catch: v1 shipped strict-only and P1
    was graded against the missing instrument)."""
    tot, hit, chit = 0, 0, 0
    for wi in range(wins.shape[0]):
        tok_in = wins[wi, :T]
        x = m.emb[tok_in]
        for b in m.bodies:
            x2, c = b.fwd(x, tab)
            h2 = c["h2"]
            outs = []
            for j in range(E):
                cc = {}
                outs.append(b._ffn_fwd(
                    h2, j, tab, cc, torch.ones(T, dtype=torch.bool)))
            for a_ in range(E):
                for b_ in range(a_ + 1, E):
                    eq = (outs[a_] == outs[b_]).all(-1)
                    hit += int(eq.sum())
                    co = ((outs[a_] - outs[b_]).abs() <= 1).all(-1)
                    chit += int(co.sum())
                    tot += T
            x = x2
    return hit / tot, chit / tot


def twin_fp64(m, tok, tgt, tops, masks=None):
    """fp64 autograd twin. ALL discrete decisions come from the
    integer side: routing tops AND (when masks is given, a list of
    (m1, m2) per body) the clamp masks — boundary elements where
    fp and integer disagree about |pre| <= 32 otherwise inject
    near-tie noise that the attention softmax cancellation
    amplifies into sign flips (measured: windows 0/4, 3-18
    boundary elements, composite -0.76/+0.64 -> the masks are the
    tie-break, same doctrine as fp16 near-ties)."""
    import math
    ps = {n: (p.double() / Q).requires_grad_(True)
          for n, p in m.param_items()}
    eps = 1e-5

    def rms(h, g):
        return g * h / torch.sqrt((h * h).mean(-1, keepdim=True)
                                  + eps)

    half = M.DH // 2
    freq = torch.exp(-math.log(10000.0)
                     * torch.arange(half, dtype=torch.float64)
                     / half)
    ang = (torch.arange(T, dtype=torch.float64)[:, None]
           * freq[None, :])
    cos, sin = ang.cos(), ang.sin()

    def rope(v):
        v1, v2 = v[:, :half], v[:, half:]
        return torch.cat([v1 * cos - v2 * sin,
                          v1 * sin + v2 * cos], -1)

    def iclamp(pre, mk):
        if mk is None:
            return torch.clamp(pre, -32.0, 32.0)
        keep = mk.bool()
        return torch.where(keep, pre,
                           pre.detach().clamp(-32.0, 32.0))

    x = ps["emb"][tok]
    for i in range(M.NBLK):
        w = {k: ps[f"b{i}.{k}"] for k in MoBody.KEYS}
        mk1, mk2 = masks[i] if masks else (None, None)
        h1 = rms(x, w["g1"])
        q, k_, v = h1 @ w["wq"].T, h1 @ w["wk"].T, h1 @ w["wv"].T
        qr, kr = rope(q), rope(k_)
        s = (qr @ kr.T) / M.DH ** 0.5
        if TAU:
            s = s / w["tt"]
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        p = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
        x1 = iclamp(x + (p @ v) @ w["wo"].T, mk1)
        h2 = rms(x1, w["g2"])
        p_r = torch.softmax(h2 @ w["wr"].T, -1)
        top = tops[i]
        top_p = p_r[torch.arange(T), top]
        out = torch.zeros(T, D, dtype=torch.float64)
        for j in range(E):
            sel = (top == j)
            if not bool(sel.any()):
                continue
            hj = h2[sel]
            fj = (torch.nn.functional.silu(hj @ w[f"e{j}.wg"].T)
                  * (hj @ w[f"e{j}.wu"].T))
            out[sel] = fj @ w[f"e{j}.wd"].T
        x = iclamp(x1 + out * top_p[:, None], mk2)
    logits = rms(x, ps["g_f"]) @ ps["emb"].T
    loss = torch.nn.functional.cross_entropy(logits, tgt)
    loss.backward()
    return {n: p.grad for n, p in ps.items()}


def build_model():
    torch.manual_seed(M.SEED)
    return GMB()


def run_loss(m, wins, tab, t_exp, regions=None):
    """Exact cycle-mean loss over the 8 windows (no training)."""
    tot = 0
    for wi in range(wins.shape[0]):
        tok_in, tgt = wins[wi, :T], wins[wi, 1:T + 1]
        lg, _ = m.fwd(tok_in, tab)
        pp = softmax_rows(lg, t_exp)
        region = regions[wi] if regions is not None else None
        tot += loss_proxy(pp, tgt, region)
    return tot // wins.shape[0]


def main():
    train_regions = None
    artifact_rows = None
    if ARTIFACT_MODE:
        artifact_rows = load_contiguous_windows(
            Path(WINDOWS_BIN), Path(WINDOWS_CONTRACT), T)
        wins = torch.tensor(artifact_rows, dtype=torch.int64)
        if wins.shape != (8, T + 1):
            raise ValueError(
                "gravmoe artifacts must contain exactly eight T+1 rows")
        print(f"[gmoe] artifact windows {WINDOWS_BIN} contract "
              f"{WINDOWS_CONTRACT} rows {wins.shape[0]}", flush=True)

    if GATE and ARTIFACT_MODE:
        from scripts.train_mathnative import MathTokenizer
        tok = MathTokenizer()
        assert len(tok.vocab) == V, f"vocab drifted: {len(tok.vocab)}"
        mark = tok.encode("Step: ")
        terminator_ids = [tok.id["\n"], tok.eos_id]
        splits = [find_split(wins[wi], mark)
                  for wi in range(wins.shape[0])]
        assert all(s is not None for s in splits)
        assert mark == [4, 26]
        assert tok.id["\n"] == 27 and tok.eos_id == 1
        assert splits == [15, 10, 15, 15, 19, 15, 12, 15]
        if ANSWER_ONLY:
            train_regions = [
                answer_region(wins[wi], mark, terminator_ids)
                for wi in range(wins.shape[0])
            ]
            print(f"[gmoe] answer regions {train_regions}", flush=True)
        print(f"[gmoe] marker ids {mark} terminator ids "
              f"{terminator_ids}", flush=True)
        print(f"[gmoe] ANSWER_ONLY {int(ANSWER_ONLY)}", flush=True)
        print("[gmoe] GATE artifact mode: 8 committed train rows; "
              "trajectory-only readouts", flush=True)
        if SS:
            print(f"[gmoe] SS mode: parallel scheduled sampling "
                  f"after warmup {SSW}, splits {splits}", flush=True)
    elif GATE:
        all_ids, truths, tok = draw_complete(16)
        train_rows_sha, full_rows_sha = assert_gate_row_shas(all_ids)
        mark = tok.encode("Step: ")
        terminator_ids = [tok.id["\n"], tok.eos_id]
        regions = [answer_region(all_ids[wi], mark, terminator_ids)
                   for wi in range(all_ids.shape[0])]
        train_regions = regions[:8]
        splits = [find_split(all_ids[wi], mark)
                  for wi in range(all_ids.shape[0])]
        assert all(s is not None for s in splits)
        prompt_counts = assert_disjoint_prompts(all_ids, splits, 8)
        assert mark == [4, 26]
        assert tok.id["\n"] == 27 and tok.eos_id == 1
        assert splits[:8] == [15, 10, 15, 15, 19, 15, 12, 15]
        print(f"[gmoe] marker ids {mark} terminator ids "
              f"{terminator_ids}", flush=True)
        print(f"[gmoe] answer regions {train_regions}", flush=True)
        print(f"[gmoe] diet sha {PINNED_DIET_SHA}", flush=True)
        print(f"[gmoe] train-row sha {train_rows_sha}", flush=True)
        print(f"[gmoe] full 16-row sha {full_rows_sha}", flush=True)
        print(f"[gmoe] prompt overlap train {prompt_counts[0]} "
              f"heldout {prompt_counts[1]} overlap {prompt_counts[2]}",
              flush=True)
        print(f"[gmoe] ANSWER_ONLY {int(ANSWER_ONLY)}", flush=True)
        wins = all_ids[:8]          # train on the first 8
        print("[gmoe] GATE mode: 8 complete train rows + "
              "8 held-out, oracle-scored free-run", flush=True)
        if SS:
            print(f"[gmoe] SS mode: parallel scheduled sampling "
                  f"after warmup {SSW}, splits {splits[:8]}",
                  flush=True)
    elif not ARTIFACT_MODE:
        wins = draw_windows()
    print(f"[gmoe] LN/LD {LN}/{LD} K {K} STEPS {STEPS} "
          f"SHIFT {SHIFT} E {E}")
    m = build_model()
    names = [n for n, _ in m.param_items()]
    print(f"[gmoe] params {sum(p.numel() for _, p in m.param_items())} "
          f"windows sha "
          f"{hashlib.sha256(wins.numpy().tobytes()).hexdigest()[:16]}")
    ts, td = build_silu_tables()
    t_exp = build_exp_table()
    cos, sin = rope_tables()
    tab = {"silu": ts, "dsilu": td, "exp": t_exp,
           "cos": cos, "sin": sin}
    eye = torch.eye(V, dtype=torch.int64)
    # init conditioning diagnostics: clamp + zero-prob fractions
    cl, zp, tot_cl, tot_zp = 0, 0, 0, 0
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for wi in range(wins.shape[0]):
        _, cc0 = m.fwd(wins[wi, :T], tab)
        for cb in cc0["bodies"]:
            cl += int((cb["m1"] == 0).sum()) + int((cb["m2"] == 0).sum())
            tot_cl += 2 * T * D
            zp += int((cb["p"][causal] == 0).sum())
            tot_zp += int(causal.sum())
    print(f"[gmoe] INIT clamp-frac {cl / tot_cl:.3f} "
          f"zero-prob-frac {zp / tot_zp:.3f}", flush=True)
    if os.environ.get("TWIN") == "1":
        tok_in, tgt = wins[0, :T], wins[0, 1:T + 1]
        lg, cc = m.fwd(tok_in, tab)
        pp = softmax_rows(lg, t_exp)
        G = m.bwd((pp - Q * eye[tgt]) * GB, cc, tab)
        tops = [cb["top"] for cb in cc["bodies"]]
        msk = [(cb["m1"], cb["m2"]) for cb in cc["bodies"]]
        ref = twin_fp64(m, tok_in, tgt, tops, msk)
        worst, argw = 1.0, ""
        for n in names:
            a = G[n].double().flatten()
            if ref[n] is None:      # expert unrouted in the twin
                assert int(a.abs().sum()) == 0, \
                    f"{n}: integer grad nonzero but twin unrouted"
                continue
            b = ref[n].flatten()
            cosv = float((a @ b) / (a.norm() * b.norm() + 1e-12))
            if cosv < worst:
                worst, argw = cosv, n
        print(f"[gmoe] TWIN worst cosine {worst:.6f} ({argw})")
    if os.environ.get("EXPORT"):
        # P4 lab-leg artifacts: init bytes (param_items order,
        # int64 LE) + window token ids + contract JSON; no training.
        import json
        out = os.environ["EXPORT"]
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out + "_init.bin", "wb") as f:
            for _, p in m.param_items():
                f.write(p.numpy().tobytes())
        # axiom record format: NW records of tok[T] ++ tgt[T]
        # (int64 LE); NW inferred from length engine-side.
        with open(out + "_windows.bin", "wb") as f:
            for wi in range(wins.shape[0]):
                f.write(wins[wi, :T].numpy().tobytes())
                f.write(wins[wi, 1:T + 1].numpy().tobytes())
        contract = {
            # the engine-consumed dict, axiom key spelling
            # (docs/plans/2026-08-01-gravmoe-engine.md; LN/LD/
            # SHIFT/STEPS overridden per-arm from pins.json)
            "contract": {"V": V, "T": T, "D": D, "DH": M.DH,
                         "F": F, "n_blocks": len(m.bodies),
                         "SHIFT": SHIFT, "E": E, "K": K,
                         "LN": LN, "LD": LD},
            "env": {k: os.environ.get(k, "") for k in
                    ("COND", "QK", "TAU", "GATE", "SS", "LN", "LD",
                     "K", "E", "STEPS", "SHIFT")},
            "dims": {"V": V, "T": T, "D": D, "DH": M.DH, "F": F,
                     "E": E, "NBLK": len(m.bodies)},
            "GB": GB, "seed": M.SEED,
            "param_order": names,
            "draw_bounds": {"wq/wk": QKW, "wo/wd": QW, "other": Q},
            "windows_sha": hashlib.sha256(
                open(out + "_windows.bin", "rb").read()).hexdigest(),
            # the 33-token-row sha the house logs print (pins
            # 99caaa64 truncated / 32cc24 gate), for cross-ref:
            "windows_rows_sha":
                hashlib.sha256(wins.numpy().tobytes()).hexdigest(),
            "init_sha": hashlib.sha256(
                open(out + "_init.bin", "rb").read()).hexdigest(),
        }
        with open(out + "_contract.json", "w") as f:
            json.dump(contract, f, indent=1)
        print(f"[gmoe] EXPORT -> {out}_(init|windows).bin + contract")
        return
    flat = dict(m.param_items())
    wide = {n: flat[n] << SHIFT for n in names}
    opt = IntAdamWQw([wide[n] for n in names], SHIFT, lrd=1000)
    # SCHED=1: the mb integer lr decay (doublings of lrd at the
    # quarter points, scaled to STEPS — mb's 250/500/750 at 1000)
    sched = os.environ.get("SCHED") == "1"
    losses, th = [], hashlib.sha256()
    for step in range(1, STEPS + 1):
        if sched and step in (STEPS // 4, STEPS // 2, 3 * STEPS // 4):
            opt.lrd *= 2
        row_index = (step - 1) % wins.shape[0]
        w = wins[row_index]
        region = train_regions[row_index] if ANSWER_ONLY else None
        tok_in, tgt = w[:T], w[1:T + 1]
        nar = {n: rdiv(wide[n], 1 << SHIFT) for n in names}
        m.emb, m.g_f = nar["emb"], nar["g_f"]
        for i, b in enumerate(m.bodies):
            b.w = {k: nar[f"b{i}.{k}"] for k in MoBody.KEYS}
        if SS and GATE and step > SSW:
            # one-step exposure: greedy preds from the truth-input
            # forward spliced in after the row's "Step: " marker
            lg0, _ = m.fwd(tok_in, tab)
            preds = lg0.argmax(-1)
            sp = splits[(step - 1) % wins.shape[0]]
            tok_in = tok_in.clone()
            tok_in[sp:T] = preds[sp - 1:T - 1]
        lg, cc = m.fwd(tok_in, tab)
        pp = softmax_rows(lg, t_exp)
        losses.append(loss_proxy(pp, tgt, region))
        GG = m.bwd(loss_dlogits(pp, tgt, eye, GB, region), cc, tab)
        opt.step([rdiv(GG[n], Q * GB) for n in names])
        if TAU:
            # temperature floor: tt >= 1 narrow (the b3 crash —
            # the model drives tt toward 0 = infinite sharpening;
            # a zero temperature is a division, not a preference)
            for i in range(M.NBLK):
                wide[f"b{i}.tt"].clamp_(min=1 << SHIFT)
        if step % K == 0 and LN:
            relax(wide)
        if step % max(125, STEPS // 8) == 0:
            for n in names:
                th.update(wide[n].numpy().tobytes())
            cyc = sum(losses[-8:]) // 8
            print(f"[gmoe] step {step} loss(cyc8) {cyc} "
                  f"nz {opt.nz_last:.3f} "
                  f"traj-sha {th.hexdigest()[:16]}", flush=True)
    # final narrow weights for probes
    nar = {n: rdiv(wide[n], 1 << SHIFT) for n in names}
    m.emb, m.g_f = nar["emb"], nar["g_f"]
    for i, b in enumerate(m.bodies):
        b.w = {k: nar[f"b{i}.{k}"] for k in MoBody.KEYS}
    if TAU:
        tts = [int(b.w["tt"]) for b in m.bodies]
        print(f"[gmoe] learned tt (Q=512=1.0): {tts}")
    if GATE and TRAJECTORY_ONLY:
        print("[gmoe] trajectory-only: skipping free-run gate; "
              "SymPy solve scoring requires the uncommitted row text",
              flush=True)
    elif GATE:
        gate(m, all_ids[:8], truths[:8], tok, tab, "TRAIN")
        gate(m, all_ids[8:], truths[8:], tok, tab, "HELDOUT")
    agr, agr_c = agreement(m, wins, tab)
    diagnostic_regions = train_regions if ANSWER_ONLY else None
    base = run_loss(m, wins, tab, t_exp, diagnostic_regions)
    # merge test: average experts to dense, in place, exactly
    for i, b in enumerate(m.bodies):
        for kind in ("wg", "wu", "wd"):
            mean = rdiv(sum(b.w[f"e{j}.{kind}"]
                            for j in range(E)), E)
            for j in range(E):
                b.w[f"e{j}.{kind}"] = mean
    merged = run_loss(m, wins, tab, t_exp, diagnostic_regions)
    c0 = sum(losses[:8]) // 8
    cf = sum(losses[-8:]) // 8
    print(f"[gmoe] cycle-mean {c0} -> {cf}  falling: {cf < c0}")
    print(f"[gmoe] expert agreement strict {agr:.4f} "
          f"coarse(+-1) {agr_c:.4f}")
    print(f"[gmoe] merge test: loss {base} -> {merged} "
          f"(delta {merged - base})")
    print(f"[gmoe] FINAL trajectory sha {th.hexdigest()}")


if __name__ == "__main__":
    main()
