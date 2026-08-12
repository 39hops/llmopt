"""Merge API over house .pt state dicts — average / task_vector / shell_graft.

Every op materializes a NEW checkpoint (never overwrites an input;
frozen-in-place doctrine, 2026-08-06) and writes a provenance sidecar
`<out>.merge.json`: {op, inputs (paths+sha256), alpha, ts, git_sha} —
merges carry provenance from birth.

NO gating happens inside this module. Cross-device gate comparisons
are forbidden and sigma never transports (device doctrine, precision
closure 2026-07-24); `gate_cmd(row, device)` only RETURNS the shell
command for the standing 120-problem chain gate
(scratch/gate_ckpt.py / scratch/gate_ckpt_cuda.py) — the caller runs
it on ONE device and compares within that device only.

Never score merges by weight distance (joint-perm closure, RESULTS
L6163): the same function lives at many weight arrangements. A merge
is judged by RUNNING it — the gate, function MSE, the oracle.

stdlib + torch only; torch imports are deferred so the module stays
importable on torch-less environments (tests skip cleanly).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from llmopt.lab import hash as lab_hash

__all__ = ["average", "task_vector", "shell_graft", "gate_cmd",
           "is_ternary_lattice"]


# ---------------------------------------------------------------- helpers

def _sha256(path: str) -> str:
    return lab_hash.sha256_file(path)


def _git_sha() -> str:
    return lab_hash.git_sha()


def _load(path: str):
    import torch
    # weights_only=True: house checkpoints are plain tensor state dicts;
    # never unpickle arbitrary objects from a merge input.
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError(f"{path}: not a state-dict .pt")
    return sd


def _check_out(out: str, *inputs: str) -> None:
    o = Path(out).resolve()
    for p in inputs:
        if Path(p).resolve() == o:
            raise ValueError(
                f"out={out} would overwrite input {p} — merges "
                "materialize NEW files, inputs stay frozen in place")
    # 2026-08-11 review (M8): out may not be ANY existing checkpoint —
    # only the two inputs were guarded, so a typo'd out could clobber a
    # third, possibly RESULTS-cited, file. Frozen-in-place means all of
    # them.
    if o.exists():
        raise ValueError(
            f"out={out} already exists — merges never overwrite; pick "
            "a fresh path (delete the stale file yourself if it is "
            "truly disposable)")


def _check_match(a: dict, b: dict, la: str, lb: str) -> None:
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))[:3]
        only_b = sorted(set(b) - set(a))[:3]
        raise ValueError(
            f"key mismatch {la} vs {lb}: only-in-{la}={only_a} "
            f"only-in-{lb}={only_b}")
    for k in a:
        if tuple(a[k].shape) != tuple(b[k].shape):
            raise ValueError(
                f"shape mismatch at {k}: {la}{tuple(a[k].shape)} vs "
                f"{lb}{tuple(b[k].shape)}")


def _row(op: str, out: str, inputs: list[str], alpha, arch=None,
         label=None) -> dict:
    row = {
        "op": op,
        "out": str(out),
        "inputs": [{"path": str(p), "sha256": _sha256(p)} for p in inputs],
        "alpha": alpha,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
    }
    if arch is not None:
        row["arch"] = dict(arch)
    if label is not None:
        row["label"] = label
    Path(str(out) + ".merge.json").write_text(
        json.dumps(row, indent=1) + "\n")
    return row


def is_ternary_lattice(sd: dict, min_numel: int = 16) -> bool:
    """True if any 2D weight looks absmean-lattice / ternary-quantized:
    <= 3 unique values per tensor after sign/scale normalization
    (|w| / max|w| rounded, plus zero). Ternary parents refuse growth —
    identity-init grafts are non-preserving on a quantized lattice.

    HONESTY FENCE (2026-08-11 review, M3, verified): house "ternary"
    checkpoints trained via the RAT_Q straight-through path store
    fp32 LATENTS, not lattice values (mathnative_45m_ternary.pt gate
    tensor: 4091 unique values) — this check does NOT detect them.
    It catches only genuinely-snapped state dicts (per-TENSOR scale;
    per-row scales also evade it). shell_graft therefore ALSO refuses
    on a "ternary"/"tern" name hint — callers merging STE-latent
    checkpoints under an innocent name are on their own recognizance."""
    import torch
    for t in sd.values():
        if not torch.is_tensor(t) or t.dim() != 2 or t.numel() < min_numel:
            continue
        a = t.detach().abs().float()
        m = a.max()
        if m == 0:
            return True
        u = torch.unique(torch.round(a / m * 1e6))
        if u.numel() <= 3:
            return True
    return False


# -------------------------------------------------------------------- ops

def average(a: str, b: str, out: str, alpha: float = 0.5, *,
            shared_lineage: bool = False, arch: dict | None = None,
            label: str | None = None) -> dict:
    """out = (1-alpha)*a + alpha*b. REFUSES unless the caller asserts
    shared_lineage=True, because model-soup averaging is DEAD on
    independent births: twin soups crater and even 1e-6-perturbed
    twins cross basins (RESULTS 9135, NIGHT-28b); mean-merge is free
    ONLY where training itself manufactured the redundancy — same
    birth, shared trajectory (RESULTS 12356, MERGE-1). Passing
    shared_lineage=True is the caller's claim that a and b descend
    from one birth; this function cannot check it, only demand it."""
    if not shared_lineage:
        raise ValueError(
            "average() refused: model-soup averaging is dead on "
            "independent births (RESULTS 9135 NIGHT-28b; RESULTS 12356 "
            "MERGE-1 — mean-merge free only under shared lineage). "
            "Pass shared_lineage=True only if a and b share a birth.")
    import torch
    _check_out(out, a, b)
    sa, sb = _load(a), _load(b)
    _check_match(sa, sb, "a", "b")
    merged = {}
    for k in sa:
        ta, tb = sa[k], sb[k]
        if ta.is_floating_point():
            merged[k] = ((1.0 - alpha) * ta.float()
                         + alpha * tb.float()).to(ta.dtype)
        else:
            merged[k] = ta.clone()
    torch.save(merged, out)
    return _row("average", out, [a, b], alpha, arch, label)


def task_vector(base: str, a: str, b: str, out: str, alpha: float = 1.0,
                *, arch: dict | None = None,
                label: str | None = None) -> dict:
    """out = base + alpha*((a-base) + (b-base)). PROBE-GRADE: no booked
    house task-vector result exists — the spec INDEX claims a
    COMPLETE-BOOKED 'addition annihilates' verdict but RESULTS carries
    zero entries (flagged 2026-08-11), so treat any prior claim as
    unbooked. The split law says the exploitable structure is
    routing-side, not weight-side (RESULTS 11197) — expect nothing
    from weight-space addition until a gate says otherwise."""
    import torch
    _check_out(out, base, a, b)
    sbase, sa, sb = _load(base), _load(a), _load(b)
    _check_match(sbase, sa, "base", "a")
    _check_match(sbase, sb, "base", "b")
    merged = {}
    for k in sbase:
        t0 = sbase[k]
        if t0.is_floating_point():
            # fp32 accumulate, matching average() (M9: three bf16
            # rounding steps otherwise)
            f0 = t0.float()
            merged[k] = (f0 + alpha * ((sa[k].float() - f0)
                                       + (sb[k].float() - f0))
                         ).to(t0.dtype)
        else:
            merged[k] = t0.clone()
    torch.save(merged, out)
    return _row("task_vector", out, [base, a, b], alpha, arch, label)


def shell_graft(small: str, large_arch: dict, out: str, *,
                seed: int = 6, arch: dict | None = None,
                label: str | None = None) -> dict:
    """Grow `small` into large_arch's FFN shells function-preservingly.

    Replicates scripts/grow_mathnative.py's growth (that file is the
    cited gen-6 instrument; its logic lives inline in main() and is
    not importable, so the operator is replicated minimally here,
    citing it): gate/up gain +grow template-spray rows (near-orthogonal
    + 3% anchor tilt, norms drawn from existing rows), down gains
    +grow ZERO columns => identical function at step 0.

    HONESTY FENCE: the crown +10.7 belongs to the grow+RE-FEED
    PIPELINE, line-v-line (RESULTS 26092). The graft ALONE is
    untested — that is rung R5's question. This function only
    manufactures the graft; it proves nothing about its value.

    REFUSES ternary/absmean-lattice parents (unique-value count per
    2D tensor <= 3 after sign/scale normalization): zero-column
    identity init is off-lattice, ternary growth is non-preserving.

    large_arch: {"ffn": <target ffn width>} (d fixed, FFN-only growth,
    matching grow_mathnative.py's scope)."""
    import torch
    _check_out(out, small)
    stem = Path(small).name.lower()
    if "ternary" in stem or "tern" in stem:
        raise ValueError(
            "shell_graft() refused on NAME HINT: parent filename says "
            "ternary — house RAT_Q checkpoints store fp32 latents the "
            "lattice check cannot see (M3, 2026-08-11); ternary growth "
            "is non-preserving.")
    sd = _load(small)
    if is_ternary_lattice(sd):
        raise ValueError(
            "shell_graft() refused: parent looks ternary/absmean-lattice "
            "quantized (<=3 unique values per 2D tensor after sign/scale "
            "normalization) — ternary growth is non-preserving.")
    # anchor to the BLOCK FFN gate only (M1: a bare endswith matched
    # MoE router gate.weight of shape [n_experts, d] — growing that
    # adds phantom experts, not FFN rows)
    _gate_re = re.compile(r"^blocks\.\d+\.gate\.weight$")
    gate_keys = [k for k in sd if _gate_re.match(k)]
    if not gate_keys:
        raise ValueError("no blocks.N.gate.weight keys — not a house "
                         "dense-FFN state dict (routers not graftable)")
    widths = {sd[k].shape[0] for k in gate_keys}
    if len(widths) != 1:
        raise ValueError(f"non-uniform FFN widths across layers "
                         f"({sorted(widths)}) — refuse rather than "
                         "grow to a wrong uniform target")
    cur_ffn = sd[gate_keys[0]].shape[0]
    tgt_ffn = int(large_arch["ffn"])
    grow = tgt_ffn - cur_ffn
    if grow <= 0:
        raise ValueError(f"large_arch ffn={tgt_ffn} <= current {cur_ffn}")
    g = torch.Generator().manual_seed(seed)
    new = {}
    _ffn_re = re.compile(r"^blocks\.\d+\.(gate|up|down)\.weight$")
    for k, W in sd.items():
        _m = _ffn_re.match(k)
        if _m and _m.group(1) in ("gate", "up"):
            n, d = W.shape
            anchors = torch.randn(5, d, generator=g)
            anchors = anchors / anchors.norm(dim=1, keepdim=True)
            fam = torch.randint(0, 5, (grow,), generator=g)
            rows = torch.randn(grow, d, generator=g)
            rows = rows / rows.norm(dim=1, keepdim=True)
            rows = rows + 0.03 * anchors[fam]
            rows = rows / rows.norm(dim=1, keepdim=True)
            src_norms = W.norm(dim=1)
            idx = torch.randint(0, n, (grow,), generator=g)
            new[k] = torch.cat(
                [W, (rows * src_norms[idx].unsqueeze(1)).to(W.dtype)])
        elif _m and _m.group(1) == "down":
            d, n = W.shape
            new[k] = torch.cat(
                [W, torch.zeros(d, grow, dtype=W.dtype)], dim=1)
        elif k.endswith((".gate.bias", ".up.bias", ".down.bias")):
            # house models are bias-free; a biased FFN would need its
            # own growth rule — refuse rather than ship a shape bomb
            raise ValueError(f"FFN bias key {k}: bias growth is not "
                             "implemented (house models are bias-free)")
        else:
            new[k] = W.clone() if torch.is_tensor(W) else W
    torch.save(new, out)
    # grow_mathnative.py writes the "-1" .ep sidecar so the resume
    # gate reads the graft as epoch-0-ready instead of refusing (M2)
    Path(str(out) + ".ep").write_text("-1")
    row = _row("shell_graft", out, [small], None,
               arch or dict(large_arch), label)
    row["grow"] = grow
    return row


# ------------------------------------------------------------------- gate

def gate_cmd(row: dict, device: str) -> str:
    """Return (never run) the shell command for the standing 120-problem
    chain gate on this merge's output. Device picks the script lineage
    (scratch/gate_ckpt.py on mps/cpu, scratch/gate_ckpt_cuda.py on
    cuda) — and that is where comparability ENDS: gates compare only
    within one device/lineage (sigma never transports; batched gate is
    a separate lineage too, see scratch/gate_batched.py note).

    row must carry row["arch"] = {d, layers, ffn, heads} (pass arch=
    to the op); label defaults to the out-file stem."""
    arch = dict(row.get("arch") or {})
    if "d" not in arch and "d_model" in arch:   # catalog rows say d_model
        arch["d"] = arch["d_model"]
    if not all(arch.get(k) for k in ("d", "layers", "ffn", "heads")):
        raise ValueError(
            "row['arch'] must carry d(/d_model)/layers/ffn/heads with "
            "non-null values — pass arch= to the merge op (heads is "
            "not inferable from the state dict; catalog rows carry "
            "heads=None and cannot satisfy this alone)")
    script = ("scratch/gate_ckpt_cuda.py" if device == "cuda"
              else "scratch/gate_ckpt.py")
    label = row.get("label") or Path(row["out"]).stem
    return (f".venv/bin/python {script} {row['out']} "
            f"{arch['d']} {arch['layers']} {arch['ffn']} {arch['heads']} "
            f"{label}")
