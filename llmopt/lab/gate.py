"""lab.gate — the standard 120 gate. CANONICAL BODY since 2026-08-12
(Phase 3 module 5): scripts/step_grpo_micro.py re-exports
sample_wave_lp and gate_eval from here via a LINE-COUNT-PRESERVING
shim (RESULTS cites lines 65 and 184 inside the original bodies, so
the shim keeps those line numbers and quotes the cited fragments in
place). Behavior is pinned by tests/test_gate_battery.py (the booked
SOFT-PROMPT-1-SAMPLER replay + gate problem-grid pins) and
tests/test_lab_adoption.py (shim-binds + lineage-constant + line
anchors).

Notes:
- gate_eval's oracle imports resolve to llmopt.lab.gen /
  llmopt.lab.verify directly (the pre-shim sys.modules aliases of
  bench_step_tokens/bench_verify_fast are gone — the scripts are
  shims of these modules now, so the alias carried no information
  and poisoned bare imports of the real scripts).
- Constants below are the GRPO-MICRO lineage values (the lineage
  every 120-gate verdict cites). GateSpec carries them per lineage;
  apply_spec swaps the module globals (single-threaded lab use —
  gates never run concurrently in-process).
- Provenance rule (2026-07-31) travels: gate_eval prints the weights
  sha first; a gate number books WITH that hash, and shas never
  compare across precisions.
"""
from __future__ import annotations

from llmopt.common.device import pick_device
from dataclasses import dataclass

# GRPO-MICRO lineage constants (scripts/step_grpo_micro.py values —
# the source-identity guard pins the FUNCTIONS; these pin the numbers)
B = 8
GATE_LEVELS = (3, 4, 5, 6, 7)
GATE_N = 24  # 12/level left +-1 solve inside the gate's noise floor (run 2b)
GATE_BAND = 9_900_000


@dataclass(frozen=True)
class GateSpec:
    """Per-lineage gate constants. GRPO_MICRO is the standard 120."""
    wave: int = 8            # B — sampling wave width
    levels: tuple = (3, 4, 5, 6, 7)
    n: int = 24              # problems per level
    band: int = 9_900_000    # seed band base


GRPO_MICRO = GateSpec()


def apply_spec(spec: GateSpec) -> None:
    """Point the module constants at a lineage. Single-threaded use."""
    global B, GATE_LEVELS, GATE_N, GATE_BAND
    B = spec.wave
    GATE_LEVELS = spec.levels
    GATE_N = spec.n
    GATE_BAND = spec.band


def sample_wave_lp(model, tok, prompt_ids, seeds, dev, max_new=120):
    """KV-cached (2026-07-22): token-identical to the eager
    full-recompute path — proven 20/20 waves on cpu/cuda/mps
    (scratch/kv_equiv.py); 4.5x cpu / 3.4x mps / 1.1x cuda."""
    import torch
    Bn = len(seeds)
    ids = torch.tensor([prompt_ids] * Bn, device=dev)
    gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
    out = [[] for _ in range(Bn)]
    lps = [0.0] * Bn
    done = [False] * Bn
    nl = tok.id["\n"]
    logits, past = model(ids, use_cache=True)
    step_logits = logits[:, -1]
    for _ in range(max_new):
        probs = torch.softmax(step_logits.float().cpu() / 0.7, -1)
        nxts = []
        for b in range(Bn):
            if done[b]:
                nxts.append(tok.pad_id)
                continue
            nxt = int(torch.multinomial(probs[b], 1, generator=gens[b]))
            lps[b] += float(torch.log(probs[b, nxt] + 1e-20))
            if nxt in (nl, tok.eos_id, tok.pad_id):
                done[b] = True
            else:
                out[b].append(nxt)
            nxts.append(nxt)
        if all(done):
            break
        col = torch.tensor(nxts, device=dev)[:, None]
        logits, past = model(col, past=past)
        step_logits = logits[:, -1]
    return [tok.decode(o).strip() for o in out], out, lps


def gate_eval(model, tok, dev, n=None):
    """Honest chain gate. n<GATE_N = cheap proxy tier (same seeds,
    prefix subset — noisier per reading, never used for promotion).
    Prints the WEIGHTS sha first (provenance rule, 2026-07-31: a
    gate number books with the hash of the weights that produced
    it — true even for in-process gates, which is exactly where
    the lb-s1 45-v-37 provenance bug lived)."""
    import hashlib
    import sympy as sp
    import torch

    from llmopt.lab.gen import _gen_isolated
    from llmopt.lab.verify import verify_wave
    wh = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        v = v.detach().cpu().contiguous()
        if v.dtype == torch.bfloat16:      # numpy can't view bf16
            v = v.view(torch.int16)
        wh.update(v.numpy().tobytes())
    print(f"[gate] weights sha {wh.hexdigest()[:16]}", flush=True)
    solves = {}
    valid = tried = 0
    with torch.no_grad():
        for lv in GATE_LEVELS:
            s = 0
            for i in range(n or GATE_N):
                p = _gen_isolated(lv, GATE_BAND + 1000 * lv + i)
                if p is None:
                    continue
                cur = f"Integral({sp.sstr(p._expr)}, x)"
                visited = {cur.replace(" ", "")}
                done = False
                for ply in range(12):
                    prompt = tok.encode(
                        f"Current: {cur}\nHints: none\nStep: ")
                    texts, _, _ = sample_wave_lp(
                        model, tok, prompt,
                        [GATE_BAND + i * 31 + ply * 7 + b
                         for b in range(B)], dev)
                    tried += len(texts)
                    distinct = [t_ for t_ in dict.fromkeys(texts)
                                if t_ and t_.replace(" ", "") not in visited]
                    wv = verify_wave(cur, distinct) if distinct else {}
                    nxt = None
                    for t_ in texts:
                        ok, so = wv.get(t_, (False, False))
                        if ok and t_.replace(" ", "") not in visited:
                            valid += 1
                            if nxt is None:
                                nxt = "SOLVED" if so else t_
                    if nxt == "SOLVED":
                        done = True
                        break
                    if nxt is None:
                        break
                    cur = nxt
                    visited.add(cur.replace(" ", ""))
                s += done
            solves[lv] = s
    return solves, 100 * valid / max(tried, 1)


def gate_checkpoint(ckpt: str, d: int, layers: int, ffn: int,
                    heads: int, label: str, device: str | None = None,
                    spec: GateSpec | None = None):
    """The scratch/gate_ckpt.py behavior as a callable: load a house
    checkpoint, run the standard gate, print the receipt line, return
    (solves_dict, valid_pct, total). Device doctrine is the CALLER's
    problem — this runs wherever it is invoked, and the printed line
    carries no device tag; book device alongside (lake gates schema
    requires it)."""
    import torch

    from llmopt.train.mathnative import MathTokenizer, build_model
    if spec is not None:
        apply_spec(spec)
    tok = MathTokenizer()
    dev = device or pick_device()
    model = build_model(len(tok.vocab), d=d, layers=layers, heads=heads,
                        ffn=ffn).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    solves, valid = gate_eval(model, tok, dev)
    tot = sum(solves.values())
    print(f"{label} gate: {solves} = {tot}/120 @ {valid:.2f}%", flush=True)
    return solves, valid, tot
