"""Teacher v2d SIDECAR: cached-v-uncached gate on the locked
rollout record (AMENDMENT QWEN-MODEL1-TREE-PINS item 3).

The rollout was produced through model.generate with use_cache=True
(one KV-cached step per token). This pass teacher-forces the SAME
padded batch (input_ids ++ generated tokens, reconstructed from the
locked rollout_tokens.jsonl + manifest gen_lengths, eos-padded to
the recorded step count with attention 1s exactly as generate
extends the mask) through the SAME streamed vendor model with
use_cache=False, and compares per generated position (b, t):

  1. TOKEN EQUALITY FIRST (primary, registered): argmax of the
     no-cache logits at (b, t) must equal the recorded token, for
     every live position t < gen_lengths[b]. A mismatch fails the
     gate regardless of norms. Diagnostic margins are recorded.
  2. NORM BOUND: after fp32 upcast (the no-cache reference is
     quantized to fp16 first, symmetric with the stored record),
     max over live (b, t) of
       ||l_nc - l_c||_2 / ||l_nc||_2  <=  5e-3.

Full tower only (the 2-layer smoke is disqualified by
registration). The receipt books the max, per-prompt maxima, and
the full per-token matrix (sidecar_rel.npy). Refuse-if-exists.

    .venv_teacher/bin/python scratch/qwen_teacher_sidecar.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_teacher_pass import build_streamed_model  # noqa: E402

TDIR = "logs/qwenteacher_v2"
OUT_JSON = os.path.join(TDIR, "sidecar_gate.json")
OUT_REL = os.path.join(TDIR, "sidecar_rel.npy")
BAR = 5e-3

torch.set_grad_enabled(False)


def main() -> int:
    if os.path.exists(OUT_JSON):
        raise SystemExit(f"REFUSING: {OUT_JSON} exists")
    man = json.load(open(os.path.join(TDIR, "teacher_manifest.json")))
    assert man["smoke"] is False, "smoke record disqualified"
    assert man["code_commit"] == "0ca4151", man["code_commit"]
    ro = man["records"]["rollouts"]
    steps, B = ro["steps"], ro["n_prompts"]
    rl = np.load(os.path.join(TDIR, "rollout_logits.npy"))
    assert list(rl.shape) == ro["shape"], (rl.shape, ro["shape"])
    import hashlib
    assert hashlib.sha256(rl.tobytes()).hexdigest() == ro["sha256"], \
        "rollout record sha mismatch"
    rows = [json.loads(l) for l in
            open(os.path.join(TDIR, "rollout_tokens.jsonl"))]
    assert len(rows) == B
    gl = ro["gen_lengths"]
    ids = torch.tensor(ro["input_ids"])
    mask = torch.tensor(ro["attention_mask"])
    plen = ids.shape[1]

    # reconstruct the full generated block exactly as generate held
    # it: recorded tokens, then eos padding to the step count
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.expanduser(
        "~/qwen_vendor"))
    eos = tok.eos_token_id
    genm = torch.full((B, steps), eos, dtype=torch.long)
    for b, r in enumerate(rows):
        t = torch.tensor(r["tokens"], dtype=torch.long)
        assert len(t) == gl[b]
        genm[b, :len(t)] = t
    full_ids = torch.cat([ids, genm], 1)
    full_mask = torch.cat([mask, torch.ones(B, steps,
                                            dtype=torch.long)], 1)

    model, trav = build_streamed_model()
    t0 = time.time()
    out = model(input_ids=full_ids, attention_mask=full_mask,
                use_cache=False)
    wall = time.time() - t0
    calls = trav["layer_calls"]
    assert len(calls) == 64 and min(calls) > 0, "traversal"
    print(f"[sc] no-cache forward {wall:.0f}s", flush=True)

    # logits at position plen-1+t predict generated token t
    lg = out.logits[:, plen - 1:plen - 1 + steps].float()
    rel = np.zeros((B, steps), np.float32)
    tok_ok, margins = True, []
    mismatches = []
    for b in range(B):
        for t in range(gl[b]):
            l_nc = lg[b, t]
            a = int(l_nc.argmax())
            want = int(genm[b, t])
            if a != want:
                s = torch.sort(l_nc, descending=True)
                mismatches.append(
                    {"b": b, "t": t, "got": a, "want": want,
                     "margin": float(s.values[0] - s.values[1]),
                     "want_logit_gap": float(l_nc[a] - l_nc[want])})
                tok_ok = False
            l_nc16 = l_nc.to(torch.float16).float()
            l_c = torch.from_numpy(rl[t, b].astype(np.float32))
            d = float((l_nc16 - l_c).norm()
                      / l_nc16.norm().clamp_min(1e-30))
            rel[b, t] = d
    live = [(b, t) for b in range(B) for t in range(gl[b])]
    mx = float(max(rel[b, t] for b, t in live))
    per_prompt = [float(rel[b, :gl[b]].max()) for b in range(B)]
    passed = tok_ok and mx <= BAR
    np.save(OUT_REL, rel)
    rec = {
        "gate": "cached-v-uncached (TREE-PINS item 3)",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "--short",
             "HEAD"]).decode().strip(),
        "teacher_code_commit": man["code_commit"],
        "bar": BAR, "n_prompts": B, "steps": steps,
        "n_live_positions": len(live),
        "token_equality": tok_ok,
        "token_mismatches": mismatches[:20],
        "n_token_mismatches": len(mismatches),
        "max_rel_l2": mx,
        "per_prompt_max_rel_l2": per_prompt,
        "nc_forward_s": round(wall, 1),
        "reference_quantization": "fp16-then-fp32, symmetric with "
                                  "the stored record",
        "passed": passed,
    }
    with open(OUT_JSON, "x") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[sc] token_equality={tok_ok} "
          f"(mismatches {len(mismatches)}) max_rel={mx:.3e} "
          f"bar={BAR} PASSED={passed}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
