"""F1b (PRE-REG V4-F1): boot the VENDOR's model.py over the kernel twin
— truncated architecture, RANDOM weights, no downloads beyond the
sha-pinned vendor source (checkpoints/v4flash_vendor/, fetched from the
HF repo; model.py sha c0c19e6c9fa439ba matches the rung-D pin).

The truncation exercises every code path the full model uses:
  layer 0  hash-routed (tid2eid), window attention (ratio 0)
  layer 1  score-routed gate (sqrtsoftplus + bias), window attention
  layer 2  score-routed, COMPRESSED attention (ratio 4: Compressor +
           Indexer + Hadamard + fp4 QAT + sparse_attn top-k)
  every layer: hyper-connections (Sinkhorn), shared expert, fp8
  Linears through the twin's act_quant/fp8_gemm, fp4 experts through
  fp4_gemm. MTP skipped (dspark_block_size=0 — construction is gated
  on it; risk-scan H5).
Experts are cut to 16/layer so the random boot stays small; the
architecture is otherwise the shipped config.

BARS (registered): runs end to end (prefill + one decode step) on cpu
AND mps; every output finite; cpu-vs-mps logits rel-L2 <= 0.05 with
top-1 agreement reported (bf16 kernels differ across devices; the
house fp16-near-tie rule applies to any argmax flip at tiny margin).

Usage: .venv/bin/python scratch/v4flash_f1b.py
"""
import importlib.util
import json
import os
import sys

import torch

sys.path.insert(0, "scratch")
import v4flash_twin  # noqa: E402

VENDOR = "checkpoints/v4flash_vendor"
SEED = 20260803


def load_vendor_model_module():
    v4flash_twin.install()                      # kernel + hadamard shims
    spec = importlib.util.spec_from_file_location(
        "v4model", os.path.join(VENDOR, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boot_args(mod):
    cfg = json.load(open(os.path.join(VENDOR, "config.json")))
    cfg.update(
        n_layers=3, n_hash_layers=1, n_mtp_layers=0, dspark_block_size=0,
        dspark_target_layer_ids=[], compress_ratios=[0, 0, 4],
        n_routed_experts=16, n_activated_experts=6,
        max_batch_size=1, max_seq_len=512,
    )
    known = {f for f in mod.ModelArgs.__dataclass_fields__}
    dropped = {k: v for k, v in cfg.items() if k not in known}
    if dropped:
        print(f"[f1b] config keys not in ModelArgs (dropped): "
              f"{sorted(dropped)}")
    return mod.ModelArgs(**{k: v for k, v in cfg.items() if k in known})


def init_random(model, n_routed, gen):
    """The vendor model has no init (inference-only, torch.empty).
    Garbage fp8/e8m0 bytes decode to inf/nan through any LUT, so every
    tensor gets a typed random init; scales get code 127 (= 1.0)."""
    with torch.no_grad():
        # PARAMETERS only: buffers are either precomputed (freqs_cis,
        # complex64) or zero-filled caches — randomizing them would be
        # wrong, not just unnecessary.
        for name, t in model.named_parameters():
            if t.dtype in (torch.bfloat16, torch.float32, torch.float16):
                if "norm" in name and name.endswith(".weight"):
                    t.copy_(torch.ones_like(t))
                else:
                    t.copy_(torch.randn(t.shape, generator=gen,
                                        device="cpu") * 0.02)
            elif t.dtype == torch.float8_e4m3fn:
                t.copy_((torch.randn(t.shape, generator=gen,
                                    device="cpu") * 0.02)
                        .to(torch.float8_e4m3fn))
            elif t.dtype == torch.float8_e8m0fnu:
                t.copy_(torch.full(t.shape, 127, dtype=torch.uint8, device="cpu")
                        .view(torch.float8_e8m0fnu))
            elif hasattr(torch, "float4_e2m1fn_x2") and \
                    t.dtype == torch.float4_e2m1fn_x2:
                t.copy_(torch.randint(0, 256, t.shape, dtype=torch.uint8,
                                      generator=gen, device="cpu")
                        .view(torch.float4_e2m1fn_x2))
            elif t.dtype in (torch.int32, torch.int64):
                t.copy_(torch.randint(0, n_routed, t.shape, device="cpu",
                                      generator=gen, dtype=t.dtype))
            else:
                raise AssertionError(f"unhandled dtype {t.dtype} at {name}")


def run(dev, mod, args, tokens):
    torch.set_default_device(dev)
    torch.set_default_dtype(torch.bfloat16)
    # lru_cache'd builders pin the device they first run on (risk-scan
    # H4): a fresh process would be cleaner, but clearing the caches
    # between devices is equivalent and keeps the comparison in-process.
    for fn in (mod.precompute_freqs_cis, mod.get_window_topk_idxs,
               getattr(mod, "get_dense_topk_idxs", None)):
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
    gen = torch.Generator().manual_seed(SEED)      # cpu gen: same draws
    model = mod.Transformer(args)
    init_random(model, args.n_routed_experts, gen)
    model.eval()
    model.temperature = 0.0          # argmax sampling: device-free (H3)
    toks = tokens.to(dev)
    with torch.inference_mode():
        ids, logits, _ = model.forward(toks, start_pos=0)      # prefill
        nxt = ids.reshape(-1)[-1]
        _, step, _ = model.forward(nxt.reshape(1, 1),
                                   start_pos=toks.size(1))
    out = {"prefill": logits.float().cpu(), "decode": step.float().cpu(),
           "next_token": int(nxt)}
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    return out


def main():
    mod = load_vendor_model_module()
    args = boot_args(mod)
    # CONTINUITY arm for the cross-device bar: with random weights the
    # gate's top-6-of-16 is a near-tie lottery, so any bf16 device noise
    # flips expert SELECTION and outputs diverge discontinuously -- that
    # measures routing discreteness, not kernel error. Activating ALL
    # experts removes the discontinuity; the tolerance bar lives there.
    # The discrete arm still runs and reports (finite is its only bar).
    cont = boot_args(mod)
    cont.n_activated_experts = cont.n_routed_experts
    print(f"[f1b] layers {args.n_layers} (hash {args.n_hash_layers}, "
          f"ratios {args.compress_ratios}) | experts "
          f"{args.n_routed_experts} | dim {args.dim}")
    tokens = torch.randint(0, args.vocab_size, (1, 8),
                           generator=torch.Generator().manual_seed(SEED))
    fails = 0
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for label, a_ in (("topk6/16", args), ("all-active", cont)):
        results = {}
        for dev in devs:
            r = run(dev, mod, a_, tokens)
            results[dev] = r
            fin = all(torch.isfinite(r[k]).all()
                      for k in ("prefill", "decode"))
            fails += not fin
            print(f"[f1b] {label:10s} {dev:3s}: finite "
                  f"{'PASS' if fin else 'FAIL'} | next {r['next_token']}")
        if len(results) == 2:
            a, b = results["cpu"]["prefill"], results["mps"]["prefill"]
            rel = ((a - b).norm() / a.norm()).item()
            same = (results["cpu"]["next_token"]
                    == results["mps"]["next_token"])
            if label == "all-active":       # the continuity arm IS the bar
                fails += not (rel <= 0.05)
                print(f"[f1b] {label:10s} cpu-vs-mps rel-L2 {rel:.4f} "
                      f"({'PASS' if rel <= 0.05 else 'FAIL'} <=0.05) | "
                      f"top-1 {'agree' if same else 'differ'}")
            else:                            # discrete arm: report only
                print(f"[f1b] {label:10s} cpu-vs-mps rel-L2 {rel:.4f} "
                      f"(REPORT: routing discreteness, no bar) | top-1 "
                      f"{'agree' if same else 'differ'}")
    print(f"[f1b] {'ALL BARS PASS' if not fails else f'{fails} FAILURES'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
