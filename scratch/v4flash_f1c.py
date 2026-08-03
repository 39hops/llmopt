"""F1c (PRE-REG V4-F1): REAL weights into the vendor model — embed +
layers 0-2 (all hash-routed, so expert demand is EXACT from tid2eid),
prefill + one decode step on cpu and mps.

Loader facts (verified against shard headers, RECEIPT V4-CENSUS):
shards ship VENDOR-NAMED tensors ("layers.N.attn...", "embed.weight",
scales as ".scale") — convert.py's HF renames are no-ops here. The two
real transforms are wo_a (block-128x128 fp8 -> bf16 dequant, the
convert.py:123-127 semantics) and expert weights (I8 bytes -> packed
fp4 view). tid2eid ships I64 and the model registers int32.

Budget honesty: the pre-reg estimated ~1.5 GB of fetches. Measured
here and printed at exit; the head is NOT fetched (zeroed instead,
+1.06 GB saved) so logits are plumbing-only — the bars live in the
hidden states and the expert oracle, as registered.

BARS (registered): finite everywhere; per-layer hidden-state rms in a
plausible band; the twin's output for one REAL demanded expert within
bf16 rounding of the exact fp64 dequant reference. REPORTED (not a
bar): cpu-vs-mps per-layer rel-L2 with real weights.

Usage: .venv/bin/python scratch/v4flash_f1c.py
"""
import json
import os
import sys
import urllib.request

import torch

sys.path.insert(0, "scratch")
import v4flash_twin  # noqa: E402
from v4flash_f1b import VENDOR, load_vendor_model_module  # noqa: E402

REPO = ("https://huggingface.co/deepseek-ai/"
        "DeepSeek-V4-Flash-0731/resolve/main")
CACHE = "checkpoints/v4flash_f1"
SEED = 20260803
NL = 3
PROMPT_IDS = [0, 3271, 5426, 315, 9622, 374]     # short; ids arbitrary-ok
fetched_bytes = 0


def _get(url, lo=None, hi=None):
    req = urllib.request.Request(url)
    if lo is not None:
        req.add_header("Range", f"bytes={lo}-{hi - 1}")
    with urllib.request.urlopen(req) as r:
        raw = r.read()
    if lo is not None:
        assert len(raw) == hi - lo, f"truncated: {len(raw)} != {hi - lo}"
    return raw


def index_map():
    """tensor name -> (shard file, [lo, hi]) from per-shard headers of
    the shards that carry embed + layers 0-2 (1..5 by census layout)."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "manifest_l012.json")
    if os.path.exists(p):
        return json.load(open(p))
    import struct
    m = {}
    for s in range(1, 6):
        fn = f"model-{s:05d}-of-00048.safetensors"
        hlen = struct.unpack("<Q", _get(f"{REPO}/{fn}", 0, 8))[0]
        hdr = json.loads(_get(f"{REPO}/{fn}", 8, 8 + hlen))
        hdr.pop("__metadata__", None)
        for k, v in hdr.items():
            m[k] = [fn, 8 + hlen, v["data_offsets"], v["dtype"], v["shape"]]
    json.dump(m, open(p, "w"))
    return m


def fetch(man, name):
    """Sha-side-cached byte-range fetch with the length assert."""
    global fetched_bytes
    p = os.path.join(CACHE, name.replace("/", "_") + ".bin")
    if not os.path.exists(p):
        fn, base, (lo, hi), _, _ = man[name]
        raw = _get(f"{REPO}/{fn}", base + lo, base + hi)
        fetched_bytes += len(raw)
        with open(p, "wb") as f:
            f.write(raw)
    return open(p, "rb").read()


DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3":
      torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu,
      "I8": torch.uint8, "I64": torch.int64}


def tensor(man, name):
    fn, base, off, dt, shape = man[name]
    raw = bytearray(fetch(man, name))
    t = torch.frombuffer(raw, dtype=DT[dt]).reshape(shape)
    return t


def demanded_experts(man, ids):
    """Exact per-layer expert demand for hash layers, from tid2eid."""
    out = {}
    for lay in range(NL):
        tid = tensor(man, f"layers.{lay}.ffn.gate.tid2eid")
        out[lay] = sorted(set(tid[ids].reshape(-1).tolist()))
    return out


def load_real(model, man, demand):
    """Load embed + layers 0-2 dense + demanded experts; zero the head."""
    sd = dict(model.named_parameters())
    loaded, skipped = 0, 0
    for name in man:
        if not (name == "embed.weight" or any(
                name.startswith(f"layers.{L}.") for L in range(NL))):
            continue
        if ".experts." in name and ".shared_experts." not in name:
            lay, idx = int(name.split(".")[1]), int(name.split(".")[4])
            if idx not in demand[lay]:
                skipped += 1
                continue
        t = tensor(man, name)
        if name.endswith("wo_a.weight"):        # convert.py:123-127
            scale = tensor(man, name.replace("weight", "scale"))
            w = (t.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
                 * v4flash_twin._e8m0_to_f32(scale)[:, None, :, None])
            sd[name].data.copy_(w.flatten(2, 3).flatten(0, 1).bfloat16())
            loaded += 1
            continue
        if name.endswith("wo_a.scale"):
            continue                             # folded into wo_a.weight
        if name not in sd:
            skipped += 1
            continue
        p = sd[name]
        if p.dtype == torch.int32:               # tid2eid ships I64
            p.data.copy_(t.to(torch.int32))
        elif hasattr(torch, "float4_e2m1fn_x2") and \
                p.dtype == torch.float4_e2m1fn_x2:
            p.data.copy_(t.view(torch.float4_e2m1fn_x2))
        elif p.dtype == t.dtype:
            p.data.copy_(t.reshape(p.shape))
        else:
            # checkpoint dtype != module dtype (e.g. bf16-stored fp32
            # norms): the vendor's load_model CASTS on copy; a byte
            # view would reinterpret (risk-scan N6). CPU casts fp8/bf16
            # fine.
            p.data.copy_(t.float().to(p.dtype).reshape(p.shape))
        loaded += 1
    # head + final norm are NOT fetched (budget): zero/one them so the
    # logits are finite plumbing, not model output. Disclosed in the bars.
    with torch.no_grad():
        model.head.weight.zero_()
        model.norm.weight.fill_(1.0)
        for n, p in sd.items():
            if n.startswith("hc_head"):
                p.data.copy_(torch.zeros_like(p) if "base" in n or
                             "fn" in n else torch.ones_like(p))
    return loaded, skipped


def run(dev, mod, args, man, demand, ids):
    torch.set_default_device(dev)
    torch.set_default_dtype(torch.bfloat16)
    for fn in (mod.precompute_freqs_cis, mod.get_window_topk_idxs):
        fn.cache_clear()
    model = mod.Transformer(args)
    loaded, skipped = load_real(model, man, demand)
    model.eval()
    model.temperature = 0.0
    hs = []
    for blk in model.layers:
        blk.register_forward_hook(
            lambda m, i, o, h=hs: h.append(o.float().mean(2).cpu()))
    toks = torch.tensor([ids], device=dev)
    with torch.inference_mode():
        out_ids, logits, _ = model.forward(toks, start_pos=0)
        nxt = int(out_ids.reshape(-1)[-1])
        model.forward(torch.tensor([[nxt]], device=dev),
                      start_pos=toks.size(1))
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    return {"h": hs[:NL], "logits": logits.float().cpu(),
            "loaded": loaded, "skipped": skipped}


def main():
    mod = load_vendor_model_module()
    man = index_map()
    cfg = json.load(open(os.path.join(VENDOR, "config.json")))
    cfg.update(n_layers=NL, n_mtp_layers=0, dspark_block_size=0,
               dspark_target_layer_ids=[],
               compress_ratios=cfg["compress_ratios"][:NL],
               max_batch_size=1, max_seq_len=512)
    known = set(mod.ModelArgs.__dataclass_fields__)
    args = mod.ModelArgs(**{k: v for k, v in cfg.items() if k in known})
    assert args.n_hash_layers >= NL, "layers 0-2 must all be hash-routed"
    demand = demanded_experts(man, torch.tensor(PROMPT_IDS))
    n_dem = sum(len(v) for v in demand.values())
    print(f"[f1c] prompt {len(PROMPT_IDS)} ids -> demanded experts/layer "
          f"{[len(demand[k]) for k in demand]} ({n_dem} total, "
          f"{n_dem * 13.37:.0f} MB)")

    fails, res = 0, {}
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devs:
        r = run(dev, mod, args, man, demand, PROMPT_IDS)
        res[dev] = r
        fin = all(torch.isfinite(h).all() for h in r["h"]) and \
            torch.isfinite(r["logits"]).all()
        rms = [float(h.pow(2).mean().sqrt()) for h in r["h"]]
        band = all(1e-3 <= v <= 1e3 for v in rms)
        fails += (not fin) + (not band)
        print(f"[f1c] {dev:3s}: loaded {r['loaded']} skipped {r['skipped']}"
              f" | finite {'PASS' if fin else 'FAIL'} | per-layer h rms "
              f"{[f'{v:.3f}' for v in rms]} band "
              f"{'PASS' if band else 'FAIL'}")
    if len(res) == 2:
        for L in range(NL):
            a, b = res["cpu"]["h"][L], res["mps"]["h"][L]
            print(f"[f1c] layer {L} cpu-vs-mps rel-L2 "
                  f"{((a - b).norm() / a.norm()).item():.5f} (REPORT)")

    # expert oracle: one REAL demanded expert vs exact fp64 reference
    lay, eid = 0, demand[0][0]
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(4, 4096, generator=g).bfloat16()
    a, a_s = v4flash_twin.act_quant(x, 128, "ue8m0",
                                    torch.float8_e8m0fnu)
    name = f"layers.{lay}.ffn.experts.{eid}.w1"
    w = tensor(man, name + ".weight")
    ws = tensor(man, name + ".scale").view(torch.float8_e8m0fnu)
    got = v4flash_twin.fp4_gemm(a, a_s, w, ws).double()
    ref = (v4flash_twin._deq_act(a, a_s, 128).double()
           @ (v4flash_twin._unpack_fp4(w).double()
              * v4flash_twin._e8m0_to_f32(ws).double()
              .repeat_interleave(32, dim=1)).T)
    rel = ((got - ref).abs() / ref.abs().clamp(min=1e-2)).max().item()
    ok = rel <= 1 / 128
    fails += not ok
    print(f"[f1c] expert oracle L{lay}/e{eid}: rel {rel:.2e} "
          f"({'PASS' if ok else 'FAIL'} <=1/128)")
    print(f"[f1c] bytes fetched this run: {fetched_bytes / 1e9:.2f} GB "
          f"(cache {sum(os.path.getsize(os.path.join(CACHE, f)) for f in os.listdir(CACHE)) / 1e9:.2f} GB)")
    print(f"[f1c] {'ALL BARS PASS' if not fails else f'{fails} FAILURES'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
