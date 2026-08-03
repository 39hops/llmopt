"""F1d (PRE-REG V4-F1): DeepSeek-V4-Flash GENERATES TOKENS ON THE MAC —
full 43-layer dense path resident, K experts/layer subset-resident,
greedy decode, tok/s and RSS measured, text logged verbatim.

Design, per the pre-reg + F1c loader facts:
  * dense path loaded whole (fp8 stays fp8-stored; the twin dequants
    via uint8 LUT per use — MPS cannot compute on fp8 natively);
  * score layers (3..42): the K most-NEGATIVE-bias experts resident
    (under noaux_tc the balancer pushes naturally strong experts DOWN,
    so most-negative bias = most naturally demanded), and masking is
    ZERO code change: bias.data[non-resident] = -1e9 — selection uses
    scores + bias, output weights gather the UNBIASED scores, so
    masked experts are simply never picked and renormalization is the
    vendor's own weights /= weights.sum();
  * hash layers (0..2): exact tid2eid demand for the prompt, plus
    FETCH-ON-MISS during decode (a generated token's demand is
    unknowable in advance; a miss costs one ~13 MB range fetch);
  * MPS move is PER-PARAMETER for loaded tensors only — model.to()
    would materialize all 43x256 garbage expert allocations (risk-scan
    B1);
  * MTP skipped (dspark_block_size=0); temperature 0.

FENCES (pre-reg): NO capability claim. K/256 residency is far below
the house pruning cliff; degraded text is the EXPECTED outcome. The
deliverables are: it runs, tok/s, RSS, and honest verbatim samples.

Env: K (default 16), NTOK (default 64), DEV (cpu|mps, default mps),
     PROMPT (default a fixed sentence).
Usage: .venv/bin/python scratch/v4flash_f1d.py
"""
import json
import os
import resource
import struct
import sys
import time

import torch

sys.path.insert(0, "scratch")
import v4flash_twin  # noqa: E402
from v4flash_f1b import VENDOR, load_vendor_model_module  # noqa: E402
from v4flash_f1c import DT, _get, fetch  # noqa: E402

REPO = ("https://huggingface.co/deepseek-ai/"
        "DeepSeek-V4-Flash-0731/resolve/main")
CACHE = "checkpoints/v4flash_f1"
K = int(os.environ.get("K", "16"))
NTOK = int(os.environ.get("NTOK", "64"))
DEV = os.environ.get("DEV", "mps" if torch.backends.mps.is_available()
                     else "cpu")
PROMPT = os.environ.get(
    "PROMPT", "The three most important ideas in computer science are")
SEED = 20260803
NL, NE = 43, 256
DEQ = os.environ.get("DEQ", "")          # "bf16": F1e arm 1
PROFILE = os.environ.get("PROFILE", "") == "1"
BATCH = os.environ.get("BATCH", "") == "1"   # F1e arm 5 (rider 3)
RECALL = os.environ.get("RECALL", "") == "1"  # F2a instrument
ORACLE = os.environ.get("ORACLE", "")         # F2b: demand-json row idx
RULE = os.environ.get("RULE", "neg")          # neg | pos bias rank (F2c)
OUT = "logs/opus/v4_f1d.jsonl"


def manifest():
    p = os.path.join(CACHE, "manifest_all.json")
    if os.path.exists(p):
        return json.load(open(p))
    m = {}
    for s in range(1, 49):
        fn = f"model-{s:05d}-of-00048.safetensors"
        hlen = struct.unpack("<Q", _get(f"{REPO}/{fn}", 0, 8))[0]
        hdr = json.loads(_get(f"{REPO}/{fn}", 8, 8 + hlen))
        hdr.pop("__metadata__", None)
        for k, v in hdr.items():
            m[k] = [fn, 8 + hlen, v["data_offsets"], v["dtype"], v["shape"]]
        print(f"[f1d] header {s}/48", flush=True)
    json.dump(m, open(p, "w"))
    return m


def tensor(man, name):
    _, _, _, dt, shape = man[name]
    return torch.frombuffer(bytearray(fetch(man, name)),
                            dtype=DT[dt]).reshape(shape)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30


class ExpertProvider:
    """Loads one routed expert's three tensors into the module tree,
    moving them to the run device. Counts fetch-on-miss events."""

    def __init__(self, model, man, dev):
        self.model, self.man, self.dev = model, man, dev
        self.resident = {L: set() for L in range(NL)}
        self.misses = 0

    def load(self, lay, eid, miss=False):
        if eid in self.resident[lay]:
            return
        exp = self.model.layers[lay].ffn.experts[eid]
        for proj in ("w1", "w2", "w3"):
            base = f"layers.{lay}.ffn.experts.{eid}.{proj}"
            w = tensor(self.man, base + ".weight")
            s = tensor(self.man, base + ".scale")
            m = getattr(exp, proj)
            m.weight.data = w.view(torch.float4_e2m1fn_x2).to(self.dev)
            sc = s.view(torch.float8_e8m0fnu).to(self.dev)
            m.scale.data = sc
            m.weight.scale = m.scale     # re-tie after data swap
        self.resident[lay].add(eid)
        self.misses += miss

    def ensure_hash(self, tid2eid, ids, miss=True):
        """miss=False for the prompt's initial preload — V4-F1d booked
        the caveat that preloads were being counted as misses."""
        for lay in range(3):
            need = set(tid2eid[lay][ids].reshape(-1).tolist())
            for eid in need - self.resident[lay]:
                self.load(lay, eid, miss=miss)


def choose_residents(man):
    """Score layers, two rules:
    default: K most-NEGATIVE-bias experts — RETAINED FOR THE RECORD but
      measured ANTI-selective (VERDICT V4-F2a: recall 0.035, below the
      random floor; the deployed selection ADDS the bias, so these are
      the most-penalized experts — V4-F1c's direction pin was wrong);
    ORACLE=<jsonl row index>: the K most-DEMANDED experts per layer from
      a prior RECALL run's logged demand counts (F2b profile-then-swap).
    """
    keep = {}
    if ORACLE:
        rows = [l for l in open(OUT)]
        row = __import__("json").loads(rows[int(ORACLE)])
        dem = row["demand_counts"]
        for lay in range(3, NL):
            d = {int(k): v for k, v in dem.get(str(lay), {}).items()}
            top = sorted(d, key=lambda e: d[e], reverse=True)[:K]
            while len(top) < K:          # pad rare layers deterministically
                for e in range(NE):
                    if e not in top:
                        top.append(e)
                        break
            keep[lay] = sorted(top)
        return keep
    for lay in range(3, NL):
        b = tensor(man, f"layers.{lay}.ffn.gate.bias").float()
        order = torch.argsort(b)
        pick = order[-K:] if RULE == "pos" else order[:K]
        keep[lay] = sorted(pick.tolist())
    return keep


def load_dense(model, man, dev):
    """Everything except routed experts, straight onto dev."""
    sd = dict(model.named_parameters())
    n = 0
    for name in man:
        if ".experts." in name and ".shared_experts." not in name:
            continue
        if name.startswith("mtp."):
            continue
        if name.endswith("wo_a.weight"):
            s = tensor(man, name.replace("weight", "scale"))
            t = tensor(man, name)
            w = (t.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
                 * v4flash_twin._e8m0_to_f32(s)[:, None, :, None])
            sd[name].data = w.flatten(2, 3).flatten(0, 1).bfloat16().to(dev)
            n += 1
            continue
        if name.endswith("wo_a.scale") or name not in sd:
            continue
        p, t = sd[name], tensor(man, name)
        if DEQ == "bf16" and p.dtype == torch.float8_e4m3fn:
            # F1e arm 1: value-exact fp8 -> bf16 (per-128x128 block
            # scales); model.py's dtype dispatch then takes F.linear.
            sc = tensor(man, name.replace("weight", "scale"))
            O, I = t.shape
            w = (v4flash_twin._f8_to_f32(t)
                 .reshape(O // 128, 128, I // 128, 128)
                 * v4flash_twin._e8m0_to_f32(sc)[:, None, :, None])
            p.data = w.reshape(O, I).bfloat16().to(dev)
            n += 1
            continue
        if DEQ == "bf16" and name.endswith(".scale"):
            wname = name[:-6] + ".weight"
            if wname in sd and sd[wname].dtype == torch.bfloat16:
                continue                 # scale folded into the weight
        if p.dtype == torch.int32:
            p.data = t.to(torch.int32).to(dev)
        elif p.dtype == t.dtype:
            p.data = t.reshape(p.shape).to(dev)
        else:
            p.data = t.float().to(p.dtype).reshape(p.shape).to(dev)
        if name.endswith(".scale") and name[:-6] + ".weight" in sd:
            w = sd[name[:-6] + ".weight"]
            w.scale = p                   # re-tie weight.scale after swap
        n += 1
        if n % 200 == 0:
            print(f"[f1d] dense {n} tensors, rss {rss_gb():.1f} GB",
                  flush=True)
    return n


def install_batched_moe(mod):
    """F1e arm 5 (RIDER 3): batch the <= 6 hit experts of a SINGLE-TOKEN
    step into one unpack + one bmm per projection, replicating the
    vendor Expert math exactly (fp32; up clamped both sides, gate
    max-only; silu(gate)*up; gate weights before w2). Multi-token calls
    (prefill) fall back to the vendor loop — decode is where the
    measured 84% lives. Also removes the vendor's per-layer hidden
    counts.tolist() device sync on the batched path. Vendor SOURCE is
    untouched; this replaces vendor LOGIC at runtime, hence the
    equivalence bar run at startup (see check_batched_equiv)."""
    import torch.nn.functional as F
    tw = v4flash_twin
    orig = mod.MoE.forward

    def fwd(self, x, input_ids):
        shape = x.size()
        xf = x.view(-1, self.dim)
        if xf.size(0) != 1:
            return orig(self, x, input_ids)
        weights, indices = self.gate(xf, input_ids.flatten())
        hits = indices[0].tolist()                 # 6 distinct experts
        exps = [self.experts[i] for i in hits]
        a, a_s = tw.act_quant(xf, 128, "ue8m0", torch.float8_e8m0fnu)
        af = tw._deq_act(a, a_s, 128)                      # [1, D] fp32

        def stack_w(attr, rows, cols):
            wb = torch.stack([getattr(e, attr).weight.view(torch.uint8)
                              for e in exps])              # [6,rows,cols/2]
            sc = torch.stack([getattr(e, attr).scale for e in exps])
            wf = tw._unpack_fp4(wb)                        # [6,rows,cols]
            # fp32, matching the vendor twin's fp32 matmul exactly —
            # the first launch failed its equivalence gate at 3.1e-3
            # because these were bf16 (accumulation inputs rounded).
            return (wf.reshape(len(exps), rows, cols // 32, 32)
                    * tw._scale_f32(sc).unsqueeze(-1)).reshape(
                len(exps), rows, cols)
        W1 = stack_w("w1", self.experts[hits[0]].w1.out_features,
                     self.dim)
        W3 = stack_w("w3", W1.size(1), self.dim)
        # the vendor path rounds EVERY projection output to bf16 (the
        # twin's get_default_dtype cast) before Expert.forward's
        # .float() — reproduce those roundings or equivalence fails
        # (measured: 1.1e-2 without them).
        gate = torch.einsum("td,eid->ei", af, W1)\
            .to(torch.bfloat16).float()                    # [6, I]
        up = torch.einsum("td,eid->ei", af, W3)\
            .to(torch.bfloat16).float()
        lim = exps[0].swiglu_limit
        if lim > 0:
            up = up.clamp(-lim, lim)
            gate = gate.clamp(max=lim)
        h = (F.silu(gate) * up) * weights[0, :, None].float()  # [6, I]
        h = h.to(x.dtype)
        ha, ha_s = tw.act_quant(h, 128, "ue8m0", torch.float8_e8m0fnu)
        hf = tw._deq_act(ha, ha_s, 128)                        # [6, I] fp32
        W2 = stack_w("w2", self.dim, W1.size(1))
        # per-expert bf16 rounding BEFORE the sum, as the vendor loop does
        y = (torch.einsum("ei,edi->ed", hf, W2).to(torch.bfloat16)
             .float().sum(0, keepdim=True))
        y = y + self.shared_experts(xf).float()
        return y.type_as(xf).view(shape)

    mod.MoE.forward = fwd
    mod.MoE._vendor_forward = orig


def check_batched_equiv(model, dev):
    """Equivalence bar (RIDER 3): patched vs vendor MoE on a real
    loaded layer, same routing, single token. rel L2 <= 1e-3 gates."""
    moe = model.layers[22].ffn
    g = torch.Generator().manual_seed(SEED)
    x = (torch.randn(1, 1, 4096, generator=g, device="cpu") * 0.5)\
        .to(dev).to(torch.bfloat16)
    ids = torch.tensor([5426], device=dev)
    with torch.inference_mode():
        yb = type(moe).forward(moe, x, ids)
        yv = type(moe)._vendor_forward(moe, x, ids)
    rel = ((yb.float() - yv.float()).norm() / yv.float().norm()).item()
    print(f"[f1d] BATCH equivalence rel-L2 {rel:.2e} "
          f"({'PASS' if rel <= 1e-3 else 'FAIL'} <=1e-3)", flush=True)
    assert rel <= 1e-3, "batched MoE diverges from the vendor loop"


def install_recall(model, mod, orig_bias):
    """F2a READOUT 2: per score-layer per token, log the UNMASKED top-6
    (scores + ORIGINAL bias) and expert recall vs the resident set.
    The wrapper recomputes selection identically for the masked path —
    asserted against the unwrapped gate on first use (equivalence
    fence)."""
    stats = {"hits": 0, "slots": 0, "per_layer": {}, "demand": {},
             "checked": False}
    orig_fwd = mod.Gate.forward

    def fwd(self, x, input_ids=None):
        w, idx = orig_fwd(self, x, input_ids)
        if not self.hash and self.bias is not None and x.size(0) == 1:
            lay = getattr(self, "_lay", None)
            if lay is not None:
                import torch as T
                s = T.nn.functional.softplus(
                    (x.float() @ self.weight.float().T)).sqrt()
                un = T.topk(s + orig_bias[lay].to(s.device), self.topk,
                            dim=-1)[1][0].tolist()
                res = set(idx[0].tolist())      # resident == selected set
                hit = len(set(un) & stats["res_sets"][lay])
                stats["hits"] += hit
                stats["slots"] += self.topk
                pl = stats["per_layer"].setdefault(lay, [0, 0])
                pl[0] += hit
                pl[1] += self.topk
                d = stats["demand"].setdefault(lay, {})
                for e in un:
                    d[e] = d.get(e, 0) + 1
                if not stats["checked"]:        # equivalence fence
                    w2, idx2 = orig_fwd(self, x, input_ids)
                    assert T.equal(idx, idx2) and T.equal(w, w2)
                    stats["checked"] = True
        return w, idx
    mod.Gate.forward = fwd
    for li, blk in enumerate(model.layers):
        blk.ffn.gate._lay = li
    return stats


def main():
    os.makedirs("logs/opus", exist_ok=True)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(CACHE, "tokenizer.json"))
    mod = load_vendor_model_module()
    man = manifest()
    cfg = json.load(open(os.path.join(VENDOR, "config.json")))
    cfg.update(n_mtp_layers=0, dspark_block_size=0,
               dspark_target_layer_ids=[], max_batch_size=1,
               max_seq_len=512)
    known = set(mod.ModelArgs.__dataclass_fields__)
    args = mod.ModelArgs(**{k: v for k, v in cfg.items() if k in known})

    ids = tok.encode(PROMPT).ids
    print(f"[f1d] K={K} dev={DEV} prompt {len(ids)} tokens: {PROMPT!r}",
          flush=True)
    t0 = time.time()
    # Construct on CPU: torch.empty there is lazy malloc, so the 43x256
    # garbage expert allocations cost nothing until touched. Under a
    # default device of mps they MATERIALIZE on Metal — 47.7 GiB at
    # __init__, measured (risk-scan B1). Loaded tensors are moved
    # per-parameter; non-resident experts stay as untouched cpu garbage
    # that masked routing never reads.
    torch.set_default_dtype(torch.bfloat16)
    model = mod.Transformer(args)
    model.eval()
    model.temperature = 0.0
    print(f"[f1d] constructed {time.time()-t0:.0f}s rss {rss_gb():.1f} GB",
          flush=True)

    nden = load_dense(model, man, DEV)
    print(f"[f1d] dense loaded: {nden} tensors, {time.time()-t0:.0f}s, "
          f"rss {rss_gb():.1f} GB", flush=True)

    # gate masking by bias: zero code change (see docstring)
    keep = choose_residents(man)
    orig_bias = {}
    prov = ExpertProvider(model, man, DEV)
    with torch.no_grad():
        for lay, eids in keep.items():
            gate = model.layers[lay].ffn.gate
            orig_bias[lay] = gate.bias.data.clone().cpu()
            mask = torch.full((NE,), -1e9, device=DEV)
            mask[torch.tensor(eids, device=DEV)] = 0.0
            gate.bias.data = gate.bias.data + mask
            for eid in eids:
                prov.load(lay, eid)
            if lay % 10 == 0:
                print(f"[f1d] layer {lay} residents loaded, "
                      f"rss {rss_gb():.1f} GB", flush=True)
    tid = {L: tensor(man, f"layers.{L}.ffn.gate.tid2eid") for L in range(3)}
    prov.ensure_hash(tid, torch.tensor(ids), miss=False)
    # buffers (KV/compressor caches, freqs_cis) follow the loaded params;
    # do this BEFORE any forward so the lazy Compressor cache aliasing
    # (risk-scan N4) is set up on the right device.
    for _, b in model.named_buffers():
        b.data = b.data.to(DEV)
    torch.set_default_device(DEV)   # lru_cache'd index builders (H4)
    if BATCH:
        install_batched_moe(mod)
        check_batched_equiv(model, DEV)
    rec = None
    if RECALL:
        rec = install_recall(model, mod, orig_bias)
        rec["res_sets"] = {lay: set(keep[lay]) for lay in keep}
    n_res = sum(len(v) for v in prov.resident.values())
    print(f"[f1d] residents {n_res} experts ({n_res*13.37/1e3:.1f} GB "
          f"packed) | load total {time.time()-t0:.0f}s | "
          f"rss {rss_gb():.1f} GB", flush=True)

    if PROFILE:
        # Component partition of a decode step, SYNCHRONIZED timers
        # (enqueue != complete on MPS — measured 65 vs 108 ms earlier).
        acc = {"attn": 0.0, "ffn": 0.0}

        def timed(fn, key):
            def w(*a, **kw):
                torch.mps.synchronize()
                t = time.perf_counter()
                r = fn(*a, **kw)
                torch.mps.synchronize()
                acc[key] += time.perf_counter() - t
                return r
            return w
        for blk in model.layers:
            blk.attn.forward = timed(blk.attn.forward, "attn")
            blk.ffn.forward = timed(blk.ffn.forward, "ffn")
        toks = torch.tensor([ids], device=DEV)
        with torch.inference_mode():
            o, _, _ = model.forward(toks, start_pos=0)
            cur = int(o.reshape(-1)[-1])
            acc["attn"] = acc["ffn"] = 0.0
            t0p = time.perf_counter()
            for j in range(3):
                prov.ensure_hash(tid, torch.tensor([cur], device="cpu"))
                o, _, _ = model.forward(torch.tensor([[cur]], device=DEV),
                                        start_pos=len(ids) + j)
                torch.mps.synchronize()
                cur = int(o.reshape(-1)[-1])
        tot = (time.perf_counter() - t0p) / 3
        a_, f_ = acc["attn"] / 3, acc["ffn"] / 3
        print(f"[prof] per-token {tot*1e3:.0f} ms | attn {a_*1e3:.0f} ms"
              f" | ffn {f_*1e3:.0f} ms | other {(tot-a_-f_)*1e3:.0f} ms",
              flush=True)
        return
    toks = torch.tensor([ids], device=DEV)
    tp0 = time.time()
    with torch.inference_mode():
        out_ids, _, _ = model.forward(toks, start_pos=0)
    t_prefill = time.time() - tp0
    cur = int(out_ids.reshape(-1)[-1])
    gen, t_dec0 = [cur], time.time()
    with torch.inference_mode():
        for i in range(NTOK - 1):
            prov.ensure_hash(tid, torch.tensor([cur], device="cpu"))
            o, _, _ = model.forward(
                torch.tensor([[cur]], device=DEV),
                start_pos=len(ids) + len(gen) - 1)
            cur = int(o.reshape(-1)[-1])
            gen.append(cur)
            if (i + 1) % 16 == 0:
                print(f"[f1d] {i+1}/{NTOK-1} tokens, "
                      f"{(i+1)/(time.time()-t_dec0):.3f} tok/s", flush=True)
    t_dec = time.time() - t_dec0
    text = tok.decode(gen)
    grams = [tuple(gen[i:i + 4]) for i in range(len(gen) - 3)]
    distinct4 = len(set(grams)) / max(len(grams), 1)
    drv = (torch.mps.driver_allocated_memory() / 2 ** 30
           if DEV == "mps" else 0.0)
    row = {"K": K, "dev": DEV, "deq": DEQ, "ntok": NTOK,
           "recall": (rec["hits"] / max(rec["slots"], 1)) if rec else None,
           "recall_per_layer": ({k: v[0] / max(v[1], 1) for k, v in
                                 sorted(rec["per_layer"].items())}
                                if rec else None),
           "demand_counts": ({str(k): v for k, v in rec["demand"].items()}
                             if rec else None),
           "distinct4": distinct4, "metal_gb": drv, "prompt": PROMPT,
           "prompt_tokens": len(ids), "residents": n_res,
           "hash_misses": prov.misses, "prefill_s": t_prefill,
           "decode_s": t_dec, "tok_per_s": (NTOK - 1) / t_dec,
           "rss_gb": rss_gb(), "load_s": tp0 - t0,
           "generated_ids": gen, "text": text}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\n[f1d] prefill {t_prefill:.1f}s | decode {row['tok_per_s']:.3f}"
          f" tok/s | hash misses {prov.misses} | RSS {row['rss_gb']:.1f} GB")
    print(f"[f1d] TEXT (verbatim): {text!r}")
    print(f"[f1d] RSS bar (<=30 GB): "
          f"{'PASS' if row['rss_gb'] <= 30 else 'FAIL'}")


if __name__ == "__main__":
    main()
