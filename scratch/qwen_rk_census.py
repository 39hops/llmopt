"""QWEN-RK-CENSUS-0: is the 2-bit artifact a faithful ROUTER?

Registered design (PRE-REG QWEN-RK-CENSUS-0): compressed A v the
PINNED VENDOR teacher (never A-v-CPU-reference — that is a backend
parity test), coupled FFN channels, teacher-forced on the FROZEN
corpus + prefix token lists only. The hot signal at layer l,
position t is z = act(W_gate h) * (W_up h) — captured as the INPUT
to down_proj via a pre-hook. Frozen sampled layer INDICES:
{0, 21, 42, 11, 32, 55} — family labels corrected post-freeze
against the manifest (0/21/42/32 linear-attn, 11/55 full-attn;
the pre-reg labeled 32 full by mistake — indices frozen, label
disclosed at booking; the manifest's 17th "self_attn" layer is
mtp.layers.0, an EXCLUDED module). Frozen k ladder {64, 256,
1024, 4096} of the 17,408 channels.

Two capture legs, then analysis:
    MODE=vendor .venv_teacher/bin/python scratch/qwen_rk_census.py
    MODE=arm ART_DIR=~/qwen_whole0t/A .venv_teacher/bin/python ...
    MODE=analyze .venv/bin/python scratch/qwen_rk_census.py

Per layer/position, analysis reports: Jaccard overlap@k of top-|z|
sets; R_k = teacher-|z|-mass captured by A's top-k (the decisive
number); and teacher-FFN-OUTPUT reconstruction error using
A-selected v teacher-oracle v random-k channels (y_S = down_proj
restricted to S applied to the TEACHER z). Weight/activation-space
census — no capability claim. Receipt: logs/qwenrouter/rk_census.json.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

MODE = os.environ.get("MODE", "analyze")
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
EV = "evals/qwen_model1"
TEACHER_DIR = "logs/qwenteacher_v2"
LAYERS = (0, 21, 42, 11, 32, 55)
KS = (64, 256, 1024, 4096)
OUT = "logs/qwenrouter"


def frozen_token_lists():
    man = json.load(open(os.path.join(TEACHER_DIR,
                                      "teacher_manifest.json")))
    seqs = [man["records"]["corpus"]["tokens"]]
    seqs += man["records"]["prefixes"]["tokens"]
    return seqs


def capture(build_model, tag):
    """Run the frozen sequences through a model, capturing down_proj
    INPUT (= z) at the sampled layers. Saves fp32 [n_positions, C]
    per layer (fp16 would inject top-k sensitivity for ~2x space —
    total is ~181 MiB/model, cheap) plus a capture-time PROVENANCE
    SIDECAR the analyzer fails closed on."""
    import torch
    os.makedirs(OUT, exist_ok=True)
    for li in LAYERS:
        p = os.path.join(OUT, f"z_{tag}_L{li}.npy")
        if os.path.exists(p):
            raise SystemExit(f"REFUSING: {p} exists")
    meta_path = os.path.join(OUT, f"capture_{tag}_meta.json")
    if os.path.exists(meta_path):
        raise SystemExit(f"REFUSING: {meta_path} exists")
    model = build_model()
    store = {li: [] for li in LAYERS}

    def make_hook(li):
        def h(module, args, kwargs):
            z = args[0] if args else kwargs["input"]
            store[li].append(z.detach()[0].float().numpy())
            return None
        return h

    for li in LAYERS:
        model.model.layers[li].mlp.down_proj.register_forward_pre_hook(
            make_hook(li), with_kwargs=True)
    seqs = frozen_token_lists()
    for ids in seqs:
        t = time.time()
        model(input_ids=torch.tensor([ids]), use_cache=False)
        print(f"[rk] {tag} len {len(ids)} {time.time()-t:.0f}s",
              flush=True)
    shas = {}
    for li in LAYERS:
        z = np.concatenate(store[li])
        p = os.path.join(OUT, f"z_{tag}_L{li}.npy")
        np.save(p, z)
        shas[f"z_{tag}_L{li}"] = hashlib.sha256(
            z.tobytes()).hexdigest()
        print(f"[rk] saved z_{tag}_L{li} {z.shape}", flush=True)
    meta = {"tag": tag, "layers": list(LAYERS), "dtype": "float32",
            "n_sequences": len(seqs),
            "token_lists_sha256": hashlib.sha256(json.dumps(
                seqs).encode()).hexdigest(),
            "z_sha256": shas,
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "--short",
                 "HEAD"]).decode().strip(),
            "tree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain",
                 "-uno"]).decode().strip()),
            "vendor_dir": VDIR,
            "vendor_index_sha256": hashlib.sha256(open(os.path.join(
                VDIR, "model.safetensors.index.json"),
                "rb").read()).hexdigest()}
    if tag == "arm":
        art = os.path.expanduser(os.environ["ART_DIR"])
        arm = os.path.basename(art.rstrip("/"))
        chain = f"logs/qwenwhole/artifact_digest_{arm}.txt"
        meta["artifact"] = {"dir": art, "arm": arm,
                            "chain_sha256": hashlib.sha256(open(
                                chain, "rb").read()).hexdigest()}
    with open(meta_path, "w") as f:
        f.write(json.dumps(meta, indent=1) + "\n")
    print(f"[rk] sidecar -> {meta_path}", flush=True)


def build_vendor():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen_teacher_pass", "scratch/qwen_teacher_pass.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    model, _trav = tp.build_streamed_model()
    return model


def build_arm():
    # runtime0r qualifies + builds on import-time globals; env set by
    # the caller (ART_DIR). Reuse its build() verbatim.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen_runtime0r", "scratch/qwen_runtime0r.py")
    r0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r0)
    model, _trav = r0.build()
    return model


def analyze():
    from safetensors import safe_open
    import torch
    # FAIL-CLOSED protocol check: both capture sidecars must exist,
    # agree on layers/dtype/token-list hash, match every z file's
    # recomputed sha, and (arm side) name the artifact chain
    metas = {}
    for tag in ("vendor", "arm"):
        mp = os.path.join(OUT, f"capture_{tag}_meta.json")
        if not os.path.exists(mp):
            raise SystemExit(f"REFUSING: no sidecar {mp}")
        metas[tag] = json.load(open(mp))
    tok_sha = hashlib.sha256(json.dumps(
        frozen_token_lists()).encode()).hexdigest()
    for tag, m in metas.items():
        if m["layers"] != list(LAYERS) or m["dtype"] != "float32":
            raise SystemExit(f"REFUSING: {tag} protocol mismatch "
                             f"{m['layers']}/{m['dtype']}")
        if m["token_lists_sha256"] != tok_sha:
            raise SystemExit(f"REFUSING: {tag} token lists differ "
                             "from the locked teacher lists")
        for name, sha in m["z_sha256"].items():
            z = np.load(os.path.join(OUT, name + ".npy"))
            if hashlib.sha256(z.tobytes()).hexdigest() != sha:
                raise SystemExit(f"REFUSING: {name} sha mismatch")
            del z
    if "artifact" not in metas["arm"]:
        raise SystemExit("REFUSING: arm sidecar lacks artifact block")
    idx = json.load(open(os.path.join(
        VDIR, "model.safetensors.index.json")))["weight_map"]
    rng = np.random.default_rng(20260818)
    per_layer = {}
    for li in LAYERS:
        zt = np.load(os.path.join(OUT, f"z_vendor_L{li}.npy"))
        za = np.load(os.path.join(OUT, f"z_arm_L{li}.npy"))
        assert zt.shape == za.shape, (zt.shape, za.shape)
        key = f"model.language_model.layers.{li}.mlp.down_proj.weight"
        with safe_open(os.path.join(VDIR, idx[key]), framework="pt",
                       device="cpu") as h:
            Wd = h.get_tensor(key).float().numpy()   # [5120, 17408]
        P, C = zt.shape
        # selection strategies: arm top-|z|; teacher top-|z| (NOT an
        # exact subset optimum — named honestly); teacher
        # CONTRIBUTION ranking |z_i| * ||Wd[:,i]|| (the better greedy
        # proxy for FFN-output mass; still not the combinatorial
        # optimum); random-k
        wd_col = np.linalg.norm(Wd, axis=0)          # [C]
        res = {k: {"jaccard": [], "r_k": [],
                   "recon_arm": [], "recon_teacher_topz": [],
                   "recon_teacher_contrib": [],
                   "recon_random": []} for k in KS}
        for t in range(P):
            zt32 = zt[t].astype(np.float32)
            za32 = za[t].astype(np.float32)
            y = Wd @ zt32
            ynorm = float(np.linalg.norm(y)) or 1e-30
            amag = np.abs(za32)
            tmag = np.abs(zt32)
            cmag = tmag * wd_col
            tsum = float(tmag.sum()) or 1e-30
            for k in KS:
                Sa = np.argpartition(-amag, k)[:k]
                St = np.argpartition(-tmag, k)[:k]
                Sc = np.argpartition(-cmag, k)[:k]
                Sr = rng.choice(C, k, replace=False)
                inter = len(np.intersect1d(Sa, St,
                                           assume_unique=True))
                res[k]["jaccard"].append(inter / (2 * k - inter))
                res[k]["r_k"].append(float(tmag[Sa].sum()) / tsum)
                for nm, S in (("recon_arm", Sa),
                              ("recon_teacher_topz", St),
                              ("recon_teacher_contrib", Sc),
                              ("recon_random", Sr)):
                    yS = Wd[:, S] @ zt32[S]
                    res[k][nm].append(
                        float(np.linalg.norm(yS - y)) / ynorm)
        per_layer[li] = {
            "n_positions": int(P),
            "ks": {str(k): {nm: round(float(np.mean(v)), 4)
                            for nm, v in res[k].items()}
                   for k in KS}}
        print(f"[rk] L{li}: " + " | ".join(
            f"k={k} R_k={per_layer[li]['ks'][str(k)]['r_k']:.3f} "
            f"jac={per_layer[li]['ks'][str(k)]['jaccard']:.3f}"
            for k in KS), flush=True)
    rcpt = {"gate": "QWEN-RK-CENSUS-0", "layers": list(LAYERS),
            "ks": list(KS),
            "per_layer": per_layer,
            "z_files_sha256": {
                f"z_{tag}_L{li}": hashlib.sha256(open(os.path.join(
                    OUT, f"z_{tag}_L{li}.npy"), "rb").read()).hexdigest()
                for tag in ("vendor", "arm") for li in LAYERS},
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
            "tree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain",
                 "-uno"]).decode().strip())}
    p = os.path.join(OUT, "rk_census.json")
    if os.path.exists(p):
        raise SystemExit(f"REFUSING: {p} exists")
    with open(p, "w") as f:
        f.write(json.dumps(rcpt, indent=1) + "\n")
    print(f"[rk] -> {p}", flush=True)


def main():
    if MODE == "vendor":
        import torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(os.cpu_count())
        capture(build_vendor, "vendor")
    elif MODE == "arm":
        import torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(os.cpu_count())
        capture(build_arm, "arm")
    elif MODE == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown MODE {MODE}")


if __name__ == "__main__":
    main()
