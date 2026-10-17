"""E0: greedy speculative-acceptance census for one compressed arm against
the 24 preserved teacher-greedy rollouts (logs/qwenteacher_v2).

Mac CPU only. The arm is run teacher-forced over prompt + recorded
rollout tokens in ONE forward per prompt, so the draft logits are taken
at exactly the teacher-greedy contexts; position s of rollout i is
compared with rollout_logits[s, i, :v_live]. Nothing is generated and
no weight is modified. Per-position rows (llmopt.lab.spec_reference.
position_stats) and the accepted-prefix replay for k = 1..8 are written
under OUT (refuse-if-exists). Full draft logits are never stored; each
row keeps the draft top-16 only (about 1 MB total).

Env: ART_DIR (default ~/qwen_whole0t/C), VENDOR_DIR (~/qwen_vendor: only
config.json, tokenizer files and model.safetensors.index.json are read),
TEACHER_DIR (logs/qwenteacher_v2, source-pinned), OUT (logs/qwen_e0/<arm>),
SMOKE=1 (one prompt, 8 steps, OUT forced to logs/qwen_e0_smoke/<arm>).
Receipts: prompts, teacher record shas, artifact digest chain, code
commit, wall per forward, peak RSS.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

TEACHER_PINS = {"logs/qwenteacher_v2": "0ca4151"}
TOPK = 16
KS = tuple(range(1, 9))
MARGIN_EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")]


def fsha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=os.path.dirname(HERE)).stdout.strip()


def margin_bin(m: float) -> int:
    for i in range(len(MARGIN_EDGES) - 1):
        if MARGIN_EDGES[i] <= m < MARGIN_EDGES[i + 1]:
            return i
    return len(MARGIN_EDGES) - 2


def main() -> None:
    import torch
    from llmopt.lab import qartifact, qrope, spec_reference as sr
    from llmopt.lab.qcodec import decode_entry
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights

    smoke = os.environ.get("SMOKE") == "1"
    if os.environ.get("DEVICE", "cpu") != "cpu":
        raise SystemExit("REFUSING: E0 runs on cpu only (device rule of the tree measurements)")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    tdir = os.environ.get("TEACHER_DIR", "logs/qwenteacher_v2")
    if tdir not in TEACHER_PINS:
        raise SystemExit(f"REFUSING: no source-frozen pin for teacher {tdir}")
    out = os.environ.get("OUT", f"logs/qwen_e0/{arm}")
    if smoke:
        out = f"logs/qwen_e0_smoke/{arm}"
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists (refuse-if-exists)")

    # ---- teacher records: pinned, sha-checked ----
    man = json.load(open(os.path.join(tdir, "teacher_manifest.json")))
    if man.get("smoke"):
        raise SystemExit("REFUSING: smoke teacher")
    if man["code_commit"] != TEACHER_PINS[tdir]:
        raise SystemExit(f"REFUSING: teacher code_commit {man['code_commit']}")
    ro = man["records"]["rollouts"]
    rl_path = os.path.join(tdir, "rollout_logits.npy")
    rl_sha = fsha(rl_path)
    if rl_sha != ro["sha256"]:
        raise SystemExit("REFUSING: rollout_logits.npy sha != manifest")
    rollouts = np.load(rl_path, mmap_mode="r")           # [steps, n_prompts, V]
    recs = [json.loads(line) for line in open(os.path.join(tdir, "rollout_tokens.jsonl"))]
    input_ids = np.asarray(ro["input_ids"])
    mask = np.asarray(ro["attention_mask"])
    gen_lengths = list(ro["gen_lengths"])
    if ro["padding_side"] != "left":
        raise SystemExit("REFUSING: unexpected padding side")
    n_prompts = len(recs) if not smoke else 1

    # ---- tokenizer, eos ids, live vocab (vendor WEIGHTS are never read) ----
    for f in os.listdir(vdir):
        if f.endswith(".safetensors"):
            raise SystemExit("REFUSING: vendor weight shard present; E0 reads config/tokenizer/index only")
    tok = AutoTokenizer.from_pretrained(vdir)
    gcfg = json.load(open(os.path.join(vdir, "generation_config.json")))
    eos_ids = gcfg.get("eos_token_id", tok.eos_token_id)
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    pad_id = gcfg.get("pad_token_id", tok.pad_token_id)
    v_live = len(tok)
    v_rec = rollouts.shape[2]
    if v_live > v_rec:
        raise SystemExit("REFUSING: live vocab exceeds record width")

    # ---- artifact qualification: digest chain is the identity ----
    chain = f"logs/qwenwhole/artifact_digest_{arm}.txt"
    if not os.path.exists(chain):
        raise SystemExit(f"REFUSING: no digest chain {chain}")
    q = qartifact.qualify_artifact(art, os.path.join(vdir, "model.safetensors.index.json"), chain)
    MAN = q["manifest"]
    print(f"[e0] qualified {arm}: {q['report']}", flush=True)

    handles: dict = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(art, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    def decode(name):
        e = MAN[name]
        if e["codec"] == "excluded":
            raise SystemExit(f"REFUSING: excluded tensor {name}")
        return torch.from_numpy(np.ascontiguousarray(decode_entry(payload(e), e)))

    # ---- streamed model (same construction as scratch/qwen_model1_score.py) ----
    cfg = AutoConfig.from_pretrained(vdir)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    model.eval()
    layer_pref = "model.layers."
    for nm, _ in list(model.named_parameters()):
        if nm.startswith(layer_pref):
            continue
        mod = model.get_submodule(nm.rsplit(".", 1)[0])
        leaf = nm.rsplit(".", 1)[1]
        full = nm if not nm.startswith("model.") else "model.language_model." + nm[len("model."):]
        mod._parameters[leaf] = torch.nn.Parameter(decode(full), requires_grad=False)
    rp = cfg.text_config.rope_parameters
    qrope.check_inv_freq(model.model.rotary_emb.inv_freq.numpy(), float(rp["rope_theta"]),
                         int(cfg.text_config.head_dim * rp.get("partial_rotary_factor", 1.0)))
    layers = model.model.layers
    trav = {"layer_calls": [0] * len(layers),
            "families": ["full_attn" if hasattr(ly, "self_attn") else "linear_attn" for ly in layers]}

    def make_pre(i):
        def pre(module, args, kwargs):
            trav["layer_calls"][i] += 1
            for nm, _ in module.named_parameters():
                full = f"model.language_model.layers.{i}.{nm}"
                m2 = module.get_submodule(nm.rsplit(".", 1)[0]) if "." in nm else module
                leaf = nm.rsplit(".", 1)[1] if "." in nm else nm
                m2._parameters[leaf] = torch.nn.Parameter(decode(full), requires_grad=False)
        return pre

    def post(module, args, kwargs, output):
        for nm, p in list(module.named_parameters()):
            m2 = module.get_submodule(nm.rsplit(".", 1)[0]) if "." in nm else module
            leaf = nm.rsplit(".", 1)[1] if "." in nm else nm
            m2._parameters[leaf] = torch.nn.Parameter(p.to("meta"), requires_grad=False)
        return output

    for i, ly in enumerate(layers):
        ly.register_forward_pre_hook(make_pre(i), with_kwargs=True)
        ly.register_forward_hook(post, with_kwargs=True)

    os.makedirs(out, exist_ok=True)
    rows_f = open(os.path.join(out, "positions.jsonl"), "w")
    per_prompt = []
    t_stream_all: list[int] = []
    d_stream_all: list[int] = []
    bins = np.zeros((len(MARGIN_EDGES) - 1, 2), dtype=int)   # [n, agree]
    t_all0 = time.time()
    for i in range(n_prompts):
        rec = recs[i]
        # padding: left side only; the prompt is the unmasked tail
        m_i = mask[i]
        if m_i.sum() == 0 or (m_i[np.argmax(m_i == 1):] != 1).any():
            raise SystemExit(f"REFUSING: prompt {i} mask is not left-padded")
        prompt = [int(x) for x in input_ids[i][m_i == 1]]
        if pad_id is not None and pad_id in prompt:
            raise SystemExit(f"REFUSING: pad id inside prompt {i}")
        gen_full = [int(x) for x in rec["tokens"][:gen_lengths[i]]]
        if len(gen_full) != gen_lengths[i] or len(rec["tokens"]) != gen_lengths[i]:
            raise SystemExit(f"REFUSING: rollout {i} token count != gen_length")
        eos_stop = gen_lengths[i] < ro["steps"]
        if eos_stop and gen_full[-1] not in eos_ids:
            raise SystemExit(f"REFUSING: rollout {i} stopped early without eos")
        gen = gen_full[:8] if smoke else gen_full
        ids = prompt + gen
        if ids[len(prompt):] != gen:
            raise SystemExit("REFUSING: alignment")
        t0 = time.time()
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float().numpy()
        wall = time.time() - t0
        if not np.isfinite(lg).all():
            raise SystemExit("REFUSING: non-finite arm logits")
        if lg.shape[1] != v_rec:
            raise SystemExit(f"REFUSING: arm logit width {lg.shape[1]} != record {v_rec}")
        p_len = len(prompt)
        # traversal: every layer exactly once per forward, 48 linear + 16 full
        calls = trav["layer_calls"]
        if any(c != i + 1 for c in calls) or trav["families"].count("linear_attn") != 48 \
                or trav["families"].count("full_attn") != 16:
            raise SystemExit(f"REFUSING: traversal {calls[:4]}... families "
                             f"{trav['families'].count('linear_attn')}/{trav['families'].count('full_attn')}")
        t_stream, d_stream = [], []
        for s, tok_s in enumerate(gen):
            # alignment: logits at p_len-1+s predict ids[p_len+s] == gen[s]
            if ids[p_len + s] != tok_s:
                raise SystemExit("REFUSING: alignment")
            zt16 = rollouts[s, i, :v_live]            # fp16 as stored
            zt = np.asarray(zt16, dtype=np.float32)
            zd = lg[p_len - 1 + s, :v_live]
            row = sr.position_stats(zt, zd, tok_s, topk=TOPK)
            row.update(sr.teacher_ambiguity(np.asarray(zt16), tok_s))
            row.update({"prompt_id": rec["id"], "category": rec["category"], "step": s,
                        "eos_stop_rollout": eos_stop})
            rows_f.write(json.dumps(row) + "\n")
            t_stream.append(tok_s)
            d_stream.append(row["draft_argmax"])
            b = margin_bin(row["teacher_margin_logit"])
            bins[b, 0] += 1
            bins[b, 1] += int(row["agree"])
        t_stream_all += t_stream
        d_stream_all += d_stream
        replay = {k: sr.replay_blocks(t_stream, d_stream, k) for k in KS}
        per_prompt.append({"prompt_id": rec["id"], "category": rec["category"],
                           "prompt_len": p_len, "gen_len": len(gen), "forward_wall_s": wall,
                           "s_per_position_teacher_forced": wall / len(ids),
                           "agree_rate": float(np.mean([t == d for t, d in zip(t_stream, d_stream)])),
                           "replay_mean_emitted": {k: r["mean_emitted"] for k, r in replay.items()}})
        if smoke:
            print(f"[e0 smoke] {rec['id']}: {len(ids)} tokens, {wall:.0f}s, finite logits, "
                  f"traversal ok, alignment ok (substantive numbers withheld)", flush=True)
        else:
            print(f"[e0] {rec['id']}: {len(ids)} tokens, {wall:.0f}s, agree "
                  f"{per_prompt[-1]['agree_rate']:.3f}", flush=True)
    rows_f.close()
    pooled = {k: sr.replay_blocks(t_stream_all, d_stream_all, k) for k in KS}
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = {
        "arm": arm, "artifact": art, "smoke": smoke, "device_actual": "cpu",
        "code_commit": git_commit(), "teacher": {"dir": tdir, "code_commit": man["code_commit"],
                                                 "revision": man["revision"], "rollout_sha": rl_sha},
        "artifact_chain": chain, "artifact_chain_sha256": fsha(chain),
        "qualification": q["report"], "v_live": v_live, "v_record": v_rec,
        "traversal": trav, "n_prompts": n_prompts, "n_positions": len(t_stream_all),
        "agree_rate_pooled": float(np.mean([t == d for t, d in zip(t_stream_all, d_stream_all)])),
        "margin_edges": MARGIN_EDGES[:-1] + ["inf"],
        "agree_by_teacher_margin_bin": bins.tolist(),
        "certified_fraction_nonambiguous": None,  # filled below from rows
        "replay_pooled": {k: {kk: v for kk, v in r.items() if kk not in
                              ("accepted_per_block", "emitted_per_block", "drafted_per_block")}
                          for k, r in pooled.items()},
        "geometric_model_for_reference": {k: sr.geometric_emitted(
            float(np.mean([t == d for t, d in zip(t_stream_all, d_stream_all)])), k) for k in KS},
        "geometric_model_label": sr.GEOMETRIC_MODEL_LABEL,
        "per_prompt": per_prompt, "wall_s": time.time() - t_all0,
        "peak_rss_bytes": int(peak_rss) if sys.platform == "darwin" else int(peak_rss) * 1024,
    }
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    rows_all = [json.loads(line) for line in open(os.path.join(out, "positions.jsonl"))]
    nonamb = [r for r in rows_all if not r["ambiguous"]]
    summary["certified_fraction_nonambiguous"] = (
        float(np.mean([r["margin_certified"] for r in nonamb])) if nonamb else None)
    summary["n_nonambiguous"] = len(nonamb)
    summary["n_ambiguous"] = int(sum(1 for line in open(os.path.join(out, "positions.jsonl"))
                                    if json.loads(line)["ambiguous"]))
    summary["eos_ids"] = sorted(eos_ids)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    if smoke:
        print(f"[e0 smoke] wrote {out}: {summary['n_positions']} positions, wall "
              f"{summary['wall_s']:.0f}s, peak rss {summary['peak_rss_bytes']/1e9:.2f} GB; "
              f"instrument checks passed; agreement withheld until the pre-reg is read", flush=True)
    else:
        print(f"[e0] wrote {out}: {summary['n_positions']} positions, pooled agree "
              f"{summary['agree_rate_pooled']:.4f}, wall {summary['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
