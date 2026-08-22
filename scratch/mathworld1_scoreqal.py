"""MATH-CYBER-1 cached-scorer qualification + runtime microbench
(AMENDMENT MATH-CYBER-1-SUBSTRATE-DESK-0-KV registration). NO
TRAINED MODEL: a fresh random-weight MicroLM (seeded birth,
BIRTH_SEED=77, stock base vocab) is the instrument — the
qualification tests SCORER MECHANICS, not capability.

Stage QUAL — full teacher-forced score v cached score, exact
semantics: for every calibration decision whose candidates all
encode under the stock tokenizer (the joint-eligible class), score
each candidate child two ways:
  FULL:   one forward over the whole sequence (causal), score =
          sum log p of child tokens (terminating newline included);
  CACHED: one causal prefill of the shared parent prefix
          ("Current: {parent}\\nHints: none\\nStep: "), then
          ONE-TOKEN steps through the child (T=1 past path — the
          only causal cached mode MicroLM has).
Bars (registered in the -KV amendment): max |delta| over all
scored candidates, and 100% per-decision argmax agreement.

Stage BENCH — random-weight Mac (mps) wall + peak memory at ctx
{512, 1024, 2048, 4096}: scoring a K=8 candidate batch (parent
prefix = ctx/2, child = ctx/2) FULL v CACHED (prefill + T=1
steps), and the training path forward+backward on an 8 x ctx
batch with and without grad checkpointing. Cost model: measured
wall on this Mac, fp32, batch shapes as stated — not a general
law.

Receipt: logs/mathworld1/scoreqal.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_scoreqal.py           (Mac)
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import (MathTokenizer,  # noqa: E402
                                     build_model)

OUT = Path("logs/mathworld1/scoreqal.json")
BIRTH_SEED = 77


def full_score(model, ids_prefix, ids_child, dev):
    ids = torch.tensor([ids_prefix + ids_child], device=dev)
    with torch.no_grad():
        logits = model(ids)
    lp = torch.log_softmax(logits[0].float(), -1)
    s = 0.0
    for i, t in enumerate(ids_child):
        s += lp[len(ids_prefix) + i - 1, t].item()
    return s


def cached_score(model, past, prefix_len, ids_child, dev):
    """past = KV of the causal prefix prefill; T=1 steps only."""
    s = 0.0
    cur_past = [(k.clone(), v.clone()) for k, v in past]
    prev_logits = None
    for i, t in enumerate(ids_child):
        if i == 0:
            s += prev_logits[t] if prev_logits is not None else 0.0
        with torch.no_grad():
            lg, cur_past = model(
                torch.tensor([[t]], device=dev), past=cur_past)
        if i + 1 < len(ids_child):
            lp = torch.log_softmax(lg[0, -1].float(), -1)
            s += lp[ids_child[i + 1]].item()
    return s


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_scoreqal.py",
         "llmopt/train/mathnative.py"])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = MathTokenizer()
    torch.manual_seed(BIRTH_SEED)
    model = build_model(len(tok.vocab)).to(dev).eval()

    # ---- QUAL over joint-eligible calibration decisions ----
    states = [json.loads(line) for line in
              open("logs/mathworld1/states.jsonl")]
    acts = {}
    for line in open("logs/mathworld1/actions.jsonl"):
        a = json.loads(line)
        acts.setdefault((a["episode_id"], a["step_id"]),
                        []).append(a)

    def enc(s):
        try:
            return tok.encode(s)
        except ValueError:
            return None

    max_delta = 0.0
    n_scored = n_decisions = agree = 0
    for srow in states:
        if srow["row_class"] != "decision":
            continue
        key = (srow["episode_id"], srow["step_id"])
        prefix = (f"Current: {srow['state_before']}\n"
                  f"Hints: none\nStep: ")
        ip = enc(prefix)
        childs = [enc(a["child"] + "\n") for a in acts[key]]
        if ip is None or any(c is None for c in childs):
            continue
        if any(len(ip) + len(c) > 512 for c in childs):
            continue  # joint-eligible class only
        n_decisions += 1
        with torch.no_grad():
            lg, past = model(torch.tensor([ip], device=dev),
                             use_cache=True)
        first_lp = torch.log_softmax(lg[0, -1].float(), -1)
        fulls, cacheds = [], []
        for c in childs:
            f = full_score(model, ip, c, dev)
            # first-token conditional comes from the prefill logits
            s = first_lp[c[0]].item() + cached_score(
                model, past, len(ip), c, dev)
            fulls.append(f)
            cacheds.append(s)
            max_delta = max(max_delta, abs(f - s))
            n_scored += 1
        agree += int(max(range(len(fulls)), key=lambda i: fulls[i])
                     == max(range(len(cacheds)),
                            key=lambda i: cacheds[i]))
    qual = {"decisions": n_decisions, "candidates_scored": n_scored,
            "max_abs_delta": max_delta,
            "argmax_agree": agree,
            "argmax_agree_pct": (100.0 * agree / n_decisions
                                 if n_decisions else None)}
    print("[qual]", json.dumps(qual), flush=True)

    # ---- BENCH ----
    def mem():
        return (torch.mps.current_allocated_memory()
                if dev == "mps" else 0)

    bench = {}
    K_FULL = 8
    K_CACHED = 2   # per-candidate cost is candidate-independent;
    # mps shape-growing KV steps churn the allocator, so the
    # cached bench runs 2 candidates with periodic empty_cache and
    # reports per-candidate wall (multiply by K to compare)
    for ctx in (512, 1024, 2048, 4096):
        half = ctx // 2
        ids = torch.randint(2, len(tok.vocab), (1, ctx), device=dev)
        cell = {}
        try:
            torch.manual_seed(0)
            if dev == "mps":
                torch.mps.empty_cache()
            m0 = mem()
            t0 = time.monotonic()
            with torch.no_grad():
                for _ in range(K_FULL):
                    model(ids)
            if dev == "mps":
                torch.mps.synchronize()
            cell["score_full_perK_wall_s"] = round(
                (time.monotonic() - t0) / K_FULL, 3)
            cell["score_full_mem_delta_mb"] = round(
                (mem() - m0) / 2**20, 1)
        except RuntimeError as e:
            cell["score_full"] = f"OOM: {str(e)[:80]}"
        try:
            if dev == "mps":
                torch.mps.empty_cache()
            m0 = mem()
            t0 = time.monotonic()
            with torch.no_grad():
                _, past = model(ids[:, :half], use_cache=True)
                for _ in range(K_CACHED):
                    cur = [(k.clone(), v.clone()) for k, v in past]
                    for j in range(half):
                        _, cur = model(
                            ids[:, half + j:half + j + 1],
                            past=cur)
                        if dev == "mps" and j % 128 == 127:
                            torch.mps.empty_cache()
            if dev == "mps":
                torch.mps.synchronize()
            cell["score_cached_perK_wall_s"] = round(
                (time.monotonic() - t0) / K_CACHED, 3)
            cell["score_cached_mem_delta_mb"] = round(
                (mem() - m0) / 2**20, 1)
        except RuntimeError as e:
            cell["score_cached"] = f"OOM: {str(e)[:80]}"
        tr = {}
        for ck in (False, True):
            try:
                model.train()
                model.grad_ckpt = ck
                batch = torch.randint(2, len(tok.vocab), (8, ctx),
                                      device=dev)
                if dev == "mps":
                    torch.mps.empty_cache()
                m0 = mem()
                t0 = time.monotonic()
                logits = model(batch)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    batch[:, 1:].reshape(-1))
                loss.backward()
                if dev == "mps":
                    torch.mps.synchronize()
                tr[f"ckpt_{ck}"] = {
                    "wall_s": round(time.monotonic() - t0, 3),
                    "peak_delta_mb": round((mem() - m0) / 2**20, 1)}
            except RuntimeError as e:
                tr[f"ckpt_{ck}"] = f"OOM: {str(e)[:80]}"
            finally:
                model.zero_grad(set_to_none=True)
                model.eval()
                model.grad_ckpt = False
                if dev == "mps":
                    torch.mps.empty_cache()
        cell["train_fwd_bwd_8x"] = tr
        bench[ctx] = cell
        print(f"[bench ctx={ctx}]", json.dumps(cell), flush=True)

    receipt = {"device": dev, "birth_seed": BIRTH_SEED,
               "vocab": len(tok.vocab), "qual": qual,
               "bench": bench, "start": START,
               "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(receipt, indent=1))
    print("[scoreqal] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
