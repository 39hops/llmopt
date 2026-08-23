"""MATH-CYBER-1 execution pricing under REAL K/length profiles
(outside-review ask after SCOREQAL-0: the accepted path is full
teacher-forced scoring; SCOREQAL's "full" bench was SERIAL B=1
repeated K times — batching was never priced). RANDOM-WEIGHT
mechanics only, no capability claims.

Vocabulary: the grammar-closed proposal realized minimally for
mechanics — stock ATOMS ids + 256 byte-fallback ids appended
(vocab 296+) so ALL 725 calibration actions encode; the shipped
tokenizer decision is unchanged by this instrument.

Stage SCORE — over all 101 calibration decisions (from the
exported corpus logs/mathworld1/{states,actions}.jsonl):
  SERIAL:  one B=1 forward per candidate (725 forwards);
  BATCHED: per decision, candidates RIGHT-PADDED into microbatches
           under a token budget (pad tokens after the sequence are
           causally invisible to valid positions, so no attention
           mask is needed; score reads logits at valid positions).
Correctness bars (same class as SCOREQAL): max |batched - serial|
score delta over all 725 candidates + 101/101 argmax agreement.
Wall + allocator-delta memory for both.

Stage UPDATE — one frozen ACTIVE optimizer update per the -DESIGN
dose law, mechanics-only: for a SOLVED episode, its chosen
transitions (variable length) right-padded into microbatches
under the same token budget, child-token-only mean CE
(prefix/pad labels masked), gradient accumulation across
microbatches, exactly one AdamW(lr 1e-4, betas 0.9/0.95, wd 0)
step. Benched on the median-length and the longest solved
calibration episodes.

Receipt: logs/mathworld1/execbench.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_execbench.py          (Mac)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import (MathTokenizer,  # noqa: E402
                                     build_model)

OUT = Path("logs/mathworld1/execbench.json")
BIRTH_SEED = 77
TOKEN_BUDGET = 16384      # max padded tokens per scoring microbatch


class ByteFallbackTok:
    """ATOMS greedy longest-match + byte-fallback ids (mechanics
    realization of the grammar-closed proposal)."""

    def __init__(self):
        self.base = MathTokenizer()
        self.n_base = len(self.base.vocab)
        self.vocab_size = self.n_base + 256

    def encode(self, s: str) -> list[int]:
        out, i = [], 0
        while i < len(s):
            for t in self.base._by_len:
                if s.startswith(t, i):
                    out.append(self.base.id[t])
                    i += len(t)
                    break
            else:
                for byte in s[i].encode():
                    out.append(self.n_base + byte)
                i += 1
        return out


def microbatches(items, budget):
    """Greedy pack (idx, ids) into padded microbatches under a
    total padded-token budget; items sorted by length first."""
    items = sorted(items, key=lambda x: len(x[1]))
    out, cur = [], []
    for it in items:
        trial = cur + [it]
        if cur and len(trial) * len(trial[-1][1]) > budget:
            out.append(cur)
            cur = [it]
        else:
            cur = trial
    if cur:
        out.append(cur)
    return out


def score_batch(model, batch, prefix_lens, dev):
    """batch: list of (idx, full_ids); returns {idx: score}."""
    mx = max(len(ids) for _, ids in batch)
    pad = torch.zeros(len(batch), mx, dtype=torch.long, device=dev)
    for r, (_, ids) in enumerate(batch):
        pad[r, :len(ids)] = torch.tensor(ids, device=dev)
    with torch.no_grad():
        logits = model(pad)
    lp = torch.log_softmax(logits.float(), -1)
    scores = {}
    for r, (idx, ids) in enumerate(batch):
        pl = prefix_lens[idx]
        s = 0.0
        for i in range(pl, len(ids)):
            s += lp[r, i - 1, ids[i]].item()
        scores[idx] = s
    return scores


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_execbench.py",
         "llmopt/train/mathnative.py",
         "logs/mathworld1/states.jsonl",
         "logs/mathworld1/actions.jsonl"])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = ByteFallbackTok()
    torch.manual_seed(BIRTH_SEED)
    model = build_model(tok.vocab_size).to(dev).eval()

    states = [json.loads(line) for line in
              open("logs/mathworld1/states.jsonl")]
    acts = {}
    for line in open("logs/mathworld1/actions.jsonl"):
        a = json.loads(line)
        acts.setdefault((a["episode_id"], a["step_id"]),
                        []).append(a)

    decisions = []      # (key, prefix_ids, [child_ids], chosen_idx)
    for srow in states:
        if srow["row_class"] != "decision":
            continue
        key = (srow["episode_id"], srow["step_id"])
        ip = tok.encode(f"Current: {srow['state_before']}\n"
                        f"Hints: none\nStep: ")
        cand = sorted(acts[key], key=lambda a: a["idx"])
        childs = [tok.encode(a["child"] + "\n") for a in cand]
        chosen = [i for i, a in enumerate(cand)
                  if a["child_hash"] == srow["chosen_child_hash"]]
        decisions.append((key, ip, childs, chosen[0]))

    # ---- SCORE: serial ----
    if dev == "mps":
        torch.mps.empty_cache()
    t0 = time.monotonic()
    serial = {}
    with torch.no_grad():
        for key, ip, childs, _ in decisions:
            for ci, c in enumerate(childs):
                ids = torch.tensor([ip + c], device=dev)
                logits = model(ids)
                lp = torch.log_softmax(logits[0].float(), -1)
                s = sum(lp[len(ip) + i - 1, t].item()
                        for i, t in enumerate(c))
                serial[(key, ci)] = s
    if dev == "mps":
        torch.mps.synchronize()
    serial_wall = time.monotonic() - t0

    # ---- SCORE: batched/microbatched ----
    if dev == "mps":
        torch.mps.empty_cache()
    m0 = (torch.mps.current_allocated_memory()
          if dev == "mps" else 0)
    t0 = time.monotonic()
    batched = {}
    for key, ip, childs, _ in decisions:
        items = [(ci, ip + c) for ci, c in enumerate(childs)]
        plens = {ci: len(ip) for ci, _ in items}
        for mb in microbatches(items, TOKEN_BUDGET):
            for ci, s in score_batch(model, mb, plens,
                                     dev).items():
                batched[(key, ci)] = s
    if dev == "mps":
        torch.mps.synchronize()
    batched_wall = time.monotonic() - t0
    mem_delta = ((torch.mps.current_allocated_memory() - m0)
                 if dev == "mps" else 0)

    max_delta = max(abs(serial[k] - batched[k]) for k in serial)
    agree = 0
    for key, ip, childs, _ in decisions:
        ks = range(len(childs))
        a1 = max(ks, key=lambda ci: serial[(key, ci)])
        a2 = max(ks, key=lambda ci: batched[(key, ci)])
        agree += int(a1 == a2)
    score_res = {
        "decisions": len(decisions),
        "candidates": len(serial),
        "serial_wall_s": round(serial_wall, 2),
        "batched_wall_s": round(batched_wall, 2),
        "speedup": round(serial_wall / batched_wall, 2),
        "batched_mem_delta_mb": round(mem_delta / 2**20, 1),
        "max_abs_delta": max_delta,
        "argmax_agree": f"{agree}/{len(decisions)}"}
    print("[score]", json.dumps(score_res), flush=True)

    # ---- UPDATE: one frozen ACTIVE update, mechanics ----
    solved_eps = {}
    for srow in states:
        if (srow["row_class"] == "decision"):
            solved_eps.setdefault(srow["episode_id"],
                                  []).append(srow)
    solved_ids = [e for e, rr in solved_eps.items()
                  if rr[-1]["outcome"] == "solved"]

    def episode_items(eid):
        items = []
        for srow in solved_eps[eid]:
            ip = tok.encode(f"Current: {srow['state_before']}\n"
                            f"Hints: none\nStep: ")
            c = tok.encode(srow["state_after"] + "\n")
            items.append((srow["step_id"], ip, c))
        return items

    sized = sorted(solved_ids, key=lambda e: sum(
        len(ip) + len(c) for _, ip, c in episode_items(e)))
    picks = {"median_episode": sized[len(sized) // 2],
             "longest_episode": sized[-1]}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4,
                            betas=(0.9, 0.95), weight_decay=0.0)
    upd = {}
    for label, eid in picks.items():
        items = episode_items(eid)
        model.train()
        if dev == "mps":
            torch.mps.empty_cache()
        t0 = time.monotonic()
        mbs = microbatches(
            [(sid, ip + c) for sid, ip, c in items], TOKEN_BUDGET)
        plens = {sid: len(ip) for sid, ip, _ in items}
        n_edges = len(items)
        for mb in mbs:
            mx = max(len(ids) for _, ids in mb)
            pad = torch.zeros(len(mb), mx, dtype=torch.long,
                              device=dev)
            lab = torch.full((len(mb), mx), -100,
                             dtype=torch.long, device=dev)
            for r, (sid, ids) in enumerate(mb):
                pad[r, :len(ids)] = torch.tensor(ids, device=dev)
                pl = plens[sid]
                lab[r, pl:len(ids)] = torch.tensor(
                    ids[pl:], device=dev)
            logits = model(pad)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                lab[:, 1:].reshape(-1), ignore_index=-100)
            (loss * len(mb) / n_edges).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if dev == "mps":
            torch.mps.synchronize()
        upd[label] = {"episode": eid, "edges": n_edges,
                      "microbatches": len(mbs),
                      "wall_s": round(time.monotonic() - t0, 3)}
        model.eval()
        print(f"[update {label}]", json.dumps(upd[label]),
              flush=True)

    receipt = {"device": dev, "birth_seed": BIRTH_SEED,
               "vocab": tok.vocab_size,
               "token_budget": TOKEN_BUDGET,
               "score": score_res, "update": upd,
               "start": START,
               "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(receipt, indent=1))
    print("[execbench] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
