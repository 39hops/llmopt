"""KV-cache sampler + equivalence oracle (house rule: token-
identical to eager full-recompute, or it doesn't ship)."""
import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model


def sample_wave_lp_kv(model, tok, prompt_ids, seeds, dev,
                      max_new=120):
    Bn = len(seeds)
    ids = torch.tensor([prompt_ids] * Bn, device=dev)
    gens = [torch.Generator(device="cpu").manual_seed(s)
            for s in seeds]
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
            nxt = int(torch.multinomial(probs[b], 1,
                                        generator=gens[b]))
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


if __name__ == "__main__":
    import sympy as sp
    from bench_step_tokens import _gen_isolated
    tok = MathTokenizer()
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    model = build_model(len(tok.vocab), d=384, layers=8, heads=6,
                        ffn=1536).to(dev)
    model.load_state_dict(torch.load(
        "checkpoints/mathnative_19m_v21.pt", map_location="cpu"))
    model.eval()
    match = mismatch = 0
    t_e = t_k = 0.0
    with torch.no_grad():
        for trial in range(20):
            lv = [3, 4, 5, 6, 7][trial % 5]
            p = _gen_isolated(lv, 55_000_000 + trial)
            if p is None:
                continue
            prompt = tok.encode(f"Current: Integral({sp.sstr(p._expr)},"
                                f" x)\nHints: none\nStep: ")
            seeds = [trial * 100 + b for b in range(8)]
            t0 = time.time()
            te, _, le = G.sample_wave_lp(model, tok, prompt, seeds, dev)
            t_e += time.time() - t0
            t0 = time.time()
            tk, _, lk = sample_wave_lp_kv(model, tok, prompt, seeds,
                                          dev)
            t_k += time.time() - t0
            if te == tk:
                match += 1
            else:
                mismatch += 1
                for a, b in zip(te, tk):
                    if a != b:
                        print(f"  DIVERGE: '{a}' vs '{b}'")
    print(f"EQUIVALENCE: {match} waves identical, {mismatch} diverged")
    print(f"TIMING ({dev}): eager {t_e:.1f}s vs kv {t_k:.1f}s "
          f"= {t_e/max(t_k,1e-9):.2f}x")
