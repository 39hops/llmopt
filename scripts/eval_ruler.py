"""RULER long-context eval against a real HF model.

Runs context/ruler.py's synthetic suite (NIAH single/multi-key, variable
tracking) at increasing lengths and reports accuracy per (task, length)
plus RULER's headline effective-context-length (>= 85% mean accuracy).

Lengths are in filler words (~1.3 tokens/word for Qwen tokenizers); the
script prints the actual prompt token count per length so results are
interpretable against the model's claimed window.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.context.ruler import effective_context_length, evaluate, make_suite

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LENGTHS_WORDS = (500, 1000, 2000, 4000, 8000, 16000, 24000, 27000)
SAMPLES = 5
MAX_NEW = 24
PREFILL_CHUNK = 2048


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

    def generate(prompt: str) -> str:
        # chunked prefill: a one-shot 20k-token SDPA prefill OOMs on 10GB
        ids = tok(prompt).input_ids
        new: list[int] = []
        with torch.inference_mode():
            past = None
            for i in range(0, len(ids), PREFILL_CHUNK):
                chunk = torch.tensor([ids[i : i + PREFILL_CHUNK]], device="cuda")
                out = m(input_ids=chunk, past_key_values=past, use_cache=True)
                past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            for _ in range(MAX_NEW):
                new.append(nxt)
                if nxt == tok.eos_token_id:
                    break
                out = m(
                    input_ids=torch.tensor([[nxt]], device="cuda"),
                    past_key_values=past, use_cache=True,
                )
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
        del past
        torch.cuda.empty_cache()
        return tok.decode(new, skip_special_tokens=True)

    suite = make_suite(LENGTHS_WORDS, samples_per_length=SAMPLES)
    seen_tokens: dict[int, int] = {}
    for t in suite:
        lw = t.meta["length_words"]
        if lw not in seen_tokens:
            seen_tokens[lw] = len(tok(t.prompt).input_ids)
    print("prompt tokens per length:",
          {k: v for k, v in sorted(seen_tokens.items())})

    results = evaluate(generate, suite)
    kinds = sorted({k for k, _ in results})
    print(f"\n{'length_words':>12} " + " ".join(f"{k:>18}" for k in kinds))
    for lw in LENGTHS_WORDS:
        row = " ".join(f"{results[(k, lw)]:>17.0%} " for k in kinds)
        print(f"{lw:>12} {row}")

    ecl = effective_context_length(results)
    print(f"\neffective context length (>=85% mean): {ecl} words"
          f" (~{seen_tokens.get(ecl, 0)} tokens)")


if __name__ == "__main__":
    main()
