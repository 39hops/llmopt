"""Local inference for the qualified Qwen3.8-27B arm C (QWEN-INFERENCE-
BASELINE-0): a library entry point that loads the frozen artifact under
its digest chain, serves greedy generation on the qualified Metal path
(generation 5 kernels, chunk-256 GEMM prefill with the final-position
head, fused cached decode, the explicit 2 GiB allocator-cache limit) and
keeps exact token accounting across turns.

Nothing here changes the model, the codec, the kernels or their defaults:
the tower is llmopt.lab.qtower.build_metal_model (adopted verbatim from
the qualification driver, with the mlx-lm GatedDeltaNet source guard), the
prefill is llmopt.lab.qprefill.prefill_chunked, and the decode step is the
plain model call the serving receipts measured.

Token accounting (the law this module keeps and tests pin):
  prefill feeds the prompt ids p[0..n) and returns the logits at p[n-1];
  the FIRST generated token g[0] is the argmax of those logits and has NOT
  been fed; each later token g[k] is the argmax of the logits returned by
  feeding g[k-1]. When generation stops on a stop token s (the argmax after
  feeding g[m-1]), every emitted token has been fed and the cache holds
  n + m tokens; s itself is recorded as stop_token_id and is neither emitted
  nor fed. When generation stops at max_tokens, the last emitted token has
  NOT been fed and the cache holds n + m - 1 tokens. GenerationResult
  records both counts; nothing assumes the last emitted token is in the
  cache.

Conversations: Session keeps the message list and renders every turn with
the pinned tokenizer's own chat template (enable_thinking passed through
to the template only when given; no template token is invented). The
reference implementation replays the FULL transcript into a fresh cache
each turn (mode "replay"). Mode "incremental" reuses the cache only when
the template's rendering of the full transcript has the previously fed
token sequence as an exact prefix (checked mechanically every turn) and
falls back to a full replay, recorded in the result, when it does not.

Template disclosure: with thinking on (enable_thinking True or not
given) the pinned template prepends its own system message ("Reasoning
effort is set to xhigh...") and opens the assistant turn with <think>;
with enable_thinking False it opens with an empty think block. A past
assistant turn is re-rendered from reasoning_content and content, so
Session splits a generated text at its last </think> when storing it.
Incremental mode and replay are token-equal where observed (the live
tests and the serving receipts) and are not bit-equivalent by
construction: earlier generated tokens entered the cache through the
fused decode path in one and the GEMM prefill path in the other.

Stop semantics: the tokenizer's eos token (<|im_end|>, the end of the
assistant turn) and <|endoftext|> stop generation; decoding never
continues past them. Context: prompt tokens plus the output budget must
fit the engine's context cap (default 8,192, far below the model's
262,144 positions and chosen for this machine's memory); anything larger
raises ContextLimitError, never a silent truncation. Empty or malformed
input raises InputError.
"""
from __future__ import annotations

import dataclasses
import os
import time
from typing import Iterable, Optional

CACHE_LIMIT_BYTES = 2 * 1024 ** 3
DEFAULT_CONTEXT_CAP = 8192
DEFAULT_MAX_TOKENS = 512
REQUIRED_KERNEL_GENERATION = 5
ENDOFTEXT = "<|endoftext|>"


class InputError(ValueError):
    """Empty or malformed input."""


class ContextLimitError(ValueError):
    """Prompt plus output budget exceed the context cap; nothing is truncated."""


@dataclasses.dataclass
class GenerationResult:
    prompt_ids: list
    generated_ids: list
    text: str
    stop_reason: str                 # "stop_token" | "max_tokens"
    stop_token_id: Optional[int]
    n_prompt: int
    n_generated: int
    fed_generated: int               # generated tokens fed back into the cache
    cache_tokens: int                # tokens the cache holds after generation (by accounting)
    cache_offset_measured: Optional[int]   # the KV cache's own offset after generation (None if no KV layer)
    last_token_in_cache: bool
    timings: dict
    memory: dict
    mode: str = "fresh"              # "fresh" | "incremental" | "replay-fallback"
    note: str = ""
    cache: object = dataclasses.field(default=None, repr=False, compare=False)   # the cache object after generation


def cache_offset(cache) -> Optional[int]:
    """The token offset the KV cache objects report (every KV layer must
    agree; the DeltaNet ArraysCache objects carry no offset and are not
    checked); None when no object carries one."""
    offs = [int(c.offset) for c in (cache or []) if hasattr(c, "offset")]
    if not offs:
        return None
    if len(set(offs)) != 1:
        raise RuntimeError(f"KV layers disagree on their offset: {sorted(set(offs))}")
    return offs[0]


def split_thinking(text: str) -> tuple:
    """(reasoning_content, content): the pinned template renders a past
    assistant turn as <think>\n{reasoning}\n</think>\n\n{content}, so a
    generated text that carries a </think> is split at its last one and the
    opening <think> (which the generation prompt supplied) is stripped."""
    if "</think>" in text:
        head, _, tail = text.rpartition("</think>")
        head = head.strip()
        if head.startswith("<think>"):
            head = head[len("<think>"):].strip()
        return head, tail.strip()
    return "", text


def _check_ids(ids) -> list:
    if not isinstance(ids, (list, tuple)) or len(ids) == 0:
        raise InputError("ids must be a non-empty list of token ids")
    out = []
    for t in ids:
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise InputError(f"token id {t!r} is not a non-negative int")
        out.append(int(t))
    return out


class Engine:
    """The loaded arm C tower with its tokenizer and serving configuration."""

    def __init__(self, art_dir: Optional[str] = None, vendor_dir: Optional[str] = None,
                 context_cap: int = DEFAULT_CONTEXT_CAP, cache_limit_bytes: int = CACHE_LIMIT_BYTES, chunk: int = 256,
                 model=None, tokenizer=None, info=None):
        """Load (or, for tests, wrap a given model and tokenizer). Loading verifies
        the artifact under logs/qwenwhole/artifact_digest_<arm>.txt, refuses a
        kernel generation other than 5, applies the serving cache limit and
        records the cold load time."""
        from llmopt.lab import qmetal, qprefill
        if qmetal.VERSION != REQUIRED_KERNEL_GENERATION:
            raise RuntimeError(f"kernel generation {qmetal.VERSION} is not the qualified {REQUIRED_KERNEL_GENERATION}")
        self.context_cap = int(context_cap)
        self.cfg = qprefill.ServingConfig(chunk=chunk, cache_limit_bytes=cache_limit_bytes)
        t0 = time.perf_counter()
        if model is None:
            from llmopt.lab.qtower import build_metal_model
            self.art_dir = os.path.expanduser(art_dir or os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
            self.vendor_dir = os.path.expanduser(vendor_dir or os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
            self.model, self.info = build_metal_model(self.art_dir, self.vendor_dir)
            if self.info["kernel_version"] != REQUIRED_KERNEL_GENERATION:
                raise RuntimeError("the tower was built under another kernel generation")
        else:
            self.art_dir, self.vendor_dir = art_dir, vendor_dir
            self.model, self.info = model, dict(info or {})
        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.vendor_dir)
            import hashlib
            self.tokenizer_files_sha256 = {}
            for fn in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                p = os.path.join(self.vendor_dir, fn)
                if os.path.exists(p):
                    self.tokenizer_files_sha256[fn] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        else:
            self.tokenizer = tokenizer
            self.tokenizer_files_sha256 = {}
        self.previous_cache_limit = qprefill.configure_serving_memory(self.cfg)
        self.load_s = time.perf_counter() - t0
        eos = self.tokenizer.eos_token_id
        eot = self.tokenizer.convert_tokens_to_ids(ENDOFTEXT)
        self.stop_ids = {int(t) for t in (eos, eot) if t is not None and t >= 0}
        if not self.stop_ids:
            raise RuntimeError("no stop token ids")

    # ---- templates ----
    def render_chat(self, messages: list, enable_thinking: Optional[bool] = None) -> list:
        """Token ids of the conversation rendered by the pinned tokenizer's chat
        template with the generation prompt appended; enable_thinking is passed
        through only when given (the template's own control)."""
        if not isinstance(messages, list) or not messages:
            raise InputError("messages must be a non-empty list")
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
                raise InputError(f"malformed message {m!r}")
            if m["role"] == "user" and not m["content"].strip():
                raise InputError("empty user message")
        kw = {} if enable_thinking is None else {"enable_thinking": bool(enable_thinking)}
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kw)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return [int(t) for t in ids]

    # ---- generation ----
    def generate_ids(self, ids: list, max_tokens: int = DEFAULT_MAX_TOKENS, stop_ids: Optional[Iterable[int]] = None,
                     cache=None, cache_tokens: int = 0, mode: str = "fresh") -> GenerationResult:
        """Greedy generation from token ids. With cache=None a fresh cache is
        made and all ids are prefilled; with a cache holding cache_tokens
        tokens, ids are the NEW tokens to feed on top of it (the caller
        guarantees the prefix). Returns the result with exact accounting."""
        import mlx.core as mx
        from llmopt.lab import qprefill
        ids = _check_ids(ids)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise InputError("max_tokens must be a positive int")
        n_total_prompt = cache_tokens + len(ids)
        if n_total_prompt + max_tokens > self.context_cap:
            raise ContextLimitError(f"{n_total_prompt} prompt tokens plus a budget of {max_tokens} exceed the context cap "
                                    f"{self.context_cap}; shorten the input or the budget (nothing is truncated)")
        stops = set(int(t) for t in stop_ids) if stop_ids is not None else self.stop_ids
        if cache is None:
            cache = self.model.make_cache()
            cache_tokens = 0
        else:
            off = cache_offset(cache)
            if off is not None and off != cache_tokens:
                raise RuntimeError(f"cache offset {off} does not match the claimed {cache_tokens} tokens")
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        lg = qprefill.prefill_chunked(self.model, ids, cache, chunk=self.cfg.chunk, head="final", matmul=self.cfg.matmul)
        mx.eval(lg)
        cur = int(lg[0, -1].argmax())
        t_prefill = time.perf_counter() - t0
        generated: list = []
        step_walls: list = []
        stop_reason, stop_tok = "max_tokens", None
        t1 = time.perf_counter()
        while True:
            if cur in stops:
                stop_reason, stop_tok = "stop_token", cur
                break
            generated.append(cur)
            if len(generated) >= max_tokens:
                break
            ts = time.perf_counter()
            lz = self.model(mx.array([[cur]]), cache=cache)[0, -1]
            cur = int(lz.argmax())
            step_walls.append(time.perf_counter() - ts)
        t_decode = time.perf_counter() - t1
        fed = len(generated) if stop_reason == "stop_token" else len(generated) - 1
        measured = cache_offset(cache)
        if measured is not None and measured != cache_tokens + len(ids) + fed:
            raise RuntimeError(f"cache offset {measured} != accounted {cache_tokens + len(ids) + fed}")
        mem = qprefill.memory_report()
        text = self.tokenizer.decode(generated, skip_special_tokens=True) if generated else ""
        steady = step_walls[4:]
        timings = {"prefill_s": t_prefill, "first_token_s": t_prefill, "decode_s": t_decode, "decode_steps": len(step_walls),
                   "decode_tok_per_s": (len(step_walls) / t_decode) if step_walls and t_decode > 0 else None,
                   "decode_steady_tok_per_s": (len(steady) / sum(steady)) if steady and sum(steady) > 0 else None,
                   "prefill_tok_per_s": len(ids) / t_prefill if t_prefill > 0 else None,
                   "total_s": time.perf_counter() - t0}
        return GenerationResult(prompt_ids=ids, generated_ids=generated, text=text, stop_reason=stop_reason, stop_token_id=stop_tok,
                                n_prompt=len(ids), n_generated=len(generated), fed_generated=fed,
                                cache_tokens=cache_tokens + len(ids) + fed, cache_offset_measured=measured,
                                last_token_in_cache=(stop_reason == "stop_token" and bool(generated)),
                                timings=timings, memory=mem, mode=mode, cache=cache)

    def chat(self, messages: list, max_tokens: int = DEFAULT_MAX_TOKENS, enable_thinking: Optional[bool] = None) -> GenerationResult:
        return self.generate_ids(self.render_chat(messages, enable_thinking), max_tokens)


class Session:
    """A conversation against one Engine. mode "incremental" (the default
    since VERDICT QWEN-CHAT-DEFAULT-0) keeps the cache across turns when the
    rendered transcript extends the fed token sequence exactly, else replays
    into a fresh cache and says so; mode "replay" (the reference) re-renders
    and re-prefills the whole transcript into a fresh cache each turn."""

    def __init__(self, engine: Engine, system: Optional[str] = None, mode: str = "incremental", enable_thinking: Optional[bool] = None):
        if mode not in ("replay", "incremental"):
            raise ValueError(mode)
        self.engine, self.mode, self.enable_thinking = engine, mode, enable_thinking
        self.system = system
        self.reset()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system}] if self.system else []
        self.cache = None
        self.fed_ids: list = []       # every token the cache has seen, in order
        self.turns: list = []

    def ask(self, user: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> GenerationResult:
        if not isinstance(user, str) or not user.strip():
            raise InputError("user text must be a non-empty string")
        trial = self.messages + [{"role": "user", "content": user}]
        try:
            full = self.engine.render_chat(trial, self.enable_thinking)
            extends = self.cache is not None and len(full) > len(self.fed_ids) and full[: len(self.fed_ids)] == self.fed_ids
            if self.mode == "incremental" and extends:
                res = self.engine.generate_ids(full[len(self.fed_ids):], max_tokens, cache=self.cache, cache_tokens=len(self.fed_ids), mode="incremental")
            else:
                res = self.engine.generate_ids(full, max_tokens, mode=("fresh" if self.mode == "replay" or self.cache is None else "replay-fallback"))
                if res.mode == "replay-fallback":
                    res.note = "the rendered transcript does not extend the fed token sequence; replayed into a fresh cache"
        except BaseException:
            # a refused, failed or interrupted turn leaves the transcript as it
            # was and discards the cache (it may be partly fed)
            self.cache = None
            raise
        self.messages = trial
        self.fed_ids = full + res.generated_ids[: res.fed_generated]
        if res.cache_tokens != len(self.fed_ids) or (res.cache_offset_measured is not None and res.cache_offset_measured != len(self.fed_ids)):
            raise RuntimeError(f"cache accounting mismatch: {res.cache_tokens} / {res.cache_offset_measured} != {len(self.fed_ids)}")
        # replay mode holds no cache between turns; incremental holds exactly one
        self.cache = res.cache if self.mode == "incremental" else None
        reasoning, content = split_thinking(res.text)
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        self.messages.append(msg)
        res.cache = None if self.mode == "replay" else res.cache
        self.turns.append(dataclasses.replace(res, cache=None))
        return res
