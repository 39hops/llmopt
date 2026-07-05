"""RULER-style synthetic long-context eval.

Claimed context length != usable context length. RULER's trick: generate
retrieval tasks at any target length from templates, so degradation can be
measured along the length axis with exact-match scoring — no dataset
needed, no judge model.

Tasks (the core RULER families, minus the summarization-ish ones):

- niah: needle-in-a-haystack — k (key, value) needles buried in filler
  text, query one key. `num_keys > 1` is the multi-key variant (harder:
  distractor needles compete with the target).
- variable_tracking: chained aliasing (X2 = X1, X3 = X2, ...) spread
  through filler; resolving the final variable needs every hop.

`evaluate` takes a plain `generate_fn(prompt) -> str` so any backend
(HF, MLX, an API) plugs in. `effective_context_length` applies RULER's
rule: the longest length where accuracy stays >= threshold (85%).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

_FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)
_WORDS = (
    "apple ocean tiger violet ember crystal falcon meadow ripple canyon "
    "harbor lantern mosaic nectar orchid prism quartz saffron thistle willow"
).split()


@dataclass(frozen=True)
class Task:
    prompt: str
    answer: str
    kind: str
    meta: dict = field(default_factory=dict, compare=False)


def _fill(rng: random.Random, num_words: int) -> list[str]:
    filler = _FILLER.split()
    return [filler[i % len(filler)] for i in range(num_words)]


def _scatter(words: list[str], inserts: Sequence[str], rng: random.Random) -> str:
    """Insert each sentence at a random (sorted) position in the filler."""
    slots = sorted(rng.sample(range(1, len(words)), len(inserts)))
    out, prev = [], 0
    for slot, ins in zip(slots, inserts):
        out += words[prev:slot] + [ins]
        prev = slot
    return " ".join(out + words[prev:])


def make_niah(
    length_words: int, *, num_keys: int = 1, seed: int = 0
) -> Task:
    """k needles "The magic number for <key> is <value>." in filler; query
    one of them. num_keys > 1 = RULER's multi-key variant."""
    rng = random.Random(seed)
    keys = rng.sample(_WORDS, num_keys)
    vals = [str(rng.randint(100000, 999999)) for _ in keys]
    needles = [f"The magic number for {k} is {v}." for k, v in zip(keys, vals)]
    target = rng.randrange(num_keys)
    body = _scatter(_fill(rng, length_words), needles, rng)
    prompt = (
        f"{body}\n\nWhat is the magic number for {keys[target]}? "
        "Answer with the number only."
    )
    return Task(prompt, vals[target], "niah" if num_keys == 1 else "niah_multikey",
                {"num_keys": num_keys, "length_words": length_words})


def make_variable_tracking(
    length_words: int, *, hops: int = 3, seed: int = 0
) -> Task:
    """X1 = <value>; X2 = X1; ... Xn = Xn-1 scattered through filler.
    Resolving Xn requires following every hop."""
    rng = random.Random(seed)
    value = str(rng.randint(10000, 99999))
    names = [f"VAR_{w.upper()}" for w in rng.sample(_WORDS, hops)]
    chain = [f"Set {names[0]} = {value}."]
    chain += [f"Set {names[i]} = {names[i - 1]}." for i in range(1, hops)]
    rng.shuffle(chain)  # hops appear out of order — no positional shortcut
    body = _scatter(_fill(rng, length_words), chain, rng)
    prompt = (
        f"{body}\n\nWhat is the value of {names[-1]}? "
        "Answer with the number only."
    )
    return Task(prompt, value, "variable_tracking",
                {"hops": hops, "length_words": length_words})


def make_suite(
    lengths: Sequence[int],
    *,
    samples_per_length: int = 5,
    tasks: Sequence[str] = ("niah", "niah_multikey", "variable_tracking"),
    seed: int = 0,
) -> list[Task]:
    out = []
    for length in lengths:
        for i in range(samples_per_length):
            s = seed * 100003 + length * 101 + i
            if "niah" in tasks:
                out.append(make_niah(length, seed=s))
            if "niah_multikey" in tasks:
                out.append(make_niah(length, num_keys=4, seed=s))
            if "variable_tracking" in tasks:
                out.append(make_variable_tracking(length, seed=s))
    return out


def score(prediction: str, task: Task) -> bool:
    """Exact-match containment — the RULER metric."""
    return task.answer in prediction


def evaluate(
    generate_fn: Callable[[str], str], suite: Sequence[Task]
) -> dict[tuple[str, int], float]:
    """Accuracy per (task kind, length_words)."""
    hits: dict[tuple[str, int], list[bool]] = {}
    for task in suite:
        key = (task.kind, task.meta["length_words"])
        hits.setdefault(key, []).append(score(generate_fn(task.prompt), task))
    return {k: sum(v) / len(v) for k, v in hits.items()}


def effective_context_length(
    results: dict[tuple[str, int], float], *, threshold: float = 0.85
) -> int:
    """Longest length where mean accuracy across kinds stays >= threshold
    (0 if even the shortest fails). RULER's headline number."""
    by_len: dict[int, list[float]] = {}
    for (_, length), acc in results.items():
        by_len.setdefault(length, []).append(acc)
    good = 0
    for length in sorted(by_len):
        if sum(by_len[length]) / len(by_len[length]) >= threshold:
            good = length
        else:
            break
    return good
