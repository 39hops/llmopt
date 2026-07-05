"""Backend-agnostic lookup loop tested against a fake deterministic backend.

The fake model's next-token depends on the whole prefix (sum of tokens),
so any error in fed positions, pad handling, or rewind bookkeeping breaks
equivalence with the plain greedy reference.
"""

from llmopt.decoding.lookup_generic import generate_lookup

VOCAB = 17


def _next_token(prefix: list[int]) -> int:
    return (sum(prefix) * 31 + 7) % VOCAB


def _greedy_reference(prompt: list[int], n: int) -> list[int]:
    tokens = list(prompt)
    for _ in range(n):
        tokens.append(_next_token(tokens))
    return tokens


class FakeBackend:
    """Simulates a position-indexed KV cache over the fake model."""

    def __init__(self):
        self.buf: list[int] = []

    def begin(self, prompt_ids, max_len):
        self.buf = list(prompt_ids) + [0] * (max_len - len(prompt_ids))
        self.valid = len(prompt_ids)
        return _next_token(self.buf[: self.valid])

    def step_argmax(self, fed, start_pos, n_real):
        for i, t in enumerate(fed):
            self.buf[start_pos + i] = t
        return [
            _next_token(self.buf[: start_pos + j + 1]) for j in range(n_real)
        ]

    def rewind(self, length):
        self.valid = length


def test_lookup_generic_matches_greedy_reference():
    # Arrange: repetitive prompt so lookup actually drafts
    prompt = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5]
    n = 40

    # Act
    out, stats = generate_lookup(
        FakeBackend(), prompt, max_new_tokens=n, num_draft=4, max_ngram=3
    )

    # Assert
    assert out == _greedy_reference(prompt, n)
    assert stats["forward_passes"] <= n + 1


def test_lookup_generic_no_draft_prompt():
    # Arrange: no repeated ngrams -> pure 1-token steps via pads
    prompt = [3, 9, 12, 5]
    n = 10

    # Act
    out, stats = generate_lookup(
        FakeBackend(), prompt, max_new_tokens=n, num_draft=4, max_ngram=3
    )

    # Assert
    assert out == _greedy_reference(prompt, n)
    assert len(out) == len(prompt) + n


def test_lookup_generic_eos_stops():
    prompt = [1, 2, 3, 4, 1, 2, 3, 4]
    full = _greedy_reference(prompt, 40)
    eos = full[len(prompt) + 5]  # some token that will appear

    out, _ = generate_lookup(
        FakeBackend(), prompt, max_new_tokens=40, num_draft=4, eos_token_id=eos
    )

    assert out[-1] == eos
    assert out == full[: len(out)]
