"""REST-style datastore drafting (He et al. 2023, "REST: Retrieval-Based
Speculative Decoding").

Prompt-lookup can only draft text that already appears in the current
context. A retrieval datastore generalizes it: index every sequence the
model has produced (or any corpus), and draft by longest-suffix match
against the whole store. Models repeat themselves across requests —
boilerplate, code idioms, phrasing — so accept rates climb with use, and
the draft source stays a lookup table: no draft model, still exact
verify-or-reject.

SuffixDatastore indexes n-gram positions (n = 1..max_ngram). draft()
tries the longest context suffix first; among occurrences it majority-
votes the next token (REST's trie vote, one level), then copies the
continuation from the most recent winning occurrence.

make_draft_fn() composes datastore + prompt-lookup for generate_lookup:
whichever source produces the longer draft wins ties to the datastore.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from llmopt.decoding.prompt_lookup import find_ngram_continuation


class SuffixDatastore:
    """N-gram position index over stored token sequences."""

    def __init__(self, max_ngram: int = 4, max_sequences: int = 10_000):
        self.max_ngram = max_ngram
        self.max_sequences = max_sequences
        self.seqs: list[list[int]] = []
        # (n-gram tuple) -> list of (seq_idx, end_pos); end_pos = index
        # just past the n-gram, i.e. where the continuation starts
        self.index: dict[tuple, list[tuple[int, int]]] = {}

    def add(self, tokens: Sequence[int]) -> None:
        tokens = list(tokens)
        if len(self.seqs) >= self.max_sequences:
            return  # simple cap; a real store would evict
        sid = len(self.seqs)
        self.seqs.append(tokens)
        for n in range(1, self.max_ngram + 1):
            for i in range(len(tokens) - n):
                key = tuple(tokens[i : i + n])
                self.index.setdefault(key, []).append((sid, i + n))

    def draft(self, context: Sequence[int], num_draft: int) -> list[int]:
        """Longest-suffix match, majority vote on the next token, copy the
        continuation from the most recent winning occurrence."""
        context = list(context)
        for n in range(min(self.max_ngram, len(context)), 0, -1):
            hits = self.index.get(tuple(context[-n:]))
            if not hits:
                continue
            votes = Counter(
                self.seqs[sid][pos] for sid, pos in hits
                if pos < len(self.seqs[sid])
            )
            if not votes:
                continue
            winner = votes.most_common(1)[0][0]
            for sid, pos in reversed(hits):  # most recent occurrence wins
                seq = self.seqs[sid]
                if pos < len(seq) and seq[pos] == winner:
                    return seq[pos : pos + num_draft]
        return []


def make_draft_fn(datastore: SuffixDatastore, *, max_ngram: int = 3):
    """Draft function for generate_lookup: datastore retrieval with
    prompt-lookup as fallback; the longer draft wins."""

    def draft_fn(tokens: Sequence[int], num_draft: int) -> list[int]:
        from_store = datastore.draft(tokens, num_draft)
        from_prompt = find_ngram_continuation(
            list(tokens), max_ngram=max_ngram, num_draft=num_draft
        )[:num_draft]
        return from_store if len(from_store) >= len(from_prompt) else from_prompt

    return draft_fn
