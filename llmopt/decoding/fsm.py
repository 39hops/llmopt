"""Regex-constrained decoding via a token-level FSM (Outlines-style).

Pipeline: regex (small subset) -> Thompson NFA -> DFA over characters ->
token-level transition table over the tokenizer vocab. At each decode
step only tokens whose full string walks the DFA from the current state
are allowed; EOS is allowed only in accepting states. Guarantees every
completed generation matches the regex.

Supported regex subset: literals, ``.``, ``[abc]``/``[a-z]``/``[^...]``,
``\\d``/``\\w``/``\\s``, ``*``, ``+``, ``?``, ``|``, ``(...)``, ``\\``
escapes. Char universe is printable ASCII.
"""

from __future__ import annotations

import string
from typing import Mapping, Sequence

ALPHABET = frozenset(string.printable)
EPS = ""  # epsilon edge label

_CLASSES = {
    "d": frozenset(string.digits),
    "w": frozenset(string.ascii_letters + string.digits + "_"),
    "s": frozenset(" \t\n\r\x0b\x0c"),
}


# --- regex -> NFA (Thompson construction) --------------------------------
# NFA: dict state -> list[(label, next_state)]; label is a frozenset of
# chars or EPS. Fragments are (start, accept) pairs.


class _NFA:
    def __init__(self):
        self.edges: dict[int, list[tuple[frozenset | str, int]]] = {}
        self.n = 0

    def state(self) -> int:
        s = self.n
        self.n += 1
        self.edges[s] = []
        return s

    def add(self, a: int, label, b: int) -> None:
        self.edges[a].append((label, b))


def _parse_class(pattern: str, i: int) -> tuple[frozenset, int]:
    """Parse [...] starting after '['; return (chars, index past ']')."""
    negate = pattern[i] == "^"
    if negate:
        i += 1
    chars: set[str] = set()
    while pattern[i] != "]":
        if pattern[i] == "\\":
            c = pattern[i + 1]
            chars |= _CLASSES.get(c, frozenset(c))
            i += 2
        elif pattern[i + 1 : i + 2] == "-" and pattern[i + 2 : i + 3] != "]":
            chars |= set(map(chr, range(ord(pattern[i]), ord(pattern[i + 2]) + 1)))
            i += 3
        else:
            chars.add(pattern[i])
            i += 1
    result = frozenset(chars)
    return (ALPHABET - result if negate else result), i + 1


def _compile_nfa(pattern: str) -> tuple[_NFA, int, int]:
    """Recursive-descent regex -> NFA. Returns (nfa, start, accept)."""
    nfa = _NFA()

    def atom(i: int) -> tuple[int, int, int]:
        c = pattern[i]
        if c == "(":
            s, a, i = alt(i + 1)
            assert pattern[i] == ")", f"unbalanced paren at {i}"
            return s, a, i + 1
        if c == "[":
            chars, i = _parse_class(pattern, i + 1)
        elif c == "\\":
            nxt = pattern[i + 1]
            chars = _CLASSES.get(nxt, frozenset(nxt))
            i += 2
        elif c == ".":
            chars, i = frozenset(ALPHABET - {"\n"}), i + 1
        else:
            chars, i = frozenset(c), i + 1
        s, a = nfa.state(), nfa.state()
        nfa.add(s, chars, a)
        return s, a, i

    def repeat(i: int) -> tuple[int, int, int]:
        s, a, i = atom(i)
        while i < len(pattern) and pattern[i] in "*+?":
            op = pattern[i]
            ns, na = nfa.state(), nfa.state()
            nfa.add(ns, EPS, s)
            nfa.add(a, EPS, na)
            if op in "*?":
                nfa.add(ns, EPS, na)
            if op in "*+":
                nfa.add(a, EPS, s)
            s, a, i = ns, na, i + 1
        return s, a, i

    def concat(i: int) -> tuple[int, int, int]:
        s, a, i = repeat(i)
        while i < len(pattern) and pattern[i] not in "|)":
            s2, a2, i = repeat(i)
            nfa.add(a, EPS, s2)
            a = a2
        return s, a, i

    def alt(i: int) -> tuple[int, int, int]:
        s, a, i = concat(i)
        while i < len(pattern) and pattern[i] == "|":
            s2, a2, i = concat(i + 1)
            ns, na = nfa.state(), nfa.state()
            for x, y in ((s, a), (s2, a2)):
                nfa.add(ns, EPS, x)
                nfa.add(y, EPS, na)
            s, a = ns, na
        return s, a, i

    s, a, i = alt(0)
    assert i == len(pattern), f"trailing regex at {i}: {pattern[i:]!r}"
    return nfa, s, a


# --- NFA -> DFA (subset construction) -------------------------------------


class DFA:
    """transitions[state][char] -> state; state 0 is the start."""

    def __init__(
        self, transitions: list[dict[str, int]], accepting: frozenset[int]
    ):
        self.transitions = transitions
        self.accepting = accepting

    def walk(self, state: int, text: str) -> int | None:
        for ch in text:
            nxt = self.transitions[state].get(ch)
            if nxt is None:
                return None
            state = nxt
        return state


def compile_regex(pattern: str) -> DFA:
    nfa, start, accept = _compile_nfa(pattern)

    def closure(states: frozenset[int]) -> frozenset[int]:
        stack, seen = list(states), set(states)
        while stack:
            for label, nxt in nfa.edges[stack.pop()]:
                if label == EPS and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return frozenset(seen)

    start_set = closure(frozenset([start]))
    index = {start_set: 0}
    transitions: list[dict[str, int]] = [{}]
    todo = [start_set]
    while todo:
        cur = todo.pop()
        moves: dict[str, set[int]] = {}
        for st in cur:
            for label, nxt in nfa.edges[st]:
                if label == EPS:
                    continue
                for ch in label:
                    moves.setdefault(ch, set()).add(nxt)
        for ch, nxts in moves.items():
            tgt = closure(frozenset(nxts))
            if tgt not in index:
                index[tgt] = len(transitions)
                transitions.append({})
                todo.append(tgt)
            transitions[index[cur]][ch] = index[tgt]
    accepting = frozenset(
        i for s, i in index.items() if accept in s
    )
    return DFA(transitions, accepting)


# --- token-level FSM -------------------------------------------------------


class TokenFSM:
    """Precompute, per DFA state, which vocab tokens are walkable.

    vocab maps token_id -> decoded string. allowed[state] maps
    token_id -> next_state; built lazily per visited state so huge
    vocabs only pay for states actually reached.
    """

    def __init__(self, dfa: DFA, vocab: Mapping[int, str]):
        self.dfa = dfa
        self.vocab = vocab
        self._allowed: dict[int, dict[int, int]] = {}

    def allowed(self, state: int) -> dict[int, int]:
        cached = self._allowed.get(state)
        if cached is None:
            cached = {}
            for tid, text in self.vocab.items():
                if not text:
                    continue
                nxt = self.dfa.walk(state, text)
                if nxt is not None:
                    cached[tid] = nxt
            self._allowed[state] = cached
        return cached

    def is_accepting(self, state: int) -> bool:
        return state in self.dfa.accepting


class FSMConstraint:
    """Stateful logits processor: mask to tokens walkable from the
    current DFA state; EOS allowed only in accepting states.

    Use with SamplerPipeline processors, then call ``advance`` with the
    chosen token before the next step.
    """

    def __init__(self, fsm: TokenFSM, eos_token_id: int):
        self.fsm = fsm
        self.eos = eos_token_id
        self.state: int | None = 0  # None once finished (EOS chosen)

    def __call__(self, logits, ctx: Sequence[int] = ()):
        import torch

        assert self.state is not None, "generation already finished (EOS)"
        allowed = self.fsm.allowed(self.state)
        keep = torch.zeros_like(logits, dtype=torch.bool)
        for tid in allowed:
            keep[tid] = True
        if self.fsm.is_accepting(self.state):
            keep[self.eos] = True
        assert keep.any(), f"dead FSM state {self.state}: nothing decodable"
        return logits.masked_fill(~keep, float("-inf"))

    def advance(self, token_id: int) -> None:
        if token_id == self.eos:
            self.state = None
            return
        assert self.state is not None, "generation already finished (EOS)"
        self.state = self.fsm.allowed(self.state)[token_id]

    @property
    def finished(self) -> bool:
        return self.state is None
