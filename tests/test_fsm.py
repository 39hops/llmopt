"""Regex->DFA compiler and token-FSM constrained decoding tests (CPU)."""

import re

import torch

from llmopt.decoding.fsm import DFA, FSMConstraint, TokenFSM, compile_regex
from llmopt.decoding.samplers import SamplerPipeline


def _matches(dfa: DFA, text: str) -> bool:
    end = dfa.walk(0, text)
    return end is not None and end in dfa.accepting


def test_dfa_matches_python_re():
    cases = {
        r"ab+c?": ["ab", "abbb", "abbc", "abc"],
        r"(foo|bar)*": ["", "foo", "barfoo", "foofoo"],
        r"\d+\.\d\d": ["3.14", "10.00"],
        r"[a-c]+[^x]": ["abcz", "ay"],
        r'"[\w ]*"': ['"hello world"', '""'],
    }
    for pattern, positives in cases.items():
        dfa = compile_regex(pattern)
        for text in positives:
            assert _matches(dfa, text), (pattern, text)
            assert re.fullmatch(pattern, text)
        for text in ["xx", "3.1415x", "abx!q"]:
            assert _matches(dfa, text) == bool(
                re.fullmatch(pattern, text)
            ), (pattern, text)


def _vocab():
    # multi-char tokens force mid-token DFA walks; id 0 is EOS
    toks = ["<eos>", "f", "o", "fo", "oo", "ba", "r", "bar", "1", "23",
            ".", "9", "x", "ab", "c"]
    return {i: t for i, t in enumerate(toks)}


def test_constrained_generation_always_matches():
    pattern = r"(foo|bar)+"
    fsm = TokenFSM(compile_regex(pattern), _vocab())
    for seed in range(20):
        con = FSMConstraint(fsm, eos_token_id=0)
        pipe = SamplerPipeline(con, seed=seed)
        gen = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(12):
            logits = torch.randn(len(_vocab()), generator=gen)
            tok = pipe(logits)
            con.advance(tok)
            if con.finished:
                break
            out.append(tok)
        text = "".join(_vocab()[t] for t in out)
        if con.finished:
            assert re.fullmatch(pattern, text), text
        else:  # ran out of budget: still a valid prefix (walkable)
            assert fsm.dfa.walk(0, text) is not None


def test_eos_only_at_accepting_state():
    fsm = TokenFSM(compile_regex(r"\d\.\d"), _vocab())
    con = FSMConstraint(fsm, eos_token_id=0)
    logits = torch.zeros(len(_vocab()))
    masked = con(logits)
    assert masked[0] == float("-inf")  # empty string not accepting
    for tok_text in ("1", ".", "9"):
        tid = next(i for i, t in _vocab().items() if t == tok_text)
        con.advance(tid)
    assert con(logits)[0] == 0.0  # "1.9" accepting -> EOS allowed


def test_multi_char_token_crosses_states():
    fsm = TokenFSM(compile_regex(r"\d+"), _vocab())
    allowed = fsm.allowed(0)
    tid_23 = next(i for i, t in _vocab().items() if t == "23")
    assert tid_23 in allowed  # "23" walks two transitions at once
    tid_x = next(i for i, t in _vocab().items() if t == "x")
    assert tid_x not in allowed
