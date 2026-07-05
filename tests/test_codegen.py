"""Compiler oracle + C generator (skips without MSVC)."""

import pytest

from llmopt.codegen.generator import ALL_KINDS, make_arith, make_programs
from llmopt.codegen.oracle import compile_c, msvc_available

pytestmark = pytest.mark.skipif(not msvc_available(), reason="MSVC not found")


def test_generator_deterministic_and_unique():
    a = make_programs(20, seed=3)
    b = make_programs(20, seed=3)
    assert [p.source for p in a] == [p.source for p in b]
    assert len({p.source for p in a}) == 20


def test_clean_program_full_labels():
    p = make_programs(1, kinds=("loop",), seed=1)[0]
    r = compile_c(p.source, run=True)
    assert r.ok and not r.diagnostics
    assert r.stdout is not None and r.stdout.strip().lstrip("-").isdigit()
    assert "main" in r.asm.lower()
    assert len(r.encodings) > 3  # (bytes, mnemonic) pairs harvested
    by, mn = r.encodings[0]
    assert all(len(x) == 2 for x in by.split())
    assert mn


def test_program_output_matches_python_semantics():
    # loop programs have transparent semantics: check the oracle's stdout
    # against an independent Python evaluation of the same loop
    p = make_programs(1, kinds=("loop",), seed=7)[0]
    src = p.source
    lo = int(src.split("int i = ")[1].split(";")[0])
    hi = int(src.split("i < ")[1].split(";")[0])
    body = src.split("{\n        ")[1].split("\n")[0]
    acc = 0
    for i in range(lo, hi):
        if body == "acc += i;":
            acc += i
        elif body == "acc += i * i;":
            acc += i * i
        else:
            acc += i * int(body.split("* ")[1].rstrip(";"))
    r = compile_c(p.source, run=True)
    assert int(r.stdout.strip()) == acc


def test_planted_bug_yields_exact_diagnostic():
    from llmopt.codegen.generator import make_bug_undeclared

    r = compile_c(make_bug_undeclared(1, 5).source)
    assert not r.ok
    assert any("C2065" in d for d in r.diagnostics)


def test_optimization_levels_differ():
    p = make_programs(1, kinds=("loop",), seed=2)[0]
    o0 = compile_c(p.source, opt="/Od")
    o2 = compile_c(p.source, opt="/O2")
    assert o0.ok and o2.ok
    # /O2 folds the constant loop; /Od keeps it — different code
    assert o0.asm != o2.asm


def test_all_kinds_compile_or_fail_as_designed():
    for kind in ALL_KINDS:
        p = make_programs(1, kinds=(kind,), seed=11)[0]
        r = compile_c(p.source)
        if kind == "bug_undeclared":
            assert not r.ok
        else:
            assert r.ok, (kind, r.diagnostics)
