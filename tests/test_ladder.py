"""LLVM oracle + capability ladder (skips without MSYS clang/llvm-mc)."""

import pytest

from llmopt.codegen.llvm import assemble, compile_c, llvm_available, norm_bytes

pytestmark = pytest.mark.skipif(not llvm_available(), reason="LLVM toolchain not found")


def test_clang_labels_and_run():
    src = '#include <stdio.h>\nint main(void){ printf("%d\\n", 6*7); return 0; }'
    r = compile_c(src, run=True)
    assert r.ok and not r.diagnostics
    assert r.stdout.strip() == "42"
    assert len(r.encodings) > 2


def test_clang_diagnostic():
    r = compile_c("int main(void){ return nope; }")
    assert not r.ok
    assert any("undeclared" in d for d in r.diagnostics)


def test_assemble_and_norm():
    assert norm_bytes(assemble("mov eax, 5")) == "b8 05 00 00 00"
    assert assemble("frobnicate rax, 7") is None  # hallucination -> None


def test_ladder_targets_self_verify():
    from llmopt.codegen.ladder import build_ladder

    tasks = build_ladder(6, seed=42)
    assert all(len(ts) > 0 for r, ts in tasks.items() if r != "o2_asm")
    for ts in tasks.values():
        for t in ts:
            assert t.check(t.target), (t.rung, t.prompt[:60], t.target[:60])


def test_decode_scored_by_assembling_not_text():
    from llmopt.codegen.ladder import _decode_task

    t = _decode_task("b8 05 00 00 00", "mov    eax,0x5")
    assert t.check("mov eax, 5")        # different spelling, same bytes
    assert not t.check("mov eax, 6")    # wrong immediate
    assert not t.check("blorp eax, 5")  # does not assemble


def test_mangle_scored_by_demangling():
    from llmopt.codegen.ladder import _mangle_task
    from llmopt.codegen.llvm import mangle

    t = _mangle_task(1, 3)
    assert t is not None and t.check(t.target)
    assert not t.check("_Z3fooid")
    assert not t.check("not_a_symbol")


def test_lifetime_output_is_scope_ordered():
    from llmopt.codegen.ladder import _lifetime_task

    t = _lifetime_task(1, 5)
    assert t is not None
    lines = t.target.splitlines()
    ctors = [l[1:] for l in lines if l.startswith("+")]
    dtors = [l[1:] for l in lines if l.startswith("-")]
    assert sorted(ctors) == sorted(dtors)          # every ctor has a dtor
    assert dtors[-1] == ctors[0]                   # outermost dies last
    assert t.check(t.target) and not t.check("+A\n-A")


def test_train_eval_exclusion():
    from llmopt.codegen.ladder import build_ladder

    ev = build_ladder(6, seed=42)
    banned = frozenset(t.prompt for ts in ev.values() for t in ts)
    tr = build_ladder(10, seed=7, exclude=banned)
    tr_prompts = {t.prompt for ts in tr.values() for t in ts}
    assert not (tr_prompts & banned)
