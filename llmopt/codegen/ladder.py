"""The capability ladder: five rungs from table lookup to optimizer
simulation, every answer scored by the toolchain.

Rungs, easiest to hardest (hypothesis: small models climb until the task
stops being a learned mapping and starts requiring simulation):

1. encode   — intel mnemonic -> hex bytes. Deterministic table.
2. decode   — hex bytes -> mnemonic. Scored by *assembling the model's
              answer* (llvm-mc) and comparing bytes: any valid syntax
              for the right instruction passes; hallucinations fail to
              assemble and score zero.
3. output   — tiny C program -> what it prints. Interpretation.
4. diagnose — C file -> "compiles" or the compiler's error line.
5. o2_asm   — C function -> the exact -O2 asm body clang emits.
              Optimizer simulation; scored by normalized text match
              (the compiler is deterministic) with assemble-equivalence
              as a fallback for syntax variation.

Rung tasks all carry (prompt, target, check); datasets are built from
seeded generator programs, so train/eval splits use disjoint seeds and
dedup like mathgen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from llmopt.codegen.generator import make_programs
from llmopt.codegen.llvm import assemble, compile_c, norm_bytes

RUNGS = ("encode", "decode", "mangle", "lifetime", "output", "diagnose", "o2_asm")


@dataclass(frozen=True)
class RungTask:
    rung: str
    prompt: str
    target: str
    check: Callable[[str], bool] = field(compare=False, repr=False)


def _norm_mnemonic(m: str) -> str:
    return re.sub(r"\s+", " ", m).strip()


def _norm_asm(a: str) -> str:
    lines = [re.sub(r"\s+", " ", l).strip() for l in a.strip().splitlines()]
    return "\n".join(l for l in lines if l)


def _first_line(pred: str) -> str:
    for line in pred.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def _encode_task(by: str, mn: str) -> RungTask:
    target = norm_bytes(by)
    return RungTask(
        "encode",
        f"Assemble this x86-64 instruction to hex machine-code bytes: {mn}",
        target,
        lambda pred, t=target: norm_bytes(_first_line(pred)) == t,
    )


def _decode_task(by: str, mn: str) -> RungTask:
    target_bytes = norm_bytes(by)

    def check(pred: str, t=target_bytes) -> bool:
        got = assemble(_first_line(pred))
        return got is not None and norm_bytes(got) == t

    return RungTask(
        "decode",
        f"Disassemble these x86-64 bytes to one Intel-syntax instruction: {target_bytes}",
        _norm_mnemonic(mn), check,
    )


def _output_task(source: str, stdout: str) -> RungTask:
    target = stdout.strip()
    return RungTask(
        "output",
        f"What does this C program print?\n```c\n{source}\n```",
        target,
        lambda pred, t=target: _first_line(pred) == t,
    )


def _diagnose_task(source: str, diags: list[str]) -> RungTask:
    if not diags:
        return RungTask(
            "diagnose",
            f"Does this C file compile without errors? Answer 'compiles' or "
            f"give the error.\n```c\n{source}\n```",
            "compiles",
            lambda pred: "compile" in pred.lower() and "not" not in pred.lower()[:40],
        )
    ident = re.search(r"'(\w+)'", diags[0])
    name = ident.group(1) if ident else ""
    return RungTask(
        "diagnose",
        f"Does this C file compile without errors? Answer 'compiles' or "
        f"give the error.\n```c\n{source}\n```",
        diags[0],
        lambda pred, n=name: "undeclared" in pred.lower() and n in pred,
    )


def _extract_fn_body(asm: str, fn: str) -> str:
    lines = asm.splitlines()
    out, active = [], False
    for l in lines:
        if l.strip() == f"{fn}:":
            active = True
            continue
        if active:
            if l and not l[0] in " \t" and l.rstrip().endswith(":") and not l.startswith("."):
                break  # next symbol
            if l.strip() == "ret":
                out.append(l.strip())
                break
            out.append(l.strip())
    return _norm_asm("\n".join(out))


def _o2_asm_task(source: str, asm: str) -> RungTask | None:
    body = _extract_fn_body(asm, "pick")
    if not body or len(body.splitlines()) > 16:
        return None
    fn_src = source.split("\n")[1]  # the pick() definition line

    def check(pred: str, t=body) -> bool:
        p = _norm_asm(pred)
        if p == t:
            return True
        # syntax-variation fallback: assemble both line by line
        pl, tl = p.splitlines(), t.splitlines()
        if len(pl) != len(tl):
            return False
        for a, b in zip(pl, tl):
            ea, eb = assemble(a), assemble(b)
            if ea is None or ea != eb:
                return False
        return True

    return RungTask(
        "o2_asm",
        "Give the exact x86-64 Intel-syntax assembly body clang -O2 emits "
        f"for this C function (instructions only, one per line):\n```c\n{fn_src}\n```",
        body, check,
    )


def _mangle_task(level: int, seed: int) -> RungTask | None:
    """Itanium name mangling: another encode-class rung (learned mapping,
    like instruction encoding). Prediction scored by demangling it — any
    symbol that demangles to the same signature passes."""
    from llmopt.codegen.generator import make_signature
    from llmopt.codegen.llvm import demangle, mangle

    src, fn, human = make_signature(level, seed)
    target = mangle(src, fn)
    if target is None:
        return None
    ref = demangle(target)

    def check(pred: str, t=target, r=ref) -> bool:
        p = _first_line(pred)
        return p == t or (r is not None and demangle(p) == r)

    return RungTask(
        "mangle",
        f"Give the Itanium-ABI mangled symbol name for this C++ function: {human}",
        target, check,
    )


def _lifetime_task(level: int, seed: int) -> RungTask | None:
    from llmopt.codegen.generator import make_lifetime
    from llmopt.codegen.llvm import compile_cpp

    prog = make_lifetime(level, seed)
    r = compile_cpp(prog.source, run=True)
    if not r.ok or not r.stdout:
        return None
    target = "\n".join(l.strip() for l in r.stdout.strip().splitlines())

    def check(pred: str, t=target) -> bool:
        got = [l.strip() for l in pred.strip().splitlines() if l.strip()]
        return "\n".join(got) == t

    return RungTask(
        "lifetime",
        "This C++ program prints on every construction and destruction. "
        f"Give its exact output, one line per event:\n```cpp\n{prog.source}\n```",
        target, check,
    )


def build_ladder(
    n_programs: int, *, seed: int = 0, exclude: frozenset[str] = frozenset(),
) -> dict[str, list[RungTask]]:
    """Compile seeded programs once; harvest every rung from the results."""
    tasks: dict[str, list[RungTask]] = {r: [] for r in RUNGS}
    seen: set[str] = set(exclude)

    def add(t: RungTask | None):
        if t is not None and t.prompt not in seen:
            seen.add(t.prompt)
            tasks[t.rung].append(t)

    clean = make_programs(n_programs, kinds=("arith", "loop", "branch"), seed=seed)
    buggy = make_programs(
        max(n_programs // 3, 1), kinds=("bug_undeclared",), seed=seed
    )
    for prog in clean:
        r = compile_c(prog.source, run=True)
        if not r.ok:
            continue
        for by, mn in r.encodings:
            got = assemble(mn)  # keep only self-consistent pairs (no symbols)
            if got is not None and norm_bytes(got) == norm_bytes(by):
                add(_encode_task(by, mn))
                add(_decode_task(by, mn))
        if r.stdout:
            add(_output_task(prog.source, r.stdout))
        add(_diagnose_task(prog.source, r.diagnostics))
        if prog.kind == "branch":
            add(_o2_asm_task(prog.source, r.asm))
    for prog in buggy:
        r = compile_c(prog.source)
        add(_diagnose_task(prog.source, r.diagnostics))
    for i in range(n_programs):
        add(_mangle_task(1 + i % 3, seed * 1_000_000 + i))
        if i < n_programs // 2:
            add(_lifetime_task(1 + i % 3, seed * 1_000_000 + i))
    return tasks


def evaluate_ladder(
    generate_fn: Callable[[str], str], tasks: Mapping[str, Sequence[RungTask]],
) -> dict[str, float]:
    """generate_fn(prompt) -> completion; accuracy per rung."""
    return {
        rung: sum(t.check(generate_fn(t.prompt)) for t in ts) / len(ts)
        for rung, ts in tasks.items() if ts
    }
