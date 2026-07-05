"""Seeded tiny-C program generator for the capability ladder.

Same philosophy as mathgen: programs are constructed, labels come from
the oracle (codegen/oracle.py), determinism comes from string seeds.
Programs are total by construction — bounded loops, no UB in the
'clean' variants, deterministic printf output — so `run` labels are
well-defined. The `bug` variants plant one specific defect (use of an
undeclared identifier, missing return value use, format mismatch) so
diagnostic-prediction tasks have exact expected labels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class CProgram:
    source: str
    kind: str    # arith | loop | branch | array | bug_undeclared
    level: int
    seed: int


def _expr(rng: random.Random, vars_: list[str], depth: int) -> str:
    if depth == 0 or rng.random() < 0.4:
        return rng.choice(vars_ + [str(rng.randint(1, 9))])
    op = rng.choice(["+", "-", "*"])
    return f"({_expr(rng, vars_, depth - 1)} {op} {_expr(rng, vars_, depth - 1)})"


def make_arith(level: int, seed: int) -> CProgram:
    rng = random.Random(f"arith-{level}-{seed}")
    n = 2 + level
    decls = [f"    int v{i} = {rng.randint(1, 20)};" for i in range(n)]
    vars_ = [f"v{i}" for i in range(n)]
    body = [f"    int r = {_expr(rng, vars_, level + 1)};"]
    src = "\n".join(
        ["#include <stdio.h>", "int main(void) {"]
        + decls + body + ['    printf("%d\\n", r);', "    return 0;", "}"]
    )
    return CProgram(src, "arith", level, seed)


def make_loop(level: int, seed: int) -> CProgram:
    rng = random.Random(f"loop-{level}-{seed}")
    lo, hi = rng.randint(0, 3), rng.randint(5, 8 + 4 * level)
    step_expr = rng.choice(["acc += i;", "acc += i * i;", f"acc += i * {rng.randint(2, 5)};"])
    src = "\n".join([
        "#include <stdio.h>",
        "int main(void) {",
        "    int acc = 0;",
        f"    for (int i = {lo}; i < {hi}; i++) {{",
        f"        {step_expr}",
        "    }",
        '    printf("%d\\n", acc);',
        "    return 0;",
        "}",
    ])
    return CProgram(src, "loop", level, seed)


def make_branch(level: int, seed: int) -> CProgram:
    rng = random.Random(f"branch-{level}-{seed}")
    a, b = rng.randint(1, 30), rng.randint(1, 30)
    cond = rng.choice(["<", ">", "<=", ">=", "==", "%"])
    arm = lambda: rng.choice([
        "x + y", "x * y", "x - y", "y - x", f"x * {rng.randint(2, 9)} + y",
        f"x + y * {rng.randint(2, 9)}", "x * x + y", f"(x + y) * {rng.randint(2, 5)}",
        f"x - {rng.randint(1, 9)} * y",
    ])
    src = "\n".join([
        "#include <stdio.h>",
        f"int pick(int x, int y) {{ return x {cond} y ? {arm()} : {arm()}; }}",
        "int main(void) {",
        f'    printf("%d\\n", pick({a}, {b}));',
        "    return 0;",
        "}",
    ])
    return CProgram(src, "branch", level, seed)


def make_bug_undeclared(level: int, seed: int) -> CProgram:
    """Plants exactly one C2065 (undeclared identifier) at a guaranteed use."""
    base = make_arith(level, seed)
    src = base.source.replace('printf("%d\\n", r);',
                              'printf("%d\\n", r + missing_var);')
    return CProgram(src, "bug_undeclared", level, seed)


_CPP_TYPES = ["int", "double", "float", "char", "long", "unsigned int",
              "int*", "const char*", "bool", "short"]


def make_signature(level: int, seed: int) -> tuple[str, str, str]:
    """(source, fn_name, human signature) for mangling tasks."""
    rng = random.Random(f"sig-{level}-{seed}")
    name = "".join(rng.choice("abcdefghikmnoprstuvw") for _ in range(rng.randint(3, 8)))
    nargs = rng.randint(0, 2 + level)
    args = [rng.choice(_CPP_TYPES) for _ in range(nargs)]
    ret = rng.choice(["int", "void", "double"])
    ns = rng.choice([None, None, "util", "core"]) if level >= 2 else None
    sig = f"{ret} {name}({', '.join(args) or 'void'})"
    decl = f"{ret} {name}({', '.join(args)}) {{ {'' if ret == 'void' else 'return {};'.format('0')} }}"
    src = f"namespace {ns} {{ {decl} }}" if ns else decl
    human = f"{ns}::{sig.split(' ', 1)[1]}" if ns else sig.split(" ", 1)[1]
    human = f"{ret} {human}"
    return src, name, human


def make_lifetime(level: int, seed: int) -> CProgram:
    """C++ program printing ctor/dtor order — deterministic, oracle-run.
    Nested scopes make destruction order (reverse of construction, inner
    scope first) the thing to predict."""
    rng = random.Random(f"life-{level}-{seed}")
    names = rng.sample(["A", "B", "C", "D", "E"], 3 + min(level, 2))
    lines = [
        "#include <cstdio>",
        "struct T { const char* n; T(const char* x):n(x){ std::printf(\"+%s\\n\", n);} "
        "~T(){ std::printf(\"-%s\\n\", n);} };",
        "int main() {",
        f'    T {names[0].lower()}("{names[0]}");',
        "    {",
    ]
    for nm in names[1:-1]:
        lines.append(f'        T {nm.lower()}("{nm}");')
    lines += [
        "    }",
        f'    T {names[-1].lower()}("{names[-1]}");',
        "    return 0;",
        "}",
    ]
    return CProgram("\n".join(lines), "lifetime", level, seed)


_MAKERS = {
    "arith": make_arith,
    "loop": make_loop,
    "branch": make_branch,
    "bug_undeclared": make_bug_undeclared,
}
ALL_KINDS = tuple(_MAKERS)


def make_programs(
    n: int, *, kinds=("arith", "loop", "branch"), levels=(1, 2), seed: int = 0,
) -> list[CProgram]:
    out, seen, i = [], set(), 0
    while len(out) < n:
        kind = kinds[i % len(kinds)]
        level = levels[(i // len(kinds)) % len(levels)]
        p = _MAKERS[kind](level, seed * 1_000_000 + i)
        i += 1
        if p.source in seen:
            continue
        seen.add(p.source)
        out.append(p)
    return out
