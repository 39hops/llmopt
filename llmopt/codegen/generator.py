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
    src = "\n".join([
        "#include <stdio.h>",
        f"int pick(int x, int y) {{ return x {rng.choice(['<', '>', '<=', '%'])} y ? x + y : x * y; }}",
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
