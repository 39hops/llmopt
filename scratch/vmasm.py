"""vm-asm closed system (code continent rung 1): straight-line
mini-ISA over r0-r3, one-rule rewrite chains, EXACT symbolic oracle
(programs are polynomial register maps; sympy decides equivalence).
Emits diet + probe with standing doctrine: stable string seeds,
determinable one-rule rows, every row oracle-verified before write.
Usage: vmasm.py <n_train_rows> <out_prefix>"""
import sys, json, random
import sympy as sp

REGS = ["r0", "r1", "r2", "r3"]
SYMS = sp.symbols("a b c d")

def run(prog):
    """Symbolic execution -> tuple of 4 polynomial maps."""
    st = list(SYMS)
    for op, x, y in prog:
        i = REGS.index(x)
        if op == "neg":
            st[i] = -st[i]
            continue
        if op == "shl":
            st[i] = st[i] * 2**int(y)
            continue
        v = st[REGS.index(y)] if y in REGS else sp.Integer(y)
        if op == "mov": st[i] = v
        elif op == "add": st[i] = st[i] + v
        elif op == "sub": st[i] = st[i] - v
        elif op == "mul": st[i] = st[i] * v
    return tuple(sp.expand(s) for s in st)

def show(prog):
    out = []
    for op, x, y in prog:
        out.append(f"{op} {x}" if op == "neg" else f"{op} {x}, {y}")
    return " ; ".join(out)

def gen(rng, n):
    prog = []
    for _ in range(n):
        op = rng.choice(["mov", "add", "sub", "mul", "shl", "neg",
                         "add", "mul", "mov"])
        x = rng.choice(REGS)
        if op == "neg":
            prog.append((op, x, None))
        elif op == "shl":
            prog.append((op, x, rng.randint(1, 3)))
        else:
            y = rng.choice(REGS + [rng.randint(-9, 9)])
            prog.append((op, x, y))
    return prog

def step(prog):
    """One rule application, first match. Returns (nxt, rule) or None."""
    for i, (op, x, y) in enumerate(prog):
        if op == "add" and y == 0 or op == "sub" and y == 0 or \
           op == "mul" and y == 1:
            return prog[:i] + prog[i+1:], "elim_identity"
        if op == "mov" and y == x:
            return prog[:i] + prog[i+1:], "elim_selfmov"
        if op == "mul" and isinstance(y, int) and y in (2, 4, 8):
            return prog[:i] + [("shl", x, {2:1,4:2,8:3}[y])] + prog[i+1:], \
                "strength_reduce"
        if op == "mul" and y == 0:
            return prog[:i] + [("mov", x, 0)] + prog[i+1:], "mul_zero"
        if op == "neg" and i + 1 < len(prog) and prog[i+1] == (op, x, None):
            return prog[:i] + prog[i+2:], "negneg"
        # dead store: write to x, next instruction overwrites x w/o read
        if i + 1 < len(prog):
            op2, x2, y2 = prog[i+1]
            writes = op in ("mov",)
            if writes and x2 == x and op2 == "mov" and y2 != x:
                return prog[:i] + prog[i+1:], "dead_store"
    return None

def farm(n_rows, seed_base, exclude=None):
    rows, seen = [], exclude or set()
    s = 0
    while len(rows) < n_rows:
        rng = random.Random(f"vmasm-{seed_base}-{s}")
        s += 1
        prog = gen(rng, rng.randint(4, 9))
        while True:
            r = step(prog)
            if r is None:
                break
            nxt, rule = r
            cur_s, nxt_s = show(prog), show(nxt)
            if run(prog) != run(nxt):
                raise AssertionError(f"ORACLE FAIL {rule}: {cur_s} -> {nxt_s}")
            if cur_s != nxt_s and cur_s not in seen:
                seen.add(cur_s)
                rows.append({"cur": cur_s, "nxt": nxt_s, "level": 1,
                             "rule": rule, "seed": s - 1})
            prog = nxt
    return rows, seen

if __name__ == "__main__":
    n, out = int(sys.argv[1]), sys.argv[2]
    train, seen = farm(n, "train")
    probe, _ = farm(400, "probe", exclude=seen)
    from collections import Counter
    print("rules:", Counter(r["rule"] for r in train))
    with open(f"data/{out}_diet.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps({"cur": r["cur"], "nxt": r["nxt"],
                                "level": 1}) + "\n")
    with open(f"data/{out}_probe.jsonl", "w") as f:
        for r in probe:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(train)} train / {len(probe)} probe")

def parse(s):
    prog = []
    for ins in s.split(" ; "):
        parts = ins.replace(",", "").split()
        if not parts:
            return None
        op = parts[0]
        if op == "neg" and len(parts) == 2 and parts[1] in REGS:
            prog.append((op, parts[1], None))
        elif op in ("mov", "add", "sub", "mul", "shl") and len(parts) == 3 \
                and parts[1] in REGS:
            y = parts[2]
            if y not in REGS:
                try:
                    y = int(y)
                except ValueError:
                    return None
            if op == "shl" and (not isinstance(y, int) or y < 0):
                return None
            prog.append((op, parts[1], y))
        else:
            return None
    return prog
