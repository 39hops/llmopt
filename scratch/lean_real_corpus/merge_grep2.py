"""Live-merge grep, refined (relay 2026-08-03-1 joint-amendment trigger).

Flags, per EQUIVALENT verdict, sites in the RAW input (sympy
evaluate=False, so nothing merges before we look) where axiom's expr
constructor would fire a fractional-pow merge on a NON-NUMERIC base:

  ring-mul   u**a * u**b at ring level, u symbolic, a or b fractional
  infn-mul   same, strictly inside a fn-atom argument
  powpow     (u**frac)**n written literally, u symbolic, n != -1
             (n == -1 is just division syntax, no merge)

Positive numeric bases (2*sqrt(2) -> 2**(3/2)) are sound and excluded.
equiv rows contribute lhs+rhs; equiv_mod_const rows candidate+integrand.
"""
import json
from collections import Counter
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

FARM = '/Users/artin/code/llmopt/data/axiom_parity_farm.jsonl'
OUT = 'parity_out.jsonl'

def is_frac(e):
    try:
        return not e.is_Integer
    except Exception:
        return True

def base_exp(f):
    if isinstance(f, sp.Pow):
        return f.args[0], f.args[1]
    return f, sp.Integer(1)

def sites(root):
    hits = []
    stack = [(root, False)]
    while stack:
        n, fn = stack.pop()
        loc = 'infn' if fn else 'ring'
        if isinstance(n, sp.Mul):
            seen = {}
            for f in n.args:
                b, e = base_exp(f)
                if b.is_Number:
                    continue
                key = sp.srepr(b)
                if key in seen and (is_frac(e) or is_frac(seen[key])):
                    hits.append((loc + '-mul', str(b)))
                seen.setdefault(key, e)
        if isinstance(n, sp.Pow):
            b, e = n.args
            if (isinstance(b, sp.Pow) and not b.args[0].is_Number
                    and is_frac(b.args[1]) and e != sp.Integer(-1)):
                hits.append(('powpow', str(b.args[0])))
        is_fn = isinstance(n, sp.Function)
        for a in n.args:
            stack.append((a, fn or is_fn))
    return hits

farm = {}
for line in open(FARM):
    r = json.loads(line)
    farm[r['id']] = r

counts = Counter()
examples = {}
n_eq = 0
parse_fail = 0
for line in open(OUT):
    r = json.loads(line)
    if r.get('verdict') != 'EQUIVALENT':
        continue
    n_eq += 1
    row = farm[r['id']]
    fields = ('lhs', 'rhs') if row['task'] == 'equiv' else ('candidate', 'integrand')
    row_kinds = set()
    for side in fields:
        try:
            e = parse_expr(row[side], evaluate=False)
        except Exception:
            parse_fail += 1
            continue
        for kind, b in sites(e):
            row_kinds.add(kind)
            examples.setdefault(kind, []).append((r['id'], side, b, row[side][:110]))
    for k in row_kinds:
        counts[k] += 1

print('EQUIVALENT rows:', n_eq, ' parse_fail sides:', parse_fail)
print('rows by kind:', dict(counts))
for k, ex in examples.items():
    print(f'-- {k}: {len(ex)} sites, first 6:')
    for t in ex[:6]:
        print('  ', t)
