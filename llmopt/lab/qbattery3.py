"""The v3 confirmation collection for QWEN-LOW-RELEASE-0: 60 fresh tasks
whose families and prompt templates differ from the exposed v2 battery
(llmopt.lab.qbattery, which keeps its own generators frozen). The kinds
and validators are shared with v2 (qbattery.score dispatches by kind;
the new instruction and termination checks are the v3 branches of
check_instruct and check_degen); the task families are new:

  arithmetic (12)  sum of three numbers, difference of two products,
                   integer quotient, lcm, exact percent change, range of a
                   list, a square, a digit sum, an integer mean, a product
                   modulo m (v2: sum, difference, product, modulus, gcd,
                   percentage)
  structured (12)  an invoice with a percent discount (subtotal, discount,
                   total); an inventory CSV with categories (count per
                   category, total quantity); key-value readings (max key,
                   min key, mean); a two-score roster (best average, the
                   count above a cut) (v2: order total and costliest sku;
                   roster mean and top)
  code (12)        palindrome ignoring non-alphanumerics, second largest,
                   flatten, caesar shift, roman to int, merge intervals,
                   vowel count, top-k word frequency, binary search, main
                   diagonal sum, chunking, running maximum (v2: reverse
                   words, balanced brackets, two-sum, run-length encoding,
                   primality, transpose, anagram groups, fibonacci mod);
                   every code task carries a reference solution that the
                   build asserts against its own hidden tests
  instruction (10) exact sentence count, a required first word, no commas,
                   dash bullets, an alphabetical list, a closing question
                   mark, an exact word count, all lower case, two required
                   words, a three-sentence count (v2: line counts, an
                   include/exclude pair, capitals, a closing word, a
                   numbered list, a word band)
  two-turn (8)     the length of a remembered list; the product of two
                   remembered numbers; the first word of a remembered
                   sentence; the animal of a remembered colour-animal pair
                   (v2: recall a code word or number; continue a sum)
  termination (6)  a phrase repeated comma-separated; a countdown; a word
                   spelt with hyphens; a one-word answer; the first N
                   letters (v2: a word repeated space-separated; a count
                   up; a one-line answer)

Seed "qwen-battery-v3-seed-23" (a stable string). overlap_report()
states the template-level and family-level overlap with v2 explicitly.
"""
from __future__ import annotations

import json
import math
import random

from llmopt.lab import qbattery

VERSION = 3
SEED = "qwen-battery-v3-seed-23"
RULES = dict(qbattery.RULES)      # the same frozen rules: greedy, thinking off, no system prompt, 384 tokens, context cap 8,192; arms override budget and thinking in execution_config


def gen_arith(r: random.Random, n: int) -> list:
    out = []
    kinds = ["sum3", "diff_prod", "quotient", "lcm", "pct_change", "range", "square", "digit_sum", "mean", "prod_mod"]
    for i in range(n):
        k = kinds[i % len(kinds)]
        if k == "sum3":
            a, b, c = (r.randint(1_000, 99_999) for _ in range(3))
            q, ans = f"What is {a} + {b} + {c}?", a + b + c
        elif k == "diff_prod":
            a, b, c, d = r.randint(10, 99), r.randint(10, 99), r.randint(10, 99), r.randint(10, 99)
            q, ans = f"What is {a} * {b} - {c} * {d}?", a * b - c * d
        elif k == "quotient":
            b = r.randint(7, 97)
            a = b * r.randint(100, 999) + r.randint(1, b - 1)
            q, ans = f"What is the integer quotient of {a} divided by {b} (ignore the remainder)?", a // b
        elif k == "lcm":
            a, b = r.randint(6, 60), r.randint(6, 60)
            q, ans = f"What is the least common multiple of {a} and {b}?", a * b // math.gcd(a, b)
        elif k == "pct_change":
            v = r.randint(20, 400) * 5
            p = r.choice([-40, -25, -20, 10, 20, 25, 40, 60])
            assert (v * p) % 100 == 0
            q, ans = f"A price goes from {v} to {v + v * p // 100}. What is the percent change (a positive or negative whole number)?", p
        elif k == "range":
            xs = r.sample(range(100, 9999), 6)
            q, ans = f"What is the range (largest minus smallest) of these numbers: {', '.join(map(str, xs))}?", max(xs) - min(xs)
        elif k == "square":
            a = r.randint(100, 999)
            q, ans = f"What is {a} squared?", a * a
        elif k == "digit_sum":
            a = r.randint(10_000_000, 99_999_999)
            q, ans = f"What is the sum of the digits of {a}?", sum(int(ch) for ch in str(a))
        elif k == "mean":
            m = r.randint(50, 500)
            offs = [r.randint(-30, 30) for _ in range(4)]
            xs = [m + o for o in offs] + [m - sum(offs)]
            r.shuffle(xs)
            q, ans = f"What is the mean (average) of these five numbers: {', '.join(map(str, xs))}?", m
        else:
            a, b, m = r.randint(100, 999), r.randint(100, 999), r.randint(11, 97)
            q, ans = f"What is ({a} * {b}) mod {m} (the remainder when the product is divided by {m})?", (a * b) % m
        out.append({"id": f"arith-{i + 1:02d}", "kind": "arith", "family": k, "turns": [q + " Answer with the number only."], "expected": ans})
    return out


def gen_json(r: random.Random, n: int) -> list:
    out = []
    structures = ["invoice", "inventory", "readings", "roster2"]
    for i in range(n):
        st = structures[i % 4]
        if st == "invoice":
            items = [{"item": r.choice(["bolt", "nut", "washer", "hinge", "bracket", "screw"]) + f"-{r.randint(1, 9)}", "qty": r.randint(1, 12), "price": round(r.uniform(0.5, 30), 2)} for _ in range(r.randint(2, 4))]
            disc = r.choice([5, 10, 15, 20, 25])
            sub = round(sum(it["qty"] * it["price"] for it in items), 2)
            d = round(sub * disc / 100, 2)
            q = ("Here is an invoice as JSON:\n" + json.dumps({"lines": items, "discount_percent": disc}) +
                 "\nReturn ONLY a JSON object {\"subtotal\": <number>, \"discount\": <number>, \"total\": <number>} where subtotal is the sum of qty times price, "
                 "discount is discount_percent of the subtotal, and total is subtotal minus discount, each rounded to 2 decimals.")
            exp = {"subtotal": sub, "discount": d, "total": round(sub - d, 2)}
            schema = {"subtotal": "number", "discount": "number", "total": "number"}
            tol = {"subtotal": 0.011, "discount": 0.011, "total": 0.021}
        elif st == "inventory":
            cats = r.sample(["tools", "paint", "wood", "glass", "wire"], 3)
            rows = [(f"P{r.randint(100, 999)}", r.choice(cats), r.randint(1, 40)) for _ in range(r.randint(5, 7))]
            q = ("Here is an inventory as CSV:\nproduct,category,qty\n" + "\n".join(f"{a},{b},{c}" for a, b, c in rows) +
                 "\nReturn ONLY a JSON object {\"rows_per_category\": {<category>: <integer count of rows>}, \"total_qty\": <integer>, \"largest_row\": <product with the largest qty>}.")
            per = {}
            for _, b, _c in rows:
                per[b] = per.get(b, 0) + 1
            largest = max(rows, key=lambda x: x[2])
            assert sum(1 for x in rows if x[2] == largest[2]) == 1
            exp = {"rows_per_category": per, "total_qty": sum(c for _, _, c in rows), "largest_row": largest[0]}
            schema = {"rows_per_category": "dict", "total_qty": "integer", "largest_row": "string"}
            tol = {}
        elif st == "readings":
            keys = r.sample(["mon", "tue", "wed", "thu", "fri", "sat", "sun"], r.randint(4, 6))
            vals = r.sample(range(-5, 40), len(keys))
            q = ("Here are temperature readings, one per line as key=value:\n" + "\n".join(f"temp_{k}={v}" for k, v in zip(keys, vals)) +
                 "\nReturn ONLY a JSON object {\"max_key\": <the key of the largest value, e.g. temp_mon>, \"min_key\": <the key of the smallest value>, \"mean\": <number, the mean rounded to 1 decimal>}.")
            mx, mn = max(zip(vals, keys)), min(zip(vals, keys))
            exp = {"max_key": f"temp_{mx[1]}", "min_key": f"temp_{mn[1]}", "mean": sum(vals) / len(vals)}
            schema = {"max_key": "string", "min_key": "string", "mean": "number"}
            tol = {"mean": 0.051}
        else:
            names = r.sample(["ida", "jon", "kai", "lee", "mia", "ned", "ola", "pip"], r.randint(3, 5))
            rows = {nm: (r.randint(30, 100), r.randint(30, 100)) for nm in names}
            avgs = {nm: (a + b) / 2 for nm, (a, b) in rows.items()}
            best = max(avgs.values())
            assert sum(1 for v in avgs.values() if v == best) == 1
            cut = r.choice([60, 70, 80])
            q = ("Here is a roster with two scores per student as CSV:\nname,s1,s2\n" + "\n".join(f"{nm},{a},{b}" for nm, (a, b) in rows.items()) +
                 f"\nReturn ONLY a JSON object {{\"best_avg_name\": <the name with the highest average of s1 and s2>, \"n_above\": <integer, how many students have an average strictly above {cut}>, \"count\": <integer, number of students>}}.")
            exp = {"best_avg_name": max(avgs, key=avgs.get), "n_above": sum(1 for v in avgs.values() if v > cut), "count": len(rows)}
            schema = {"best_avg_name": "string", "n_above": "integer", "count": "integer"}
            tol = {}
        out.append({"id": f"json-{i + 1:02d}", "kind": "json", "family": st, "turns": [q], "expected": exp, "schema": schema, "tol": tol})
    return out


CODE_TASKS = [
    ("is_palindrome", "Write a Python function is_palindrome(s) that returns True if s reads the same forwards and backwards after removing every non-alphanumeric character and ignoring case, else False.",
     ["assert is_palindrome('A man, a plan, a canal: Panama') is True", "assert is_palindrome('race a car') is False", "assert is_palindrome('') is True", "assert is_palindrome('No lemon, no melon!') is True"],
     "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n"),
    ("second_largest", "Write a Python function second_largest(xs) that returns the second largest DISTINCT value in a list of integers, or None if there is no such value.",
     ["assert second_largest([3, 1, 4, 4, 2]) == 3", "assert second_largest([5]) is None", "assert second_largest([2, 2, 2]) is None", "assert second_largest([-1, -5, -3]) == -3"],
     "def second_largest(xs):\n    u = sorted(set(xs))\n    return u[-2] if len(u) >= 2 else None\n"),
    ("flatten", "Write a Python function flatten(x) that flattens arbitrarily nested lists into one flat list, preserving order; non-list items are kept as they are.",
     ["assert flatten([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]", "assert flatten([]) == []", "assert flatten([[], [[]], 1]) == [1]", "assert flatten(['a', ['b']]) == ['a', 'b']"],
     "def flatten(x):\n    out = []\n    for v in x:\n        if isinstance(v, list):\n            out.extend(flatten(v))\n        else:\n            out.append(v)\n    return out\n"),
    ("caesar", "Write a Python function caesar(s, k) that shifts every ASCII letter of s by k positions (wrapping within a-z and A-Z, case preserved) and leaves other characters unchanged; k may be negative or larger than 26.",
     ["assert caesar('abc XYZ', 3) == 'def ABC'", "assert caesar('Hello, World!', 0) == 'Hello, World!'", "assert caesar('def', -3) == 'abc'", "assert caesar('a', 27) == 'b'"],
     "def caesar(s, k):\n    out = []\n    for c in s:\n        if 'a' <= c <= 'z':\n            out.append(chr((ord(c) - 97 + k) % 26 + 97))\n        elif 'A' <= c <= 'Z':\n            out.append(chr((ord(c) - 65 + k) % 26 + 65))\n        else:\n            out.append(c)\n    return ''.join(out)\n"),
    ("roman_to_int", "Write a Python function roman_to_int(s) that converts a valid Roman numeral string (I V X L C D M, subtractive forms allowed) to an integer.",
     ["assert roman_to_int('III') == 3", "assert roman_to_int('IV') == 4", "assert roman_to_int('MCMXCIV') == 1994", "assert roman_to_int('LVIII') == 58"],
     "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    t = 0\n    for i, c in enumerate(s):\n        if i + 1 < len(s) and v[c] < v[s[i + 1]]:\n            t -= v[c]\n        else:\n            t += v[c]\n    return t\n"),
    ("merge_intervals", "Write a Python function merge_intervals(iv) that merges overlapping or touching closed intervals given as a list of [start, end] pairs and returns the merged list sorted by start.",
     ["assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]", "assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]", "assert merge_intervals([]) == []", "assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]"],
     "def merge_intervals(iv):\n    out = []\n    for a, b in sorted(iv):\n        if out and a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out\n"),
    ("count_vowels", "Write a Python function count_vowels(s) that returns the number of vowels (a, e, i, o, u, either case) in s.",
     ["assert count_vowels('Hello World') == 3", "assert count_vowels('') == 0", "assert count_vowels('AEIOU aeiou') == 10", "assert count_vowels('rhythm') == 0"],
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n"),
    ("top_words", "Write a Python function top_words(text, k) that returns the k most frequent lower-cased words (letters only, split on non-letters) as a list of (word, count) tuples, most frequent first, ties broken alphabetically.",
     ["assert top_words('a b a c b a', 2) == [('a', 3), ('b', 2)]", "assert top_words('Dog dog cat. Cat bird', 3) == [('cat', 2), ('dog', 2), ('bird', 1)]", "assert top_words('', 2) == []", "assert top_words('x y', 5) == [('x', 1), ('y', 1)]"],
     "import re\nfrom collections import Counter\ndef top_words(text, k):\n    c = Counter(w for w in re.findall(r'[A-Za-z]+', text.lower()))\n    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]\n"),
    ("bsearch", "Write a Python function bsearch(xs, t) that returns the index of t in the sorted list xs using binary search, or -1 if absent; if t occurs more than once return the lowest index.",
     ["assert bsearch([1, 3, 5, 7], 5) == 2", "assert bsearch([1, 3, 5, 7], 4) == -1", "assert bsearch([], 1) == -1", "assert bsearch([2, 2, 2, 3], 2) == 0"],
     "def bsearch(xs, t):\n    lo, hi = 0, len(xs)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if xs[mid] < t:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo if lo < len(xs) and xs[lo] == t else -1\n"),
    ("diag_sum", "Write a Python function diag_sum(m) that returns the sum of the main diagonal of a square list-of-lists matrix m; diag_sum([]) returns 0.",
     ["assert diag_sum([[1, 2], [3, 4]]) == 5", "assert diag_sum([[7]]) == 7", "assert diag_sum([]) == 0", "assert diag_sum([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == 6"],
     "def diag_sum(m):\n    return sum(m[i][i] for i in range(len(m)))\n"),
    ("chunk", "Write a Python function chunk(xs, n) that splits the list xs into consecutive chunks of length n (the last chunk may be shorter); chunk([], n) returns [].",
     ["assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "assert chunk([], 3) == []", "assert chunk([1, 2, 3], 3) == [[1, 2, 3]]", "assert chunk([1, 2], 5) == [[1, 2]]"],
     "def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n"),
    ("running_max", "Write a Python function running_max(xs) that returns a list where element i is the maximum of xs[0..i].",
     ["assert running_max([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]", "assert running_max([]) == []", "assert running_max([2, 2, 1]) == [2, 2, 2]", "assert running_max([-3, -1, -2]) == [-3, -1, -1]"],
     "def running_max(xs):\n    out, m = [], None\n    for v in xs:\n        m = v if m is None or v > m else m\n        out.append(m)\n    return out\n"),
]


def gen_code(r: random.Random, n: int) -> list:
    out = []
    for i, (name, spec, tests, ref) in enumerate(CODE_TASKS[:n]):
        ns: dict = {}
        exec(ref, ns)                     # the reference solution must pass its own hidden tests at build time
        for a in tests:
            exec(a, ns)
        out.append({"id": f"code-{i + 1:02d}", "kind": "code", "name": name, "tests": tests, "reference": ref,
                    "turns": [spec + " Reply with a single Python code block containing only the function (and any helpers), no prints and no tests."]})
    return out


def gen_instruct(r: random.Random, n: int) -> list:
    specs = [
        ("sentences", {"n": 2}, "Describe a lighthouse in exactly two sentences."),
        ("starts_with", {"word": "indeed"}, "Agree that reading is useful in one sentence that begins with the word Indeed."),
        ("no_commas", {}, "Explain what a thermostat does in one or two sentences without using any commas."),
        ("bullets", {"n": 4}, "List four tools found in a kitchen as a bullet list: four lines, each starting with '- ', nothing else."),
        ("alphabetical", {"n": 5}, "List five fruits in alphabetical order, one per line, lower case, nothing else."),
        ("ends_question", {}, "Write one sentence inviting a friend to lunch; it must end with a question mark."),
        ("exact_words", {"n": 12}, "Describe the sea in exactly 12 words."),
        ("all_lower", {}, "Write one sentence about autumn entirely in lower-case letters."),
        ("include_two", {"words": ["river", "bridge"]}, "Write two sentences about a town. Use both the word 'river' and the word 'bridge'."),
        ("sentences", {"n": 3}, "Explain why the sky looks blue in exactly three sentences."),
    ]
    return [{"id": f"instruct-{i + 1:02d}", "kind": "instruct", "check": c, "params": p, "turns": [q]} for i, (c, p, q) in enumerate(specs[:n])]


def gen_twoturn(r: random.Random, n: int) -> list:
    out = []
    for i in range(n):
        k = i % 4
        if k == 0:
            words = r.sample(["maple", "copper", "falcon", "ember", "willow", "granite", "tulip", "cobalt"], r.randint(3, 6))
            t1, t2, exp = f"Remember this list: {', '.join(words)}. Reply only: noted.", "How many items were in the list I gave you? Reply with the number only.", str(len(words))
        elif k == 1:
            a, b = r.randint(3, 12), r.randint(3, 12)
            t1, t2, exp = f"Remember two numbers: {a} and {b}. Reply only: noted.", "Multiply the two numbers I gave you and reply with the product only.", str(a * b)
        elif k == 2:
            first = r.choice(["Seagulls", "Lanterns", "Bicycles", "Orchards", "Meteors"])
            t1, t2, exp = f"Remember this sentence: {first} drift over the quiet harbour at dusk. Reply only: noted.", "What was the first word of the sentence I asked you to remember? Reply with that one word only.", first.lower()
        else:
            colour, animal = r.choice(["crimson", "indigo", "ochre", "teal"]), r.choice(["otter", "heron", "lynx", "badger"])
            t1, t2, exp = f"My favourite colour is {colour} and my favourite animal is the {animal}. Reply only: noted.", "Which animal did I say was my favourite? Reply with that one word only.", animal
        out.append({"id": f"twoturn-{i + 1:02d}", "kind": "twoturn", "family": ["list_len", "product", "first_word", "pair"][k], "turns": [t1, t2], "expected": exp})
    return out


def gen_degen(r: random.Random, n: int) -> list:
    specs = []
    for i in range(n):
        k = i % 5
        if k == 0:
            ph, m = r.choice(["blue sky", "green hill", "red door"]), r.randint(4, 8)
            specs.append({"check": "repeat_phrase", "params": {"phrase": ph, "n": m}, "turns": [f"Write the phrase '{ph}' exactly {m} times, separated by commas, nothing else."]})
        elif k == 1:
            a = r.randint(12, 30)
            specs.append({"check": "countdown", "params": {"a": a, "b": a - 7}, "turns": [f"Count down from {a} to {a - 7} in digits, separated by commas, nothing else."]})
        elif k == 2:
            w = r.choice(["river", "candle", "window"])
            specs.append({"check": "hyphen_spell", "params": {"word": w}, "turns": [f"Spell the word {w} with its letters separated by hyphens, nothing else."]})
        elif k == 3:
            q, exp = r.choice([("What is the capital of Japan? Answer with exactly one word.", "tokyo"), ("Which planet is closest to the Sun? Answer with exactly one word.", "mercury")])
            specs.append({"check": "one_word", "params": {}, "turns": [q], "expected": exp})
        else:
            m = r.randint(5, 9)
            specs.append({"check": "letters", "params": {"n": m}, "turns": [f"Write the first {m} letters of the English alphabet in lower case, separated by commas, nothing else."]})
    return [{"id": f"degen-{i + 1:02d}", "kind": "degen", **s} for i, s in enumerate(specs)]


def build(seed: str = SEED) -> dict:
    r = random.Random(seed)
    tasks = gen_arith(r, 12) + gen_json(r, 12) + gen_code(r, 12) + gen_instruct(r, 10) + gen_twoturn(r, 8) + gen_degen(r, 6)
    return {"collection": f"qwen_battery_v{VERSION}", "version": VERSION, "seed": seed, "rules": RULES, "tasks": tasks}


def overlap_report(v3: dict, v2: dict) -> dict:
    """Template-level and family-level overlap with v2, stated explicitly."""
    p3 = {t["turns"][0] for t in v3["tasks"]}
    p2 = {t["turns"][0] for t in v2["tasks"]}
    fam2 = {"arith": ["sum", "diff", "prod", "mod", "gcd", "pct"], "json": ["order", "roster"], "code": [c[0] for c in qbattery.CODE_TASKS],
            "instruct": sorted({t["check"] for t in v2["tasks"] if t["kind"] == "instruct"}), "twoturn": ["recall", "continue_sum"], "degen": sorted({t["check"] for t in v2["tasks"] if t["kind"] == "degen"})}
    fam3 = {k: sorted({t.get("family") or t.get("check") or t.get("name") for t in v3["tasks"] if t["kind"] == k}) for k in fam2}
    shared_fam = {k: sorted(set(fam2[k]) & set(fam3[k])) for k in fam2}
    near = {"all_lower": "mirrors v2 all_caps", "starts_with": "mirrors v2 ends_with", "include_two": "a two-word form of v2 include_exclude", "bullets": "a dash form of v2 numbered and lines",
            "countdown": "the descending form of v2 count (same comma-separated digits contract)", "repeat_phrase": "a comma-separated two-word form of v2 repeat", "one_word": "a one-word form of v2 one_line (capital and planet facts)",
            "pair": "a recall task like v2's code word and number", "product": "a two-number recall with an operation, near v2's continue-a-sum", "invoice": "its subtotal is v2's order total (sum of qty times price) before the discount step", "roster2": "a two-score form of v2 roster"}
    return {"shared_prompts": len(p3 & p2), "shared_code_functions": len({c[0] for c in CODE_TASKS} & {c[0] for c in qbattery.CODE_TASKS}),
            "families_v3": fam3, "families_v2": fam2, "shared_families": shared_fam,
            "near_relatives_of_v2_families": near,
            "independence_note": "the family names are all new and no prompt template or instance repeats, but eleven v3 families are near relatives of a v2 family (listed); since the low-effort candidate was selected on v2, v3 is a fresh-instance confirmation of moderate structural independence, not an unrelated benchmark",
            "shared_structure_note": "the kinds, the reply contracts ('Answer with the number only', 'Return ONLY a JSON object', 'Reply with a single Python code block', 'Reply only: noted.') and the validators are shared with v2 by design; every task family, prompt template and instance is new"}
