"""The functional-quality battery for the local Qwen3.8-27B arm C
(QWEN-INFERENCE-BASELINE-0): task generators (seeded, versioned),
validators (programs, never a text match where several answers are
right) and an isolated code runner. The frozen collection a run scores is
scratch/evalsets/qwen_battery_v2.json, written from these generators
(v1, the first freeze, is kept as a record of its three audit-found
defects: a wrong expected answer, a non-integer percentage, a rounding
tie); a run records the collection's sha256 and never
edits it.

Kinds and their checks:
  arith      seeded integer arithmetic (sum, difference, product, modulus,
             gcd, percentage); the LAST integer in the reply must equal the
             expected value (thousands separators accepted)
  json       a seeded structured task (orders with items and quantities;
             a roster with scores); the FIRST JSON object in the reply must
             parse, carry the required keys with the required types and
             equal the expected values computed from the seed
  code       a short Python function specification with hidden tests; the
             reply's first python code block runs in the sandbox against
             the tests; pass means every test passed and the process exited
             cleanly within the time limit; a run with no sandbox books HOLD
  instruct   constrained instruction following checked by programs: exact
             line count, a required or forbidden word, all caps, a required
             final word, a numbered list of N items, a word-count band
  twoturn    turn 1 states facts; turn 2 asks something only the facts
             answer; the expected value (a number or a code word) must
             appear in turn 2's reply and the reply must be short
  degen      termination behaviour: repeat a word exactly N times, count
             from A to B, or answer in one line; checked by exact counts,
             the stop reason (a stop token, not the budget) and no runaway
Scoring rules are frozen with the collection (greedy, enable_thinking
False, no system prompt, max 384 output tokens per turn, context cap
8,192); a task is "pass" or "fail" with a reason, and a plumbing failure
(an exception, a context refusal) is recorded as "error", separately from
a model failure.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from typing import Optional

VERSION = 2            # v1 (same seed) had three mis-specified tasks found by audit before any result was read
SEED = "qwen-battery-v1-seed-11"
RULES = {"decoding": "greedy", "enable_thinking": False, "system": None, "max_tokens_per_turn": 384, "context_cap": 8192,
         "code_time_limit_s": 10, "code_sandbox": "sandbox-exec: no network, no writes outside the task's temp dir, no reads under /Users, /Volumes or /private/etc, no exec but the interpreter, empty environment; CPU 10 s, 1 MiB file-size and one-process limits inside the sandboxed process; memory is NOT limited (macOS refuses to lower RLIMIT_DATA)"}

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


# ---------------------------------------------------------------- generators

def gen_arith(r: random.Random, n: int) -> list:
    out = []
    kinds = ["sum", "diff", "prod", "mod", "gcd", "pct"]
    for i in range(n):
        k = kinds[i % len(kinds)]
        if k == "sum":
            a, b = r.randint(10_000, 99_999), r.randint(1_000, 99_999)
            q, ans = f"What is {a} + {b}?", a + b
        elif k == "diff":
            a, b = r.randint(50_000, 99_999), r.randint(1_000, 49_999)
            q, ans = f"What is {a} - {b}?", a - b
        elif k == "prod":
            a, b = r.randint(100, 999), r.randint(100, 999)
            q, ans = f"What is {a} * {b}?", a * b
        elif k == "mod":
            a, b = r.randint(1_000, 99_999), r.randint(7, 97)
            q, ans = f"What is {a} mod {b} (the remainder of {a} divided by {b})?", a % b
        elif k == "gcd":
            g = r.randint(3, 40)
            a, b = g * r.randint(10, 99), g * r.randint(10, 99)
            q, ans = f"What is the greatest common divisor of {a} and {b}?", math.gcd(a, b)
        else:
            p, v = r.choice([5, 10, 15, 20, 25, 40, 50]), r.randint(10, 999) * 20
            assert (v * p) % 100 == 0
            q, ans = f"What is {p} percent of {v}?", v * p // 100
        out.append({"id": f"arith-{i + 1:02d}", "kind": "arith", "turns": [q + " Answer with the number only."], "expected": ans})
    return out


def gen_json(r: random.Random, n: int) -> list:
    out = []
    for i in range(n):
        if i % 2 == 0:
            items = [{"sku": f"{r.choice('ABCD')}-{r.randint(10, 99)}", "qty": r.randint(1, 9), "unit": round(r.uniform(1, 50), 2)} for _ in range(r.randint(2, 4))]
            total = round(sum(it["qty"] * it["unit"] for it in items), 2)
            most = max(items, key=lambda it: it["qty"] * it["unit"])["sku"]
            q = ("Here is an order as JSON:\n" + json.dumps({"items": items}) +
                 "\nReturn ONLY a JSON object {\"total\": <number>, \"n_items\": <integer>, \"costliest_sku\": <string>} where total is the sum of qty times unit "
                 "rounded to 2 decimals, n_items is the number of line items and costliest_sku is the sku with the largest qty times unit.")
            exp = {"total": total, "n_items": len(items), "costliest_sku": most}
            schema = {"total": "number", "n_items": "integer", "costliest_sku": "string"}
        else:
            names = r.sample(["ava", "ben", "cy", "dee", "eli", "fay", "gus", "hal"], r.randint(3, 5))
            scores = {nm: r.randint(40, 100) for nm in names}
            top = max(scores.values())
            q = ("Here is a roster as CSV:\nname,score\n" + "\n".join(f"{k},{v}" for k, v in scores.items()) +
                 "\nReturn ONLY a JSON object {\"mean\": <number>, \"top\": [<names with the highest score, alphabetical>], \"count\": <integer>} with mean rounded to 1 decimal.")
            exp = {"mean": sum(scores.values()) / len(scores), "top": sorted(k for k, v in scores.items() if v == top), "count": len(scores)}
            schema = {"mean": "number", "top": "list", "count": "integer"}
            tol = {"mean": 0.051}      # any rounding of the exact mean to one decimal is accepted
        if i % 2 == 0:
            tol = {"total": 0.011}
        out.append({"id": f"json-{i + 1:02d}", "kind": "json", "turns": [q], "expected": exp, "schema": schema, "tol": tol})
    return out


CODE_TASKS = [
    ("reverse_words", "Write a Python function reverse_words(s) that returns the words of s in reverse order joined by single spaces, treating any whitespace run as a separator.",
     ["assert reverse_words('a b  c') == 'c b a'", "assert reverse_words('hello') == 'hello'", "assert reverse_words('  one two ') == 'two one'", "assert reverse_words('') == ''"]),
    ("balanced", "Write a Python function balanced(s) that returns True if the brackets ()[]{} in s are balanced and properly nested, ignoring other characters, else False.",
     ["assert balanced('a(b[c]{d})') is True", "assert balanced('(]') is False", "assert balanced('') is True", "assert balanced('((') is False", "assert balanced('{[()]}x') is True"]),
    ("two_sum", "Write a Python function two_sum(nums, target) that returns a tuple (i, j) with i < j and nums[i] + nums[j] == target, the lexicographically smallest such pair, or None.",
     ["assert two_sum([2, 7, 11, 15], 9) == (0, 1)", "assert two_sum([3, 3], 6) == (0, 1)", "assert two_sum([1, 2], 7) is None", "assert two_sum([1, 4, 2, 3], 5) == (0, 1)"]),
    ("rle", "Write a Python function rle(s) that run-length encodes s as pairs, e.g. rle('aaab') returns [('a', 3), ('b', 1)]; rle('') returns [].",
     ["assert rle('aaab') == [('a', 3), ('b', 1)]", "assert rle('') == []", "assert rle('abc') == [('a', 1), ('b', 1), ('c', 1)]", "assert rle('zz') == [('z', 2)]"]),
    ("is_prime", "Write a Python function is_prime(n) that returns True for prime integers n >= 2 and False otherwise (including n < 2).",
     ["assert is_prime(2) and is_prime(97) and is_prime(7919)", "assert not is_prime(1) and not is_prime(0) and not is_prime(-7)", "assert not is_prime(91) and not is_prime(1000)", "assert is_prime(2_147_483_647)"]),
    ("transpose", "Write a Python function transpose(m) that returns the transpose of a rectangular list-of-lists matrix m (a list of lists); transpose([]) returns [].",
     ["assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]", "assert transpose([[1]]) == [[1]]", "assert transpose([]) == []", "assert transpose([[1, 2]]) == [[1], [2]]"]),
    ("group_anagrams", "Write a Python function group_anagrams(words) that groups anagrams: return a list of groups (each a list of the original words in input order), groups ordered by first appearance.",
     ["assert group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']) == [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]", "assert group_anagrams([]) == []", "assert group_anagrams(['a']) == [['a']]"]),
    ("fib_mod", "Write a Python function fib_mod(n, m) that returns the n-th Fibonacci number (fib(0)=0, fib(1)=1) modulo m, efficient enough for n up to 10**6.",
     ["assert fib_mod(0, 10) == 0 and fib_mod(1, 10) == 1", "assert fib_mod(10, 1000) == 55", "assert fib_mod(50, 10**9 + 7) == 12586269025 % (10**9 + 7)", "assert fib_mod(10**6, 1000) == 875"]),
]


def gen_code(r: random.Random, n: int) -> list:
    out = []
    for i, (name, spec, tests) in enumerate(CODE_TASKS[:n]):
        out.append({"id": f"code-{i + 1:02d}", "kind": "code", "name": name, "tests": tests,
                    "turns": [spec + " Reply with a single Python code block containing only the function (and any helpers), no prints and no tests."]})
    return out


def gen_instruct(r: random.Random, n: int) -> list:
    specs = [
        ("lines", {"n": 3}, "Name three oceans, one per line, exactly three lines, nothing else."),
        ("include_exclude", {"include": "glacier", "exclude": "ice"}, "Describe a mountain in two sentences. You must use the word 'glacier' and you must not use the word 'ice'."),
        ("all_caps", {}, "Write one sentence about rain entirely in capital letters."),
        ("ends_with", {"word": "done"}, "Give a two-sentence tip about sleep. The reply must end with the single word: done"),
        ("numbered", {"n": 5}, "List five vegetables as a numbered list, 1. to 5., one per line, nothing else."),
        ("word_band", {"lo": 20, "hi": 40}, "Explain what a compiler does in between 20 and 40 words."),
        ("lines", {"n": 4}, "Give four colours, one per line, exactly four lines, no numbering, no extra text."),
        ("include_exclude", {"include": "orbit", "exclude": "planet"}, "In one sentence, say what a satellite does. Use the word 'orbit' and do not use the word 'planet'."),
    ]
    return [{"id": f"instruct-{i + 1:02d}", "kind": "instruct", "check": c, "params": p, "turns": [q]} for i, (c, p, q) in enumerate(specs[:n])]


def gen_twoturn(r: random.Random, n: int) -> list:
    out = []
    for i in range(n):
        if i % 2 == 0:
            word = r.choice(["kestrel", "lantern", "quartz", "velvet", "harbour", "meadow"])
            num = r.randint(1000, 9999)
            t1 = f"Remember two things for later: the code word is {word} and the ticket number is {num}. Reply only: noted."
            t2 = r.choice([("What was the code word? Reply with that one word only.", word), ("What was the ticket number? Reply with the number only.", str(num))])
            out.append({"id": f"twoturn-{i + 1:02d}", "kind": "twoturn", "turns": [t1, t2[0]], "expected": t2[1]})
        else:
            a, b = r.randint(12, 99), r.randint(12, 99)
            t1 = f"Compute {a} + {b} and reply with the number only."
            t2 = "Now add 10 to the number you just gave and reply with the number only."
            out.append({"id": f"twoturn-{i + 1:02d}", "kind": "twoturn", "turns": [t1, t2], "expected": str(a + b + 10), "expected_turn1": str(a + b)})
    return out


def gen_degen(r: random.Random, n: int) -> list:
    specs = []
    for i in range(n):
        k = i % 3
        if k == 0:
            w, m = r.choice(["echo", "delta", "amber"]), r.randint(5, 9)
            specs.append({"check": "repeat", "params": {"word": w, "n": m}, "turns": [f"Write the word {w} exactly {m} times separated by single spaces, then stop."]})
        elif k == 1:
            a = r.randint(3, 20)
            specs.append({"check": "count", "params": {"a": a, "b": a + 9}, "turns": [f"Count from {a} to {a + 9} in digits, separated by commas, nothing else."]})
        else:
            q, exp = r.choice([("What is the capital of France? One line.", "paris"), ("Name the largest planet of the solar system. One line.", "jupiter")])
            specs.append({"check": "one_line", "params": {}, "turns": [q], "expected": exp})
    return [{"id": f"degen-{i + 1:02d}", "kind": "degen", **s} for i, s in enumerate(specs)]


def build(seed: str = SEED) -> dict:
    r = random.Random(seed)
    tasks = gen_arith(r, 10) + gen_json(r, 8) + gen_code(r, 8) + gen_instruct(r, 8) + gen_twoturn(r, 6) + gen_degen(r, 6)
    return {"collection": f"qwen_battery_v{VERSION}", "version": VERSION, "seed": seed, "rules": RULES, "tasks": tasks}


# ---------------------------------------------------------------- validators

def _last_int(text: str) -> Optional[int]:
    nums = _NUM.findall(text)
    if not nums:
        return None
    s = nums[-1].replace(",", "")
    try:
        return int(float(s)) if float(s) == int(float(s)) else None
    except ValueError:
        return None


def _first_json(text: str):
    start = text.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:j + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _code_block(text: str) -> Optional[str]:
    m = re.search(r"```(?:python3?|py)?[ \t]*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    return text if "def " in text else None


def check_arith(task, replies) -> tuple[bool, str]:
    got = _last_int(replies[0])
    return got == task["expected"], f"expected {task['expected']}, last integer in reply {got}"


def check_json(task, replies) -> tuple[bool, str]:
    obj = _first_json(replies[0])
    if not isinstance(obj, dict):
        return False, "no JSON object parsed"
    for k, ty in task["schema"].items():
        if k not in obj:
            return False, f"missing key {k}"
        v = obj[k]
        ok = {"number": isinstance(v, (int, float)) and not isinstance(v, bool), "integer": isinstance(v, int) and not isinstance(v, bool),
              "string": isinstance(v, str), "list": isinstance(v, list)}[ty]
        if not ok:
            return False, f"key {k} is not {ty}"
    exp = task["expected"]
    for k, ev in exp.items():
        gv = obj[k]
        if isinstance(ev, float):
            if abs(float(gv) - ev) > task.get("tol", {}).get(k, 0.011):
                return False, f"{k}: {gv} != {ev}"
        elif isinstance(ev, list):
            if [str(x).lower() for x in gv] != [str(x).lower() for x in ev]:
                return False, f"{k}: {gv} != {ev}"
        elif isinstance(ev, str):
            if str(gv).strip().lower() != ev.lower():
                return False, f"{k}: {gv} != {ev}"
        elif gv != ev:
            return False, f"{k}: {gv} != {ev}"
    return True, "schema and values match"


def sandbox_available() -> bool:
    return sys.platform == "darwin" and os.path.exists("/usr/bin/sandbox-exec")


def run_code_sandboxed(code: str, tests: list, time_limit_s: int = 10) -> dict:
    """Run the candidate code plus the tests in a deny-default macOS sandbox:
    the interpreter tree readable, the task's temp dir the only writable
    place, no network, CPU and file-size limits. Returns the outcome."""
    if not sandbox_available():
        return {"status": "hold", "reason": "no sandbox on this platform"}
    import secrets
    py = os.path.realpath(sys.executable)
    d = os.path.realpath(tempfile.mkdtemp(prefix="qbat_"))
    prog = os.path.join(d, "task.py")
    nonce = secrets.token_hex(8)
    # the harness: resource limits set INSIDE the sandboxed process (no preexec
    # hook in the multithreaded runner), stdout bound to a file in the task dir
    # under the file-size limit, a start sentinel (a run without it is a
    # harness or sandbox failure, not the candidate's), the candidate, the
    # tests, and a per-run nonce printed only after every test passed
    harness = (
        "import resource, sys\n"
        "def _lim(name, v):\n"
        "    r = getattr(resource, name); h = resource.getrlimit(r)[1]\n"
        "    v = v if h == resource.RLIM_INFINITY else min(v, h)\n"
        "    resource.setrlimit(r, (v, v))\n"
        f"_lim('RLIMIT_CPU', {time_limit_s}); _lim('RLIMIT_FSIZE', 1 << 20); _lim('RLIMIT_NPROC', 1)\n"   # macOS refuses to lower RLIMIT_DATA; memory is not limited
        f"sys.stdout = sys.stderr = open({os.path.join(d, 'out.txt')!r}, 'w', buffering=1)\n"
        "print('QBAT-START')\n")
    with open(prog, "w") as f:
        f.write(harness + code + "\n\n" + "\n".join(tests) + f"\nprint('ALL-TESTS-PASSED-{nonce}')\n")
    # allow-default with explicit denials (a deny-default profile aborts this
    # interpreter at start-up): no network, no writes outside the task dir, no
    # reads under /Users, /Volumes or /private/etc (home, secrets, keys), no
    # exec of anything but the interpreter; the environment is emptied
    profile = ("(version 1)(allow default)(deny network*)(deny file-write*)"
               f"(allow file-write* (subpath \"{d}\") (literal \"/dev/null\"))"
               "(deny file-read* (subpath \"/Users\") (subpath \"/Volumes\") (subpath \"/private/etc\"))"
               f"(allow file-read* (subpath \"{d}\"))(deny process-exec)(allow process-exec (literal \"{py}\"))")

    try:
        p = subprocess.run(["/usr/bin/sandbox-exec", "-p", profile, py, "-I", "-B", prog], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, timeout=time_limit_s + 2, cwd=d, env={})
        rc, err = p.returncode, p.stderr[-800:]
        timed_out = False
    except subprocess.TimeoutExpired:
        rc, err, timed_out = None, "wall timeout", True
    try:
        with open(os.path.join(d, "out.txt"), errors="replace") as f:
            out = f.read(1 << 20)
    except OSError:
        out = ""
    if "QBAT-START" not in out:
        return {"status": "error", "reason": "the sandboxed interpreter did not start the harness", "returncode": rc, "stderr": err}
    ok = rc == 0 and f"ALL-TESTS-PASSED-{nonce}" in out
    return {"status": "pass" if ok else "fail", "returncode": rc, "timed_out": timed_out, "stdout": out[-800:], "stderr": err}


def check_code(task, replies) -> tuple[Optional[bool], str]:
    code = _code_block(replies[0])
    if code is None:
        return False, "no code block"
    if re.search(r"^\s*(?:import|from)\s+(?:os|subprocess|socket|shutil|pathlib|ctypes|multiprocessing)\b", code, re.M) or re.search(r"\bopen\(", code):
        return False, "disallowed host access in code (defence in depth; the sandbox is the guard)"
    res = run_code_sandboxed(code, task["tests"], RULES["code_time_limit_s"])
    if res["status"] == "hold":
        return None, res["reason"]
    if res["status"] == "error":
        raise RuntimeError(res["reason"])
    return res["status"] == "pass", json.dumps({k: v for k, v in res.items() if k != "stdout"})[:300]


def check_instruct(task, replies) -> tuple[bool, str]:
    t = replies[0].strip()
    p = task["params"]
    c = task["check"]
    if c == "lines":
        lines = [ln for ln in t.splitlines() if ln.strip()]
        return len(lines) == p["n"], f"{len(lines)} non-empty lines, wanted {p['n']}"
    if c == "include_exclude":
        low = t.lower()
        inc = re.search(r"\b" + re.escape(p["include"]) + r"\b", low) is not None
        exc = re.search(r"\b" + re.escape(p["exclude"]) + r"\b", low) is not None
        return inc and not exc, f"include {inc}, exclude present {exc}"
    if c == "all_caps":
        letters = [ch for ch in t if ch.isalpha()]
        return bool(letters) and all(ch.isupper() for ch in letters), "all letters upper" if letters else "no letters"
    if c == "ends_with":
        last = re.sub(r"[^a-z]", "", t.split()[-1].lower()) if t.split() else ""
        return last == p["word"], f"last word {last!r}"
    if c == "numbered":
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        ok = len(lines) == p["n"] and all(re.match(rf"{i + 1}[.)]\s*\S", ln) for i, ln in enumerate(lines))
        return ok, f"{len(lines)} lines, numbered {ok}"
    if c == "word_band":
        n = len(re.findall(r"[A-Za-z0-9']+", t))
        return p["lo"] <= n <= p["hi"], f"{n} words, band {p['lo']} to {p['hi']}"
    raise ValueError(c)


def check_twoturn(task, replies) -> tuple[bool, str]:
    t2 = replies[1].strip()
    exp = task["expected"]
    if exp.isdigit():
        ok = _last_int(t2) == int(exp) and len(t2.split()) <= 12
    else:
        ok = re.search(r"\b" + re.escape(exp) + r"\b", t2.lower()) is not None and len(t2.split()) <= 12
    return ok, f"turn 2 reply {t2[:60]!r}, expected {exp}"


def check_degen(task, replies, results) -> tuple[bool, str]:
    t = replies[0].strip()
    r0 = results[0]
    if r0["stop_reason"] != "stop_token":
        return False, f"did not stop on a stop token ({r0['stop_reason']})"
    c = task["check"]
    p = task["params"]
    if c == "repeat":
        toks = t.split()
        ok = len(toks) == p["n"] and all(x.strip(".,").lower() == p["word"] for x in toks)
        return ok, f"{len(toks)} tokens"
    if c == "count":
        nums = [int(x) for x in re.findall(r"-?\d+", t)]
        ok = nums == list(range(p["a"], p["b"] + 1))
        return ok, f"numbers {nums[:12]}"
    if c == "one_line":
        lines = [ln for ln in t.splitlines() if ln.strip()]
        ok = len(lines) == 1 and task["expected"] in t.lower()
        return ok, f"{len(lines)} lines; expected {task['expected']}"
    raise ValueError(c)


def score(task: dict, replies: list, results: list) -> dict:
    kind = task["kind"]
    if kind == "arith":
        ok, why = check_arith(task, replies)
    elif kind == "json":
        ok, why = check_json(task, replies)
    elif kind == "code":
        ok, why = check_code(task, replies)
    elif kind == "instruct":
        ok, why = check_instruct(task, replies)
    elif kind == "twoturn":
        ok, why = check_twoturn(task, replies)
    elif kind == "degen":
        ok, why = check_degen(task, replies, results)
    else:
        raise ValueError(kind)
    return {"status": "hold" if ok is None else ("pass" if ok else "fail"), "why": why}
