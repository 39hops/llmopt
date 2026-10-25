"""QWEN-GDN-PROMOTE-0: the fresh CLI collection (scratch/evalsets/
qwen_battery_v4_cli.json), 16 tasks drawn from the v3 generators
(llmopt.lab.qbattery3) under a fresh stable string seed, with an exclusion
guard: no task whose FIRST user turn appears in the v1, v2 or v3
collections is kept (the farm skill's exclude= rule; a two-turn task's
second turn is one of four fixed v3 question templates and is not the
instance). Code and instruction
tasks are not drawn (their templates and function lists are fixed menus
already exhausted by v3; a fresh instance there would repeat a prompt).
The arithmetic, structured, two-turn and termination families are fresh
INSTANCES of v3 templates, so this collection confirms behaviour on
unread prompts, not on an unrelated benchmark. Written once; a run records
the file's sha256 and never edits it.

    .venv/bin/python scratch/qwen_gdn_promote_collection.py        # refuses if the file exists
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab import qbattery3  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "evalsets", "qwen_battery_v4_cli.json")
SEED = "qwen-battery-v4cli-seed-31"
PRIOR = ["qwen_battery_v1.json", "qwen_battery_v2.json", "qwen_battery_v3.json"]


def build(seed: str = SEED) -> dict:
    r = random.Random(seed)
    tasks = qbattery3.gen_arith(r, 12) + qbattery3.gen_json(r, 8) + qbattery3.gen_twoturn(r, 8) + qbattery3.gen_degen(r, 6)
    prior = set()
    for fn in PRIOR:
        for t in json.load(open(os.path.join(HERE, "evalsets", fn)))["tasks"]:
            prior.update(t["turns"])
    kept, excluded = [], []
    for t in tasks:
        (excluded if t["turns"][0] in prior else kept).append(t)       # the first turn carries the instance (a two-turn task's second turn is a fixed v3 question template)
    want = {"arith": 6, "json": 4, "twoturn": 4, "degen": 2}
    out, seen = [], {k: 0 for k in want}
    for t in kept:
        if seen[t["kind"]] < want[t["kind"]]:
            seen[t["kind"]] += 1
            out.append(dict(t, id=f"cli-{t['kind']}-{seen[t['kind']]:02d}"))
    assert seen == want, seen
    return {"collection": "qwen_battery_v4_cli", "version": 4, "seed": seed, "rules": dict(qbattery3.RULES),
            "generators": "llmopt.lab.qbattery3 (the v3 templates; fresh instances)", "excluded_by_prior_prompt": len(excluded), "prior_collections": PRIOR,
            "independence_note": "fresh instances of the v3 arithmetic, structured, two-turn and termination templates; no first turn shared with v1, v2 or v3 (asserted at build; two-turn second turns are the fixed v3 question templates); code and instruction tasks omitted because their menus are fixed and exhausted by v3",
            "tasks": out}


if __name__ == "__main__":
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    c = build()
    json.dump(c, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(c['tasks'])} tasks, {c['excluded_by_prior_prompt']} excluded by a prior prompt")
