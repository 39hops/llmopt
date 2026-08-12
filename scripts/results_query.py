"""Query docs/results-index.jsonl (the RESULTS.md index).

    results_query.py --thread streaming        # entries by thread
    results_query.py --type null               # by type
    results_query.py --live                    # not superseded
    results_query.py --chain <id-substring>    # walk amendment links
    results_query.py --grep <regex>            # title regex
Prints: date  type  id  [line]  title/verdict.
"""
import argparse
import json
import re
import sys

E = [json.loads(l) for l in open("docs/results-index.jsonl")]
by_id = {e["id"]: e for e in E}

ap = argparse.ArgumentParser()
ap.add_argument("--thread")
ap.add_argument("--type")
ap.add_argument("--live", action="store_true")
ap.add_argument("--chain")
ap.add_argument("--grep")
ap.add_argument("--repro")
a = ap.parse_args()


def repro(rows, entry_id: str) -> int:
    row = next((r for r in rows if r["id"] == entry_id), None)
    if row is None:
        print(f"no entry with id {entry_id}", file=sys.stderr)
        return 1
    sha = row.get("code_commit")
    if not sha:
        print(f"{entry_id}: code_commit is null (ambiguous backfill); "
              f"fall back to the booking commit via "
              f"git log -S over docs/RESULTS.md", file=sys.stderr)
        return 1
    print(f"git worktree add ../repro-{entry_id[:24]} {sha}")
    for f in row.get("files", []):
        print(f"  # cited: {f}")
    return 0


if a.repro:
    raise SystemExit(repro(E, a.repro))


def show(e, mark=""):
    v = e.get("verdict", "")
    print(f"{e.get('date') or '????-??-??'}  {e['type']:9s} "
          f"{mark}{e['id'][:58]}  [L{e['line']}]"
          + (f"\n    {v}" if v else ""))


if a.chain:
    hits = [e for e in E if a.chain in e["id"]]
    for root in hits:
        seen, stack = set(), [root["id"]]
        chain = []
        while stack:
            i = stack.pop()
            if i in seen or i not in by_id:
                continue
            seen.add(i)
            chain.append(by_id[i])
            e = by_id[i]
            # amends/superseded_by are a str in old entries, a
            # list in newer ones — normalize (Sol review find)
            ids = lambda x, k: (v := x.get(k, [])) and \
                ([v] if isinstance(v, str) else list(v)) or []
            stack += ids(e, "amends") + ids(e, "superseded_by")
            stack += [x["id"] for x in E
                      if i in ids(x, "amends")
                      or i in ids(x, "superseded_by")]
        for e in sorted(chain, key=lambda x: x["line"]):
            show(e, mark="* " if e["id"] == root["id"] else "  ")
        print()
    raise SystemExit

out = E
if a.thread:
    out = [e for e in out if a.thread in e.get("threads", [])
           or a.thread.lower() in e["title"].lower()]
if a.type:
    out = [e for e in out if e["type"] == a.type]
if a.live:
    out = [e for e in out if not e.get("superseded_by")]
if a.grep:
    out = [e for e in out if re.search(a.grep, e["title"], re.I)]
for e in out:
    show(e)
print(f"-- {len(out)} entries")
