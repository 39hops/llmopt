"""MATH-CYBER-1 SVP-TOKEN-CHANNEL-DESK-0 — identifiability
census of ActionProgram token channels: which token IDs carry
rule / site kind / site ordinal / param kind / param index /
separators in the 73,324 training program targets, and how much
each ID is ALSO exposed through STATE-view text (state targets +
shared prompts). Pure counting; no model, no training.

Output logs/mathworld1/svptokdesk_receipt.json
(refuse-if-exists).

    .venv/bin/python scratch/mathworld1_svptokdesk.py         (Mac)
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402

PAIRED = "data/matsub_paired.jsonl"
PAIRED_SHA = ("a943ba7fc581db743b07192e5d951fadd"
              "dd2ba19bca3225b75d8402351d468e8")
RECEIPT = Path("logs/mathworld1/svptokdesk_receipt.json")
TOK = ActionGCTok()


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if RECEIPT.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} exists")
    gate(fsha(PAIRED) == PAIRED_SHA, "PIN")
    START = start_provenance(
        ["scratch/mathworld1_svptokdesk.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_svpbirth.py",
         "llmopt/lab/provenance.py"])
    # channel decomposition of program_text:
    # "<r:RULE>" + (" KIND" + "ORD") + (" uK" | " tK")? + "\n"
    prog_chan = {k: Counter() for k in
                 ("rule_opcode", "site_kind", "site_ordinal",
                  "param_marker", "param_index", "separator")}
    state_ids = Counter()
    prompt_ids = Counter()
    n = 0
    for l in open(PAIRED):
        r = json.loads(l)
        n += 1
        state_ids.update(TOK.encode(r["state_target"] + "\n"))
        prompt_ids.update(TOK.encode(
            f"Current: {r['cur']}\nHints: none\nStep: "))
        # channel split via the serializer's own fields
        rid = TOK.encode(f"<r:{r['rule']}>")
        gate(len(rid) == 1 and rid[0] >= 296, "RULE NOT OPCODE")
        prog_chan["rule_opcode"].update(rid)
        full = TOK.encode(r["program_text"])
        # everything after the opcode is base-alphabet; classify
        # by re-encoding each printed field
        sk = r["site_kind"]
        if sk and sk != "W":
            ktoks = TOK.encode(f" {sk}")
            otoks = TOK.encode(str(r["site_ordinal"]))
            prog_chan["site_kind"].update(ktoks)
            prog_chan["site_ordinal"].update(otoks)
        if r["param_kind"] == "u_choice":
            ptoks = TOK.encode(f" u{r['param_index']}")
            prog_chan["param_marker"].update(ptoks[:2])
            prog_chan["param_index"].update(ptoks[2:] or ptoks[-1:])
        elif r["param_kind"] == "term_index":
            ptoks = TOK.encode(f" t{r['param_index']}")
            prog_chan["param_marker"].update(ptoks[:2])
            prog_chan["param_index"].update(ptoks[2:] or ptoks[-1:])
        prog_chan["separator"].update(TOK.encode("\n"))
        # sanity: channels reconstruct the full token stream
        gate(TOK.decode(full) == r["program_text"], "ROUNDTRIP")
    gate(n == 73324, "ROWS")

    def report(chan):
        ids = prog_chan[chan]
        rows = []
        for tid, cnt in ids.most_common():
            rows.append({
                "id": tid,
                "atom": TOK.decode([tid]),
                "program_count": cnt,
                "state_target_count": state_ids.get(tid, 0),
                "prompt_count": prompt_ids.get(tid, 0),
                "action_only": (state_ids.get(tid, 0) == 0
                                and prompt_ids.get(tid, 0) == 0)})
        return rows

    channels = {c: report(c) for c in prog_chan}
    summary = {c: {
        "distinct_ids": len(prog_chan[c]),
        "action_only_ids": sum(1 for r in channels[c]
                               if r["action_only"]),
        "shared_ids": sum(1 for r in channels[c]
                          if not r["action_only"])}
        for c in prog_chan}
    receipt = {
        "rows": n,
        "summary": summary,
        "channels": channels,
        "distinct_state_ids": len(state_ids),
        "distinct_prompt_ids": len(prompt_ids),
        "pins": {PAIRED: fsha(PAIRED)},
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({"summary": summary,
                      "channels": {c: channels[c][:12]
                                   for c in channels}},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
