"""Checkpoint DELETE pass (Artin sign-off 2026-08-08 on
logs/triage/revive_cite_plan.md). Builds per-host rm manifests for
the signed DELETE-AFTER-SIGNOFF families, with the safety interlock
Artin mandated: the manifest carries the inventory sha256 and the
executor re-hashes each file AT DELETE TIME — mismatch means stop
that row, keep the file, report. Classifier class is never the
interlock.

Mechanical exclusions (freeze rules): git-tracked paths, the plan's
load-bearing named set, everything in uncited_but_consumed.json,
and any CITED/DUP-CITED row. merged_grown_identity.pt is HELD until
the crown-tie thread decides (plan).

MODE=plan (default) prints; MODE=emit writes
logs/triage/delete_pass_{mac,wsl}.json.
"""
import json
import os
import re
import subprocess

FREEZE = {
    "mathnative_gen6_grown.pt", "mathnative_gen6_ternary.pt",
    "mathnative_gen6_ternary_latent.pt", "merged_grown.pt",
    "merged_grown_latent.pt", "mathnative_19m.pt",
    "mathnative_19m_infixtwin.pt", "metab_v4.pt", "metab_v5_s1.pt",
    "seedvar_1.pt", "mathnative_grpo.pt", "mathnative_grpo_c010.pt",
    "merged_grown_identity.pt",  # HELD, not frozen forever
}

DELETE_PATTERNS = [
    r"^mathnative_grpo_(c\d+|cand\d+|run_c\d+|run_cand\d+)\.pt$",
    r"^mathnative_45m_gen5mine.*cand\d+\.pt$",
    r"^grpo_shaped_(c|cand)\d+.*\.pt$",
    r"^snapm_q\d+.*\.pt$",
    r"^u45_(fq|rat)\d+\.pt$",
    r"^pred_syndromes_5b\.pt$",
    r"^layers\.\d+\.ffn\.experts\..*\.bin$",   # v4flash_sample cache
    r".*_gen4.*\.pt$", r".*_gen5.*\.pt$",
    r"^mathnative_110m_(v21_lr25|grpo)\.pt$",
    r"^grid_(T_768|fp32_768).*\.pt$",
    r"^tourn_[PZ].*\.pt$",
]

tracked = set(subprocess.run(["git", "ls-files", "checkpoints/"],
              capture_output=True, text=True).stdout.split())
consumed = set(json.load(open("logs/triage/uncited_but_consumed.json")))
rows = [json.loads(l) for l in open("logs/triage/triage_table.jsonl")]

out = {"mac": {}, "wsl": {}}
kept = []
for r in rows:
    name = r["path"].split("/")[-1]
    if not any(re.match(p, name) for p in DELETE_PATTERNS):
        continue
    if (r["path"] in tracked or r["path"] in consumed
            or name in FREEZE or r["cls"] in ("CITED", "DUP-CITED")):
        kept.append((r["host"], r["path"], "excluded-by-freeze"))
        continue
    out[r["host"]][r["path"]] = r["sha256"]

for host in ("mac", "wsl"):
    n = len(out[host])
    gb = sum(1 for _ in out[host])
    print(f"{host}: {n} paths marked")
for h, p, why in kept:
    print(f"  KEPT {h}:{p} ({why})")

if os.environ.get("MODE") == "emit":
    for host in ("mac", "wsl"):
        fn = f"logs/triage/delete_pass_{host}.json"
        json.dump(out[host], open(fn, "w"), indent=1)
        print("wrote", fn)
