"""ARTIFACT ARENA: two compressed towers, one prompt, side by side.

Left lane v right lane (default: artifact A v artifact B), token
streams rendered in parallel with the divergence point marked —
the qualitative instrument for watching io-precision change a
trajectory. Reads paired rung-4 receipts (produced on the 3080 by
scratch/qwen_cuda_rung4.py); with --run it fires a fresh PAIRED
run first (same prompt, same commit, same settings — the pairing
discipline the 2026-08-17 same-commit incident bought).

    .venv/bin/python scripts/arena_qwen.py \
        --left logs/qwencuda/rung4_A_qm1400_paired.json \
        --right logs/qwencuda/rung4_B_qm1400_paired.json
    .venv/bin/python scripts/arena_qwen.py --run \
        --prompt "..." --n-new 700

Chat reads never gate anything (registered); this is a viewing
instrument, not a scorer.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import textwrap

BOLD, DIM, RED, GREEN, CYAN, END = ("\033[1m", "\033[2m", "\033[91m",
                                    "\033[92m", "\033[96m", "\033[0m")
WIDTH = 58


def load(path: str) -> dict:
    r = json.load(open(path))
    return {"artifact": r.get("artifact", "?"),
            "commit": r.get("code_commit", "?"),
            "tok_s": r.get("gen", {}).get("tok_s"),
            "n": r.get("gen", {}).get("n_new"),
            "text": r.get("gen", {}).get("output", ""),
            "prompt": r.get("prompt", "")}


def run_pair(prompt: str, n_new: int) -> tuple[str, str]:
    """Fire a fresh paired run on the 3080 (one wsl.sh call, both
    artifacts, same commit): receipts land under logs/qwencuda/
    with a shared suffix and pull back here."""
    import time
    tag = time.strftime("%H%M%S")
    outs = []
    q = shlex.quote(prompt)
    cmds = []
    for arm in ("A", "B"):
        rec = f"logs/qwencuda/arena_{arm}_{tag}.json"
        outs.append(rec)
        cmds.append(
            f"TORCH_DISABLE_NATIVE_JIT=1 ART_DIR=~/qwen_whole0t/{arm} "
            f"N_NEW={n_new} PROMPT={q} RECEIPT={rec} "
            f".venv/bin/python scratch/qwen_cuda_rung4.py "
            f"2>&1 | tail -1")
    remote = ("cd ~/code/llmopt && git pull --ff-only -q && "
              + " && ".join(cmds))
    subprocess.run(["scratch/wsl.sh", "run", remote], check=True)
    for rec in outs:
        raw = subprocess.check_output(
            ["scratch/wsl.sh", "run", f"cat ~/code/llmopt/{rec}"])
        os.makedirs(os.path.dirname(rec), exist_ok=True)
        with open(rec, "wb") as f:
            f.write(raw)
    return outs[0], outs[1]


def render(lpath: str, rpath: str) -> None:
    L, R = load(lpath), load(rpath)
    print(f"{BOLD}PROMPT{END}: {L['prompt']}")
    same_commit = L["commit"] == R["commit"]
    cc = f"{GREEN}same commit {L['commit']}{END}" if same_commit \
        else f"{RED}COMMITS DIFFER {L['commit']} v {R['commit']} — " \
             f"not a paired read{END}"
    print(f"{DIM}{cc}{END}\n")
    lw = textwrap.wrap(L["text"], WIDTH) or [""]
    rw = textwrap.wrap(R["text"], WIDTH) or [""]
    # first divergence in the raw text
    div = next((i for i, (a, b) in enumerate(
        zip(L["text"], R["text"])) if a != b),
        min(len(L["text"]), len(R["text"])))
    hdr_l = (f"{BOLD}{L['artifact']}{END} "
             f"{DIM}{L['tok_s']} tok/s, {L['n']} tok{END}")
    hdr_r = (f"{BOLD}{R['artifact']}{END} "
             f"{DIM}{R['tok_s']} tok/s, {R['n']} tok{END}")
    print(f"{hdr_l:<{WIDTH + 16}}| {hdr_r}")
    print("-" * (WIDTH + 8) + "+" + "-" * (WIDTH + 8))
    seen = 0
    for i in range(max(len(lw), len(rw))):
        lt = lw[i] if i < len(lw) else ""
        rt = rw[i] if i < len(rw) else ""
        mark = " "
        if seen <= div <= seen + len(lt):
            mark = f"{CYAN}>{END}"      # divergence enters this line
        seen += len(lt) + 1
        print(f"{mark}{lt:<{WIDTH + 6}} | {rt}")
    print(f"\n{DIM}first character divergence at offset {div} "
          f"({CYAN}>{END}{DIM} marks the left-lane line){END}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left",
                    default="logs/qwencuda/rung4_A_qm1400_paired.json")
    ap.add_argument("--right",
                    default="logs/qwencuda/rung4_B_qm1400_paired.json")
    ap.add_argument("--run", action="store_true",
                    help="fire a fresh paired A/B run on the 3080")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n-new", type=int, default=256)
    a = ap.parse_args()
    if a.run:
        lp, rp = run_pair(a.prompt, a.n_new)
    else:
        lp, rp = a.left, a.right
    render(lp, rp)


if __name__ == "__main__":
    main()
