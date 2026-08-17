"""Thin CLI over llmopt.lab.qartifact — the qualification ladder.

All logic lives in the library so consumers (runtime, sidecar,
scorer) share it in code rather than by workflow. See
llmopt/lab/qartifact.py for the rungs and /qualify for the law.

    ART_DIR=~/qwen_whole0t/A python scratch/qwen_qualify.py
    ALLOW_UNCHAINED=1 ...    explicit identity override (recorded)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab import qartifact  # noqa: E402

ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/A"))
VENDOR_INDEX = os.path.expanduser(os.environ.get(
    "VENDOR_INDEX", "~/qwen_vendor/model.safetensors.index.json"))


def main():
    arm = os.path.basename(ART.rstrip("/"))
    chain = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "qwenwhole",
        f"artifact_digest_{arm}.txt")
    out = qartifact.qualify_artifact(
        ART, VENDOR_INDEX,
        chain if os.path.exists(chain) else None,
        allow_unchained=os.environ.get("ALLOW_UNCHAINED") == "1")
    print(json.dumps(out["report"], indent=1))
    print("[q] PASS")


if __name__ == "__main__":
    main()
