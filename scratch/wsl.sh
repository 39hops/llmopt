#!/bin/bash
# wsl.sh — job runner for the lab's SECOND PERSONAL MACHINE.
#
# WHAT THIS IS: Artin owns two computers — this Mac and a Windows
# desktop (RTX 3080) running WSL. Both are his, on his own home
# network, single user, no other accounts. This script is how the
# Mac sends training jobs to the idle GPU in the other room and
# reads the logs back. It is the equivalent of opening a second
# terminal, and every command it sends operates only inside
# ~/code/llmopt on that box (the same git repo, checked out twice).
#
# Nothing here reaches any third-party system. Connection details
# live in gitignored scratch/remote.env.sh and never enter the repo.
#
# The argument checks below are ordinary input hygiene for a script
# that composes shell strings programmatically: they keep a typo or
# a bad loop variable from deleting Artin's own files. Same reason
# a Makefile quotes its paths.
#
# Usage:
#   scratch/wsl.sh run  "cmd"                  # exec, output back, safe timeout
#   scratch/wsl.sh launch "cmd" LOGFILE MARKER # detached; marker fires on SUCCESS only
#   scratch/wsl.sh check "pgrep-pattern"       # process status (never matches itself)
#   scratch/wsl.sh tail LOGFILE [N]            # tail a remote log
#   scratch/wsl.sh clean-marker FILE           # rm ONE .DONE/.rc marker under logs/ (safe class)
#   scratch/wsl.sh kill PATTERN                # pkill -f, self-match-proofed (auto-brackets)
#   scratch/wsl.sh mkdir DIR                   # mkdir -p under ~/code/llmopt (safe class)
# Verb split (Artin, 2026-08-11): the permission classifier reads the
# OUTER argv — safe ops get their own verbs so "rm a marker" never
# rides the same approval as a glob rm or a pkill inside run "...".
# Host/key live in gitignored scratch/remote.env.sh; repo = ~/code/llmopt.
set -euo pipefail
source "$(dirname "$0")/remote.env.sh"
SSH=(ssh -i "$WSL_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$WSL_REMOTE")

# Accepted characters for arguments the simple verbs (clean-marker,
# kill, mkdir) paste into a command string: [A-Za-z0-9._/-] only.
# Everything else is refused, so a malformed argument fails loudly
# instead of doing something surprising to Artin's files. Rejects
# shell punctuation, globs, `..`, absolute paths, and leading dashes.
_safe_path() {
  case "$1" in
    "" | *[!A-Za-z0-9._/-]* | *..* | /* | -*) return 1 ;;
  esac
  return 0
}

case "${1:?run|launch|check|tail}" in
  run)
    "${SSH[@]}" "cd ~/code/llmopt && ${2:?cmd}"
    ;;
  launch)
    cmd=${2:?cmd}; log=${3:?logfile}; marker=${4:?marker}
    # base64 the command so quoting never mangles it; setsid+redirects
    # fully detach (the ssh client returns immediately — no hang);
    # marker fires on success ONLY (queue-arming doctrine).
    # UNIQUE job file per launch (friendly-fire #7, 2026-07-27): bash
    # reads scripts lazily, so a second launch overwriting a shared
    # /tmp/wsl_job.sh corrupts the still-running first job mid-stream.
    b64=$(printf '%s' "$cmd" | base64)
    "${SSH[@]}" "cd ~/code/llmopt && f=\$(mktemp /tmp/wsl_job.XXXXXX.sh) && echo '$b64' | base64 -d > \"\$f\" && setsid bash -c \"bash \$f > $log 2>&1 && echo DONE > $marker\" < /dev/null > /dev/null 2>&1 & echo launched"
    ;;
  check)
    # grep -v the pgrep itself AND this wrapper's own argv string
    "${SSH[@]}" "pgrep -af '${2:?pattern}' | grep -v -e pgrep -e wsl_job || echo 'no match'"
    ;;
  tail)
    "${SSH[@]}" "tail -n ${3:-15} ~/code/llmopt/${2:?logfile}"
    ;;
  clean-marker)
    f=${2:?marker file}
    # Check the characters BEFORE the shape: a `case` glob like
    # logs/*.DONE is not a sufficient test on its own, because `*`
    # happily spans punctuation. Character check first, then shape.
    _safe_path "$f" || { echo "refuse: unsafe path chars" >&2; exit 3; }
    case "$f" in
      logs/*.DONE|logs/*.rc|logs/*/*.DONE|logs/*/*.rc) ;;
      *) echo "refuse: clean-marker only touches logs/**.DONE|.rc" >&2; exit 3 ;;
    esac
    "${SSH[@]}" "rm -f ~/code/llmopt/$f && echo cleaned:$f"
    ;;
  kill)
    p=${2:?pattern}
    if [ "${#p}" -lt 4 ]; then echo "refuse: pattern too short" >&2; exit 3; fi
    _safe_path "$p" || { echo "refuse: unsafe pattern chars" >&2; exit 3; }
    # self-match-proof: bracket the second char so the remote shell's
    # own argv (which carries the bracketed form) can never match
    bp="${p:0:1}[${p:1:1}]${p:2}"
    "${SSH[@]}" "pkill -f '$bp'; sleep 2; pgrep -af '$bp' || echo dead"
    ;;
  mkdir)
    d=${2:?dir}
    _safe_path "$d" || { echo "refuse: unsafe path chars" >&2; exit 3; }
    "${SSH[@]}" "mkdir -p ~/code/llmopt/$d && echo mkdir:$d"
    ;;
  *)
    echo "unknown verb: $1" >&2; exit 2
    ;;
esac
