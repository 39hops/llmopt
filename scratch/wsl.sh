#!/bin/bash
# wsl.sh — standardized 3080/WSL remote ops (doctrine: remote-ops,
# friendly-fire, ssh-nohup hang). Usage:
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

# Character allowlist for any argument that gets interpolated into a
# remote shell string by the safe-class verbs (clean-marker/kill/mkdir).
# Allows [A-Za-z0-9._/-] only: no quotes, no $ ` ; & | newline, no
# globs, no `..` traversal, no absolute paths, no leading dash.
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
    # SECURITY (2026-08-11 review): these three verbs interpolate their
    # argument into a REMOTE shell string, and the permission hook
    # auto-allows two of them — so the argument must be validated by a
    # strict CHARACTER ALLOWLIST, not a glob whitelist. A `case` pattern
    # like logs/*.DONE matches `logs/x;rm -rf ~/.DONE` (the * eats the
    # semicolon), which was live command injection behind an auto-allow.
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
