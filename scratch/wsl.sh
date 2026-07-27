#!/bin/bash
# wsl.sh — standardized 3080/WSL remote ops (doctrine: remote-ops,
# friendly-fire, ssh-nohup hang). Usage:
#   scratch/wsl.sh run  "cmd"                  # exec, output back, safe timeout
#   scratch/wsl.sh launch "cmd" LOGFILE MARKER # detached; marker fires on SUCCESS only
#   scratch/wsl.sh check "pgrep-pattern"       # process status (never matches itself)
#   scratch/wsl.sh tail LOGFILE [N]            # tail a remote log
# Host/key live in gitignored scratch/remote.env.sh; repo = ~/code/llmopt.
set -euo pipefail
source "$(dirname "$0")/remote.env.sh"
SSH=(ssh -i "$WSL_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$WSL_REMOTE")

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
  *)
    echo "unknown verb: $1" >&2; exit 2
    ;;
esac
