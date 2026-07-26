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
    b64=$(printf '%s' "$cmd" | base64)
    "${SSH[@]}" "cd ~/code/llmopt && echo '$b64' | base64 -d > /tmp/wsl_job.sh && setsid bash -c 'bash /tmp/wsl_job.sh > $log 2>&1 && echo DONE > $marker' < /dev/null > /dev/null 2>&1 & echo launched"
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
