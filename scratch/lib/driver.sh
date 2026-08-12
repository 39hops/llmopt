# scratch/lib/driver.sh — shared preamble for experiment drivers.
# Source it: `. "$(dirname "$0")/lib/driver.sh"` (new drivers only;
# frozen drivers cited by booked verdicts are never retrofitted).
#
# Provides:
#   strict mode        set -euo pipefail on source. pipefail is the
#                      load-bearing part: without it a dead python in a
#                      `python ... | tee` chain logs rc=0 (the
#                      2026-08-11 dead-run-logged-green failure).
#   llmopt_cd          cd to the repo root or die loudly.
#   cuda_preamble      the 3080/WSL environment block in one place.
#   mark_done PATH RC  write a completion marker on SUCCESS ONLY
#                      (rc==0). Markers that fire on failure are the
#                      friendly-fire class: watchers treat the marker
#                      as "result exists".
#   wait_for PATTERN TIMEOUT_S [INTERVAL_S]
#                      poll pgrep -f PATTERN until it exits. The
#                      PATTERN must never match the launcher's own
#                      command line (pass a fragment unique to the
#                      child, e.g. the script basename, never the full
#                      launch string the caller itself carries).

set -euo pipefail

llmopt_cd() {
    cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/code/llmopt")" || {
        echo "driver.sh: no llmopt checkout" >&2
        exit 1
    }
    [ -f pyproject.toml ] || { echo "driver.sh: not at repo root" >&2; exit 1; }
}

cuda_preamble() {
    # WSL venv has no C compiler: stop torch's _native router from
    # JIT-ing triton kernels for aten ops even without torch.compile
    export TORCH_DISABLE_NATIVE_JIT=1
    # in-tree allocator default; expandable_segments crashes the WSL
    # driver, max_split_size is the knob that held the measured 43x
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
    export PYTHONUNBUFFERED=1
}

mark_done() {
    # mark_done <marker-path> <rc> — marker on success only
    local marker="$1" rc="$2"
    if [ "$rc" -eq 0 ]; then
        printf 'rc=0 ts=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$marker"
    else
        echo "driver.sh: rc=$rc, no marker written ($marker)" >&2
    fi
    return "$rc"
}

wait_for() {
    # wait_for <pgrep-pattern> <timeout-s> [interval-s]
    local pattern="$1" timeout="$2" interval="${3:-30}" waited=0
    while pgrep -f "$pattern" > /dev/null 2>&1; do
        if [ "$waited" -ge "$timeout" ]; then
            echo "driver.sh: wait_for '$pattern' timed out after ${timeout}s" >&2
            return 124
        fi
        sleep "$interval"
        waited=$((waited + interval))
    done
    return 0
}
