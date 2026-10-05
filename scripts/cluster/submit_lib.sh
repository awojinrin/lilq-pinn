# Shared by submit_wave1.sh ... submit_wave4.sh (sourced, with
# LILQ_WAVE set). Run the wave scripts from a login node:
#   bash scripts/cluster/submit_wave1.sh            (CLUSTER=grace is the default)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs   # sbatch does not create the --output directory
S=scripts/cluster
RESULTS="${LILQ_RESULTS:-$PWD/results}"   # LILQ_RESULTS: tests only

# Check the profile before submitting anything (a failure halfway would leave
# part of a wave queued).
CLUSTER="${CLUSTER:-grace}"
[[ -f $S/profiles/$CLUSTER.sh ]] || { echo "no profile $S/profiles/$CLUSTER.sh" >&2; exit 1; }
source $S/profiles/$CLUSTER.sh
[[ -n "$ACCOUNT" ]] || { echo "set LILQ_ACCOUNT to your HPRC allocation account (myproject -l)" >&2; exit 1; }
[[ -n "$CPU_PARTITION" && -n "${CPU_NODE_CORES:-}" ]] || { echo "the profile needs CPU_PARTITION and CPU_NODE_CORES" >&2; exit 1; }
export CLUSTER CPU_PARTITION LILQ_WAVE

# The code's commit (the bundle's, on the cluster) and a wave's locked commit.
code_commit() {
    python3 -c "import json; print(json.load(open('PROVENANCE.json'))['commit'])" 2>/dev/null \
        || git rev-parse HEAD 2>/dev/null || echo unknown
}
wave_commit() {   # wave_commit <N>: the commit results/wave<N> is locked to, or nothing
    python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['commit'])" "$RESULTS/wave$1/COMMIT" 2>/dev/null || true
}

# DRY_RUN=1: every job goes through `sbatch --test-only` -- the scheduler's own
# checks (Grace's job_submit rules, partitions, limits) and an estimated start --
# and nothing is submitted, charged or written; dependencies are left out (there
# are no job IDs). Run it before each wave.
DRY_RUN="${DRY_RUN:-0}"
dry_failed=""
if [[ "$DRY_RUN" == 1 ]]; then
    trap 'echo; if [[ -n "$dry_failed" ]]; then echo "DRY RUN FAILED for:$dry_failed (nothing was submitted)"; exit 1;
          else echo "DRY RUN OK: every job passed sbatch --test-only; nothing was submitted."; fi' EXIT
fi

# Before each wave: the balance, then an explicit go-ahead (YES=1 skips the question).
confirm_balance() {   # confirm_balance <the advisor's SU estimate for this wave>
    local what="wave $LILQ_WAVE"
    if [[ "$LILQ_WAVE" == p2s* ]]; then   # Package 2's stages write results/package2_stage<N>
        what="Package 2, stage ${LILQ_WAVE#p2s}"
        echo "Package 2, stage ${LILQ_WAVE#p2s} (results/package2_stage${LILQ_WAVE#p2s}), code at $(code_commit)."
    else
        echo "Wave $LILQ_WAVE of 4 (results/wave$LILQ_WAVE), code at $(code_commit)."
    fi
    echo "Estimated request: $1. Read the 'Requested SUs' line sbatch prints for each job."
    if command -v myproject >/dev/null; then myproject -l || true; else echo "(myproject not found: check the balance another way)"; fi
    [[ "$DRY_RUN" == 1 ]] && { echo "DRY RUN: checking each job with sbatch --test-only."; return 0; }
    if [[ "${YES:-0}" != 1 ]]; then
        read -r -p "Balance checked; submit $what? [y/N] " answer
        [[ "$answer" =~ ^[yY] ]] || { echo "Nothing submitted."; exit 1; }
    fi
}

# sbatch can fail without a usable job ID (e.g. "Socket timed out"); stop at
# the first such failure. A timeout may still have queued the job, so check
# `squeue -u $USER` (and `sacct`) before resubmitting anything.
submitted=""
submit() {   # submit <variable> <job script> [sbatch options...]: sets <variable> to the job ID
    local var=$1 script=$2 id; shift 2
    if [[ "$DRY_RUN" == 1 ]]; then
        local args=() a
        for a in "$@"; do [[ "$a" == --dependency=* ]] || args+=("$a"); done
        printf '  %-28s ' "$(basename "$script")"
        if bash $S/sbatch.sh "$script" --test-only ${args[@]+"${args[@]}"} 2>&1 | tail -3 | tr '\n' ' '; [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            echo
        else
            echo " <- FAILED"; dry_failed="$dry_failed $(basename "$script")"
        fi
        printf -v "$var" '%s' 0
        return 0
    fi
    id=$(bash $S/sbatch.sh "$script" --parsable "$@" | cut -d';' -f1) || true
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "submission failed for $script; submitted so far:${submitted:- none}." >&2
        echo "Check squeue -u \$USER before resubmitting." >&2
        exit 1
    fi
    submitted="$submitted $id"
    printf -v "$var" '%s' "$id"
    printf '  %-28s %s %s\n' "$(basename "$script")" "$id" "$*"
}
