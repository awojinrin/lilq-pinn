#!/bin/bash
# Submit one job script with the resources its class needs on this cluster:
#   bash scripts/cluster/sbatch.sh <job.slurm> [extra sbatch options...]
# CLUSTER (grace, the default, or faster) picks profiles/$CLUSTER.sh. Each job
# script declares its class in a "# lilq-resources:" line:
#   timed       the whole A100 node (--exclusive), all its cores and memory
#   shared-gpu  one A100, 8 cores, 32 GB (untimed GPU work)
#   cpu         24 cores, 32 GB, no GPU (untimed CPU work)
# Use this for resubmissions too (e.g. one array index: ... sbatch.sh job.slurm --array=2).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
CLUSTER="${CLUSTER:-grace}"
source "$here/profiles/$CLUSTER.sh"
export CLUSTER LILQ_MODULES

script="$1"; shift
class=$(sed -n 's/^# lilq-resources: \([a-z-]*\).*/\1/p' "$script")
case "$class" in
    timed)      res=(--partition="$GPU_PARTITION" --gres="$GPU_GRES" --exclusive --ntasks=1
                     --cpus-per-task="$NODE_CORES" --mem=0) ;;
    shared-gpu) res=(--partition="$GPU_PARTITION" --gres="$GPU_GRES" --ntasks=1 --cpus-per-task=8 --mem=32G) ;;
    cpu)        [[ -n "$CPU_PARTITION" ]] || { echo "set CPU_PARTITION (see sinfo -s)" >&2; exit 1; }
                res=(--partition="$CPU_PARTITION" --ntasks=1 --cpus-per-task=24 --mem=32G) ;;
    *)          echo "$script: no '# lilq-resources:' class" >&2; exit 1 ;;
esac
exec sbatch --account="$ACCOUNT" "${res[@]}" "$@" "$script"
