#!/bin/bash
# Section 3.4 / task B4, GPU pass: NiL-N, NiL-Q (seeds 0 1 2) and LiL-N at
# every size, one A100 node per benchmark (the timed class: all its cores and
# memory). Resumable: resubmit the same array index after a walltime kill.
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-b4-gpu
#SBATCH --array=0-3
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail

# Stall controls (the advisor's reply to wave 1, item 2.5): after its own
# runs, each task reruns those that ended on optimizer_stall with tolerances 0
# and F1's restart-once rule (--controls). In wave 2 the Bratu task runs only
# the controls of wave 1's Bratu runs, into its own folder (so wave 1's Bratu
# table is not overridden when the package is assembled).
PROBLEMS=(bratu burgers bl bl_gravity)
P="${PROBLEMS[$SLURM_ARRAY_TASK_ID]}"
OUT="$B/four_method_jobs/${P}_gpu"
MODE=(--controls)
if [[ "$P" == bratu && "$LILQ_WAVE" != 1 ]]; then
    OUT="$B/four_method_jobs/${P}_gpu_controls"
    MODE=(--controls-only --controls-from "$RESULTS/wave1/B_instrumentation/four_method_jobs/${P}_gpu")
fi
python experiments/four_method_tables.py --problems "$P" --passes primary "${MODE[@]}" --out-dir "$OUT"
