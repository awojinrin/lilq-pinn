#!/bin/bash
# Section 3.4 / task B4, CPU pass: the same three methods at the largest
# size of each benchmark, on an exclusive CPU node (the A100 nodes' CPU model; Addendum v2.2 4.1), all its cores.
# 4 h: gravity BL's worst case reached the earlier 3 h (the advisor's reply to wave 1, item 2.4).
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-b4-cpu
#SBATCH --array=0-3
#SBATCH --time=04:00:00
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
OUT="$B/four_method_jobs/${P}_cpu"
MODE=(--controls)
if [[ "$P" == bratu && "$LILQ_WAVE" != 1 ]]; then
    OUT="$B/four_method_jobs/${P}_cpu_controls"
    MODE=(--controls-only --controls-from "$RESULTS/wave1/B_instrumentation/four_method_jobs/${P}_cpu")
fi
python experiments/four_method_tables.py --problems "$P" --passes cpu "${MODE[@]}" --out-dir "$OUT"
