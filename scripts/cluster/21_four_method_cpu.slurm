#!/bin/bash
# Section 3.4 / task B4, CPU pass: the same three methods at the largest
# size of each benchmark, on the same node type's CPU, all its cores.
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-b4-cpu
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail

PROBLEMS=(bratu burgers bl bl_gravity)
P="${PROBLEMS[$SLURM_ARRAY_TASK_ID]}"
python experiments/four_method_tables.py --problems "$P" --passes cpu     --out-dir "$B/four_method_jobs/${P}_cpu"
