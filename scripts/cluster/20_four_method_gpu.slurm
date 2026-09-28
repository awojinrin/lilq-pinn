#!/bin/bash
# Section 3.4 / task B4, GPU pass: NiL-N, NiL-Q (seeds 0 1 2) and LiL-N at
# every size, one exclusive node per benchmark. Resumable: resubmit the same
# array index after a walltime kill.
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-b4-gpu
#SBATCH --array=0-3
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail

PROBLEMS=(bratu burgers bl bl_gravity)
P="${PROBLEMS[$SLURM_ARRAY_TASK_ID]}"
python experiments/four_method_tables.py --problems "$P" --passes primary     --out-dir "$B/four_method_jobs/${P}_gpu"
