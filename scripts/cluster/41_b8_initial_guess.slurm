#!/bin/bash
# Task B8 (untimed): Buckley-Leverett initial-guess sensitivity, one array task
# per (case, guess) -- LiL-Q, LiL-N, NiL-N and NiL-Q (seeds 0 1 2) at the four
# sizes. The networks train on the A100, as in the four-method GPU pass, so
# B8's rows are comparable with Tables 4 and 5. 90_finalize merges the CSVs.
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-b8
#SBATCH --array=0-3
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail

CASES=(viscous viscous gravity gravity)
GUESSES=(zero ic zero ic)
i=$SLURM_ARRAY_TASK_ID
python experiments/b8_initial_guess.py --cases "${CASES[$i]}" --guesses "${GUESSES[$i]}" --device cuda \
    --out-dir "$B/b8_jobs/${CASES[$i]}_${GUESSES[$i]}"
