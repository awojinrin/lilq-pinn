#!/bin/bash
# Task B9, NiL Darcy runs in float64 (untimed, so a shared A100): one array
# task per (field, seed), 4 fields x 3 seeds. Each writes its own CSV
# (the LiL row is recomputed alongside, in seconds); 90_finalize merges them.
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-b9-nil
#SBATCH --array=0-11
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail

FIELDS=(S1 S2 S3 SPE10)
FIELD="${FIELDS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED=$((SLURM_ARRAY_TASK_ID % 3))
python experiments/darcy_fv_comparison.py --fields "$FIELD" --nil --seeds "$SEED"     --out-dir "$B/darcy_fv/${FIELD}_s${SEED}"
