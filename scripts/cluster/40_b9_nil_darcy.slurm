#!/bin/bash
# Task B9, NiL Darcy runs (untimed, so a shared A100): one array task per
# (field, seed), 4 fields x 3 seeds, each training the manuscript's network
# (Table 13, 3,555 parameters) in float64 -- the result -- and in float32 --
# the manuscript's precision, for Addendum v2.2's 20% comparison. Measured on
# the laptop GPU: ~113 min float64 + ~19 min float32 per task; a walltime kill
# resumes from the 5,000-epoch checkpoint. Each task writes its own CSV (the
# LiL row is recomputed alongside, in seconds); 90_finalize merges them.
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-b9-nil
#SBATCH --array=0-11
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail

FIELDS=(S1 S2 S3 SPE10)
FIELD="${FIELDS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED=$((SLURM_ARRAY_TASK_ID % 3))
python experiments/darcy_fv_comparison.py --fields "$FIELD" --nil --seeds "$SEED"     --out-dir "$B/darcy_fv/${FIELD}_s${SEED}"
