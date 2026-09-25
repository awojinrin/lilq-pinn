#!/bin/bash
# Task B9, NiL Darcy runs in float64 (untimed, so a shared A100): one array
# task per (field, seed), 4 fields x 3 seeds. Each writes its own CSV
# (the LiL row is recomputed alongside, in seconds); 90_finalize merges them.
#SBATCH --job-name=lilq-b9-nil
#SBATCH --array=0-11
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --account=132698954494
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x.%A_%a.out

source scripts/grace/env.sh
set -euo pipefail

FIELDS=(S1 S2 S3 SPE10)
FIELD="${FIELDS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED=$((SLURM_ARRAY_TASK_ID % 3))
python experiments/darcy_fv_comparison.py --fields "$FIELD" --nil --seeds "$SEED"     --out-dir "$B/darcy_fv/${FIELD}_s${SEED}"
