#!/bin/bash
# Section 3.4, GPU pass: NiL-N / NiL-Q (seeds 0 1 2) and LiL-N at every
# paper size, one array task per benchmark, each writing its own directory
# (merged by 40_finalize.slurm). The longest part of Component B: hours per
# task. The CSV is rewritten after every run, so resubmitting the same
# array index after a walltime kill resumes; failures are logged, retried.
#SBATCH --job-name=lilq-b34-gpu
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --account=142698950992
#SBATCH --output=logs/%x.%A_%a.out

source scripts/hprc/env.sh
set -euo pipefail

PROBLEMS=(bratu burgers bl bl_gravity)
P="${PROBLEMS[$SLURM_ARRAY_TASK_ID]}"
python experiments/four_method_tables.py --problems "$P" --passes primary \
    --out-dir "$PKG/B_instrumentation/four_method_jobs/${P}_gpu"
