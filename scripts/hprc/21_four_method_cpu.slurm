#!/bin/bash
# Section 3.4, CPU pass: the same three methods at the largest size of each
# benchmark, "on the CPU with all 24 threads". One array task per benchmark;
# resumable like the GPU pass. Runs on a CPU node, not idling a GPU.
#SBATCH --job-name=lilq-b34-cpu
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --account=142698950992
#SBATCH --output=logs/%x.%A_%a.out

source scripts/hprc/env.sh
set -euo pipefail

PROBLEMS=(bratu burgers bl bl_gravity)
P="${PROBLEMS[$SLURM_ARRAY_TASK_ID]}"
# On a CPU node the "primary" device is the CPU, so select the largest size
# explicitly instead of relying on --passes cpu (which needs a GPU present).
LARGEST=(225 625 1024 1024)
python experiments/four_method_tables.py --problems "$P" --passes primary --P "${LARGEST[$SLURM_ARRAY_TASK_ID]}" \
    --out-dir "$PKG/B_instrumentation/four_method_jobs/${P}_cpu"
