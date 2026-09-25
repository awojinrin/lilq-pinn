#!/bin/bash
# Component A, Section 4.5: each family's representative on the node's CPU
# (all cores), seeds 0-4, full budget (5 h + evaluation).
#SBATCH --job-name=lilq-A-cpu
#SBATCH --array=0-1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --account=132698954494
#SBATCH --exclusive                 # the whole node: nothing else runs during timed work (Addendum Section 2)
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48          # every core of a Grace A100 node; check: sinfo -p gpu -o "%N %c %G"
#SBATCH --mem=0                     # all of the node's memory
#SBATCH --output=logs/%x.%A_%a.out

source scripts/grace/env.sh
set -euo pipefail

FAMILIES=(F1 F2)
python experiments/component_a.py cpu --family "${FAMILIES[$SLURM_ARRAY_TASK_ID]}" --root "$A"
