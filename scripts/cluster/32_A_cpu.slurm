#!/bin/bash
# Component A, Section 4.5: each family's representative on the node's CPU
# (all cores), seeds 0-4, full budget (5 h + evaluation).
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-A-cpu
#SBATCH --array=0-1
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail

FAMILIES=(F1 F2)
python experiments/component_a.py cpu --family "${FAMILIES[$SLURM_ARRAY_TASK_ID]}" --root "$A"
