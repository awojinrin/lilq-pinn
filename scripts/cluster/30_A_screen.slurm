#!/bin/bash
# Component A screening (the search itself is saved by 29_A1_gate, so the
# two array tasks never race on it): all 24 configurations of one family, seed 0, 10 minutes each (~4 h + evaluation).
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-A-screen
#SBATCH --array=0-1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail

FAMILIES=(F1 F2)
F="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
python experiments/component_a.py screen --family "$F" --device cuda --root "$A"
