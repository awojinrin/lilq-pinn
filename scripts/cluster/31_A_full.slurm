#!/bin/bash
# Component A full runs: select the top 3 of one family by training loss,
# run them with seeds 0-4 at 60 minutes each (15 h + evaluation), and write
# the representative. Check the gpu partition's time limit (sinfo) allows it.
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-A-full
#SBATCH --array=0-1
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail

FAMILIES=(F1 F2)
F="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
python experiments/component_a.py select --family "$F" --root "$A"
python experiments/component_a.py full --family "$F" --device cuda --root "$A"
