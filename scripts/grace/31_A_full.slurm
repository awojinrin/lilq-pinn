#!/bin/bash
# Component A full runs: select the top 3 of one family by training loss,
# run them with seeds 0-4 at 60 minutes each (15 h + evaluation), and write
# the representative. Check the gpu partition's time limit (sinfo) allows it.
#SBATCH --job-name=lilq-A-full
#SBATCH --array=0-1
#SBATCH --time=22:00:00
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
F="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
python experiments/component_a.py select --family "$F" --root "$A"
python experiments/component_a.py full --family "$F" --device cuda --root "$A"
