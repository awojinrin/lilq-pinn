#!/bin/bash
# Component A screening (the search itself is saved by the preflight, so the
# two array tasks never race on it): all 24 configurations of one family, seed 0, 10 minutes each (~4 h + evaluation).
#SBATCH --job-name=lilq-A-screen
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
F="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
python experiments/component_a.py screen --family "$F" --device cuda --root "$A"
