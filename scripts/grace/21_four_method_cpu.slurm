#!/bin/bash
# Section 3.4 / task B4, CPU pass: the same three methods at the largest
# size of each benchmark, on the same node type's CPU, all its cores.
#SBATCH --job-name=lilq-b4-cpu
#SBATCH --array=0-3
#SBATCH --time=24:00:00
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

PROBLEMS=(bratu burgers bl bl_gravity)
P="${PROBLEMS[$SLURM_ARRAY_TASK_ID]}"
python experiments/four_method_tables.py --problems "$P" --passes cpu     --out-dir "$B/four_method_jobs/${P}_cpu"
