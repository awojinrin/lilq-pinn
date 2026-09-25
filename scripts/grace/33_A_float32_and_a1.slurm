#!/bin/bash
# Component A: the float32 sensitivity run (F1's representative, Adam in
# float32, L-BFGS in float64, seed 0) and check A1 (the plain PINN must reach
# eps_u <= 1e-3 in the full GPU budget). Two hours of budget.
#SBATCH --job-name=lilq-A-f32-a1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=132698954494
#SBATCH --exclusive                 # the whole node: nothing else runs during timed work (Addendum Section 2)
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48          # every core of a Grace A100 node; check: sinfo -p gpu -o "%N %c %G"
#SBATCH --mem=0                     # all of the node's memory
#SBATCH --output=logs/%x.%j.out

source scripts/grace/env.sh
set -euo pipefail

python experiments/component_a.py float32 --device cuda --root "$A"
python experiments/component_a.py a1 --device cuda --root "$A"
