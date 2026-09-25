#!/bin/bash
# Component A: the float32 sensitivity run (F1's representative, Adam in
# float32, L-BFGS in float64, seed 0) and check A1 (the plain PINN must reach
# eps_u <= 1e-3 in the full GPU budget). Two hours of budget.
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-A-f32-a1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python experiments/component_a.py float32 --device cuda --root "$A"
python experiments/component_a.py a1 --device cuda --root "$A"
