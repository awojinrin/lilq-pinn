#!/bin/bash
# Component A: the float32 sensitivity run (F1's representative, Adam in
# float32, L-BFGS in float64, seed 0). One hour of budget. (Check A1 runs
# in 29_A1_gate, before any Component A run: Addendum v2.2 Section 2.6.)
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-A-f32
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python experiments/component_a.py float32 --device cuda --root "$A"
