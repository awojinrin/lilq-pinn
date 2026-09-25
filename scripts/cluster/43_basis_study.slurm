#!/bin/bash
# Task B6 / Section 3.6 (untimed): the Burgers basis study, 9 bases to K_max = 50, CPU.
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-B6
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python experiments/run_burgers_basis_comparison.py --table3 --output-dir "$B/basis_study"
