#!/bin/bash
# Component C (untimed): the oversampling sweep, 196 LiL-Q runs, CPU.
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-C
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python experiments/component_c.py --root "$C"
