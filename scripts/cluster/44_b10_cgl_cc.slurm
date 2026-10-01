#!/bin/bash
# Wave 4, task B10 (Addendum v2.3): Kovasznay on CGL grids with
# Clenshaw-Curtis weights, 16 runs (P = 300, 1,200, 1,875 x N/P = 5, 10 x
# paper and kmax passes; equal weights at P = 1,875 beside them), untimed,
# with the comparison against wave 1's paper-grid runs. About 30 minutes.
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-b10
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/b10_cgl_cc.py --out-root "$PKG" --paper-grid-root "$RESULTS/wave1/B_instrumentation"
