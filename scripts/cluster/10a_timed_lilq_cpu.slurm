#!/bin/bash
# Timed LiL-Q work on the CPU, one exclusive CPU node (the A100 nodes' CPU
# model, no GPU surcharge; Addendum v2.2 Section 4.1), one run at a time:
#   Section 3.3 CPU runs (Bratu, Burgers, both BL, elasticity, Kovasznay,
#   Beltrami, Darcy; the paper pass, and the K_max passes: 60 iterations for
#   the scalar benchmarks and Kovasznay, 8 for Beltrami), then Section 3.7
#   (Beltrami pinned) and its K_max pass. Both Beltrami runs compute the
#   pivoted QR at every iteration, off the clock (~70 s each).
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-timed-lilq-cpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python experiments/component_b.py --devices cpu --out-root "$PKG"
python experiments/run_beltrami_pinned.py --out-dir "$B/beltrami_pinned"
python experiments/run_beltrami_pinned.py --kmax --out-dir "$B/beltrami_pinned_kmax"
