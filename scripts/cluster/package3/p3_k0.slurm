#!/bin/bash
# Package 3, check K0 (Section 7): gelsy and the column-pivoted QR on a random 600,000 x 7,984
# matrix (4.79e9 entries, above 2^32; 38 GB) with a consistent right-hand side. About 1 h
# (two factorizations at Package 2's measured 44 GFLOP/s). B3 level 2 (5.3e9 entries) is
# launched only if K0 passed (submit_p3_b3l2.sh).
# Submit: through scripts/cluster/package3/submit_p3.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-k0
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p3_k0_k8.py k0 --out "$PKG"
