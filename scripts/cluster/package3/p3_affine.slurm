#!/bin/bash
# Package 3, item 3 (Section 5): the certificates for elasticity (3 solutions x 5 sizes) and
# Darcy (4 fields), each re-solve in its own process at the original run's thread count (48:
# package1; 24: P2-10), then K5 (elasticity 144 points, Darcy SPE10 n_q = 6) and the summary
# against package1 and P2-10. Laptop: 22 min.
# Submit: through scripts/cluster/package3/submit_p3.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-affine
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p3_3_affine_certificates.py all --out "$PKG" --package1 "$RESULTS/package1_v2.0.0/package1" \
    --p2-10 "$RESULTS/package2_stage2/P2_10_elasticity_manufactured"
