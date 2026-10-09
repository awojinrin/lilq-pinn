#!/bin/bash
# Package 3, Addendum 1 (item 4): Burgers LiL-Q at P = 900, 1,024, 1,225 and the P = 625 control
# (1,600 only if none reaches 1.6e-5): the logged passes, then the clean timings (an exclusive
# node, 48 threads; warm-up, median of three) and checks L1-L4. Cap 50 SU.
# Submit: through scripts/cluster/package3/submit_p3.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-burgers-large-p
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p3_4_burgers_large_P.py all --out "$PKG" --reference-dir "$RESULTS/package2_stage1/reference" \
    --p2-12 "$RESULTS/package2_stage1/P2_12_reference_errors"
