#!/bin/bash
# Package 3, item 2 (Section 4): Buckley-Leverett 2a and 2b, 48 runs (and the other-guess
# reruns of Section 8.2), six side by side with 4 BLAS threads each, with K5 and K6 in the
# same pool; then 2a's a priori constants and the summary. Laptop: 67 min for the sweep.
# Submit: through scripts/cluster/package3/submit_p3.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-bl
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p3_2_bl_certified.py all --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" \
    --workers 6 --threads 4 --with-checks
