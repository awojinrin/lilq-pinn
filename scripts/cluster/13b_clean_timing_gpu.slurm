#!/bin/bash
# Wave 4: clean timing of the Kovasznay GPU paper passes (warm-up, then a
# timed run with diagnostics off), with wave 2's logged times beside them.
# Minutes.
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-clean-gpu
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/clean_timing.py --devices cuda --benchmarks kovasznay --out-root "$PKG" \
    --logged-roots "$RESULTS/wave2/B_instrumentation"
