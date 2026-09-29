#!/bin/bash
# The end of every wave (Addendum v2.2 Section 4.2): the wave's run index,
# check B1's reproduction table, the merged four-method table, sacct, and
# results/wave<N>_report.tar.gz with everything to send the advisor
# (scripts/cluster/wave_report.py). Minutes.
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-wave-report
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python scripts/cluster/wave_report.py --wave "$LILQ_WAVE" --results "$RESULTS"
