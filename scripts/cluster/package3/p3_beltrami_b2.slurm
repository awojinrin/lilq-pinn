#!/bin/bash
# Package 3, item 1: B2 (3,171 coefficients) at levels 1 and 2, 2 h cap each (Section 3.4);
# kappa by SVD (P < 3,200). About 20 and 45 min.
# Submit: through scripts/cluster/package3/submit_p3.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-beltrami-b2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
# capped <limit> <command...>: the run under its own wall cap (the advisor's, Section 3.4). A run
# stopped by its cap keeps what it logged (Section 8.2); the job goes on to the next run.
capped() { local cap=$1; shift; timeout "$cap" "$@" || echo "STOPPED OR FAILED (exit $?, cap $cap): $*"; }
B=experiments/p3_1_beltrami_certified.py
capped 2h python $B run --size B2 --level 1 --out "$PKG"
capped 2h python $B run --size B2 --level 2 --out "$PKG"
