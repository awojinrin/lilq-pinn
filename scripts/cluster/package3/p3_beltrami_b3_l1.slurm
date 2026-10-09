#!/bin/bash
# Package 3, item 1: B3 (the paper's 7,984 coefficients) at level 1, 348,168 x 7,984, 4 h cap
# (Section 3.4; the run stops at 3 h 55 min so the job ends cleanly). About 2.6 h; peak about
# 105 GB. Its seconds per iteration and kappa time feed B3 level 2's launch rule.
# Submit: through scripts/cluster/package3/submit_p3.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-beltrami-b3-l1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
# capped <limit> <command...>: the run under its own wall cap (the advisor's, Section 3.4). A run
# stopped by its cap keeps what it logged (Section 8.2); the job goes on to the next run.
capped() { local cap=$1; shift; timeout "$cap" "$@" || echo "STOPPED OR FAILED (exit $?, cap $cap): $*"; }
capped 235m python experiments/p3_1_beltrami_certified.py run --size B3 --level 1 --out "$PKG"
