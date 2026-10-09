#!/bin/bash
# Package 3, item 1: B3 at level 2, 665,331 x 7,984 (42.5 GB; peak about 150-200 GB), 7 h cap
# (Section 3.4; the run stops at 6 h 55 min). Wave 2: submitted by submit_p3_b3l2.sh only if
# K0 passed, the launch rule holds and the budget guard holds (b3l2_gate.py).
# Submit: through scripts/cluster/package3/submit_p3_b3l2.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-beltrami-b3-l2
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
# capped <limit> <command...>: the run under its own wall cap (the advisor's, Section 3.4). A run
# stopped by its cap keeps what it logged (Section 8.2); the job goes on to the next run.
capped() { local cap=$1; shift; timeout "$cap" "$@" || echo "STOPPED OR FAILED (exit $?, cap $cap): $*"; }
capped 415m python experiments/p3_1_beltrami_certified.py run --size B3 --level 2 --out "$PKG"
