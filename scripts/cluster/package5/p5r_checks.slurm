#!/bin/bash
# Package 5, item 3, the CPU network runs again, on one whole node (48 threads): checks C2 and C4 for
# NiL-N and LM on the three problems.
# Submit: through scripts/cluster/package5/submit_p5_item3_rerun.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5r-checks
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=1800   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps the runs that finished: a run writes its records when it ends, so the one in progress is
# lost (a resubmitted job runs it again and skips the finished ones). The job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
# The OpenMP library each driver loads: the GCC module's, as Package 1's runs (DECISIONS.md, 17 November 2026)
for m in p5_3_networks p5_3_lil; do
    python -c "import sys; sys.path.insert(0, '.'); import experiments.$m; from lilq.provenance import require_openmp; require_openmp('module')"
done
python experiments/p4_checks.py provenance --out "$PKG" --name checks
# Checks C2 and C4: NiL-N and LM at the largest size, about 30 s each (sized by a pilot), without and
# with the error log
for bench in bratu burgers bl; do
    until_end python experiments/p5_3_networks.py checks --bench "$bench" --device cpu --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" --tag cpu-checks --require-openmp module
done
