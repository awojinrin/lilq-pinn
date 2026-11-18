#!/bin/bash
# Package 5, item 3, the CPU network runs again: after the jobs (afterany). The time-to-error table of
# these runs, then sacct, su_per_job.csv and results/package5_item3_cpu_rerun.tar.gz. C1 runs at the
# assembly on the laptop (Package 1's files are not on Grace).
# Submit: through scripts/cluster/package5/submit_p5_item3_rerun.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5r-report
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
# Each step under its own time limit, so that a stall cannot cost the tarball (the lesson of
# Addendum 2's first job)
timeout 10m python experiments/p5_3_table.py --out "$PKG" || echo "TABLE FAILED OR STOPPED (exit $?)"
timeout 15m python scripts/cluster/package2/stage_report.py --stage p5r --results "$RESULTS" \
    || echo "REPORT FAILED OR STOPPED (exit $?)"
