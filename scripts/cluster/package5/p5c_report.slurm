#!/bin/bash
# Package 5, item 3, the CPU set: after the jobs (afterany). The time-to-error table, then sacct,
# su_per_job.csv and results/package5_item3_cpu.tar.gz. C1 (against Package 1) and C3 (against
# Package 4) need files that are not on Grace: they run at the assembly on the laptop.
# Submit: through scripts/cluster/package5/submit_p5_item3_cpu.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5c-report
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
# Each step under its own time limit, so that a stall cannot cost the tarball (the lesson of
# Addendum 2's first job)
timeout 10m python experiments/p5_3_table.py --out "$PKG" || echo "TABLE FAILED OR STOPPED (exit $?)"
timeout 15m python scripts/cluster/package2/stage_report.py --stage p5c --results "$RESULTS" \
    || echo "REPORT FAILED OR STOPPED (exit $?)"
