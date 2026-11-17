#!/bin/bash
# Package 5, item 3, the A100 set: after the jobs (afterany). The paired-against-solo check, the
# time-to-error table, then sacct, su_per_job.csv and results/package5_item3_gpu.tar.gz.
# Submit: through scripts/cluster/package5/submit_p5_item3_gpu.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5g-report
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
# Each step under its own time limit, so that a stall cannot cost the tarball (the lesson of
# Addendum 2's first job)
timeout 5m python experiments/p5_3_networks.py pairing --out "$PKG" || echo "PAIRING FAILED OR STOPPED (exit $?)"
timeout 10m python experiments/p5_3_table.py --out "$PKG" || echo "TABLE FAILED OR STOPPED (exit $?)"
timeout 15m python scripts/cluster/package2/stage_report.py --stage p5g --results "$RESULTS" \
    || echo "REPORT FAILED OR STOPPED (exit $?)"
