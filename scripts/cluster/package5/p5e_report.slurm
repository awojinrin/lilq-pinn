#!/bin/bash
# Package 5, item 3, the C4 diagnostic: after both jobs (afterany). The summary (its last line: A100 SET
# MAY GO: yes or NO), then sacct, su_per_job.csv and results/package5_item3_c4_diagnostic.tar.gz.
# Submit: through scripts/cluster/package5/submit_p5_item3_c4_diagnostic.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5e-report
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
# Each step under its own time limit, so that a stall cannot cost the tarball
timeout 5m python experiments/p5_3_c4_diagnostic.py summary --out "$PKG" || echo "SUMMARY FAILED OR STOPPED (exit $?)"
timeout 15m python scripts/cluster/package2/stage_report.py --stage p5e --results "$RESULTS" \
    || echo "REPORT FAILED OR STOPPED (exit $?)"
