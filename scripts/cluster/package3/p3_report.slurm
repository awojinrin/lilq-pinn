#!/bin/bash
# Package 3: after the jobs (afterany). Item 1's summary (terminal.csv, checks K3, K4, K6, K7,
# over every Beltrami run so far: items 2-4 summarize in their own jobs), then sacct,
# su_per_job.csv and results/package3.tar.gz (the whole package folder and the Slurm logs) to
# download (scripts/cluster/package2/stage_report.py). A failed summary does not stop the tarball.
# Submit: through submit_p3.sh, and again by submit_p3_b3l2.sh after B3 level 2.
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-report
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p3_1_beltrami_certified.py summarize --out "$PKG" || echo "ITEM 1 SUMMARY FAILED (exit $?)"
python scripts/cluster/package2/stage_report.py --stage p3 --results "$RESULTS"
