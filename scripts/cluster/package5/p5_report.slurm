#!/bin/bash
# Package 5, items 1, 2 and 4: after the jobs (afterany). The summaries (terminal.csv and the
# tables of each item), then sacct, su_per_job.csv and results/package5.tar.gz to download
# (scripts/cluster/package2/stage_report.py). The checks C1 to C3 and the comparisons with the
# advisor's expected values need files that are not on Grace: they run at the assembly on the
# laptop. A failed summary does not stop the tarball.
# Submit: through scripts/cluster/package5/submit_p5.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5-report
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p5_1_weights.py summarize --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" || echo "ITEM 1 SUMMARY FAILED (exit $?)"
python experiments/p3_1_beltrami_certified.py summarize --out "$PKG" || echo "BELTRAMI SUMMARY FAILED (exit $?)"
python experiments/p5_2_nlls.py summarize --out "$PKG" || echo "ITEM 2 SUMMARY FAILED (exit $?)"
python experiments/p5_4_pb.py summarize --out "$PKG" || echo "ITEM 4 SUMMARY FAILED (exit $?)"
python scripts/cluster/package2/stage_report.py --stage p5 --results "$RESULTS"
