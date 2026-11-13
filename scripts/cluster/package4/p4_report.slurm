#!/bin/bash
# Package 4: after the jobs (afterany). The summaries of items 1 and 2 (terminal.csv, the sweep
# tables, constants.csv, the checks), checks.json, then sacct, su_per_job.csv and
# results/package4.tar.gz to download (scripts/cluster/package2/stage_report.py). Checks M1, M6
# and M8 compare with the advisor's expected values, which are not on Grace: they are evaluated
# at the assembly on the laptop. A failed summary does not stop the tarball.
# Submit: through scripts/cluster/package4/submit_p4.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p4-report
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p4_1_burgers_gauss.py summarize --out "$PKG" --reference-dir "$RESULTS/package2_stage1/reference" || echo "ITEM 1 SUMMARY FAILED (exit $?)"
python experiments/p4_2_pb_inclusions.py summarize --out "$PKG" || echo "ITEM 2 SUMMARY FAILED (exit $?)"
python experiments/p4_checks.py collect --out "$PKG" || echo "CHECKS.JSON FAILED (exit $?)"
python scripts/cluster/package2/stage_report.py --stage p4 --results "$RESULTS"
