#!/bin/bash
# Package 2, Stage 2: after every Stage 2 job (afterany). sacct and su_per_job.csv for the
# stage's jobs (Grace's rates, as the wave reports), then results/package2_stage2.tar.gz:
# the whole stage folder and the stage's Slurm logs, to download
# (scripts/cluster/package2/stage_report.py). Minutes.
# Submit: through scripts/cluster/package2/submit_p2s2.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-report
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python scripts/cluster/package2/stage_report.py --stage p2s2 --results "$RESULTS"
