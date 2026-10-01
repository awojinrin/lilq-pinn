#!/bin/bash
# Wave 4: B8's LiL-Q rows that ended on iteration_cap in wave 3 (three,
# gravity BL), rerun on the CPU with K_max = 60 (the advisor's follow-up of
# 1 October, item 1), into their own table beside wave 3's. Under a minute.
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-b8-kmax60
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/b8_initial_guess.py --rerun-capped-from "$RESULTS"/wave3/B_instrumentation/b8_jobs/*/b8_initial_guess.csv \
    --out-dir "$B/b8_jobs/lilq_kmax60"
