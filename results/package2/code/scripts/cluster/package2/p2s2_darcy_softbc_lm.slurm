#!/bin/bash
# Package 2, Stage 2, item 5b (the advisor's reply before the Stage-2 submission,
# 5 October 2026, item 5: an addition requested after the pre-submission note): the
# control for item 5. The paper's Darcy NiL network with its Dirichlet rows (no
# lifting), trained by Levenberg-Marquardt under item 5's cap and logging. Seed 0,
# the four fields: one array task per field. Each run is capped at 2,000 iterations
# or 30 min of training; with the set-up and the per-iteration delta_FV, 45 min.
# A shared A100 on Grace, untimed: its time is not quoted. Results in
# results/package2_stage2 (P2_15b_darcy_softbc_lm).
# Submit: through scripts/cluster/package2/submit_p2s2.sh
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-darcy-softbc-lm
#SBATCH --array=0-3
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_15_darcy_hardbc.py run --variant soft --array-task "$SLURM_ARRAY_TASK_ID" --out "$PKG" --device cuda
