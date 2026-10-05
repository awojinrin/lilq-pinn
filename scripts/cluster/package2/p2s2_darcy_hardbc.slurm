#!/bin/bash
# Package 2, Stage 2, item 5 (P2-15; the advisor's instructions of 4 October 2026,
# Section 8, as his reply of 5 October, Section 1.1, settles them): the paper's Darcy
# NiL network with the pressure output lifted (hard Dirichlet conditions), trained by
# Levenberg-Marquardt. One array task per (field, seed): 4 fields x 3 seeds, task t is
# field FIELDS[t / 3], seed t % 3. Each run is capped at 2,000 iterations or 30 min of
# training; with the set-up and the per-iteration delta_FV, 45 min. On FASTER (a
# shared A100, untimed): the time is reported with the hardware, not compared with
# Grace. Results in results/package2_stage2 on FASTER (P2_15_darcy_hardbc).
# Submit: through scripts/cluster/package2/submit_p2s2_faster.sh
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-darcy-hardbc
#SBATCH --array=0-11
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%A_%a.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_15_darcy_hardbc.py run --array-task "$SLURM_ARRAY_TASK_ID" --out "$PKG" --device cuda
