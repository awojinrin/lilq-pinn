#!/bin/bash
# Package 2, Stage 2, item 1 (P2-3), Kovasznay (Section 4.2, step 2, released in the
# advisor's Stage-1 reply): the square P_N - P_{N-2} system with the corner pin at
# p_d = 10, 15, 20, 25, with one untimed warm-up and the median of three clean runs
# each; then, the same day under the same protocol, the paper's LiL-Q (P = 75 ..
# 1,875) and wave 4's Clenshaw-Curtis CGL runs at N/P = 5 (P = 300, 1,200, 1,875).
# An exclusive CPU node, 48 threads (check C9). Minutes. Results in results/package2_stage2.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_classical_kovasznay.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-kovasznay
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_3_classical.py kovasznay --out "$PKG" --package1 "$RESULTS/package1_v2.0.0/package1"
