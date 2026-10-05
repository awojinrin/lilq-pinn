#!/bin/bash
# Package 2, Stage 2, item 1 (P2-3), Bratu (the advisor's instructions of 4 October
# 2026, Section 4.1): every classical method and size -- SQ-CGL, LS-hard-CGL-1.5,
# LS-weakMS-CGL-3 -- with one untimed warm-up and the median of three clean runs
# each, and the paper's LiL-Q (P = 25, 100, 225) the same day under the same
# protocol, against package1 rebuilt at v2.0.0. An exclusive CPU node, 48 threads
# (the package1 setting; check C9). Minutes. Results in results/package2_stage2.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_classical_bratu.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-bratu
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_3_classical.py reference --out "$PKG"
python experiments/p2_3_classical.py bratu --out "$PKG" --package1 "$RESULTS/package1_v2.0.0/package1"
