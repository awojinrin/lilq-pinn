#!/bin/bash
# Package 2, Stage 2, item 8 (P2-10; the advisor's instructions of 4 October 2026,
# Section 11, and his reply of 5 October, Section 3.2): two manufactured elasticity
# solutions outside the span of the basis, the compatible one (the result) and the
# specified one (the negative result), P = 50 .. 1,250, one solve each; errors,
# delta_P, their ratio, kappa and rank.
# Untimed. Seconds. Results in results/package2_stage2.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_elasticity_manufactured.slurm
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-elast-mms
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_10_elasticity_manufactured.py --out "$PKG"
