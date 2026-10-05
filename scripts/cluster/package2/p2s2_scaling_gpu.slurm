#!/bin/bash
# Package 2, Stage 2, item 7 (P2-1; the advisor's instructions of 4 October 2026,
# Section 10), A100: the scaling series on a whole A100 node, the same systems as the
# CPU job (check C9: nothing else on the node, a warm-up before every timing; GPU memory
# as in Package 1).
# Kovasznay: the P series (p_d = 25 .. 50, N/P about 3) and the N series (P = 1,200,
# N/P = 3, 5, 10, 20). Each run in its own process: one untimed warm-up, one timed run
# with the diagnostics off, the peak memory, then kappa, rank and errors off the clock.
# Batch 4 adds Beltrami where it fits in 40 GB, and sets the walltime.
# Results in results/package2_stage2/P2_1_scaling.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_scaling_gpu.slurm
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-scaling-gpu
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_1_scaling.py series --series kovasznay_P kovasznay_N --device cuda --out "$PKG"
