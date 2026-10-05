#!/bin/bash
# Package 2, Stage 2, item 7 (P2-1; the advisor's instructions of 4 October 2026,
# Section 10), CPU, the largest Beltrami size alone: N_vel = 8, N_p = 10 (22,288
# coefficients) on an 11^4 interior grid, the pinned configuration's rule for the
# boundary rows and pins. One untimed warm-up and one timed run (1.6-2.2 h each, scaled
# as P^3 to P^3.3 from the paper's 267 s), then kappa and rank off the clock. Exclusive
# 48-thread node (check C9). Skip rule of Section 1: dropped if this one job would
# exceed 300 SU; 6 h x 48 SU/h = 288 SU, so it runs.
# Results in results/package2_stage2/P2_1_scaling.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_scaling_cpu_b.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-scaling-cpu-b
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_1_scaling.py series --series beltrami --sizes 8 --device cpu --out "$PKG"
