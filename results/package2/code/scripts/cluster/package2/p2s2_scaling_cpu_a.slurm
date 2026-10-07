#!/bin/bash
# Package 2, Stage 2, item 7 (P2-1; the advisor's instructions of 4 October 2026,
# Section 10), CPU: the scaling series on an exclusive 48-thread CPU node (check C9:
# nothing else on the node, the paper's thread setting, a warm-up before every timing).
# Kovasznay: the P series (p_d = 25 .. 50, N/P about 3) and the N series (P = 1,200,
# N/P = 3, 5, 10, 20). Each run in its own process: one untimed warm-up, one timed run
# with the diagnostics off, the peak memory, then kappa, rank and errors off the clock.
# Beltrami: the pinned configuration at 7,984 and 13,764 coefficients (the paper's size,
# and N_vel = 7 on a 10^4 grid); the largest size (22,288) is p2s2_scaling_cpu_b.
# Expected about 1.5 h (laptop: Kovasznay 26 min, Beltrami 6 12 min, Beltrami 7 about 50 min on
# Grace from the paper's 267 s scaled as P^3).
# Results in results/package2_stage2/P2_1_scaling.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_scaling_cpu_a.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-scaling-cpu-a
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_1_scaling.py series --series kovasznay_P kovasznay_N --device cpu --out "$PKG"
python experiments/p2_1_scaling.py series --series beltrami --sizes 6 7 --device cpu --out "$PKG"
