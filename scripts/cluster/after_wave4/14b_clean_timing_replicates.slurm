#!/bin/bash
# Wave 4 fix, option B: two more independent clean timings of every quoted
# CPU time, with the same code (17b3539) and protocol as job 13a (each run: a
# warm-up, then the timed run with diagnostics off, the median of five under
# 1 s), each replicate in its own process. With 13a's, every quoted CPU time
# then has three independent clean timings, and option B reports their
# median. Into results/wave4/fixes/option_b_replicate_{1,2}/; nothing of wave
# 4 is changed. About 50 minutes (Beltrami's two solves dominate).
# Submitted by hand on 3 October 2026, from outside the code bundle (so that the
# wave 4 lock held); from the repo: LILQ_WAVE=4 bash scripts/cluster/sbatch.sh
# scripts/cluster/after_wave4/14b_clean_timing_replicates.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-fix-b-replicates
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
for rep in 1 2; do
    python experiments/clean_timing.py --devices cpu \
        --benchmarks bratu burgers bl bl_gravity elasticity kovasznay beltrami beltrami_pinned darcy \
        --out-root "$PKG/fixes/option_b_replicate_$rep" \
        --logged-roots "$B" "$RESULTS/wave2/B_instrumentation" "$RESULTS/wave1/B_instrumentation"
done
