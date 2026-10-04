#!/bin/bash
# Wave 4 fix, option A (the gravity BL P = 256 clean time): the clean timing
# of gravity BL again, on an exclusive CPU node, with the same code (17b3539)
# and protocol as job 13a: a warm-up, then the timed run with diagnostics off,
# the median of five under 1 s. All four sizes run (seconds); P = 256 is the
# one in question, the other three check consistency with 13a. Into
# results/wave4/fixes/option_a_gravity_rerun/; nothing of wave 4 is changed.
# Submitted by hand on 3 October 2026, from outside the code bundle (so that the
# wave 4 lock held); from the repo: LILQ_WAVE=4 bash scripts/cluster/sbatch.sh
# scripts/cluster/after_wave4/14a_gravity_rerun.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-fix-a-gravity
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/clean_timing.py --devices cpu --benchmarks bl_gravity \
    --out-root "$PKG/fixes/option_a_gravity_rerun" \
    --logged-roots "$B" "$RESULTS/wave2/B_instrumentation"
