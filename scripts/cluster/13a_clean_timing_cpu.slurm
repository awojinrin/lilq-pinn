#!/bin/bash
# Wave 4: clean timing on the CPU (the advisor's reply to wave 2, Section 3,
# and his follow-up of 1 October): first the logged paper pass of gravity BL
# at P = 64 again, now with K_max = 60 (its monitor values; wave 2's ended at
# K_max = 20 without its target); then, for every quoted LiL-Q time, a warm-up
# and a timed run with diagnostics off -- Bratu, Burgers, both BL,
# elasticity, Kovasznay (CPU), Beltrami, the pinned Beltrami run and Darcy --
# with the logged times beside them (wave 4's gravity rerun, wave 2's runs,
# wave 1's pinned run). About 30-40 minutes.
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-clean-cpu
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/component_b.py --passes paper --benchmarks bl_gravity --configs P64 --devices cpu --out-root "$PKG"
python experiments/clean_timing.py --devices cpu --out-root "$PKG" \
    --benchmarks bratu burgers bl bl_gravity elasticity kovasznay beltrami beltrami_pinned darcy \
    --logged-roots "$B" "$RESULTS/wave2/B_instrumentation" "$RESULTS/wave1/B_instrumentation"
