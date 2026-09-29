#!/bin/bash
# Timing reruns on the CPU (the advisor's reply to wave 1, item 2.2), with a
# warm-up run before every timed run: the paper passes of Bratu, Burgers, both
# BL, elasticity and Kovasznay. Their rows replace wave 1's in package1
# (WAVES.json lists them); wave 1's K_max passes, Beltrami and Darcy stand.
# Minutes.
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-timing-cpu
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
source scripts/cluster/env.sh
set -euo pipefail
python experiments/component_b.py --passes paper --devices cpu --out-root "$PKG" \
    --benchmarks bratu burgers bl bl_gravity elasticity kovasznay
