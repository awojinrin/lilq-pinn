#!/bin/bash
# Timing reruns of Beltrami and Darcy on the CPU (the advisor's reply to our note
# before wave 2, 30 Sept, item 3): their paper passes with the warm-up run, so
# every reported LiL-Q time follows one rule. Run folders have wave 1's names,
# so WAVES.json lists them under overrides. Submitted in wave 2 as its own job
# (11a was already queued); this file was not in the 905e58c bundle, so the
# lock's tree hash is unchanged. About 20 minutes.
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-timing-bd
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
source scripts/cluster/env.sh
set -euo pipefail
python experiments/component_b.py --passes paper --devices cpu --out-root "$PKG" --benchmarks beltrami darcy
