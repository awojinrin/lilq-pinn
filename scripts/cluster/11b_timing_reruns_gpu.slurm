#!/bin/bash
# Timing reruns on the A100 (the advisor's reply to wave 1, item 2.2), with a
# warm-up run before every timed run: the Kovasznay GPU paper passes at all
# five sizes, then check B3 (its GPU/CPU timings on this node). They replace
# wave 1's in package1 (WAVES.json lists them). Minutes.
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-timing-gpu
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
source scripts/cluster/env.sh
set -euo pipefail
python experiments/component_b.py --passes paper --benchmarks kovasznay --devices cuda --out-root "$PKG"
python experiments/component_b.py --gpu-equivalence --out-root "$PKG"
