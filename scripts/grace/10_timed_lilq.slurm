#!/bin/bash
# Timed LiL-Q work, one exclusive A100 node, one run at a time:
#   Section 3.3 CPU runs (Bratu, Burgers, both BL, elasticity, Kovasznay,
#   Beltrami, Darcy; both passes where the spec has them),
#   Section 3.3 Kovasznay GPU runs, check B3 with the gelsy/gels/GPU-QR
#   timings, and Section 3.7 (Beltrami pinned) -- its wall-clock.
#SBATCH --job-name=lilq-timed-lilq
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --account=132698954494
#SBATCH --exclusive                 # the whole node: nothing else runs during timed work (Addendum Section 2)
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48          # every core of a Grace A100 node; check: sinfo -p gpu -o "%N %c %G"
#SBATCH --mem=0                     # all of the node's memory
#SBATCH --output=logs/%x.%j.out

source scripts/grace/env.sh
set -euo pipefail

python experiments/component_b.py --devices cpu --out-root "$PKG"
python experiments/component_b.py --benchmarks kovasznay --devices cuda --out-root "$PKG"
python experiments/component_b.py --gpu-equivalence --out-root "$PKG"
python experiments/run_beltrami_pinned.py --out-dir "$B/beltrami_pinned"
