#!/bin/bash
# Package 2, Stage 2, item 1 (P2-3), Burgers (Section 4.3; optional, run because item 1
# stays below 250 SU): square space-time Chebyshev collocation at p = 7, 12, 17, 22, 27
# (30-650 free coefficients), with one untimed warm-up and the median of three clean
# runs each, and the error against the Cole-Hopf reference; then, the same day under
# the same protocol, the paper's LiL-Q Burgers runs (P = 25 .. 625) for the time ratio
# only (their errors are Stage 1's). An exclusive CPU node, 48 threads (check C9).
# Seconds on a laptop. Results in results/package2_stage2.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_classical_burgers.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-burgers
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
[[ -f "$PKG/reference/burgers_cole_hopf.npz" ]] || python experiments/p2_references.py burgers --out "$PKG"
python experiments/p2_3_classical.py burgers --out "$PKG" --package1 "$RESULTS/package1_v2.0.0/package1" \
    --reference-dir "$PKG/reference"
