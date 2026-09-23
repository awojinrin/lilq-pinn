#!/bin/bash
# Section 3.6 (Burgers basis study, 9 bases to K_max = 50) and Section 3.7
# (Beltrami with the pressure pinned at every temporal level; the spec asks
# for its wall-clock time, so it runs alone, after the study). CPU, ~1 hour.
#SBATCH --job-name=lilq-b36-b37
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --account=142698950992
#SBATCH --output=logs/%x.%j.out
##SBATCH --exclusive

source scripts/hprc/env.sh
set -euo pipefail

python experiments/run_burgers_basis_comparison.py --table3 --output-dir "$PKG/B_instrumentation/basis_study"
python experiments/run_beltrami_pinned.py --out-dir "$PKG/B_instrumentation/beltrami_pinned"
