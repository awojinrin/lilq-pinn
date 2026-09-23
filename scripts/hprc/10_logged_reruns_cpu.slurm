#!/bin/bash
# Section 3.3, CPU: every LiL-Q configuration of the paper (both passes),
# Beltrami, elasticity and Darcy included -- 52 runs, about 30 minutes.
# 24 cores = the paper's thread count. Resubmitting resumes where it stopped.
#
# Timings: Section 2 wants nothing else running on the machine. Uncomment
# --exclusive to reserve the whole node for this short job (charged for
# every core of the node for its duration); without it, other users' jobs
# may share the node and add timing noise, but not change any result.
#SBATCH --job-name=lilq-b33-cpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --account=142698950992
#SBATCH --output=logs/%x.%j.out
##SBATCH --exclusive

source scripts/hprc/env.sh
set -euo pipefail

python experiments/component_b.py --devices cpu --out-root "$PKG"
