#!/bin/bash
# Wave 4, Component A, F1 (the advisor's reply to wave 2, decision 1: option
# 3): no new full stage. Wave 2's finalists F1_04, F1_15 and F1_18 are kept
# (their screening and full runs copied into results/wave4 by
# submit_wave4.sh); the validation residual of every screening and full run
# is computed (validation.json beside the old run.json), and the
# representative is the lowest median validation residual: F1_18. Minutes.
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-A-f1-repick
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/component_a.py select --family F1 --keep-top F1_04 F1_15 F1_18 --device cuda --root "$A"
python experiments/component_a.py full --family F1 --device cuda --root "$A"
