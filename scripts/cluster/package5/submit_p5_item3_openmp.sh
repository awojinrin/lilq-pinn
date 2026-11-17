#!/bin/bash
# Package 5, item 3: the OpenMP library diagnostic (the user's decision, 17 November 2026). The CPU
# set's network processes loaded the PyTorch wheel's OpenMP library, Package 1's the GCC module's, and
# check C1 failed on 15 of 18 runs. One job on a whole CPU node, into results/package5_item3_openmp
# under its own commit lock:
#   p5o_openmp (timed-cpu, 30 min): six processes, numpy first and torch first alternating over three
#   rounds (experiments/p5_3_openmp.py); then results/package5_item3_openmp.tar.gz.
# 24 SU requested (walltime x 48 SU/h), about 16 expected.
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package5/submit_p5_item3_openmp.sh     (checks the job; submits nothing)
#   bash scripts/cluster/package5/submit_p5_item3_openmp.sh
LILQ_WAVE=p5o
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package5
confirm_balance "24 SU requested, about 16 expected (1 job on a whole CPU node)"

submit om $P/p5o_openmp.slurm
[[ "$DRY_RUN" == 1 ]] || echo "Submitted. When it has run (squeue -u \$USER), download results/package5_item3_openmp.tar.gz."
