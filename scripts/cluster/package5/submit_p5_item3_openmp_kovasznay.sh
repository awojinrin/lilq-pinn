#!/bin/bash
# Package 5, item 3: the OpenMP library of Package 1's Component A baselines (the user's decision, 17
# November 2026). Package 1's Component A processes (Kovasznay, F1 and F2) loaded the PyTorch wheel's
# OpenMP library; LiL-Q's CPU times, against which their CPU runs are compared, the GCC module's. One
# job on a whole CPU node, into results/package5_item3_openmp_kovasznay under its own commit lock:
#   p5k_openmp_kovasznay (timed-cpu, 40 min): six processes, numpy first and torch first alternating
#   over three rounds (experiments/p5_3_openmp_kovasznay.py); then
#   results/package5_item3_openmp_kovasznay.tar.gz.
# 32 SU requested (walltime x 48 SU/h), about 22 expected.
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package5/submit_p5_item3_openmp_kovasznay.sh     (checks the job; submits nothing)
#   bash scripts/cluster/package5/submit_p5_item3_openmp_kovasznay.sh
LILQ_WAVE=p5k
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package5
confirm_balance "32 SU requested, about 22 expected (1 job on a whole CPU node)"

submit omk $P/p5k_openmp_kovasznay.slurm
[[ "$DRY_RUN" == 1 ]] || echo "Submitted. When it has run (squeue -u \$USER), download results/package5_item3_openmp_kovasznay.tar.gz."
