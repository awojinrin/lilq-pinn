#!/bin/bash
# Package 5, Addendum 2 (16 November 2026): the 18 plateau runs, K1 and K2. One job, into
# results/package5_a2 under its own commit lock:
#   p5_a2 (cpu-mem, 1 h): parts A to C and K1 at 4 threads, part D at 24, the summary, then
#   results/package5_a2.tar.gz.
# 24 SU requested (walltime x 24 SU/h), about 10 expected.
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package5/submit_p5_a2.sh     (checks the job; submits nothing)
#   bash scripts/cluster/package5/submit_p5_a2.sh
LILQ_WAVE=p5a
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package5
confirm_balance "24 SU requested, about 10 expected (1 cpu-mem job, 24 cores)"

submit a2 $P/p5_a2.slurm
[[ "$DRY_RUN" == 1 ]] || echo "Submitted. When it has run (squeue -u \$USER), download results/package5_a2.tar.gz."
