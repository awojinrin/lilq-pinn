#!/bin/bash
# Package 5, item 2: one run re-timed (the user's decision, 14 November 2026). In the timed job of
# 14 November, the job's first PyTorch L-BFGS pass (Burgers p = 10) carried a one-time cost of about
# 29 s inside its only timed pass. One job on a whole node, into results/package5_retime under its
# own commit lock:
#   p5_retime (timed-cpu, 15 min): the run, after an untimed warm-up; then results/package5_retime.tar.gz.
# 12 SU requested (walltime x 48 SU/h), about 5 expected.
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package5/submit_p5_retime.sh     (checks the job; submits nothing)
#   bash scripts/cluster/package5/submit_p5_retime.sh
LILQ_WAVE=p5t
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package5
confirm_balance "12 SU requested, about 5 expected (1 job on a whole node)"

submit rt $P/p5_retime.slurm
[[ "$DRY_RUN" == 1 ]] || echo "Submitted. When it has run (squeue -u \$USER), download results/package5_retime.tar.gz."
